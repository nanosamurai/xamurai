# Modular realtime ASR providers

`rtservice` remains the application-facing realtime gateway. SamuraiBFF still
uses the existing bidirectional `RealtimeASR.Stream` RPC and receives the same
`AsrEvent` PARTIAL/FINAL data model. Model runtimes sit behind the internal
`nanosamurai.speech.v1.SpeechProvider` contract.

## Fixed provider profiles

Profiles are process-start allowlist entries, not request-time model
configuration. A stream may select only a registered profile via
`x-rt-provider-profile`; the normal deployment selects its default with
`RT_PROVIDER_PROFILE`.

| Profile | Runtime | Mode | Timing claims |
| --- | --- | --- | --- |
| `faster-whisper-medium-ctranslate2-r1` | In-process Faster-Whisper 1.2 / CTranslate2 4.6, model revision `08e178d48790749d25932bbc082711ddcfdfbc4f` | Existing windowed realtime path with VAD, diarization, partials, finals, and enrollment mapping | Existing segment/window behavior |
| `qwen3-asr-0.6b-vllm-r1` | Isolated Qwen3-ASR 0.6B provider using `qwen-asr==0.0.6` and `vllm==0.14.0`, model revision `c4468bdb552ddc559e464f6081e22dd4034f2e68` | Native stateful stream, one concurrent session in the validation profile | Cumulative sample range only; no word or segment timestamps |

Both profiles record the model weight SHA-256 in provider provenance. The Qwen
container verifies its pinned `model.safetensors` digest before loading vLLM.
No RPC accepts an arbitrary model ID, revision, URL, or runtime argument.

## Internal contract

The internal gRPC service exposes:

- capability and provenance discovery;
- a bounded window RPC for window-oriented providers;
- a bidirectional native-streaming RPC with request, session, and contiguous
  provider sequence identities;
- readiness.

All audio coordinates are integer sample indexes. Native Qwen hypotheses cover
`[0, samples_received)` and replace the prior hypothesis for that range. A
flush emits a FINAL candidate. rtservice converts those coordinates to the
existing `AsrEvent` seconds only at the public boundary. It does not synthesize
word timestamps, segment timestamps, or speaker labels.

Natural EOF on the public `RealtimeASR.Stream` request flushes each active
provider session before rtservice closes it. A canceled public stream instead
cancels the internal RPC. Provider-side client cancellation is normal teardown:
it releases the single-session permit without emitting a synthetic inference
error, so the next session can start immediately.

The remote client allows at most two queued frames and waits for exactly one
acknowledgement/candidate per frame. `RT_PROVIDER_REQUEST_TIMEOUT_SECONDS` is
clamped to 0.1-120 seconds. A timeout, provider disconnect, sequence mismatch,
or inference error cancels that provider stream and ends only the corresponding
public stream with an appropriate gRPC status. There is no silent fallback to a
different model. W3C trace metadata is propagated to the internal RPC.

## Qwen native streaming details

The Qwen provider uses `Qwen3ASRModel.LLM(...)`, initializes one official
streaming state per RPC, passes incoming PCM16 samples through
`streaming_transcribe`, and calls `finish_streaming_transcribe` on flush. The
official implementation buffers the configured chunk, re-feeds accumulated
audio, and applies token-prefix rollback. “Native streaming” here means the
model's supported vLLM streaming API, not a claim that every step has constant
cost or reuses an acoustic KV cache.

Configuration is deliberately small and bounded:

| Variable | Default | Allowed range |
| --- | ---: | ---: |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.65` | `0.1`-`0.95` |
| `QWEN_STREAM_CHUNK_SECONDS` | `2.0` | `0.5`-`10.0` |
| `QWEN_MAX_NEW_TOKENS` | `256` | `16`-`1024` |
| `QWEN_MAX_AUDIO_SECONDS` | `300` | `1`-`1800` |

The provider requires CUDA and fails startup clearly when it is unavailable.
It binds only inside its workload network in Compose; no provider port is
published on the host. Audio and transcript contents are not logged. Hugging
Face and vLLM telemetry are disabled in the image and Compose configuration.

## Validation

`tests/test_speech_provider_integration.py` starts real in-process gRPC servers
and verifies the unchanged public stream through the internal native stream.
`tests/test_qwen_provider_integration.py` exercises the concrete Qwen provider
servicer with a fake backend, including capabilities, pinned provenance,
cumulative candidates, flush, and cancellation recovery. The lightweight suite
passed 46 tests with 11 integration tests deselected; the real Faster-Whisper
gRPC suite passed four tests with one deselected.

Real GPU/model validation used the nanosamurai Compose Qwen override on an RTX
5090 Laptop GPU. A cold start including the pinned public model download took
340.7 seconds. Reusing both the model volume and preserved provider container
reached Compose readiness in 33.6 seconds. Two consecutive 12-second Czech
fixture sessions traversed BFF -> rtservice -> this provider. Each emitted six
partials and one final; the first partial arrived at 12.11 seconds and the final
at 12.15 and 12.14 seconds respectively. rtservice completed the streams in
13.856 and 13.884 seconds without provider/inference failures. The smoke runner
reported event keys and timing only and did not print transcript content.

Post-run device use was 16,687 MiB of 24,463 MiB total. Container memory was
3.853 GiB for the provider, 277.3 MiB for rtservice, and 411.3 MiB for BFF. The
provider image was 14,399,061,895 bytes and ran as UID 10002; the provider port
remained unpublished.

vLLM 0.14 emits tokenizer regex warnings during its independent tokenizer loads
even though the qwen-asr wrapper constructs its processor with the upstream fix
flag. It also makes two non-fatal safetensors metadata lookups against the
pinned local snapshot path before loading the local shard successfully. The
validation deliberately leaves these upstream diagnostics visible instead of
patching unsupported runtime internals; re-evaluate them when either dependency
is upgraded.
