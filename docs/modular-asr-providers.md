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
cumulative candidates, and flush. Real GPU/model validation is performed from
the nanosamurai Compose Qwen override so the BFF-facing path is also covered.
