# Modular realtime ASR services

Faster-Whisper and Qwen are peer services behind the same public
`RealtimeASR` gRPC API. SamuraiBFF owns the configured track list and fans one
accepted audio stream out to those peers. There is no
`rtservice -> SpeechProvider gRPC -> Qwen` hop.

```mermaid
flowchart LR
    subgraph bffImage["Docker image: samuraibff"]
        subgraph bffService["Compose service: samuraibff"]
            WS["WebSocket audio/events"]
            Fanout["bounded per-track fan-out"]
            KafkaOnce["publish audio.raw once"]
        end
    end

    subgraph fasterImage["Docker image: xamurai-rtservice"]
        subgraph fasterService["Compose service: rtservice"]
            FasterAPI["RealtimeASR\nGetCapabilities + Stream"]
            SpeechProvider["internal Python SpeechProvider"]
            Faster["LocalFasterWhisperProvider"]
        end
    end

    subgraph qwenImage["Docker image: xamurai-qwen-rtservice"]
        subgraph qwenService["Compose service: qwen-rtservice"]
            QwenAPI["RealtimeASR\nGetCapabilities + Stream"]
            QwenBackend["Qwen3ASRModel.LLM\nin-process vLLM API"]
        end
    end

    WS --> Fanout
    WS --> KafkaOnce
    Fanout -->|"track=faster-whisper"| FasterAPI
    Fanout -->|"track=qwen"| QwenAPI
    FasterAPI --> SpeechProvider --> Faster
    QwenAPI --> QwenBackend
```

## Common service contract

Every realtime service implements:

- `GetCapabilities(RealtimeCapabilitiesRequest) -> RealtimeCapabilities`;
- `Stream(stream AudioChunk) -> stream AsrEvent`.

Capabilities describe mode, timestamp and language support, statefulness,
sample rate, limits, runtime, and immutable model/implementation provenance.
`AsrEvent.provider_profile_id` identifies the implementation profile. The BFF
adds its operator-controlled `track` ID and `primary_track` compatibility flag
to WebSocket events; a client cannot supply a model endpoint or model ID.
Language identifiers at this public boundary use short codes (`cs`, `en`,
`id`, and so on). The Qwen adapter translates these to the language names its
official library accepts and translates detected names back to codes in both
capabilities and events.

Natural request EOF is the flush signal. A service drains accepted chunks,
emits its terminal result when available, and completes its response stream.
Cancellation is immediate teardown and does not synthesize a final result.

## Internal implementation boundaries

The Faster service retains the `SpeechProvider` Python Protocol because it
separates rolling-window/session policy from Faster-Whisper preprocessing and
decoding with little transport overhead. `LocalFasterWhisperProvider` is the
only registered profile in that process. VAD, window overlap, partial/final
state, diarization, and enrolled-speaker mapping remain in `rtservice`.

The Qwen service is already isolated by its public service/container boundary,
so it does not reimplement the removed remote `SpeechProvider` gRPC layer. Its
small `QwenBackend` Protocol isolates the public stream handling from the
official Qwen library for tests. This avoids daisy-chained gRPC services and
lets either realtime implementation be deployed, restarted, and scaled as a
peer.

## Fixed profiles

| BFF track | Provider profile | Runtime and mode | Timing claims |
| --- | --- | --- | --- |
| `faster-whisper` | `faster-whisper-medium-ctranslate2-r1` | Faster-Whisper 1.2 / CTranslate2 4.6; windowed realtime with VAD, diarization, enrollment mapping, partials, and finals | Existing window/segment behavior and word timestamps |
| `qwen` | `qwen3-asr-0.6b-vllm-r2` | `qwen-asr==0.0.6`; `vllm==0.14.0`; bounded native-streaming epochs; one concurrent session in this validation profile | Contiguous epoch ranges only; no word or segment timestamps |

The Qwen profile pins `Qwen/Qwen3-ASR-0.6B` revision
`c4468bdb552ddc559e464f6081e22dd4034f2e68` and verifies the
`model.safetensors` digest
`sha256:79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea`
before initializing inference. No RPC accepts an arbitrary model ID, revision,
URL, or runtime argument.

## Qwen/vLLM lifecycle

The Qwen container starts only `python -m qwen_rtservice.server`. That process
constructs `Qwen3ASRModel.LLM(...)`; the `qwen-asr` adapter initializes and
owns vLLM in-process. There is no separate `vllm serve` command or sidecar.

Each RPC remains one continuous public gRPC stream, but the service bounds the
official Qwen state to an epoch (120 seconds by default). At an epoch boundary
it calls `finish_streaming_transcribe`, emits a normal `FINAL` event for that
contiguous range, commits the novel text, and opens a fresh
`init_streaming_state` with a bounded tail of transcript context. The next
`PARTIAL` starts at the same sample boundary, so no audio is replayed and the
public track has neither a gap nor duplicated seam text. Request EOF flushes
the current epoch and closes the public stream normally.

The upstream adapter buffers the configured chunk, re-feeds audio accumulated
within the current epoch, and applies token-prefix rollback. “Native streaming”
describes this official API; it does not imply constant work per chunk or an
acoustic KV cache. Restarted epochs were chosen for the initial implementation
because the official API has no supported state-truncation operation. Rolling
audio overlap can be evaluated later if real seam-quality measurements justify
the duplicate-suppression complexity.

| Variable | Default | Allowed range |
| --- | ---: | ---: |
| `QWEN_STREAM_CHUNK_SECONDS` | `2.0` | `0.5`-`10.0` |
| `QWEN_STREAM_EPOCH_SECONDS` | `120.0` | `10.0`-`600.0` |
| `QWEN_EPOCH_CONTEXT_CHARACTERS` | `1000` | `0`-`8000` |
| `QWEN_MAX_NEW_TOKENS` | `1024` | `16`-`1024` |
| `QWEN_MAX_MODEL_LEN` | `4096` | `2048`-`65536` |
| `QWEN_KV_CACHE_MIB` | `512` | `512`-`16384` |

`QWEN_MAX_MODEL_LEN` must cover the configured epoch's estimated audio tokens,
the bounded context tail, output tokens, and safety margin. `QWEN_KV_CACHE_MIB`
must cover that context using the pinned model's KV geometry. The service
rejects incompatible settings at startup. This replaces vLLM's percentage-of-
GPU allocation, which reserved roughly 10 GiB for the earlier 0.65 setting on
the 24 GiB development GPU despite the 0.6B model size.

The profile advertises `maximum_audio_seconds=0`, meaning the public stream has
no provider-imposed duration cutoff. It still advertises the real one-session
concurrency limit and lack of timestamps. Epoch duration is an internal compute
bound, not a recording limit.

The service requires CUDA, runs as UID 10002, disables Hugging Face/vLLM
telemetry, and does not log audio or transcript contents. In the Compose
evaluation stack its gRPC port remains network-internal.

## Failure isolation and BFF orchestration

`SAMURAIBFF_GRPC_REALTIME_TRACKS` is an operator-controlled, maximum-four
allowlist in the form `track-id=host:port,...`. Each track has its own bounded
queue, gRPC stream, capability handshake, cancellation, and completion state.
A full or failed track is canceled without closing its healthy peers or the
browser session. The first configured track is the compatibility primary.

The BFF publishes every accepted chunk to `audio.raw` once, independently of
the number of realtime tracks. The existing Kafka refinement, recording, and
finalization path therefore remains unchanged and is not duplicated per model.

## Validation

The lightweight Xamurai suite covers both real localhost gRPC service
boundaries with fake model backends, including capabilities, native
partial/final delivery, contiguous sequencing, cancellation permit recovery,
safe inference failure, subsequent-session recovery, and a synthetic stream
longer than 300 seconds that crosses more than 30 bounded epochs without
duplicate or discontinuous public segments. BFF integration tests
send one WebSocket audio stream to two fake `RealtimeASR` peers and prove that
all accepted chunks, EOF, and track-labelled terminal events reach both.

The earlier single-Qwen Compose prototype validated the pinned runtime on an
RTX 5090 Laptop GPU before the peer-service refactor. Cold readiness was 340.7
seconds and preserved-cache readiness 33.6 seconds; two consecutive 12-second
fixture sessions each emitted six partials and one final. Those measurements
remain model/runtime evidence, while the peer dual-track topology requires its
own final Compose result before Phase 2a is closed.
