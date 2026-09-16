# Parakeet final track

The `parakeet` final track runs NVIDIA Parakeet TDT 0.6B v3 Q8 and embedded
Sortformer v2 through the same pinned NeMo-Speech.cpp C ABI as Nemotron.
It reuses the existing finalizer Kafka/storage loop and `SessionTranscript`.
No additional ASR server, Python ML framework, alignment model, topic or database
schema is needed. Inference is sequential within each worker; model weights stay
loaded between recordings and native recognition creates fresh speaker state.

Build from Xamurai:

```sh
docker build --target parakeet-finalizer -t xamurai-parakeet-finalizer:local -f nemotron_rtservice/Dockerfile .
```

The named Docker target shares the native build; the default target remains the
Nemotron realtime service. Run `python -m finalizer_worker.parakeet` to consume
`recordings.finished`. The image defaults `FINALIZER_TRACK_ID=parakeet`, with
consumer group `finalizer.parakeet`; replicas of that track share the group.
`PARAKEET_GPU=0` selects the GPU (`-1` permits CPU). Existing Kafka, recording
storage and tracing environment variables apply. Final model metadata is
`nvidia/parakeet-tdt-0.6b-v3`. Different tracks retain the same source audio;
multiple final tracks still require `store_recording=true`.

## Scope and artifacts

Input is the recorder's mono 16 kHz WAV. Parakeet detects its own language;
the session language remains metadata and does not constrain recognition.
Native word timestamps populate existing segment/word fields. Adjacent words
are grouped by speaker, splitting gaps longer than one second. Sortformer
supports at most four anonymous speakers per recording, labelled `SPEAKER_00`
through `SPEAKER_03`; zero/unassigned native tags remain unknown. This track
does not perform enrolled-speaker matching. Sortformer retains state throughout
each recording, including native ASR splits for long inputs.

| Artifact | Revision | SHA-256 |
| --- | --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3`, `parakeet-tdt-0.6b-v3.q8_0.gguf` | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | `e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e` |
| Sortformer v2 Q8 | See [the shared Sortformer pins](nemotron-sortformer.md) | Verified before loading |

NeMo-Speech.cpp remains pinned to `4f9676226f667d14608487df744f375db87127f8`.
Both model artifacts use CC BY 4.0; attribution is in NOTICE. Downloads require
no HF token, execute no model-repository code, and use immutable revisions and
content hashes. The worker runs as UID 10003, publishes no host ports and disables
HF telemetry. It publishes before committing input; inference/download failures
leave work for replay. Temporary recording downloads are removed after each job.

Native full-recording inference consumes memory proportional to recording length;
qualification on short recordings does not establish unlimited recording capacity.
Deployment memory and concurrency budgets must include both models and audio.

## Validation

`tests/test_parakeet_finalizer_integration.py` exercises real speech, word timing,
speaker labels, repeated independent recordings, silence, empty input and rejected
stereo input inside the native image. The Community Edition repository owns the
real Compose smoke, including recorder, both final tracks, Kafka, Postgres and
HTTP playback. See `nanosamurai/docs/parakeet-finalizer.md` for invocation and
the deployment repository's spike evidence. No synthetic inference is used there.
