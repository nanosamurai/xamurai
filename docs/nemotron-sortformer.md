# Optional Nemotron speaker diarization and enrollment

This is a local Phase 2b validation extension, stacked on
`validate-nemotron-realtime`. Keep the Nemotron implementation PR and this
diarization PR separate. The implementation uses the existing RealtimeASR API,
native recognizer, process-local admission and Compose DNS routing.

## Enable locally

Build `nemotron_rtservice/Dockerfile` from `codex/add-optional-sortformer`.
The native build uses two compiler jobs to fit local Docker Desktop memory.
The same image supports three startup configurations:

| Configuration | Result |
| --- | --- |
| `NEMOTRON_DIARIZATION=false` (default) | Original Nemotron ASR profile; neither speaker model is downloaded or loaded |
| `NEMOTRON_DIARIZATION=true`, `ENROLL_BACKEND=disabled` (default) | Native Sortformer; anonymous `SPEAKER_00` through `SPEAKER_03` finals |
| `NEMOTRON_DIARIZATION=true`, `ENROLL_BACKEND=s3_manifest` | Sortformer plus tenant-specific enrolled names where matching is confident |

These configurations have distinct immutable profile IDs. Capabilities advertise
speaker labels and segment timestamps only with Sortformer enabled. The public
event has no word-timing field, so `word_timestamps` remains false.

The operator sets this at process startup; clients cannot select models, URLs,
revisions, thresholds, or runtime options. Configuration changes require
recreating the replicas. The `nanosamurai/docker-compose.nemotron.yml` override
maps `NEMOTRON_ENROLL_BACKEND` to the service's `ENROLL_BACKEND` and supplies
the existing LocalStack enrollment bucket settings.

## Models and runtime

The ASR model and NeMo-Speech.cpp revision are unchanged. Enable the existing
native diarization build option; Python requests native speaker-tagged words
and joins consecutive words with the same speaker into FINAL events. Every
admitted native stream owns its Sortformer state. We do not restart it at ASR
utterance boundaries, or share speaker-slot identity across streams/replicas.
Word ends are clipped at the next speaker's onset so late RNNT punctuation
cannot include the next turn in enrollment audio.

| Stage | Fixed artifact | Revision | SHA-256 |
| --- | --- | --- | --- |
| Diarization | `nvidia/diar_streaming_sortformer_4spk-v2`, `diar_streaming_sortformer_4spk-v2.q8_0.gguf` | `5240a64075176943f677d30fa2171c780229f341` | `0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a` |
| Enrollment embedding | `Wespeaker/wespeaker-voxceleb-resnet34-LM`, `voxceleb_resnet34_LM.onnx` | `f0c48c298fd835726c27956a5d617bad7115627e` | `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068` |

Sortformer v2 is the model explicitly supported by the already pinned native
runtime. Adopting another checkpoint, including v2.1, needs its own parity
qualification. Both listed models are published under CC BY 4.0; attribution
is retained in NOTICE. No Hugging Face token is required for these artifacts.

WeSpeaker runs with CPU-only ONNX Runtime and two inference threads. Its
Kaldi-compatible frontend uses 80 filterbank bins, a 25 ms Hamming window,
10 ms stride, no dither and mean normalization, with PCM16 amplitude scale.
The same encoder and preprocessing compute enrollment and live embeddings.
Existing S3 WAV samples remain the source; pyannote embeddings are not reused
because they are a different embedding space. Neither PyTorch nor NeMo Python
is added to this service.

## Four speakers and the S3 gallery

Sortformer has four output channels for speakers in **one continuous audio
stream**, not four enrollment records. It does not ingest a named enrollment
gallery. Never load the first four S3 records into those slots or use record
order as identity.

The service reads the existing layout:

```text
s3://<bucket>/<prefix>/<tenant-id>/speakers/<speaker-id>/speaker.json
s3://<bucket>/<prefix>/<tenant-id>/speakers/<speaker-id>/samples/<sample-id>.wav
```

For each final, sufficiently long speaker-turn audio is compared with every
usable name in that tenant's gallery. Mean normalized sample embeddings form
each candidate. Default raw cosine similarity must be at least `0.65`
(`ENROLL_SIM_THRESHOLD`), and exceed the runner-up by `0.10`
(`NEMOTRON_ENROLL_MATCH_MARGIN`). These are initial local thresholds requiring
calibration on consented held-out recordings. Short, unknown, low-confidence,
or ambiguous matches retain an anonymous slot. Duplicate names in a gallery
are excluded. A match is not permanently attached to a slot: each final is
matched from its current audio, reducing propagation of an earlier wrong name.
Names are transcript annotations, never authentication or authorization.

More than four distinct speakers in a continuous stream is outside this
Sortformer profile's supported scope, even if they speak sequentially. It may
merge or reuse channels, and does not reliably signal a fifth speaker. A
larger enrollment gallery does not remove this limit. Use the existing
pyannote-based path for meetings expected to exceed four speakers. Splitting
the stream or selecting four enrolled names would not solve general identity
tracking and is not implemented.

## Bounds, isolation and failure behavior

- Each replica has a locked TTL/LRU enrollment cache (300 seconds, 32 tenants).
  No speaker state or capacity database is shared between replicas.
- The complete usable tenant gallery is searched, capped independently at
  256 speaker records and five samples per record. An oversized gallery falls
  back to anonymous labels; it is never silently truncated to four people.
- Manifests are limited to 64 KiB, samples to 4 MiB and 30 seconds of mono/stereo
  WAV at 8–48 kHz. Embedding uses at most ten seconds, with a minimum of 1.5
  seconds; resampling is local. Enrollment-enabled streams retain at most
  64 seconds of PCM16 (2 MiB), covering the 30-second native utterance backstop
  plus the maximum ingress chunk. The buffer is released on stream teardown.
- Tenant identifiers are fixed by the first audio chunk and must agree with
  `x-tenant-id` when supplied. Invalid tenant identifiers cannot reach S3.
  Only samples in the configured bucket and the manifest's own tenant/speaker
  prefix are read. File, HTTP and cross-tenant URLs are rejected.
- S3 uses bounded reads and short network timeouts. Cache refresh and embedding
  are serialized within each replica. A lookup has a 15-second work budget,
  checked between S3/embedding operations, and stops on stream cancellation;
  an in-flight operation may finish after that budget. Cold gallery loading adds final latency;
  large galleries need separate latency qualification.
- Enrollment failures preserve anonymous diarization. Missing or inconsistent
  native word coverage preserves the complete coarse, speakerless transcript.
  Native ASR/Sortformer errors fail that stream through the existing sanitized
  INTERNAL response and release its permit. A requested model that cannot load
  prevents readiness; the service does not advertise unavailable diarization.
- No transcript text, names, audio, sample URLs or credentials are logged by
  the adapter. Models are fixed data artifacts and verified before loading.
  Provider ports remain Docker-network internal. Use workload identity and
  read-only enrollment access outside the local LocalStack setup.

The pinned native diarizer bounds its acoustic cache, but its historical
probability/segment bookkeeping is not a proven strict constant-memory bound
for arbitrarily long streams. This local integration does not claim production
capacity, >4-speaker correctness, T4 qualification or lossless pod migration.

## Validation

Run the lightweight native, speaker and gRPC tests:

```bash
pytest -q tests/test_nemotron_native_unit.py tests/test_nemotron_speakers_unit.py \
  tests/test_nemotron_rtservice_integration.py tests/test_qwen_rtservice_integration.py
```

The internal `nemotron_rtservice.probe` supports `--require-speakers`,
`--tenant-id`, `--require-enrolled` and `--concurrent-audio`. With two one-session
replicas, the latter holds both admissions before sending fixture audio to both
processes and requires non-empty finals from both. Counts are printed without
transcripts or enrolled names. Real-GPU results belong in the Phase 2b plan;
unit tests alone do not establish diarization accuracy.

The real CPU embedding smoke on two non-overlapping ten-second halves of
`test_cs.wav` returned a 256-dimensional embedding and cosine similarity
`0.8153`. This verifies the encoder/frontend wiring for one consented speaker;
it does not establish multi-speaker accuracy or calibrate the threshold.

References: [Sortformer model](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
[pinned native API](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/include/nemo_speech/asr.h),
[official WeSpeaker model](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM).
