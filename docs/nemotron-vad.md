# Native VAD endpointing

Nemotron uses the pinned NeMo-Speech.cpp runtime's Silero VAD to detect silence,
instead of the absence of decoder tokens. The recognizer still receives every
audio chunk: `enable_masking=false`, with fixed onset `0.5` and offset `0.3`.
Each native stream owns its VAD state. There is no Python VAD inference, audio
replay, overlap, extra service or batch transcription pass.

The deployment default remains `NEMOTRON_ENDPOINTING_SILENCE_MS=2000` (integer
`1`–`30000`), with the existing per-session `endpointing_silence_ms` override.
The ASR-only profile is `nemotron-3.5-asr-streaming-0.6b-nemo-speech-cpp-q8-r3`;
Sortformer and enrolled profiles are `sortformer-q8-r4` and
`sortformer-enrolled-q8-r4`. Capabilities identify `silero:6.2.0;duration-endpointing:1` in the
implementation revision. Speaker capabilities and public events are unchanged.

This addresses premature endpoints caused by missing decoder tokens during
speech. VAD can still misclassify quiet speech or noise. Native endpoint
flush/reset is unchanged; the emergency boundary can still split a word.

## Prefer a pause before the emergency limit

The duration policy runs in the native endpointer on the decoded audio clock:

| Startup environment variable | Default | Valid integers |
| --- | --- | --- |
| `NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS` | `90` | `1` through maximum minus one |
| `NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS` | `500` | `1`–`30000` |
| `NEMOTRON_MAX_UTTERANCE_SECONDS` | `120` | `2`–`3600` |

Before the soft age, honor the normal session silence timeout. At or after it,
accept the smaller of that timeout and the soft silence interval. If no suitable
pause occurs, endpoint at the maximum, even during speech. These three settings
are instance-only; recreate the service to change them. Invalid combinations
fail startup before loading models. Lowering the maximum below 91 seconds also
requires lowering the soft age.

Elapsed time starts with the first detected speech after the previous endpoint;
leading silence does not spend that budget. Each endpoint restores the normal
silence interval for the next utterance. Timings are quantized by native chunks
and include VAD/right-context latency, so 120 seconds is not an exact wall-clock
deadline. Partials continue throughout; final speaker labels wait for an endpoint.
There is no audio replay or overlap, and no additional session setting.

The small `nemo_speech_native/patches/duration-endpointing.patch` extends the
pinned runtime's endpointer and C ABI. The library reports `0.1.0+xamurai.1`,
which the Python binding requires, so an unpatched library cannot silently ignore
the new fields. Zero-valued native fields retain upstream behavior for Parakeet.
The former Python 30-second force loop is removed.

## Artifact provenance

The shared native Docker build converts `silero-vad==6.2.0`'s full-precision
checkpoint using `convert_model.py silero` from the same NeMo-Speech.cpp commit
as the runtime: `4f9676226f667d14608487df744f375db87127f8`.
Conversion uses CPU PyTorch/Torchaudio `2.8.0+cpu`, GGUF `0.19.0`, NumPy `2.2.3`
and PyYAML `6.0.2` in a separate build stage. Only the approximately 1.24 MB GGUF
and MIT license enter the runtime image, including the shared Parakeet base;
Parakeet does not load VAD. Runtime inference uses the existing native backend.

The build verifies SHA-256
`83d702b11378ced4b1bbfcb53394bd00994f7382ff0b9a11b87605dd0898369d` for
`/opt/nemo-speech/models/silero-v6.2.0.gguf`. The path is fixed and root-owned;
an unreadable or invalid model fails recognizer startup. No model URL, threshold
or path is exposed to clients. Attribution is in `NOTICE` and the full license
ships at `/opt/nemo-speech/models/LICENSE`.

## Validation

The existing configuration test covers VAD wiring in ASR-only and Sortformer
modes, including defaults and silence overrides. The real native integration
test uses the synthetic repository WAV, 20 ms chunks, leading silence, two
speech/pause cycles and 800/2000/3000 ms timeouts. It requires finals during each
pause before EOF, no extra nonempty finals during silence/EOF, increasing
endpoint delay with the timeout, and speaker words when diarization is enabled.
Additional GPU cases check a long initial silence, short pauses before/after the
90-second transition, restoration after an endpoint, a 12/20-second override,
preservation of shorter session timeouts, and the 120-second emergency endpoint
with speech continuing afterward. The gRPC enrollment test checks that early
speaker audio survives a 170-second utterance with a configured 180-second cap.

After `docker buildx bake --load nemotron-rtservice`, run from Xamurai against
the model cache populated by Nanosamurai's Compose service:

```powershell
docker run --rm --gpus all --entrypoint /bin/sh `
  -v "${PWD}:/workspace:ro" -w /workspace `
  -v nanosamurai_nanosamurai_nemotron_hf_cache:/data/huggingface `
  xamurai-nemotron-rtservice:local -c 'pip install --target /tmp/test-deps pytest==8.3.5 && PYTHONPATH=/tmp/test-deps:$PYTHONPATH python -m pytest -q -s -p no:cacheprovider tests/test_nemotron_vad_integration.py'
```

The test adds no dependencies to the production image. For end-to-end validation,
recreate the Nemotron service with the rebuilt image and run Nanosamurai's
internal replica probe and Tier 2 BFF WebSocket smoke. Keep ports on localhost
and use only synthetic repository audio. GPU capacity and speech accuracy still
need separate qualification for production workloads.

The 2026-09-17 local RTX 5090 Laptop qualification passed both native test modes,
60 Nemotron regression tests, two concurrent Compose replicas, and BFF finals
from both realtime tracks. The full 20-second fixture followed by five seconds
of silence produced a speaker-labelled final before EOF and below the hard limit.

The 8- and 12-second excerpts followed by silence exposed a remaining native
word/text coverage mismatch: a final can include extra words in its word
metadata, causing the existing speaker conversion to preserve text as a coarse
speakerless final. The 8-second VAD endpoint was at about 10.1 seconds; the
previous decoder-silence image ended earlier at about 6.9 seconds on that input.
The fallback has not been weakened to guess word ownership. Endpoint boundary
quality and this native metadata mismatch need separate investigation; these
smokes establish integration, not the elimination of truncated words.

The initial duration-policy follow-up, using 700 ms soft silence, passed six
native GPU cases and 68 focused Nemotron regression tests on the same GPU.
Rebuilt local Compose validation
passed Tier 1, finals from both realtime tracks at EOF, and the 20-second
speaker-labelled silence smoke. A 99-second BFF
stream, built from repeated synthetic speech with one-second pauses and a
3000 ms session timeout, emitted no finals through 90 seconds and a final at
audio position 92.18 seconds before EOF. Continuous speech in the native test
hit the emergency boundary at 120.66 seconds. These timings describe this fixture,
not a guarantee of lossless words at the emergency boundary.

The 500 ms default passed the two native duration-policy cases (90/120 and
12/20 seconds, including reset) and all 68 focused regression tests.
