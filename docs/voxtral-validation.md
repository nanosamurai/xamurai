# Voxtral validation — 2026-10-30

This is a local smoke result, not a production capacity certification. The
[service guide](voxtral-realtime.md) contains configuration, architecture,
operational limits, and the remaining rollout gates.

## Environment and configuration

- Linux GPU container on Docker Desktop/WSL, RTX 5090 Laptop GPU with nominal
  24 GB VRAM, NVIDIA driver 610.74. This was not a dedicated production GPU.
- Voxtral Mini 4B Realtime 2602, revision
  `2769294da9567371363522aac9bbcfdd19447add`; all six artifacts checksum-verified.
- vLLM 0.32.0, BF16, tensor parallelism 1, PIECEWISE CUDA graphs.
- Admission and engine `max_num_seqs`: **2**. Epoch: 60 seconds. Context: 2048.
  KV allocation: **6144 MiB**. Scheduler token budget: 2048. Startup GPU memory
  utilization check: 0.7. Other controls used their documented defaults.
- Model cache mounted read-only; Hugging Face offline mode enabled. Test gRPC
  and metrics ports were published only on `127.0.0.1`.
- Two distinct locally synthesized English WAV clips, repeated for the longer
  tests. No customer recordings or transcript outputs were committed.

The initial 2048 MiB KV allocation failed vLLM initialization: this architecture
requires about 2.28 GiB for even one full 2048-token request. The corrected 6 GiB
allocation reported **5386 cache tokens / 2.63 full-request concurrency**.
This validates the cache geometry for the tested capacity, not arbitrary
session counts or an inference-throughput guarantee.

Cold startup with already downloaded weights took approximately 4 minutes
20 seconds on this environment, including about 30 seconds to verify weights
through the Windows bind mount and 135 seconds of profiling, compilation, and
warmup. Keep warm spare replicas for admission bursts; scale-from-zero is not
an instantaneous capacity response. A fresh model/image download adds more time.

## Results

| Check | Result |
| --- | --- |
| Full lightweight CI selection | 146 tests passed, including 24 focused Voxtral cases |
| Docker Bake image build | Passed; exact dependency check passed with the documented vendor NCCL exception |
| Artifact checksums | All six passed |
| Two simultaneous paced 10-second clips | First partial about 0.906 s; completion 10.515 s; one final each |
| Two simultaneous paced 70-second clips | First partial about 0.859 s; completion 70.547 s; tail flush about 0.531 s |
| Epoch ownership | Both long streams emitted exactly `[0,60]` and `[60,70]` finals |
| Basic contamination check | Each stream recognized its distinct expected fixture word and did not emit the other fixture's word |
| Two simultaneous unpaced 70-second clips | Both completed in 29.937 s; two finals each; same final character counts as paced runs |
| Overload | A third RPC was rejected with `RESOURCE_EXHAUSTED / REPLICA_FULL` before sending audio |
| Active cancellation | Eight requests cancelled after first text; service stayed healthy and both slots were reusable |
| Metrics | Active sessions returned to zero; capacity, rejection, completion, and cancellation counters reflected the probes |
| Secret scan | Gitleaks found no secrets in the change |

Latency numbers above are individual smoke observations, **not p95/p99
estimates**. The unpaced run is about 4.68 aggregate audio-seconds processed per
wall-second for this workload, but does not certify four realtime sessions.
Its roughly 29.86-second client-EOF-to-completion measurement includes queued
unpaced audio and must not be compared with the paced EOF flush latency.

One sample during the paced run showed approximately 3.59 GiB of container
memory and 22.4 GiB of device-wide GPU memory in use. These are snapshots, not
process-exclusive VRAM or peak/RSS measurements. No 4/8-session, multi-GPU,
multi-hour, real-customer WER, or autoscaler test has been completed.

Reproduce with synthetic/consented PCM16 mono 16 kHz clips:

```bash
python utilities/voxtral_realtime_probe.py first.wav second.wav \
  --sessions 2 --seconds 70 --expect bicycle --expect umbrella
python utilities/voxtral_realtime_probe.py first.wav second.wav \
  --sessions 2 --seconds 70 --unpaced --expect bicycle --expect umbrella
python utilities/voxtral_realtime_probe.py first.wav second.wav \
  --sessions 2 --seconds 10 --cancel-cycles 8 --expect bicycle --expect umbrella
```

Choose distinct expected words actually present in the corresponding fixtures.
The cancellation check needs speech in the first fixture's first three seconds.
Use `--cancel-cycles 0` for fixtures with an initial silence. The probe reports
counts/timing and performs assertions without printing transcript contents.

## Image security assessment

Trivy 0.74.0 with its 2026-10-30 vulnerability database scanned the built image.
The initial scan reported 17 critical and 392 high package/advisory matches.
The image now updates affected Ubuntu OpenSSL, GnuPG, Brotli, and certificate
packages, and pins patched certifi 2026.10.23, uv 0.13.4, and hf-xet 1.8.0.
The certificate fixes address the reviewed
[certifi advisory](https://github.com/advisories/GHSA-6q48-6c7w-9gw2) and
[uv advisory](https://github.com/advisories/GHSA-hq85-5cpj-mr62).

The final scan still reports **15 critical and 377 high matches**:

| Remaining inventory group | Scanner result | Assessment and release requirement |
| --- | --- | --- |
| `linux-libc-dev` development headers | 386 matches: 14 critical, 372 high | These are kernel-advisory matches against header-package inventory, not proof that this container ships a running vulnerable kernel. The deployment host supplies the kernel. Retain the findings for explicit applicability/VEX review and separately verify host kernel patching. |
| Embedded upstream SBOM Python entries | Six matches: one critical, five high | Trivy marks these packages `AnalyzedBy=sbom`, with no installed metadata path. The entries list older certifi, msgpack, setuptools, and urllib3 versions. Runtime imports resolve to the patched versions below. The upstream SBOM discrepancy still needs review; it has not been hidden or declared a blanket exception. |

Runtime package versions were checked inside the actual serving container:
certifi **2026.10.23**, uv **0.13.4**, hf-xet **1.8.0**, msgpack **1.2.3**,
urllib3 **2.8.0**, and setuptools **80.10.2**. The final report contained no other
high/critical matches against installed runtime-package metadata. This is a
scoped observation, not a claim that the image has no vulnerabilities.

The scanned local OCI manifest was
`sha256:c86588a11ae485d3624a056d537b7836e716abb8d2292758a1a1efcab9243a8e`;
Trivy's config artifact ID was
`sha256:b594b789cbc2b5b178bfa84307e13795b195e43d0ae0243a6d195bbbf165a66b`.
Rebuilds with newer OS security packages will have different image identities.
Re-scan the release image and finish applicability review **before production
rollout**. No ignore list or removal of SBOM metadata was used to make the scan
appear clean.

Other remaining rollout gates are sustained memory/latency testing, multilingual
and seam WER evaluation, authenticated/private deployment routing, and operational
draining. The current five-second shutdown grace cancels unfinished streams;
deployment scale-down must account for long-lived sessions. Nothing was deployed
to a shared or production environment during this validation.
