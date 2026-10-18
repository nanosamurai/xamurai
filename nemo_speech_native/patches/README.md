# Native runtime patches

`duration-endpointing.patch` applies to the pinned NeMo-Speech.cpp commit
`4f9676226f667d14608487df744f375db87127f8`. It adds optional soft-age, soft-silence
and maximum-duration fields to the endpointer's startup C ABI. All default to
zero, preserving upstream behavior unless a recognizer opts in. Existing request
silence overrides remain effective; the soft threshold can only shorten them.

The library reports `0.1.0+xamurai.1`, required by the shared Python binding.
Rebuild the shared native image and its consuming service together. The patch
applies after the cached CUDA build and VAD conversion, allowing small policy
edits without repeating those expensive steps. Keep patch files LF-terminated.

On an upstream revision change, rebase this patch and rerun the native GPU tests
described in [Nemotron VAD](../../docs/nemotron-vad.md).
