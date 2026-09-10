"""Adapt the existing WhisperX composite to the generic completed-audio interface."""

import importlib.metadata
import importlib.util
from pathlib import Path
from drsynth_common.final_tracks import ContractError, TranscriptResult, file_digest
from drsynth_common.pyannote_telemetry import disable_pyannote_telemetry
from finalizer_worker.whisperx_profile import (
    ASR, ALIGNMENT, RUNTIME, VAD_SHA256, descriptor, model_snapshot,
)


class WhisperXTrack:
    """One fixed composite per process; models load lazily after selection."""

    def __init__(self):
        self._ready = False
        self._worker = None
        self._diarization_available = False

    def describe(self):
        return descriptor()

    def _initialize(self):
        disable_pyannote_telemetry()
        for package, version in RUNTIME.items():
            if importlib.metadata.version(package) != version:
                raise ContractError("runtime_profile_mismatch")
        import torch
        import whisperx
        from whisperx_worker import whisperx_worker as worker
        if not torch.cuda.is_available():
            raise ContractError("profile_requires_cuda")
        vad = Path(importlib.util.find_spec("whisperx").origin).parent / "assets" / "pytorch_model.bin"
        if file_digest(vad) != VAD_SHA256:
            raise ContractError("vad_digest_mismatch")
        asr = whisperx.load_model(model_snapshot(ASR), device="cuda", compute_type="float16",
                                  vad_method="pyannote", vad_options={
                                      "vad_onset": 0.5, "vad_offset": 0.363, "chunk_size": 30})
        diarization = None
        try:
            # Reuse the existing revision- and checksum-verified pipeline loader.
            from qwen_rtservice.enrichment import PyannoteDiarizer
            diarization = PyannoteDiarizer()._pipeline
            self._diarization_available = True
        except Exception:
            self._diarization_available = False
        worker.configure_batch_runtime(asr, diarization, self._load_alignment)
        self._worker = worker
        self._ready = True

    @staticmethod
    def _load_alignment(language):
        import whisperx
        if language not in ALIGNMENT:
            raise ContractError("alignment_language_unavailable")
        return whisperx.load_align_model(language, "cuda", model_name=model_snapshot(ALIGNMENT[language]),
                                         model_cache_only=True)

    def process(self, audio, context):
        if not self._ready:
            self._initialize()
        import torch
        torch.cuda.reset_peak_memory_stats()
        execution = {}
        text, segments = self._worker.run_whisperx_diarized_words(
            str(audio.path), tenant=context.tenant_id, lang=context.language or None,
            use_alignment=True, diagnostics=execution)
        capabilities = {"segment_timestamps": bool(segments) and all(
                            "start_s" in s and "end_s" in s for s in segments),
                        "word_timestamps": any(s.get("words") for s in segments),
                        "speaker_labels": any(s.get("speaker") for s in segments)}
        degradations = []
        if text and not capabilities["word_timestamps"]:
            degradations.append("alignment_unavailable")
        if text and not capabilities["speaker_labels"]:
            degradations.append("diarization_unavailable_or_unassigned")
        provenance = self.describe()
        execution["no_speech"] = not text and not segments
        provenance["execution"] = execution
        provenance["diarization_loaded"] = self._diarization_available
        provenance["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        return TranscriptResult(text, segments, execution.get("language", context.language),
                                capabilities, provenance, degradations)
