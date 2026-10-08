"""WhisperX 3.8.6 adapter: existing VAD/preprocessing/decoder, shared requests."""
from dataclasses import dataclass, replace


@dataclass
class Prepared:
    audio: object
    chunks: list
    language: str
    tokenizer: object
    options: object

    @property
    def key(self):
        # All requests use the same loaded model and immutable base options.
        # Tokenizer and numeral suppression are language specific.
        return self.language

    def remaining(self, chunk):
        return len(self.chunks) - chunk[0]


class WhisperXBackend:
    def __init__(self, model):
        from importlib.metadata import version
        if version("whisperx") != "3.8.6":
            raise RuntimeError("Shared batching requires whisperx==3.8.6")
        self.model = model
        if model.options.condition_on_previous_text:
            raise ValueError("Shared batching requires independent chunk decoding")

    def prepare(self, audio, language):
        from faster_whisper.tokenizer import Tokenizer
        from whisperx.vads import Vad, Pyannote
        from whisperx.asr import find_numeral_symbol_tokens
        model = self.model
        vad = model.vad_model if isinstance(model.vad_model, Vad) else Pyannote
        segments = model.vad_model({"waveform": vad.preprocess_audio(audio), "sample_rate": 16000})
        segments = vad.merge_chunks(segments, 30, onset=model._vad_params["vad_onset"],
                                   offset=model._vad_params["vad_offset"])
        language = language or model.preset_language or model.detect_language(audio)
        tokenizer = Tokenizer(model.model.hf_tokenizer, model.model.model.is_multilingual,
                              task="transcribe", language=language)
        options = model.options
        if model.suppress_numerals:
            options = replace(options, suppress_tokens=list(set(
                list(options.suppress_tokens) + find_numeral_symbol_tokens(tokenizer))))
        return Prepared(audio, list(enumerate(segments)), language, tokenizer, options)

    def decode(self, batch):
        import torch
        features = []
        for request, (_index, segment) in batch:
            audio = request.prepared.audio[int(segment["start"] * 16000):int(segment["end"] * 16000)]
            features.append(self.model.preprocess({"inputs": audio})["inputs"])
        prepared = batch[0][0].prepared
        outputs = self.model.model.generate_segment_batched(
            torch.stack(features), prepared.tokenizer, prepared.options)
        return [dict(text=text, avg_logprob=score, start=round(chunk[1]["start"], 3),
                     end=round(chunk[1]["end"], 3))
                for (_, chunk), text, score in zip(batch, outputs["text"], outputs["avg_logprob"])]

    @staticmethod
    def finish(prepared, results):
        return {"segments": results, "language": prepared.language}
