#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import pandas as pd
import torch
from pyannote.audio import Pipeline
import whisperx


def format_srt_timestamp(seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000.0))
    hh = ms // 3_600_000
    ms -= hh * 3_600_000
    mm = ms // 60_000
    ms -= mm * 60_000
    ss = ms // 1_000
    ms -= ss * 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def diarization_to_dataframe(diar):
    # Already a DataFrame (your case)
    if isinstance(diar, pd.DataFrame):
        df = diar.copy()

        # Some variants have a 'segment' column (pyannote Segment objects)
        if "start" not in df.columns and "segment" in df.columns:
            df["start"] = df["segment"].apply(lambda s: float(s.start))
            df["end"] = df["segment"].apply(lambda s: float(s.end))

        # Normalize speaker column name
        if "speaker" not in df.columns:
            if "label" in df.columns:
                df["speaker"] = df["label"]

        return df[["start", "end", "speaker"]].sort_values(["start", "end"]).reset_index(drop=True)

    # Otherwise assume it's a pyannote Annotation-like object
    rows = []
    for seg, _, label in diar.itertracks(yield_label=True):
        rows.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})

    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)

def annotation_to_segments(annotation):
    """
    Convert pyannote.core.Annotation into JSON-serializable segments.
    """
    diar = []
    # itertracks yields (Segment, track, label)
    for seg, _, label in annotation.itertracks(yield_label=True):
        diar.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": str(label),
            }
        )
    diar.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return diar


def write_txt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            speaker = seg.get("speaker", "UNKNOWN")
            text = (seg.get("text") or "").strip()
            f.write(f"[{start:0.2f} - {end:0.2f}] {speaker}: {text}\n")


def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(f"{idx}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{speaker}: {text}\n\n")
            idx += 1


def main():
    ap = argparse.ArgumentParser(
        description="WhisperX transcription + alignment + speaker diarization for an MP4."
    )
    ap.add_argument("input", help="Path to input .mp4 (or any media file ffmpeg can read).")
    ap.add_argument("-o", "--outdir", default="whisperx_out", help="Output directory.")
    ap.add_argument("--model", default="large-v3", help="Whisper model name (e.g., small, medium, large-v2).")
    ap.add_argument("--batch_size", type=int, default=16, help="Reduce if low on GPU memory.")
    ap.add_argument("--compute_type", default=None, help='e.g. "float16" (GPU), "int8" (CPU/low-mem).')
    ap.add_argument("--device", default=None, help='e.g. "cuda" or "cpu". Default: auto.')
    ap.add_argument("--language", default=None, help="Optional language code (e.g., en, de).")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token (or set HF_TOKEN env var).")
    ap.add_argument("--min_speakers", type=int, default=None, help="Optional min # speakers.")
    ap.add_argument("--max_speakers", type=int, default=None, help="Optional max # speakers.")
    ap.add_argument("--clustering_threshold", type=float, default=None,
                    help="pyannote clustering.threshold. Lower => less merging (more speakers).")
    ap.add_argument("--segmentation_threshold", type=float, default=None,
                    help="pyannote segmentation.threshold (optional).")
    ap.add_argument("--min_cluster_size", type=int, default=None,
                    help="pyannote clustering.min_cluster_size (optional).")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem

    # Decide device/compute_type
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    # 1) Load model + audio, transcribe
    model = whisperx.load_model(args.model, device, compute_type=compute_type)

    audio = whisperx.load_audio(str(in_path))

    # model.transcribe signature can vary slightly; keep it robust
    try:
        result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
    except TypeError:
        # Older/newer versions may not accept language kwarg
        result = model.transcribe(audio, batch_size=args.batch_size)

    # 2) Align words
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 3) Diarize + assign speaker labels
    diarization_json = None
    if hf_token:
        # Use pyannote pipeline directly so we can tweak hyperparameters
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        pipeline.to(torch.device(device))

        # Get current/default hyperparameters, then override as requested
        try:
            params = pipeline.parameters(instantiated=True)
        except TypeError:
            params = pipeline.parameters()

        # Override knobs (if provided)
        if args.clustering_threshold is not None:
            params.setdefault("clustering", {})
            params["clustering"]["threshold"] = float(args.clustering_threshold)

        if args.min_cluster_size is not None:
            params.setdefault("clustering", {})
            params["clustering"]["min_cluster_size"] = int(args.min_cluster_size)

        if args.segmentation_threshold is not None:
            params.setdefault("segmentation", {})
            params["segmentation"]["threshold"] = float(args.segmentation_threshold)

        pipeline.instantiate(params)  # applies the overrides :contentReference[oaicite:1]{index=1}

        diar_kwargs = {}
        if args.min_speakers is not None:
            diar_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diar_kwargs["max_speakers"] = args.max_speakers

        # whisperx.load_audio returns mono 16k float waveform; pass it as in-memory audio
        waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
        diar_raw = pipeline({"waveform": waveform, "sample_rate": 16000}, **diar_kwargs)

        # Convert Annotation -> DataFrame (so assign_word_speakers works in many WhisperX builds) :contentReference[oaicite:2]{index=2}
        diar_df = diarization_to_dataframe(diar_raw)
        diarization_json = diar_df.to_dict("records")

        result = whisperx.assign_word_speakers(diar_df, result)
    else:
        print(
            "WARNING: No HF token provided. Skipping diarization.\n"
            "Provide --hf_token or set HF_TOKEN environment variable to enable speaker labels."
        )
    # Save outputs
    json_path = outdir / f"{stem}.whisperx.json"
    txt_path = outdir / f"{stem}.transcript.txt"
    srt_path = outdir / f"{stem}.transcript.srt"
    diar_path = outdir / f"{stem}.diarization.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    write_txt(result.get("segments", []), txt_path)
    write_srt(result.get("segments", []), srt_path)

    if diarization_json is not None:
        with diar_path.open("w", encoding="utf-8") as f:
            json.dump(diarization_json, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"- {json_path}")
    print(f"- {txt_path}")
    print(f"- {srt_path}")
    if diarization_json is not None:
        print(f"- {diar_path}")


if __name__ == "__main__":
    main()
