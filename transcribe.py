#!/usr/bin/env python3
"""
Transcribe an MP3/MP4 (or any audio/video) file to Egyptian Arabic text.

Uses faster-whisper (Whisper large-v3). Auto-detects an NVIDIA GPU and
falls back to CPU if CUDA is unavailable or fails.

The meeting recordings tend to be very quiet, so by default the audio is
first normalized with ffmpeg (loudnorm + high-pass) — this is what makes
the difference between usable text and silence-hallucinations.

Usage:
    transcribe.py INPUT [--model large-v3] [--device auto|cuda|cpu]
                        [--out-dir DIR] [--no-normalize]
                        [--initial-prompt TEXT]

Outputs (next to the input file, or in --out-dir):
    NAME.txt   plain text
    NAME.srt   subtitles with timestamps
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def human_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# Extensions this tool *writes* — passing one back in as input is almost
# certainly a mistake (e.g. an empty output stub from a previous run).
OUTPUT_EXTS = {".txt", ".srt"}


# English words spoken in the meetings get transliterated into Arabic script
# by Whisper — often inconsistently (the same word several ways). This maps the
# transliterations back to the real English word. Keys are matched longest-first
# so definite-article ("ال") variants win over the bare form. Extend as needed.
CODESWITCH_MAP = {
    # interview(s) — Whisper produced 5+ spellings
    "الانتروبيوز": "the interviews",
    "الانترويوز": "the interviews",
    "الانترويوش": "the interviews",
    "الانتربيوز": "the interviews",
    "الانتربيو": "the interview",
    "الانترويو": "the interview",
    "انتروبيوز": "interviews",
    "انترويوز": "interviews",
    "انترويوش": "interviews",
    "انتربيوز": "interviews",
    "انتربيو": "interview",
    "انترويو": "interview",
    # other mangled terms seen in the recordings
    "بارت تايم": "part time",
    "فل تايم": "full time",
    "تأبتمايز": "optimize",
    "اكسل شيت": "Excel sheet",
    "اكسل شي": "Excel sheet",
    "كارد": "card",
    "كارت": "card",
}


def postprocess_text(text: str) -> str:
    """Replace Arabic-transliterated English terms with real English.

    Longest keys first so 'الانتربيو' is handled before 'انتربيو'.
    """
    for src in sorted(CODESWITCH_MAP, key=len, reverse=True):
        text = text.replace(src, CODESWITCH_MAP[src])
    return text


def validate_input(src: Path) -> None:
    """Fail early (and clearly) if src isn't a usable audio/video file.

    Without this, a text file or an empty stub gets handed straight to
    ffmpeg, which dies with a cryptic 'Invalid data found' traceback.
    """
    ext = src.suffix.lower()
    if ext in OUTPUT_EXTS:
        sys.exit(
            f"error: {src.name} is a {ext} file — that's an *output* format, "
            f"not an audio/video source. Point me at the recording "
            f"(e.g. {src.stem}.mp4) instead."
        )

    if src.stat().st_size == 0:
        sys.exit(f"error: {src} is empty (0 bytes) — nothing to transcribe.")

    # Probe for at least one audio stream. If ffprobe is missing, skip the
    # check and let ffmpeg surface any problem later.
    try:
        probe = subprocess.run(
            ["ffprobe", "-hide_banner", "-loglevel", "error",
             "-select_streams", "a", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(src)],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return
    if probe.returncode != 0 or "audio" not in probe.stdout:
        sys.exit(
            f"error: no audio stream found in {src.name} — it doesn't look "
            f"like an audio/video file. Check that you passed the recording, "
            f"not a text/subtitle file."
        )


def normalize_audio(src: Path) -> str:
    """Run ffmpeg loudnorm -> 16 kHz mono wav in a temp file. Returns path."""
    fd, out = tempfile.mkstemp(suffix=".wav", prefix="whisper_norm_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80",
        "-ar", "16000", "-ac", "1",
        out,
    ]
    subprocess.run(cmd, check=True)
    return out


def pick_device(requested: str):
    """Return (device, compute_type). 'auto' probes for a working GPU.

    Pascal GPUs (e.g. GTX 10-series) don't do efficient float16, so we use
    int8 on CUDA, which runs well on them.
    """
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        return "cuda", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8"
    except Exception as e:
        print(f"[info] GPU probe failed ({e}); using CPU.", file=sys.stderr)
    return "cpu", "int8"


def load_model(model_name, device, compute_type):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def main():
    ap = argparse.ArgumentParser(description="MP3/MP4 -> Egyptian Arabic text")
    ap.add_argument("input", help="path to audio/video file")
    ap.add_argument("--model", default="large-v3",
                    help="whisper model (default: large-v3; try 'medium'/'small' if low on VRAM/RAM)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-dir", default=None, help="where to write .txt/.srt (default: next to input)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="skip ffmpeg loudness normalization (not recommended for quiet recordings)")
    ap.add_argument("--initial-prompt", default=None,
                    help="optional prompt to bias spelling (leave empty to avoid echo on silence)")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="skip mapping transliterated English words back to English (see CODESWITCH_MAP)")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        sys.exit(f"error: file not found: {src}")
    validate_input(src)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / (src.stem + ".txt")
    srt_path = out_dir / (src.stem + ".srt")

    device, compute_type = pick_device(args.device)
    print(f"[info] file:    {src}", flush=True)
    print(f"[info] model:   {args.model}", flush=True)
    print(f"[info] device:  {device} ({compute_type})", flush=True)

    audio_path = str(src)
    tmp_audio = None
    if not args.no_normalize:
        print("[info] normalizing audio with ffmpeg (loudnorm)...", flush=True)
        tmp_audio = normalize_audio(src)
        audio_path = tmp_audio

    try:
        try:
            model = load_model(args.model, device, compute_type)
        except Exception as e:
            if device == "cuda":
                print(f"[warn] CUDA load failed ({e}); falling back to CPU (int8).", file=sys.stderr)
                device, compute_type = "cpu", "int8"
                model = load_model(args.model, device, compute_type)
            else:
                raise

        t0 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language="ar",
            task="transcribe",
            initial_prompt=args.initial_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            condition_on_previous_text=False,   # avoids runaway repetition loops
        )
        print(f"[info] language prob: {info.language_probability:.2f}; "
              f"duration: {info.duration:.0f}s. Transcribing...", flush=True)

        count = 0
        with open(txt_path, "w", encoding="utf-8") as ftxt, \
             open(srt_path, "w", encoding="utf-8") as fsrt:
            for i, seg in enumerate(segments, 1):
                line = seg.text.strip()
                if not line:
                    continue
                if not args.no_postprocess:
                    line = postprocess_text(line)
                count += 1
                ftxt.write(line + "\n")
                fsrt.write(f"{i}\n{human_time(seg.start)} --> {human_time(seg.end)}\n{line}\n\n")
                print(f"  [{human_time(seg.start)[:-4]}] {line}", flush=True)

        dt = time.time() - t0
        print(f"\n[done] {count} segments, {dt:.0f}s elapsed", flush=True)
        if count == 0:
            print("[note] no speech detected — the recording may be near-silent.", flush=True)
        print(f"[out]  {txt_path}", flush=True)
        print(f"[out]  {srt_path}", flush=True)
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)


if __name__ == "__main__":
    main()
