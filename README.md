# Egyptian Arabic Transcriber

Convert **MP3 / MP4** (or any audio/video) into **Egyptian Arabic text** on Linux
using Whisper `large-v3` (via `faster-whisper`). Runs on an NVIDIA GPU with an
automatic CPU fallback.

- **Output:** `NAME.txt` (plain text) + `NAME.srt` (timestamped subtitles)
- **Tested on:** Ubuntu 26.04, Python 3.12, NVIDIA GTX 1070 (8 GB), driver 580

---

## 1. Prerequisites

| Requirement | Check command | Notes |
|-------------|---------------|-------|
| ffmpeg      | `ffmpeg -version` | Audio decode + normalization |
| Python 3.12 | `python3.12 --version` | 3.10–3.12 recommended (avoid 3.14 — thin wheel support) |
| NVIDIA GPU (optional) | `nvidia-smi` | Falls back to CPU if absent |
| Disk space  | `df -h ~` | ~3 GB for the model + ~1 GB for libs |

Install the system prerequisites if missing:

```bash
sudo apt update
sudo apt install -y ffmpeg python3.12 python3.12-venv
```

---

## 2. Install

```bash
cd ~/projects/transcribe

# 1. create an isolated virtual environment
python3.12 -m venv venv

# 2. upgrade pip
./venv/bin/pip install --upgrade pip

# 3. install the transcriber + CUDA runtime libraries
./venv/bin/pip install faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

> **CPU-only machine?** You can skip the two `nvidia-*` packages; the script
> auto-detects the missing GPU and runs on CPU.

The `large-v3` model (~3 GB) downloads automatically on the **first run** into
`~/.cache/huggingface`. Later runs reuse it (no internet needed).

---

## 3. Run

Use the wrapper script — it activates the venv and sets the CUDA library path
for you:

```bash
cd ~/projects/transcribe
./transcribe.sh ~/meetings/meeting.mp4
```

It prints live progress and writes `NAME.txt` and `NAME.srt` next to the input.

### Batch a whole folder

```bash
for f in ~/meetings/*.mp4; do
  ./transcribe.sh "$f"
done
```

### Run in the background (long files)

```bash
nohup ./transcribe.sh /path/to/big_meeting.mp4 > transcribe.log 2>&1 &
tail -f transcribe.log
```

---

## 4. Usage / Options

```
./transcribe.sh INPUT [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--out-dir DIR`     | next to input | Where to write `.txt` / `.srt` |
| `--device auto\|cuda\|cpu` | `auto` | Force GPU or CPU |
| `--model NAME`      | `large-v3` | Try `medium` / `small` if low on VRAM/RAM |
| `--no-normalize`    | off | Skip ffmpeg loudness pass (not recommended here) |
| `--initial-prompt "…"` | none | Bias spelling toward specific terms/names (can cause echo on silence) |

Examples:

```bash
# force CPU, custom output folder
./transcribe.sh meeting.mp4 --device cpu --out-dir ~/transcripts

# smaller/faster model
./transcribe.sh meeting.mp4 --model medium

# an mp3 works exactly the same
./transcribe.sh interview.mp3
```

---

## 5. How it works

1. **Validate** — empty files, `.txt`/`.srt` output stubs, and files with no
   audio stream are rejected up front with a clear message instead of a raw
   ffmpeg error.
2. **Normalize** — ffmpeg `loudnorm` + high-pass boosts the (very quiet)
   recordings. *This is essential:* without it Whisper hallucinates on silence.
3. **Transcribe** — Whisper `large-v3` with `language=ar`, VAD to skip silence,
   `condition_on_previous_text=False` to avoid repetition loops.
4. **Write** — plain text `.txt` and timestamped `.srt`.

The GTX 1070 is a Pascal GPU, so the code uses **int8 on CUDA** (float16 isn't
efficient there). On a CPU-only box it uses int8 on CPU automatically.

---

## 6. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| Empty transcript, "no speech detected" | Recording really is near-silent (no one spoke). Expected. |
| Repeated "شكرا لكم" / prompt echoed | Silence hallucination — make sure normalization is **on** (default). |
| `CUDA load failed … float16` | Harmless — the script auto-falls back; Pascal GPUs use int8. |
| `out of memory` on GPU | Use `--model medium` or `--device cpu`. |
| Very slow | You're likely on CPU — check the `[info] device:` line; install the `nvidia-*` libs for GPU. |
| `ffmpeg: command not found` | `sudo apt install ffmpeg`. |
| Wrong/garbled words | Audio is low quality; ensure normalization is on. See the code-switching note below. |

---

## 7. Known limitation: Arabic/English code-switching

English words mixed into Egyptian Arabic speech (e.g. "interview", "part
time") are often transliterated into Arabic script — and inconsistently
(the same word can come out several different ways). This is inherent to
Whisper decoding under a forced Arabic language; `large-v3` is already the
largest model, and seeding `--initial-prompt` with the English terms does
**not** reliably fix it. The robust fix is a post-processing pass that maps
the transliterated terms back to English.

---

## 8. Files in this project

```
transcribe/
├── transcribe.sh    # wrapper — run this
├── transcribe.py    # the transcription logic
├── venv/            # isolated Python environment
└── README.md        # this file
```
