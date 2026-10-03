# AudioRise

Audio-to-stems pipeline — one engine, two doors (so far): a **CLI** and a **Python API**.
MCP server lands in a later increment. No website, by design.

```
song.mp3 → probe → separate (Increment 2: local Demucs) → stems/*.wav (+ stems.json)
```

Stems are plain WAVs: drag the folder into **FL Studio / Audacity / Reaper / Ableton**,
each stem on its own track. Speech and singing share the `vocals` stem (per spec).
Fine-grained labels (flute, piano, …) arrive in Increment 3 as best-effort
tags on top of the 4–6 proven stems — true flute-vs-violin from a finished mix
is unsolved, so we keep the proven stems and *label*, never invent, extra stems.

Live status: **Increment 2 — `probe` + `separate` (local Demucs) live.**
Fine-grained labels land in Increment 3.

## Requirements

- Python 3.10–3.14, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux).
- `probe` needs no third-party packages. `separate` needs:
  `pip install -e packages/core[separation]` (torch + torchaudio + demucs + soundfile).
  First run downloads the `htdemucs` model (~80 MB), then works fully offline.

## Quick start (Increment 2)

```bash
pip install -e packages/core -e packages/cli
pip install -e "packages/core[separation]"   # once, for separate
audiorise probe song.mp3
audiorise separate song.mp3 --outdir stems/ --model htdemucs
audiorise separate song.mp3 --outdir stems/ --device cpu --json
```

### Python API (Increment 2)

```python
from audio_engine import probe_audio, separate
info = probe_audio("song.mp3")
result = separate("song.mp3", outdir="stems/", model="htdemucs")  # CPU auto
print([(s["name"], s["rms_db"]) for s in result["stems"]])
```

`info` keys: `path, format, duration_sec, sample_rate, channels, codec, bit_rate`.

## Output contract (stable from Increment 1)

`separate()` writes:

```
stems/
  vocals.wav   # speech + singing, always present even if silent
  drums.wav
  bass.wav
  other.wav
  stems.json   # probe info + model name + per-stem rms_db + fine labels (Inc 3)
```

44.1 kHz stereo WAV — imports cleanly into FL Studio / Audacity.

## Repo layout

```
packages/core/src/audio_engine/   the engine (probe.py live, separate.py + labels.py stubs)
packages/core/tests/              core test-suite (probe only for now)
packages/cli/src/audio_cli/       `audiorise` command (probe live, separate stub)
docs/                             (stub, one file per increment)
```

## Roadmap (one increment per submit, no accumulation)

- [x] **Inc 1:** skeleton mirroring VectoRise, `probe`, CLI, README, tests.
- [x] **Inc 2 (this):** local separation — PyTorch + Demucs `htdemucs` (4 stems),
      CPU auto (`--device cpu/cuda`), model auto-download then offline.
      `audiorise separate` live, verified on synth mix + CLI.
- [ ] **Inc 3:** fine-grained labeling — classifier tags each stem
      (`other.wav` → `other[flute,piano].wav` style aliases + `stems.json`), best-effort.
- [ ] **Inc 4:** MCP server (`probe_audio`, `separate_audio`) jailed to `AUDIORISE_MCP_ROOTS`.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli
.venv/bin/pytest packages/core/tests/ -q
