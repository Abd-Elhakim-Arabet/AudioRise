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

Live status: **Increment 1 — skeleton + `probe` only.** `separate` is a stub that
explains what Increment 2 will install.

## Requirements

- Python 3.10–3.14, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux).
- Increment 1 needs **no** third-party packages (stdlib only).

## Quick start (Increment 1)

```bash
pip install -e packages/core -e packages/cli
audiorise probe song.mp3
audiorise probe song.mp3 --json
audiorise separate song.mp3 --outdir stems/   # stub: prints Increment 2 roadmap
```

### Python API (Increment 1)

```python
from audio_engine import probe_audio
info = probe_audio("song.mp3")
print(info["duration_sec"], info["sample_rate"], info["channels"])
```

`info` keys: `path, format, duration_sec, sample_rate, channels, codec, bit_rate`.

## Output contract (stable from Increment 1)

`separate()` (landing Increment 2) will write:

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

- [x] **Inc 1 (this):** skeleton mirroring VectoRise, `probe`, CLI, README, tests.
- [ ] **Inc 2:** local separation — PyTorch + Demucs `htdemucs` (4 stems), CPU, model
      auto-download on first run, fully offline after. `audiorise separate` goes live.
- [ ] **Inc 3:** fine-grained labeling — classifier tags each stem
      (`other.wav` → `other[flute,piano].wav` style aliases + `stems.json`), best-effort.
- [ ] **Inc 4:** MCP server (`probe_audio`, `separate_audio`) jailed to `AUDIORISE_MCP_ROOTS`.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli
.venv/bin/pytest packages/core/tests/ -q
