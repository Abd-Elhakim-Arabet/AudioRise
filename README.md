# AudioRise

Audio-to-stems pipeline — one engine, three doors: a **CLI**, a **Python API**,
and an **MCP server** for AI agents. No website, by design.

```
song.mp3 → probe → separate (local Demucs) → stems/*.wav (+ stems.json) → label
```

Stems are plain WAVs: drag the folder into **FL Studio / Audacity / Reaper / Ableton**,
each stem on its own track. Speech and singing share the `vocals` stem (per spec).
Fine-grained tags (flute, piano, …) are best-effort **labels on top of the 4 proven
stems** — true flute-vs-violin from a finished mix is unsolved, so we keep the proven
stems and *label*, never invent, extra stems.

Live status: **Increment 3 — `probe` + `separate` + `label` all live.**

## Requirements

- Python 3.10–3.14, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux).
- `probe` needs no third-party packages. `separate` needs:
  `pip install -e packages/core[separation]` (torch + torchaudio + demucs + soundfile).
  First run downloads the `htdemucs` model (~80 MB), then works fully offline.

## Quick start (Increment 3)

```bash
pip install -e packages/core -e packages/cli
pip install -e "packages/core[separation]"   # once, for separate
audiorise probe song.mp3
audiorise separate song.mp3 --outdir stems/ --model htdemucs
audiorise separate song.mp3 --outdir stems/ --label --top-k 2   # separate + tag
audiorise label stems/ --top-k 2                               # tag existing stems
```

`label` writes per-stem tags into `stems.json` (`labels: {other: [{label, score}]}`)
and alias symlinks like `other[flute,synth].wav` (copied if symlinks unsupported).
Silent stems get no tags and no alias. Heuristic is numpy-only, fully local,
no download — hints for organizing tracks in your DAW, not ground truth.

### Python API (Increment 3)

```python
from audio_engine import probe_audio, separate, label_stems, label_stem
info = probe_audio("song.mp3")
result = separate("song.mp3", outdir="stems/", model="htdemucs", label=True, top_k=2)
print(result["labels"]["other"])   # [{'label': 'flute', 'score': 0.93}, ...]
print(label_stem("stems/other.wav"))

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
packages/core/src/audio_engine/   the engine (probe + separate + labels, all live)
packages/core/tests/              core test-suite (probe + separate + labels)
packages/cli/src/audio_cli/       `audiorise` command (probe + separate + label)
packages/mcp-server/              MCP server + per-client install scripts
```

## Roadmap (one increment per submit, no accumulation)

- [x] **Inc 1:** skeleton mirroring VectoRise, `probe`, CLI, README, tests.
- [x] **Inc 2:** local separation — PyTorch + Demucs `htdemucs` (4 stems),
      CPU auto (`--device cpu/cuda`), model auto-download then offline.
      `audiorise separate` live, verified on synth mix + CLI.
- [x] **Inc 3:** fine-grained labeling — numpy-only heuristic tags each stem
      (`other[flute,synth].wav` aliases + `stems.json` labels), silent stems skipped.
      `audiorise label` + `separate --label` live, verified on sine + Demucs stems.
- [x] **Inc 4 (this):** MCP server (`probe_audio`, `separate_audio`, `label_stems`,
      `summarize_stems`) jailed to `AUDIORISE_MCP_ROOTS`, verified in-process
      (tool list + probe/summarize JSON + jail rejection).

## MCP server (`packages/mcp-server/`)

`audiorise` MCP server (official SDK v2, stdio) with four tools returning JSON.
Paths are jailed to `AUDIORISE_MCP_ROOTS` (default: repo root + tmp). Details in
[`packages/mcp-server/README.md`](packages/mcp-server/README.md).

```bash
pip install -e packages/core -e packages/mcp-server
python -m audio_mcp
```

One-command client setup (idempotent, machine-local paths auto-detected):

```bash
sh packages/mcp-server/install-codex.sh     # → ~/.codex/config.toml
sh packages/mcp-server/install-opencode.sh   # → ~/.config/opencode/opencode.json
sh packages/mcp-server/install-claude.sh     # → claude mcp add (user scope)
```
- [ ] **Inc 3:** fine-grained labeling — classifier tags each stem
      (`other.wav` → `other[flute,piano].wav` style aliases + `stems.json`), best-effort.
- [ ] **Inc 4:** MCP server (`probe_audio`, `separate_audio`) jailed to `AUDIORISE_MCP_ROOTS`.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli
.venv/bin/pytest packages/core/tests/ -q
