# AudioRise

Feed it a song, get back **stems you can mix** and **notes you can edit** — one
engine, three doors: a **CLI**, a **Python API**, and an **MCP server** for AI
agents. No website, by design. Fully local after the first model download.

```
song.mp3 → probe → separate (Demucs) → stems/*.wav (+ stems.json)
                                     → label   → instrument tags
                                     → transcribe (MuScriptor) → song.mid
```

- **Stems** are plain 44.1 kHz stereo WAVs: drag them into **FL Studio / Audacity /
  Reaper / Ableton**, one per track. Speech and singing share the `vocals` stem.
- **Tags** are best-effort instrument labels on top of the 4 proven stems
  (`other[flute,synth].wav` aliases + `stems.json`). True flute-vs-violin from a
  finished mix is unsolved — we *label*, never invent, extra stems.
- **MIDI** is one multitrack `.mid` (General MIDI programs, drums on channel 10).
  Drop it into FL Studio and move notes in the piano roll. Speech and silence
  stay audio-only — speech has no notes.

## Requirements

- Python 3.10–3.14, `ffmpeg` + `ffprobe` on `PATH`
  (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux).
- `probe` / `label` need nothing else. Each AI stage is one extra install:

| Stage | Install | First run |
|---|---|---|
| `separate` (Demucs `htdemucs`) | `pip install -e "packages/core[separation]"` | downloads ~80 MB, then offline |
| `transcribe` (MuScriptor) | `pip install -e "packages/core[transcription]"` | gated weights, one-time login (below) |

> **Gated transcription weights (once per machine):** MuScriptor weights are
> CC BY-NC 4.0 (non-commercial) and need a free Hugging Face account:
> 1. Accept the license at `huggingface.co/MuScriptor/muscriptor-small`
> 2. `hf auth login` (or `export HF_TOKEN=hf_...`)
> 3. Re-run — weights cache locally, then fully offline.
> Until then, transcribe commands exit 3 with these steps.

## Quick start

```bash
pip install -e packages/core -e packages/cli
pip install -e "packages/core[separation]"      # once, for separate
pip install -e "packages/core[transcription]"  # once, for transcribe

audiorise probe song.mp3
audiorise separate song.mp3 --label --midi     # everything at once
```

Without `--outdir`, output goes to `<song>-stems/` next to the input
(e.g. `Premier-Night-stems/`).

## Commands

```bash
# Inspect any audio file
audiorise probe song.mp3 [--json]

# Split into vocals/drums/bass/other.wav + stems.json
audiorise separate song.mp3 [--outdir DIR] [--model htdemucs] [--device cpu]
                            [--label] [--midi] [--midi-model small|medium|large]

# Tag a stems dir or a single stem file (numpy-only heuristic, offline)
audiorise label stems/ [--top-k 2]
audiorise label stems/other.wav

# Stems/wav → multitrack MIDI (MuScriptor, local)
audiorise transcribe stems/ [--model-size small|medium|large] [--no-condition]
audiorise transcribe stems/other.wav
```

`--no-condition` transcribes without instrument hints; by default each stem is
conditioned on its label (known stems like drums always use their family —
separation beats guessing). Per-stem conditioning and GM programs live in
`stems.json` under `labels` / `midi`.

## Output contract

`separate song.mp3` writes (default dir `<song>-stems/`):

```
Premier-Night-stems/
  vocals.wav   # speech + singing, always present even if silent
  drums.wav
  bass.wav
  other.wav
  other[flute,synth].wav   # alias to the tagged stem (label only)
  stems.json               # probe info, model, rms_db, labels, midi
  Premier-Night.mid        # multitrack MIDI (midi only)
```

## Python API

```python
from audio_engine import probe_audio, separate, label_stems, transcribe_stems

info = probe_audio("song.mp3")
print(info["duration_sec"], info["sample_rate"], info["channels"])

result = separate("song.mp3", label=True, midi=True, midi_model="small")
print(result["labels"]["other"])   # [{'label': 'flute', 'score': 0.93}, ...]
print(result["midi"]["tracks"])    # [{'name': 'bass', 'program': 33, 'notes': 25}, ...]

print(transcribe_stems("Premier-Night-stems/"))  # transcribe later, separately
```

`probe_audio` returns `path, format, duration_sec, sample_rate, channels,
codec, bit_rate`. `separate` returns `source, model, device, sample_rate,
stems[{name, path, rms_db}], labels, midi, stems_json`.

## MCP server (`packages/mcp-server/`)

`audiorise` MCP server (official SDK v2, stdio): `probe_audio`,
`separate_audio`, `label_stems`, `summarize_stems`, `transcribe_audio` —
all returning JSON, all paths jailed to `AUDIORISE_MCP_ROOTS`
(default: repo root + tmp). Details in
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

## Repo layout

```
packages/core/src/audio_engine/   the engine (probe + separate + labels + transcribe)
packages/core/tests/              core test-suite (22 tests, mocked heavy models)
packages/cli/src/audio_cli/       `audiorise` command
packages/mcp-server/              MCP server + per-client install scripts
```

## History (one increment per commit)

- **Inc 1:** skeleton mirroring VectoRise, live `probe`, CLI, tests.
- **Inc 2:** local Demucs separation (`htdemucs`, CPU auto), verified on synth mix.
- **Inc 3:** heuristic labeling (`other[flute,synth].wav` aliases + `stems.json`).
- **Inc 4:** MCP server, jailed paths, per-client install scripts.
- **Inc 5:** `label` takes one `.wav` or a dir; input-named default outdir.
- **Inc 6:** MuScriptor transcription → multitrack `.mid`, verified on a real
  song excerpt (bass 25 / drums 57 / other 27 notes); family-first conditioning
  fix so known stems never follow a wrong heuristic tag.

## Developing

```bash
python -m venv .venv && .venv/bin/pip install -e packages/core -e packages/cli
.venv/bin/pytest packages/core/tests/ packages/mcp-server/tests/ -q
```
