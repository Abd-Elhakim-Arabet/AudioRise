"""`audiorise` MCP server (official SDK v2, stdio) with five tools.

- probe_audio: duration/rate/channels via ffprobe (light).
- separate_audio: local Demucs split into vocals/drums/bass/other.wav (heavy,
  CPU; model downloads once then works offline). Output dir is
  `<input-stem>_stems/` next to the input.
- label_stems: best-effort heuristic tags + `stem[a,b].wav` aliases.
- summarize_stems: sizes/durations/labels from an existing stems dir (light).
- transcribe_audio: stems/wav → multitrack MIDI (MuScriptor, gated weights).

All paths are jailed to `AUDIORISE_MCP_ROOTS` (default: repo root + tmp).
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from audio_engine import probe_audio as core_probe
from audio_engine import label_stems as core_label_stems
from audio_engine import separate as core_separate
from audio_engine import transcribe_stems as core_transcribe_stems

from .jail import jailed, roots

mcp = MCPServer("audiorise")


@mcp.tool()
def probe_audio(path: str) -> dict:
    """Probe an audio file (duration, sample rate, channels, codec)."""
    p = jailed(path)
    if not p.is_file():
        raise ValueError(f"audio file not found: {p}")
    return core_probe(p)


@mcp.tool()
def separate_audio(
    path: str,
    model: str = "htdemucs",
    label: bool = False,
    top_k: int = 2,
    midi: bool = False,
    midi_model: str = "small",
) -> dict:
    """Split audio into stems locally (vocals/drums/bass/other.wav + stems.json).

    Writes to `<input-stem>_stems/` next to the input. Slow on CPU
    (minutes for full songs); the model downloads once, then works offline.
    `midi=True` also transcribes to multitrack MIDI (gated MuScriptor weights).
    """
    p = jailed(path)
    if not p.is_file():
        raise ValueError(f"audio file not found: {p}")
    outdir = p.parent / f"{p.stem}_stems"
    jailed(outdir)  # must stay inside the jail (derived from input, always is)
    return core_separate(
        p, outdir=outdir, model=model, label=label, top_k=top_k,
        midi=midi, midi_model=midi_model,
    )


@mcp.tool()
def label_stems(stems_dir: str, top_k: int = 2) -> dict:
    """Tag stems in a dir with best-effort instrument labels + aliases."""
    d = jailed(stems_dir)
    if not d.is_dir():
        raise ValueError(f"stems dir not found: {d}")
    return {"stems_dir": str(d), "labels": core_label_stems(d, top_k=top_k)}


@mcp.tool()
def summarize_stems(stems_dir: str) -> dict:
    """Summarize a stems dir: per-stem bytes, probe info, labels (no ML)."""
    import json

    d = jailed(stems_dir)
    if not d.is_dir():
        raise ValueError(f"stems dir not found: {d}")
    wavs = sorted(p for p in d.glob("*.wav") if "[" not in p.stem)
    if not wavs:
        raise ValueError(f"no stem wavs in {d}")
    stems = []
    for wav in wavs:
        try:
            info = core_probe(wav)
            duration, sr, ch = info["duration_sec"], info["sample_rate"], info["channels"]
        except (RuntimeError, ValueError):
            duration, sr, ch = 0.0, 0, 0
        stems.append(
            {
                "name": wav.stem,
                "path": str(wav),
                "bytes": wav.stat().st_size,
                "duration_sec": duration,
                "sample_rate": sr,
                "channels": ch,
            }
        )
    labels: dict = {}
    meta_path = d / "stems.json"
    if meta_path.is_file():
        try:
            labels = json.loads(meta_path.read_text()).get("labels", {})
        except (json.JSONDecodeError, OSError):
            labels = {}
    return {"stems_dir": str(d), "stems": stems, "labels": labels}


@mcp.tool()
def transcribe_audio(
    path: str, model_size: str = "small", condition: bool = True
) -> dict:
    """Transcribe a stem .wav or stems dir to multitrack MIDI (MuScriptor).

    Needs `pip install -e packages/core[transcription]` plus gated weights
    (HF login + license accept). Speech/silence stay audio-only.
    """
    from pathlib import Path as _Path

    from audio_engine import transcribe_file as core_transcribe_file

    p = jailed(path)
    if p.is_file():
        data = core_transcribe_file(p, model_size=model_size)
        out = p.with_suffix(".mid")
        jailed(out)
        out.write_bytes(data)
        return {"midi_path": str(out)}
    if p.is_dir():
        return core_transcribe_stems(p, model_size=model_size, condition=condition)
    raise ValueError(f"file or dir not found: {p}")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
