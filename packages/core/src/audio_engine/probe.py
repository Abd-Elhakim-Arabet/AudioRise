"""Probe an audio file with ffprobe (stdlib only, no torch).

Returns a stable dict — the same shape Increment 2's `stems.json`
will embed under its `source` key.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_ALLOWLIST = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def probe_audio(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    if p.suffix.lower() not in _ALLOWLIST:
        raise ValueError(
            f"unsupported extension {p.suffix!r} — allowed: {sorted(_ALLOWLIST)}"
        )
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH (install ffmpeg)")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout or "{}")

    streams = data.get("streams", []) or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), streams[0] if streams else {})
    fmt = data.get("format", {}) or {}

    try:
        duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "path": str(p),
        "format": fmt.get("format_name", "unknown"),
        "duration_sec": round(duration, 3),
        "sample_rate": int(audio.get("sample_rate") or 0),
        "channels": int(audio.get("channels") or 0),
        "codec": audio.get("codec_name", "unknown"),
        "bit_rate": int(str(fmt.get("bit_rate") or 0).split(".")[0] or 0),
    }
