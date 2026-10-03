"""Increment 2 placeholder — local Demucs separation.

Stable output contract (do not change without bumping README):
    separate(input, outdir) -> dict with keys:
      stems: list of {name, path, rms_db}
      stems_json: path to stems.json
      source: probe_audio() dict

Increment 2 will implement this with `htdemucs` (vocals/drums/bass/other,
44.1 kHz stereo WAVs) fully locally. Until then, calling it raises
with install guidance instead of failing silently.
"""

from __future__ import annotations

from pathlib import Path

EXPECTED_STEMS = ("vocals", "drums", "bass", "other")


class SeparationNotAvailable(RuntimeError):
    pass


def separate(
    input_path: str | Path,
    outdir: str | Path = "stems",
    model: str = "htdemucs",
) -> dict:
    raise SeparationNotAvailable(
        "AudioRise Increment 2 not installed yet: `separate()` needs "
        "torch + torchaudio + demucs (`pip install torch torchaudio demucs`). "
        f"Requested model={model!r}, input={input_path}, outdir={outdir}. "
        "Output contract when live: vocals.wav/drums.wav/bass.wav/other.wav "
        "+ stems.json (44.1kHz stereo, FL Studio / Audacity ready)."
    )
