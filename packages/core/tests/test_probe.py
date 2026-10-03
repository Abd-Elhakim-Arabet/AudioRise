import json
import subprocess

import pytest

from audio_engine import probe_audio

HAVE_TONE = True


def _make_tone(tmp_path, name="tone.wav", seconds=1.0):
    out = tmp_path / name
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-ar", "44100", "-ac", "2", str(out),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()[:300]
    return out


def test_probe_wav(tmp_path):
    tone = _make_tone(tmp_path)
    info = probe_audio(tone)
    assert info["channels"] == 2
    assert info["sample_rate"] == 44100
    assert 0.9 <= info["duration_sec"] <= 1.1


def test_probe_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_audio(tmp_path / "nope.mp3")


def test_probe_bad_extension(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        probe_audio(f)
