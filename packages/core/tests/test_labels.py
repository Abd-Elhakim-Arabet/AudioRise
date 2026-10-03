"""Increment 3 tests — heuristic labeling, no model download."""

import json
import subprocess
from pathlib import Path

from audio_engine.labels import extract_features, label_path, label_stem, label_stems


def _tone(path: Path, freq: float = 440.0, seconds: float = 2.0):
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
            "-ar", "44100", "-ac", "2", str(path),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()[:300]
    return path


def _silence(path: Path, seconds: float = 1.0):
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={seconds}",
            str(path),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()[:300]
    return path


def test_sustained_sine_tagged_pitched_not_percussive(tmp_path):
    tone = _tone(tmp_path / "other.wav")
    feats = extract_features(tone)
    assert not feats.get("silent")
    assert feats["sustain"] > 0.8
    tags = label_stem(tone, top_k=2)
    assert len(tags) == 2
    assert all(0.0 <= t["score"] <= 1.0 for t in tags)
    top = tags[0]["label"]
    assert top in {"flute", "synth", "violin", "singing", "trumpet", "saxophone"}
    names = {t["label"] for t in tags}
    assert "drums" not in names  # pure legato sine must not read as drums


def test_silence_returns_no_tags(tmp_path):
    quiet = _silence(tmp_path / "vocals.wav")
    assert label_stem(quiet) == []


def test_label_stems_writes_aliases_and_json(tmp_path):
    d = tmp_path / "stems"
    d.mkdir()
    _tone(d / "other.wav", freq=660.0)
    _silence(d / "drums.wav")
    (d / "stems.json").write_text(json.dumps({"stems": [], "labels": {}}))
    labels = label_stems(d, top_k=2)
    assert set(labels) == {"other", "drums"}
    assert labels["drums"] == []
    assert len(labels["other"]) == 2
    meta = json.loads((d / "stems.json").read_text())
    assert meta["labels"]["other"][0]["label"] == labels["other"][0]["label"]
    aliases = list(d.glob("other[*.wav"))
    assert len(aliases) == 1  # e.g. other[flute,synth].wav


def test_label_path_single_file(tmp_path):
    tone = _tone(tmp_path / "other.wav")
    labels = label_path(tone, top_k=2)
    assert set(labels) == {"other"}
    assert len(labels["other"]) == 2
    assert len(list(tmp_path.glob("other[*.wav"))) == 1


def test_label_path_missing_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        label_path(tmp_path / "nope.wav")
