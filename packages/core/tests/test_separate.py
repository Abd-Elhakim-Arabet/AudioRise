"""Increment 2 contract test — no model download (monkeypatched)."""

import json
import subprocess

import pytest


def _make_mix(tmp_path):
    out = tmp_path / "mix.wav"
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=110:duration=1",
            "-filter_complex", "amix=inputs=2:duration=first",
            "-ar", "44100", "-ac", "2", str(out),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()[:300]
    return out


def test_separate_contract_without_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import importlib

    import torch

    sep = importlib.import_module("audio_engine.separate")

    def fake_run(wav, model_name, device):
        zeros = torch.zeros((2, wav.shape[-1]))
        names = ["vocals", "drums", "bass", "other"]
        srcs = torch.stack([zeros.clone() for _ in names])
        return names, srcs, 44100

    monkeypatch.setattr(sep, "_run_model", fake_run)

    mix = _make_mix(tmp_path)
    outdir = tmp_path / "stems"
    result = sep.separate(mix, outdir=outdir, model="htdemucs", device="cpu")

    assert {s["name"] for s in result["stems"]} == {"vocals", "drums", "bass", "other"}
    for s in result["stems"]:
        assert (outdir / f"{s['name']}.wav").is_file()
        assert isinstance(s["rms_db"], float)
    meta = json.loads((outdir / "stems.json").read_text())
    assert meta["model"] == "htdemucs"
    assert "source" in meta and meta["source"]["sample_rate"] == 44100
