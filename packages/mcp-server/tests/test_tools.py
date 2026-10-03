"""Tool tests — probe/summarize real (ffprobe), separate mocked (no model)."""

import json
import subprocess


def _tone(path, seconds=1.0, freq=440.0):
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


def test_probe_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIORISE_MCP_ROOTS", str(tmp_path))
    from audio_mcp.server import probe_audio

    tone = _tone(tmp_path / "song.wav")
    info = probe_audio(str(tone))
    assert info["channels"] == 2 and info["sample_rate"] == 44100


def test_separate_tool_outdir_and_jail(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIORISE_MCP_ROOTS", str(tmp_path))
    import audio_mcp.server as srv

    tone = _tone(tmp_path / "mix.wav")
    calls = {}

    def fake_separate(path, outdir, model, label=False, top_k=2, **kwargs):
        calls.update(path=str(path), outdir=str(outdir), model=model, label=label)
        return {"stems": [], "stems_json": str(outdir / "stems.json")}

    monkeypatch.setattr(srv, "core_separate", fake_separate)
    result = srv.separate_audio(str(tone), model="htdemucs", label=True)
    assert calls["outdir"].endswith("mix_stems")
    assert result["stems"] == []


def test_summarize_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIORISE_MCP_ROOTS", str(tmp_path))
    from audio_mcp.server import summarize_stems

    d = tmp_path / "mix_stems"
    d.mkdir()
    _tone(d / "vocals.wav")
    _tone(d / "other.wav", freq=660.0)
    (d / "stems.json").write_text(
        json.dumps({"labels": {"other": [{"label": "flute", "score": 0.9}]}})
    )
    summary = summarize_stems(str(d))
    assert {s["name"] for s in summary["stems"]} == {"vocals", "other"}
    assert summary["labels"]["other"][0]["label"] == "flute"
    assert all(s["bytes"] > 0 and s["duration_sec"] > 0 for s in summary["stems"])
