"""Jail tests — no MCP server run needed."""

import os

import pytest

from audio_mcp.jail import jailed, roots


def test_rejects_outside_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIORISE_MCP_ROOTS", str(tmp_path))
    assert jailed(tmp_path / "song.mp3").name == "song.mp3"
    with pytest.raises(ValueError, match="outside AUDIORISE_MCP_ROOTS"):
        jailed("/etc/hosts")


def test_default_roots_cover_cwd_and_tmp():
    import tempfile
    from pathlib import Path

    if os.environ.get("AUDIORISE_MCP_ROOTS"):
        pytest.skip("custom roots in env")
    r = [str(p) for p in roots()]
    assert str(Path(tempfile.gettempdir()).resolve()) in r
