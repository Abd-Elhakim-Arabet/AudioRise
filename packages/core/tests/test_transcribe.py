"""Increment 6 tests — MIDI merge/mapping logic, no weights (mocked inference)."""

import io
import json
import sys

import pytest


def _synth_midi(notes, ppq=480, program=0):
    """Build tiny MIDI bytes: notes = [(pitch, start_beats, len_beats)]."""
    from mido import Message, MidiFile, MidiTrack

    mid = MidiFile(type=0, ticks_per_beat=ppq)
    track = MidiTrack()
    track.append(Message("program_change", program=program, time=0))
    cursor = 0
    for pitch, start, length in notes:
        track.append(
            Message("note_on", note=pitch, velocity=90,
                    time=round(start * ppq) - cursor)
        )
        cursor = round(start * ppq)
        track.append(
            Message("note_off", note=pitch, velocity=0,
                    time=round(length * ppq))
        )
        cursor += round(length * ppq)
    mid.tracks.append(track)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def test_label_maps_cover_fine_labels():
    from audio_engine.transcribe import LABEL_TO_GM, LABEL_TO_MUSCRIPTOR
    from audio_engine.labels import FINE_LABELS

    for label in FINE_LABELS:
        assert label in LABEL_TO_MUSCRIPTOR
        if label != "speech":
            assert label in LABEL_TO_GM
    assert LABEL_TO_MUSCRIPTOR["speech"] is None  # audio-only
    assert all(isinstance(v, int) and 0 <= v <= 127 for v in LABEL_TO_GM.values())


def test_merge_tracks_programs_channels_names():
    from mido import MidiFile
    from audio_engine.transcribe import count_notes, merge_tracks

    a = _synth_midi([(69, 0, 1), (72, 1, 1)], ppq=480)
    b = _synth_midi([(36, 0, 0.5)], ppq=960)  # different ppq
    assert count_notes(a) == 2 and count_notes(b) == 1

    merged = MidiFile(
        file=io.BytesIO(merge_tracks({"other": (a, 73, False), "drums": (b, 0, True)}))
    )
    assert merged.type == 1 and len(merged.tracks) == 2
    by_name = {}
    for track in merged.tracks:
        name = next(m.name for m in track if m.type == "track_name")
        prog = next(m.program for m in track if m.type == "program_change")
        channels = {m.channel for m in track if hasattr(m, "channel")}
        by_name[name] = (prog, channels)
    assert by_name["other"] == (73, {0})
    assert by_name["drums"] == (0, {9})  # GM drums → channel 10
    total = sum(
        1 for t in merged.tracks for m in t
        if m.type == "note_on" and m.velocity > 0
    )
    assert total == 3


def test_transcribe_file_validates_without_weights():
    import importlib

    tr = importlib.import_module("audio_engine.transcribe")
    with pytest.raises(ValueError, match="model_size"):
        tr.transcribe_file("/tmp/x.wav", model_size="xl")
    with pytest.raises(FileNotFoundError):
        tr.transcribe_file("/tmp/definitely-not-here.wav")


def test_transcribe_file_missing_package_error(tmp_path, monkeypatch):
    import importlib

    tr = importlib.import_module("audio_engine.transcribe")
    tone = tmp_path / "a.wav"
    tone.write_bytes(b"RIFF....")  # exists; import fails first
    monkeypatch.setitem(sys.modules, "muscriptor", None)
    with pytest.raises(tr.TranscriptionNotAvailable, match="pip install"):
        tr.transcribe_file(tone)


def test_transcribe_stems_mocked(tmp_path, monkeypatch):
    import importlib

    tr = importlib.import_module("audio_engine.transcribe")
    d = tmp_path / "Song-stems"
    d.mkdir()
    for stem in ("vocals", "drums", "bass", "other"):
        (d / f"{stem}.wav").write_bytes(b"RIFF....")
    (d / "stems.json").write_text(json.dumps({
        "stems": [
            {"name": "vocals", "rms_db": -20.0},
            {"name": "drums", "rms_db": -120.0},  # silent → skipped
            {"name": "bass", "rms_db": -18.0},
            {"name": "other", "rms_db": -22.0},
        ],
        "labels": {
            "vocals": [{"label": "singing", "score": 0.8}],
            "bass": [{"label": "bass", "score": 0.9}],
            "other": [{"label": "flute", "score": 0.93}],
        },
    }))

    seen = {}

    def fake_transcribe(path, model_size="small", instruments=None, device=None):
        seen[__import__("pathlib").Path(path).stem] = instruments
        pitch = {"vocals": 60, "bass": 40, "other": 74}[__import__("pathlib").Path(path).stem]
        return _synth_midi([(pitch, 0, 1)])

    monkeypatch.setattr(tr, "transcribe_file", fake_transcribe)
    result = tr.transcribe_stems(d, model_size="small")

    assert result["midi_path"].endswith("Song.mid")
    assert (d / "Song.mid").is_file()
    assert {t["name"] for t in result["tracks"]} == {"vocals", "bass", "other"}
    assert seen["other"] == ["flutes"]  # conditioned on Inc3 label
    assert seen["bass"] == ["electric_bass"]
    progs = {t["name"]: t["program"] for t in result["tracks"]}
    assert progs == {"vocals": 52, "bass": 33, "other": 73}
    meta = json.loads((d / "stems.json").read_text())
    assert meta["midi"]["midi_path"].endswith("Song.mid")
