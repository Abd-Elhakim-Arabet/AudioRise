"""Audio → editable notes (Increment 6, MuScriptor backend).

Pipeline: Demucs stem (or any wav) → MuScriptor `transcribe_to_midi`
(conditioned on our Inc3 labels, verified against
`muscriptor list-instruments`) → merged multitrack `.mid`
(one track per instrument, General MIDI programs, drums on ch.10).

Speech stays audio-only: it has no musical notes.
Weights are gated (free HF account + license accept + HF_TOKEN);
`pip install -e packages/core[transcription]` gets the code,
the 3-step login gets the weights.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

# Our fine labels → MuScriptor conditioning groups (verified via
# `muscriptor list-instruments`; None = skip, audio-only).
LABEL_TO_MUSCRIPTOR: dict[str, str | None] = {
    "flute": "flutes",
    "piano": "acoustic_piano",
    "guitar": "acoustic_guitar",
    "violin": "violin",
    "trumpet": "trumpet",
    "saxophone": "tenor_sax",
    "synth": "synth_lead",
    "speech": None,
    "singing": "voice",
    "drums": "drums",
    "bass": "electric_bass",
}

# Our fine labels → General MIDI programs for the merged .mid.
# Drums always go on channel 10 (index 9) per GM spec.
LABEL_TO_GM: dict[str, int] = {
    "flute": 73,
    "piano": 0,
    "guitar": 25,
    "violin": 40,
    "trumpet": 56,
    "saxophone": 66,
    "synth": 81,
    "singing": 52,
    "drums": 0,
    "bass": 33,
}

MODEL_SIZES = ("small", "medium", "large")
DRUM_CHANNEL = 9


class TranscriptionNotAvailable(RuntimeError):
    pass


def _require_muscriptor():
    try:
        from muscriptor import TranscriptionModel

        return TranscriptionModel
    except ImportError as exc:
        raise TranscriptionNotAvailable(
            "transcription needs muscriptor + mido: "
            "`pip install -e packages/core[transcription]`, then unblock the "
            "gated weights: (1) accept the license at "
            "huggingface.co/MuScriptor/muscriptor-small, (2) `hf auth login` "
            "or `export HF_TOKEN=hf_...`."
        ) from exc


def transcribe_file(
    path: str | Path,
    model_size: str = "small",
    instruments: list[str] | None = None,
    device: str | None = None,
) -> bytes:
    """Transcribe one wav to MIDI bytes (MuScriptor, local)."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {MODEL_SIZES}")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"audio file not found: {p}")
    TranscriptionModel = _require_muscriptor()
    try:
        model = TranscriptionModel.load_model(model_size, device=device)
        return model.transcribe_to_midi(str(p), instruments=instruments)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "auth" in msg.lower():
            raise TranscriptionNotAvailable(
                "MuScriptor weights are gated and this machine is not logged in: "
                "(1) accept the license at "
                "huggingface.co/MuScriptor/muscriptor-small, (2) `hf auth login` "
                "or `export HF_TOKEN=hf_...`, then retry. Original error: "
                f"{msg[:200]}"
            ) from exc
        raise


def _rescale_ticks(messages: list, from_ppq: int, to_ppq: int) -> list:
    if from_ppq == to_ppq:
        return messages
    factor = to_ppq / from_ppq
    out = []
    for msg in messages:
        msg = msg.copy()
        if hasattr(msg, "time"):
            msg.time = round(msg.time * factor)
        out.append(msg)
    return out


def merge_tracks(tracks: dict[str, tuple[bytes, int, bool]]) -> bytes:
    """Merge `{stem: (midi_bytes, gm_program, is_drum)}` into one type-1 MIDI.

    One track per stem, named after the stem, program_change at tick 0,
    drums forced onto channel 10. PPQs are normalized to the max found.
    """
    from mido import Message, MetaMessage, MidiFile, MidiTrack

    parsed = {}
    for stem, (data, program, is_drum) in tracks.items():
        mid = MidiFile(file=io.BytesIO(data))
        parsed[stem] = (mid, program, is_drum)
    ppq = max(mid.ticks_per_beat for mid, _, _ in parsed.values())

    merged = MidiFile(type=1, ticks_per_beat=ppq)
    first = True
    channel = 0
    for stem, (mid, program, is_drum) in sorted(parsed.items()):
        ch = DRUM_CHANNEL if is_drum else channel
        if not is_drum:
            channel = channel + 1 if channel + 1 != DRUM_CHANNEL else channel + 2
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=stem, time=0))
        track.append(Message("program_change", program=program, channel=ch, time=0))
        for mtrack in mid.tracks:
            for msg in mtrack:
                if msg.is_meta:
                    # Keep song-level metas once (tempo/map from first stem).
                    if first and msg.type in ("set_tempo", "time_signature"):
                        track.append(msg.copy())
                    continue
                track.append(msg.copy())
        first = False
        track[:] = _rescale_ticks(track, mid.ticks_per_beat, ppq)
        # Force all channel messages onto this track's channel.
        for msg in track:
            if hasattr(msg, "channel"):
                msg.channel = ch
        merged.tracks.append(track)
    buf = io.BytesIO()
    merged.save(file=buf)
    return buf.getvalue()


def count_notes(midi_bytes: bytes) -> int:
    from mido import MidiFile

    mid = MidiFile(file=io.BytesIO(midi_bytes))
    return sum(
        1
        for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0
    )


def _condition_for(stem: str, labels: dict) -> list[str] | None:
    tags = labels.get(stem, []) if labels else []
    groups = [LABEL_TO_MUSCRIPTOR.get(t["label"]) for t in tags]
    groups = [g for g in groups if g]
    # Fall back to the Demucs stem family's best guess when heuristic is unsure.
    if not groups:
        groups = {
            "vocals": ["voice"],
            "drums": ["drums"],
            "bass": ["electric_bass"],
            "other": [],
        }.get(stem, [])
    return groups or None


def _program_for(stem: str, labels: dict) -> tuple[int, bool]:
    tags = labels.get(stem, []) if labels else []
    for t in tags:
        if t["label"] == "speech":
            continue
        if t["label"] in LABEL_TO_GM:
            return LABEL_TO_GM[t["label"]], t["label"] == "drums"
    return {"vocals": 52, "drums": 0, "bass": 33}.get(stem, 0), stem == "drums"


def transcribe_stems(
    stems_dir: str | Path,
    model_size: str = "small",
    condition: bool = True,
    device: str | None = None,
) -> dict:
    """Transcribe every non-silent stem in a dir → merged `<name>.mid`.

    Returns {midi_path, tracks: [{name, program, is_drum, notes, instruments}]}.
    Silent stems (rms_db <= -60 in stems.json) and speech-only stems are kept
    as audio and skipped. Updates stems.json with a `midi` key.
    """
    d = Path(stems_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"stems dir not found: {d}")

    meta_path = d / "stems.json"
    meta: dict = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    labels: dict = meta.get("labels", {})
    rms = {s["name"]: s.get("rms_db", 0.0) for s in meta.get("stems", [])}

    wavs = sorted(p for p in d.glob("*.wav") if "[" not in p.stem)
    if not wavs:
        raise FileNotFoundError(f"no stem wavs in {d}")

    tracks: dict[str, tuple[bytes, int, bool]] = {}
    info: list[dict] = []
    for wav in wavs:
        stem = wav.stem
        if rms.get(stem, 0.0) <= -60.0:
            continue  # silent: stays audio-only
        tags = labels.get(stem, [])
        if tags and all(t["label"] == "speech" for t in tags):
            continue  # speech has no notes: stays audio-only
        instruments = _condition_for(stem, labels) if condition else None
        midi_bytes = transcribe_file(
            wav, model_size=model_size, instruments=instruments, device=device
        )
        program, is_drum = _program_for(stem, labels)
        tracks[stem] = (midi_bytes, program, is_drum)
        info.append(
            {
                "name": stem,
                "program": program,
                "is_drum": is_drum,
                "notes": count_notes(midi_bytes),
                "instruments": instruments,
            }
        )

    if not tracks:
        raise RuntimeError(f"nothing transcribable in {d} (all silent/speech?)")

    base = d.name[:-len("-stems")] if d.name.endswith("-stems") else d.name
    midi_path = d / f"{base}.mid"
    midi_path.write_bytes(merge_tracks(tracks))

    result = {"midi_path": str(midi_path), "tracks": info}
    if meta_path.is_file():
        meta["midi"] = result
        meta_path.write_text(json.dumps(meta, indent=2))
    return result
