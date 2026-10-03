"""`audiorise` CLI — probe + separate + label + transcribe live."""

from __future__ import annotations

import argparse
import json
import sys

from audio_engine import __version__, label_path, probe_audio
from audio_engine.separate import SeparationNotAvailable, separate
from audio_engine.transcribe import (
    MODEL_SIZES,
    TranscriptionNotAvailable,
    transcribe_stems,
)


def _cmd_probe(args: argparse.Namespace) -> int:
    try:
        info = probe_audio(args.input)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(
            f"{info['path']}: {info['codec']}, "
            f"{info['sample_rate']} Hz, {info['channels']}ch, "
            f"{info['duration_sec']}s"
        )
    return 0


def _cmd_separate(args: argparse.Namespace) -> int:
    try:
        result = separate(
            args.input,
            outdir=args.outdir,
            model=args.model,
            device=args.device,
            label=args.label,
            top_k=args.top_k,
            midi=args.midi,
            midi_model=args.midi_model,
            midi_condition=not args.no_condition,
        )
    except SeparationNotAvailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Install with: pip install -e packages/core[separation]",
            file=sys.stderr,
        )
        return 3
    except TranscriptionNotAvailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # model download / OOM / corrupt file
        print(f"separation failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        outdir = result["stems_json"].rsplit("/", 1)[0]
        print(f"wrote {len(result['stems'])} stems -> {outdir}/")
        for s in result["stems"]:
            tags = ""
            if result.get("labels", {}).get(s["name"]):
                tags = "  [" + ", ".join(
                    f"{t['label']}:{t['score']}" for t in result["labels"][s["name"]]
                ) + "]"
            print(f"  {s['name']}.wav  {s['rms_db']} dB{tags}")
        if result.get("midi"):
            for t in result["midi"]["tracks"]:
                print(f"  {t['name']}: {t['notes']} notes (GM {t['program']})")
        print(f"  {result['stems_json']}")
        if result.get("midi"):
            print(f"  {result['midi']['midi_path']}")
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    try:
        labels = label_path(args.path, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(labels, indent=2))
    else:
        for stem, tags in sorted(labels.items()):
            desc = ", ".join(f"{t['label']}:{t['score']}" for t in tags) or "(silent/unsure)"
            print(f"  {stem}: {desc}")
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from pathlib import Path

    from audio_engine.transcribe import count_notes, transcribe_file

    try:
        p = Path(args.path)
        if p.is_file():
            midi_bytes = transcribe_file(
                p,
                model_size=args.model_size,
                instruments=None,
                device=args.device,
            )
            out = Path(args.out) if args.out else p.with_suffix(".mid")
            out.write_bytes(midi_bytes)
            result = {
                "midi_path": str(out),
                "tracks": [{"name": p.stem, "notes": count_notes(midi_bytes)}],
            }
        elif p.is_dir():
            result = transcribe_stems(
                p,
                model_size=args.model_size,
                condition=not args.no_condition,
                device=args.device,
            )
            if args.out and args.out != result["midi_path"]:
                Path(result["midi_path"]).rename(args.out)
                result["midi_path"] = args.out
        else:
            print(f"error: file or dir not found: {p}", file=sys.stderr)
            return 2
    except TranscriptionNotAvailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"transcription failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"wrote {result['midi_path']}")
        for t in result["tracks"]:
            extra = f" (GM {t['program']})" if "program" in t else ""
            print(f"  {t['name']}: {t['notes']} notes{extra}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiorise", description="AudioRise: audio → stems + MIDI")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="print duration/rate/channels via ffprobe")
    p.add_argument("input", help="audio file (mp3/wav/flac/ogg/m4a/...)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_probe)

    s = sub.add_parser("separate", help="split into stems locally (Demucs)")
    s.add_argument("input", help="audio file to separate")
    s.add_argument(
        "--outdir",
        default=None,
        help="output dir (default: <input-dir>/<name>-stems, e.g. Premier-Night-stems/)",
    )
    s.add_argument("--model", default="htdemucs", help="separator model (default: htdemucs)")
    s.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    s.add_argument("--label", action="store_true", help="also tag stems (heuristic)")
    s.add_argument("--top-k", type=int, default=2, help="tags per stem (default: 2)")
    s.add_argument(
        "--midi",
        action="store_true",
        help="also transcribe stems → multitrack MIDI (needs HF-gated MuScriptor)",
    )
    s.add_argument(
        "--midi-model",
        default="small",
        choices=list(MODEL_SIZES),
        help="MuScriptor size (default: small)",
    )
    s.add_argument(
        "--no-condition",
        action="store_true",
        help="don't condition transcription on instrument labels",
    )
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=_cmd_separate)

    lb = sub.add_parser("label", help="tag a stem .wav or a stems dir (best-effort)")
    lb.add_argument("path", help="stem .wav file or dir with vocals/drums/bass/other.wav")
    lb.add_argument("--top-k", type=int, default=2, help="tags per stem (default: 2)")
    lb.add_argument("--json", action="store_true", help="machine-readable output")
    lb.set_defaults(func=_cmd_label)

    t = sub.add_parser("transcribe", help="stems/wav → multitrack MIDI (MuScriptor)")
    t.add_argument("path", help="stem .wav file or stems dir")
    t.add_argument(
        "--model-size",
        default="small",
        choices=list(MODEL_SIZES),
        help="MuScriptor size (default: small)",
    )
    t.add_argument("--device", default=None, help="cpu/cuda/mps (default: auto)")
    t.add_argument(
        "--no-condition",
        action="store_true",
        help="don't condition on Inc3 instrument labels",
    )
    t.add_argument("--out", default=None, help="output .mid path (default: next to input)")
    t.add_argument("--json", action="store_true", help="machine-readable output")
    t.set_defaults(func=_cmd_transcribe)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
