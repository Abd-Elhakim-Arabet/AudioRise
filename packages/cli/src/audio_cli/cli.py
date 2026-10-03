"""`audiorise` CLI — Increment 1: `probe` live, `separate` stub."""

from __future__ import annotations

import argparse
import json
import sys

from audio_engine import __version__, probe_audio
from audio_engine.separate import separate


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
        separate(args.input, outdir=args.outdir, model=args.model)
    except RuntimeError as exc:  # SeparationNotAvailable is a RuntimeError
        print(f"not yet: {exc}", file=sys.stderr)
        print(
            "Hint: Increment 2 (`pip install torch torchaudio demucs`) "
            "makes this command live. Nothing was written.",
            file=sys.stderr,
        )
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiorise", description="AudioRise: audio → stems")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="print duration/rate/channels via ffprobe")
    p.add_argument("input", help="audio file (mp3/wav/flac/ogg/m4a/...)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_probe)

    s = sub.add_parser("separate", help="split into stems (live in Increment 2)")
    s.add_argument("input", help="audio file to separate")
    s.add_argument("--outdir", default="stems", help="output dir (default: stems/)")
    s.add_argument("--model", default="htdemucs", help="separator model (default: htdemucs)")
    s.set_defaults(func=_cmd_separate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
