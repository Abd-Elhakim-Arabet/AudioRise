"""`audiorise` CLI — Increment 2: `probe` + `separate` (local Demucs) live."""

from __future__ import annotations

import argparse
import json
import sys

from audio_engine import __version__, probe_audio
from audio_engine.separate import SeparationNotAvailable, separate


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
            args.input, outdir=args.outdir, model=args.model, device=args.device
        )
    except SeparationNotAvailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "Install with: pip install -e packages/core[separation]",
            file=sys.stderr,
        )
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
        print(f"wrote {len(result['stems'])} stems -> {args.outdir}/")
        for s in result["stems"]:
            print(f"  {s['name']}.wav  {s['rms_db']} dB")
        print(f"  {result['stems_json']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="audiorise", description="AudioRise: audio → stems")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="print duration/rate/channels via ffprobe")
    p.add_argument("input", help="audio file (mp3/wav/flac/ogg/m4a/...)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_probe)

    s = sub.add_parser("separate", help="split into stems locally (Demucs)")
    s.add_argument("input", help="audio file to separate")
    s.add_argument("--outdir", default="stems", help="output dir (default: stems/)")
    s.add_argument("--model", default="htdemucs", help="separator model (default: htdemucs)")
    s.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=_cmd_separate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
