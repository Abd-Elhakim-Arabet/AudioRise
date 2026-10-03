"""`audiorise` CLI — Increment 3: `probe` + `separate` + `label` live."""

from __future__ import annotations

import argparse
import json
import sys

from audio_engine import __version__, label_path, probe_audio
from audio_engine.separate import SeparationNotAvailable, default_outdir, separate


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
        outdir = result["stems_json"].rsplit("/", 1)[0]
        print(f"wrote {len(result['stems'])} stems -> {outdir}/")
        for s in result["stems"]:
            tags = ""
            if result.get("labels", {}).get(s["name"]):
                tags = "  [" + ", ".join(
                    f"{t['label']}:{t['score']}" for t in result["labels"][s["name"]]
                ) + "]"
            print(f"  {s['name']}.wav  {s['rms_db']} dB{tags}")
        print(f"  {result['stems_json']}")
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
    s.add_argument(
        "--outdir",
        default=None,
        help="output dir (default: <input-dir>/<name>-stems, e.g. Premier-Night-stems/)",
    )
    s.add_argument("--model", default="htdemucs", help="separator model (default: htdemucs)")
    s.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    s.add_argument("--label", action="store_true", help="also tag stems (Increment 3 heuristic)")
    s.add_argument("--top-k", type=int, default=2, help="tags per stem (default: 2)")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=_cmd_separate)

    lb = sub.add_parser("label", help="tag a stem .wav or a stems dir (best-effort)")
    lb.add_argument("path", help="stem .wav file or dir with vocals/drums/bass/other.wav")
    lb.add_argument("--top-k", type=int, default=2, help="tags per stem (default: 2)")
    lb.add_argument("--json", action="store_true", help="machine-readable output")
    lb.set_defaults(func=_cmd_label)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
