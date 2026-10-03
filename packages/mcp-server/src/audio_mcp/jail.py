"""Path jailing for the MCP server (stdlib only, independently testable).

Roots come from `AUDIORISE_MCP_ROOTS` (OS path-separator separated).
Default: repo toplevel (git) or cwd, plus the system temp dir.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def default_roots() -> list[str]:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        repo = top.stdout.strip() if top.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        repo = ""
    roots = [repo or os.getcwd(), tempfile.gettempdir()]
    return [str(Path(r).resolve()) for r in roots]


def roots() -> list[Path]:
    raw = os.environ.get("AUDIORISE_MCP_ROOTS", "")
    parts = [p for p in raw.split(os.pathsep) if p.strip()] or default_roots()
    return [Path(p).resolve() for p in parts]


def jailed(path: str | Path) -> Path:
    """Resolve `path` and require it inside one of the roots.

    Raises ValueError otherwise. Non-existent paths are allowed
    (resolution is non-strict) so tools can create outputs.
    """
    p = Path(path).expanduser().resolve()
    if not any(p == r or p.is_relative_to(r) for r in roots()):
        raise ValueError(
            f"path outside AUDIORISE_MCP_ROOTS: {p} "
            f"(roots: {[str(r) for r in roots()]})"
        )
    return p
