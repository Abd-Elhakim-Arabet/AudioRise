"""Increment 3 placeholder — fine-grained instrument labeling.

Strategy (best-effort, never invents stems): keep the proven 4–6 Demucs
stems, run a local classifier per stem, and record tags like
`other -> [flute:0.71, piano:0.22]` in stems.json. File aliases such as
`other[flute,piano].wav` are symlinks/copies, not re-separations.
"""

from __future__ import annotations

FINE_LABELS = (
    "flute",
    "piano",
    "guitar",
    "violin",
    "trumpet",
    "saxophone",
    "synth",
    "speech",
    "singing",
)


def label_stem(stem_path: str, top_k: int = 3) -> list[dict]:
    raise NotImplementedError(
        "Increment 3 not implemented yet — "
        f"would tag {stem_path} with top-{top_k} of {list(FINE_LABELS)}."
    )
