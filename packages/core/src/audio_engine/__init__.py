"""AudioRise core engine — probe + separate + labels (all live)."""

from .labels import FINE_LABELS, label_path, label_stem, label_stems
from .probe import probe_audio
from .separate import (
    EXPECTED_STEMS,
    SeparationNotAvailable,
    default_outdir,
    separate,
    slugify_stem,
)

__all__ = [
    "probe_audio",
    "separate",
    "SeparationNotAvailable",
    "EXPECTED_STEMS",
    "default_outdir",
    "slugify_stem",
    "FINE_LABELS",
    "label_path",
    "label_stem",
    "label_stems",
    "__version__",
]
__version__ = "0.4.0"
