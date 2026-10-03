"""AudioRise core engine — probe + separate + labels + transcribe (all live)."""

from .labels import FINE_LABELS, label_path, label_stem, label_stems
from .probe import probe_audio
from .separate import (
    EXPECTED_STEMS,
    SeparationNotAvailable,
    default_outdir,
    separate,
    slugify_stem,
)
from .transcribe import (
    LABEL_TO_GM,
    LABEL_TO_MUSCRIPTOR,
    MODEL_SIZES,
    TranscriptionNotAvailable,
    count_notes,
    merge_tracks,
    transcribe_file,
    transcribe_stems,
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
    "transcribe_file",
    "transcribe_stems",
    "merge_tracks",
    "count_notes",
    "TranscriptionNotAvailable",
    "LABEL_TO_GM",
    "LABEL_TO_MUSCRIPTOR",
    "MODEL_SIZES",
    "__version__",
]
__version__ = "0.5.0"
