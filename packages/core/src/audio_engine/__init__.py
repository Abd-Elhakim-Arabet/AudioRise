"""AudioRise core engine.

Increment 1: `probe_audio` is live (ffprobe, stdlib only).
`separate` / labels are stubs — Increment 2/3 fill them in.
"""

from .probe import probe_audio
from .separate import EXPECTED_STEMS, SeparationNotAvailable, separate

__all__ = ["probe_audio", "separate", "SeparationNotAvailable", "EXPECTED_STEMS", "__version__"]
__version__ = "0.2.0"
