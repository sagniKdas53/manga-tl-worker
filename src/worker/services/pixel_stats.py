"""Small, dependency-free pixel statistics shared by cleanup decisions.

Relocated out of ``handlers.ocr`` (R3 glyph-mask cleanup) so ``services.cleanup_reconstruct``
can reuse the same statistic to route reconstruction method without a services-import-handlers
dependency, which this package's other modules avoid (see ``services.region_policy``'s
docstring for the same discipline).
"""

import numpy as np


def pixel_spread(pixels: np.ndarray) -> float:
    """Per-channel median absolute deviation, maxed across channels.

    Robust in the two ways a plain stddev over the flattened array is not: a handful of
    anti-aliased text-edge pixels in an otherwise-flat sample barely moves a median-based
    measure, and computing per-channel (then taking the max) rather than over B/G/R mixed
    together means a saturated solid colour reads as flat instead of "spread" by its own
    channel separation.
    """
    medians = np.median(pixels, axis=0)
    return float(np.max(np.median(np.abs(pixels - medians), axis=0)))
