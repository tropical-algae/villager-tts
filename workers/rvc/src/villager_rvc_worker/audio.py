"""Audio format boundaries for the isolated RVC worker."""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

_FLOAT_PEAK_TOLERANCE = 1.01


def pcm_to_float32(samples: ArrayLike) -> NDArray[np.float32]:
    """Convert signed PCM or normalized floating-point samples to ``[-1, 1]``."""
    array = np.asarray(samples)
    if np.issubdtype(array.dtype, np.signedinteger):
        limits = np.iinfo(array.dtype)
        scale = float(max(abs(limits.min), limits.max))
        normalized = array.astype(np.float32) / scale
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(np.float32, copy=False)
    else:
        msg = f"unsupported RVC audio dtype: {array.dtype}"
        raise TypeError(msg)

    if not np.all(np.isfinite(normalized)):
        msg = "RVC returned non-finite audio samples"
        raise ValueError(msg)
    peak = float(np.max(np.abs(normalized), initial=0.0))
    if peak > _FLOAT_PEAK_TOLERANCE:
        msg = f"RVC returned non-normalized floating-point audio with peak {peak:.3f}"
        raise ValueError(msg)
    clipped = np.clip(normalized, -1.0, 1.0)
    return cast("NDArray[np.float32]", np.ascontiguousarray(clipped, dtype=np.float32))
