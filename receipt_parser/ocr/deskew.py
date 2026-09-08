"""Straighten a rotated receipt before OCR.

Why not measure skew from OCR tokens? That was the first approach, and it
works up to about 3 degrees. Past that it fails, and it fails in a
misleading way -- it reports a small skew rather than no skew.

The reason: token-based estimation finds pairs of words on the same
printed line by looking for words within a vertical band of each other.
As rotation increases, words genuinely on the same line drift further
apart vertically while words on *adjacent* lines drift into the band. The
median slope gets contaminated with cross-line pairs, which have arbitrary
slopes, and collapses toward zero. Measured against synthetic receipts,
the token estimate tracked true rotation to within 4% at 3 degrees, then
reported 16px of drift where the truth was 88px at 5 degrees.

This module measures the pixels instead. Sum the dark pixels in each
image row: when the page is straight, text lines produce sharp peaks and
the gaps between them produce deep troughs, so the profile has high
variance. When the page is rotated, each text line smears across many
rows and the profile flattens. Rotating through candidate angles and
taking the one with maximum variance finds the angle that makes the text
lines horizontal.

It needs no OCR pass to measure, which also means the expensive OCR call
happens once, on an already-straightened image.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Beyond this, the image is not a casually-tilted photo -- it is sideways
# or upside down, which is a different problem (orientation detection)
# and not one a fine angle search should try to solve.
MAX_SKEW_DEGREES = 15.0

# Work at this width for the search. The projection profile measures the
# gross vertical structure of the page, not glyph detail, and downscaling
# makes the angle sweep roughly an order of magnitude cheaper.
SEARCH_WIDTH = 500


def find_skew_angle(
    image: Image.Image,
    *,
    max_degrees: float = MAX_SKEW_DEGREES,
    coarse_step: float = 1.0,
    fine_step: float = 0.1,
) -> float:
    """Return the rotation, in degrees, that best straightens the image.

    Coarse-to-fine: a 1-degree sweep across the full range, then a
    0.1-degree sweep around the winner. A flat 0.1-degree sweep over 30
    degrees would be 300 rotations for the same answer.
    """
    small = _prepare(image)

    coarse = _best_angle(small, _frange(-max_degrees, max_degrees, coarse_step))
    fine = _best_angle(
        small,
        _frange(coarse - coarse_step, coarse + coarse_step, fine_step),
    )
    return fine


# Below this, no rotation makes the text lines noticeably sharper than any
# other -- which means the page is not merely tilted. A curled receipt or
# one shot at an angle has baselines that curve, and no single rotation
# straightens a curve.
#
# Calibrated on four receipts: a curled steakhouse receipt photographed at
# an angle scored 1.33, while three readable ones scored 1.78, 2.02 and
# 2.18. That is a thin basis for a threshold and it should be revisited
# against a larger sample; it is set conservatively so that it fires on
# clear warping rather than on every slightly imperfect photo.
SHARPNESS_THRESHOLD = 1.5


@dataclass
class DeskewResult:
    image_bytes: bytes
    angle: float
    # Ratio of the best projection variance to the median across all
    # candidate angles. High means one rotation clearly wins; low means
    # the page has no consistent horizontal structure to find.
    sharpness: float

    @property
    def looks_warped(self) -> bool:
        return self.sharpness < SHARPNESS_THRESHOLD


def deskew(image_bytes: bytes) -> DeskewResult:
    """Straighten an image and report how confident that straightening is.

    An angle of 0 leaves the bytes untouched rather than passing them
    through a rotation, since resampling costs a little sharpness and
    there is nothing to gain when the page is already square.
    """
    import numpy as np

    image = Image.open(io.BytesIO(image_bytes))
    prepared = _prepare(image)

    coarse_angles = _frange(-MAX_SKEW_DEGREES, MAX_SKEW_DEGREES, 1.0)
    variances = [_profile_variance(prepared, a) for a in coarse_angles]
    median = float(np.median(variances))
    sharpness = (max(variances) / median) if median > 0 else 0.0

    best_coarse = coarse_angles[variances.index(max(variances))]
    angle = _best_angle(prepared, _frange(best_coarse - 1.0, best_coarse + 1.0, 0.1))

    if abs(angle) < fine_threshold():
        return DeskewResult(image_bytes, 0.0, sharpness)

    rotated = image.rotate(
        angle, expand=True, fillcolor=255, resample=Image.BICUBIC
    )
    buffer = io.BytesIO()
    rotated.save(buffer, format="PNG")
    return DeskewResult(buffer.getvalue(), angle, sharpness)


def fine_threshold() -> float:
    """Rotations below this are not worth the resampling cost."""
    return 0.2


def _prepare(image: Image.Image) -> np.ndarray:
    """Grayscale, downscale, and invert to an ink-intensity array.

    Inverted so that "more ink" is a larger number, which makes the row
    sums a direct measure of how much text sits in each row.
    """
    gray = image.convert("L")
    if gray.width > SEARCH_WIDTH:
        scale = SEARCH_WIDTH / gray.width
        gray = gray.resize(
            (SEARCH_WIDTH, max(int(gray.height * scale), 1)), Image.BILINEAR
        )

    array = np.asarray(gray, dtype=np.float32)
    return 255.0 - array


def _best_angle(array: np.ndarray, angles) -> float:
    best_angle = 0.0
    best_score = -1.0

    for angle in angles:
        score = _profile_variance(array, angle)
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _profile_variance(array: np.ndarray, angle: float) -> float:
    """Variance of the horizontal projection profile at a given angle.

    High variance means the rows alternate sharply between "full of ink"
    and "empty" -- which is what horizontal text lines look like. A
    rotated page smears each line across many rows and flattens the
    profile.
    """
    if angle == 0.0:
        rotated = array
    else:
        image = Image.fromarray(array)
        rotated = np.asarray(
            image.rotate(angle, resample=Image.BILINEAR, fillcolor=0),
            dtype=np.float32,
        )

    profile = rotated.sum(axis=1)
    return float(np.var(profile))


def _frange(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step)) + 1
    return [start + i * step for i in range(count)]
