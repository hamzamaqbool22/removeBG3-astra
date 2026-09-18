"""Simple background replacement with optional photo enhancement."""
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


def match_vehicle_light(colors, alpha, background):
    """Bounded photo correction from bright neutral paint and neutral pavement.

    A weak asphalt color cue reduces a mismatched blue cast without changing
    geometry or synthesizing reflections. Skip when neutral samples are scarce.
    """
    small = cv2.resize(background, (200, 120)).astype(np.float32) / 255
    road = small[72:108, 30:170].reshape(-1, 3)
    neutral = road[(np.ptp(road, axis=1) < .18) & (road.mean(1) > .18) & (road.mean(1) < .8)]
    paint = colors[(alpha > .98) & (colors.mean(2) > .65) & (np.ptp(colors, axis=2) < .18)]
    gain = np.ones(3, np.float32)
    exposure = 1.
    if len(paint) > 100 and len(neutral) > 100:
        source = np.median(paint, axis=0)
        ambient = np.median(neutral, axis=0)
        target = .6 * ambient / ambient.mean() + .4
        gain = np.clip((target / (source / source.mean())) ** .65, .94, 1.06)
        exposure = float(np.clip(.98 + .12*(ambient.mean()-.5), .95, 1.02))
    # Multiplicative photo correction retains texture and luminance ordering.
    adjusted = np.clip(colors * gain[None, None, :] * exposure, 0, 1)
    return adjusted


def composite_parking(cutout: np.ndarray, shadow: np.ndarray, geometry: dict,
                      background: Image.Image, enhancement: bool = False) -> Image.Image:
    if cutout.ndim != 3 or cutout.shape[2] != 4 or shadow.shape != cutout.shape[:2]:
        raise ValueError("Expected matching RGBA vehicle and grayscale shadow")
    rgb = np.array(ImageOps.exif_transpose(background).convert("RGB"))
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = geometry["bbox"]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0 or min(w, h) <= 12:
        raise ValueError("Vehicle bounds and background must have usable dimensions")
    # Close photographic framing, with a height cap so front/rear views don't
    # get enlarged just to fill the width. Preserve the source perspective.
    scale = min(.88 * w / bw, .48 * h / bh)
    tx = .5 * w - scale * (x0 + x1) / 2
    ty = .55 * h - scale * (y0 + y1) / 2
    ty = float(np.clip(ty, 6 - scale * y0, h - 6 - scale * y1))
    matrix = np.array([[scale, 0, tx], [0, scale, ty]], np.float32)
    source_alpha = cutout[..., 3].astype(np.float32) / 255
    colors = cutout[..., :3].astype(np.float32) / 255
    if enhancement:
        colors = match_vehicle_light(colors, source_alpha, rgb)
    interpolation = cv2.INTER_LANCZOS4
    alpha = np.clip(cv2.warpAffine(source_alpha, matrix, (w, h), flags=interpolation), 0, 1)
    premult = cv2.warpAffine(colors * source_alpha[..., None], matrix, (w, h), flags=interpolation)
    premult = np.clip(premult, 0, alpha[..., None])
    ground_shadow = np.clip(cv2.warpAffine(shadow.astype(np.float32) / 255,
                            matrix, (w, h), flags=interpolation), 0, 1)
    # Ordinary black-alpha overlay, matching the existing Phase 1 shadow.
    # Do not reinterpret its opacity in linear light, which weakened it.
    shaded = rgb.astype(np.float32) / 255 * (1 - ground_shadow[..., None])
    combined = premult + shaded * (1 - alpha[..., None])
    ungraded = np.uint8(np.clip(np.rint(combined * 255), 0, 255))
    final = Image.fromarray(ungraded)
    if enhancement:
        final = ImageEnhance.Contrast(final).enhance(1.02)
    return final
