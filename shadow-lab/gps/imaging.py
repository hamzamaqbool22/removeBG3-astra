"""Image-only helpers; no GPU or diffusion imports."""
import cv2
import numpy as np


def letterbox(rgb, mask, size=512):
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        raise ValueError('Composite and object mask must have identical dimensions')
    if min(h, w) < 8 or not np.any(mask > 127):
        raise ValueError('Image is too small or the object mask is empty')
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(small, y, size-nh-y, x, size-nw-x, cv2.BORDER_REPLICATE)
    alpha = np.zeros((size, size), np.uint8)
    alpha[y:y+nh, x:x+nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    return padded, alpha, (x, y, nw, nh)


def restore_map(value, box, shape):
    x, y, w, h = box
    return cv2.resize(value[y:y+h, x:x+w], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def shadow_opacity(original, generated, support):
    """Experimental neutral darkening layer, not generated asphalt/vehicle pixels."""
    coeff = np.array([.2126, .7152, .0722], np.float32)
    before = original.astype(np.float32) @ coeff
    after = generated.astype(np.float32) @ coeff
    darkening = np.clip((before-after) / np.maximum(before, 8), 0, .95)
    # Smooth only the transfer map, never the actual photo.
    darkening = cv2.GaussianBlur(darkening, (0, 0), .7)
    return darkening * np.clip(support, 0, 1)


def apply_shadow(composite, background, alpha, opacity):
    if composite.shape != background.shape or alpha.shape != composite.shape[:2] or opacity.shape != alpha.shape:
        raise ValueError('All full-resolution layers must be aligned')
    a = alpha.astype(np.float32) / 255
    # Composite already contains premultiplied vehicle + background*(1-alpha).
    # Darken the background contribution only, including at soft vehicle edges.
    result = composite.astype(np.float32) - background.astype(np.float32) * (1-a[...,None]) * np.clip(opacity,0,.95)[...,None]
    return np.uint8(np.clip(np.rint(result), 0, 255))


def geometry_region(prediction, foreground_box, size=512):
    """Decode upstream rotated-box regression without mutating its input."""
    p = np.asarray(prediction, np.float32)
    fg = np.asarray(foreground_box, np.float32)
    if p.shape != (5,) or not np.all(np.isfinite(p)):
        raise ValueError('Invalid predicted geometry')
    x, y = p[:2] * fg[2:4] + fg[:2]
    w, h = fg[2:4] * np.exp(np.clip(p[2:4], -6, 6))
    theta = float(p[4] * 180 / np.pi + fg[4])
    if theta > 0:
        w, h = h, w
        theta -= 90
    points = cv2.boxPoints(((float(x),float(y)),(float(w),float(h)),theta))
    points = np.clip(points, 0, size-1).astype(np.int32)
    region = np.zeros((size, size), np.uint8)
    cv2.fillPoly(region, [points], 1)
    return region
