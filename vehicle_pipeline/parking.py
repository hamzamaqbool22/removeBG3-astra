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


def blend_ambient(colors, alpha, background, geometry, matrix):
    """Gentle spatial illumination cues, confined to the existing vehicle alpha.

    This is a photographic approximation, not reconstructed environment lighting.
    Robust low-resolution samples avoid projecting scene texture onto the paint.
    """
    h, w = background.shape[:2]
    scene = cv2.resize(background, (240, 160), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    x0, y0, x1, y1 = geometry['bbox']
    scale, tx, ty = float(matrix[0,0]), float(matrix[0,2]), float(matrix[1,2])

    def chroma(pixels, strength, limit):
        pixels = pixels.reshape(-1,3)
        level = pixels.mean(axis=1)
        valid = (level>.15)&(level<.92)&(np.ptp(pixels,axis=1)<.25)
        if valid.sum()<24:
            return np.ones(3,np.float32)
        sample = np.median(pixels[valid],axis=0)
        return np.clip(1+strength*(sample/sample.mean()-1),1-limit,1+limit)

    # Broad upper-scene cue; restrained so a blue sky does not paint a white car blue.
    sky_gain = chroma(scene[:64,24:216], .20, .025)
    # Three ground patches follow actual placement. Interpolate their colors,
    # never their texture. Missing/out-of-frame samples fall back to neutral.
    gy = int(np.clip((geometry['ground_y']*scale+ty)/h*160, 0, 159))
    road_gains = []
    for fraction in (.15,.50,.85):
        gx = int(np.clip(((x0+(x1-x0)*fraction)*scale+tx)/w*240,0,239))
        patch = scene[max(0,gy-4):min(160,gy+18),max(0,gx-18):min(240,gx+18)]
        road_gains.append(chroma(patch,.35,.045))
    xs = np.clip((np.arange(colors.shape[1])-x0)/max(1,x1-x0),0,1)
    local_ground = np.stack([np.interp(xs,(.15,.50,.85),np.array(road_gains)[:,c])
                             for c in range(3)],axis=1).astype(np.float32)
    height = np.clip((np.arange(colors.shape[0])-y0)/max(1,y1-y0),0,1)[:,None,None]
    lower = np.clip((height-.48)/.48,0,1)
    lower = lower*lower*(3-2*lower)
    # Saturated paint gets less chromatic adaptation. Deep blacks receive no
    # additive lift; wheel rubber and window details keep their black point.
    luminance = colors @ np.array([.2126,.7152,.0722],np.float32)
    saturation = np.ptp(colors,axis=2)/np.maximum(colors.max(axis=2),.05)
    protect = (1-.75*np.clip(saturation,0,1))[...,None]
    gain = 1+protect*((sky_gain[None,None,:]-1)*(1-lower)*.65
                      +(local_ground[None,:,:]-1)*lower)
    # A tiny diffuse ground bounce is strongest on lower midtones, vanishes
    # at black/white, and remains under ~1.6 sRGB levels at neutral pavement.
    bounce = .025*lower*luminance[...,None]*(1-luminance[...,None])
    adjusted = np.clip(colors*gain+bounce*protect*local_ground[None,:,:],0,1)
    return np.where((alpha>0)[...,None],adjusted,colors).astype(np.float32)


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
        colors = blend_ambient(colors, source_alpha, rgb, geometry, matrix)
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
