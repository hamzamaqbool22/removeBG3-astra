"""A single aspect-preserving transform anchored at the nearest tire contact."""
from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Framing:
    width: int = 1024
    height: int = 768
    width_fraction: float = .88
    height_fraction: float = .46
    ground_fraction: float = .80

    def __post_init__(self):
        if min(self.width, self.height) < 128:
            raise ValueError("Output dimensions must be at least 128 pixels")
        if not (0 < self.width_fraction < 1 and 0 < self.height_fraction < self.ground_fraction < .95):
            raise ValueError("Invalid framing fractions")


def normalize(rgb: np.ndarray, alpha: np.ndarray, geometry: dict,
              framing: Framing) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    x0, y0, x1, y1 = geometry['bbox']
    scale = min(framing.width * framing.width_fraction / (x1-x0),
                framing.height * framing.height_fraction / (y1-y0))
    tx = framing.width / 2 - scale * (x0+x1) / 2
    ty = framing.height * framing.ground_fraction - scale * geometry['ground_y']
    matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    # Resample premultiplied color so removed source background cannot bleed
    # into the edge during scaling. The same affine transforms every layer.
    a = alpha.astype(np.float32) / 255
    premult = rgb.astype(np.float32) * a[..., None]
    size = (framing.width, framing.height)
    placed_a = np.clip(cv2.warpAffine(a, matrix, size, flags=cv2.INTER_LANCZOS4), 0, 1)
    placed_p = cv2.warpAffine(premult, matrix, size, flags=cv2.INTER_LANCZOS4)
    color = np.divide(placed_p, placed_a[..., None], out=np.zeros_like(placed_p),
                      where=placed_a[..., None] > 1e-6)
    placed_rgb = np.uint8(np.clip(np.rint(color), 0, 255))
    placed_alpha = np.uint8(np.rint(placed_a * 255))
    g = deepcopy(geometry)
    point = lambda p: [float(scale * p[0] + tx), float(scale * p[1] + ty)]
    g['bbox'] = point([x0,y0]) + point([x1,y1])
    for key in ('footprint', 'chassis_footprint', 'lower_body_profile'):
        if key in g: g[key] = [point(p) for p in g[key]]
    g['ground_y'] = point([0, geometry['ground_y']])[1]
    g['ground_anchor'] = point(g['ground_anchor'])
    for c in g['contacts'] + g['inferred_contacts']:
        c['x'],c['y'] = point([c['x'],c['y']])
        c['radius'] *= scale
        c['half_width'] *= scale
        if 'ellipse' in c:
            e=c['ellipse']; e['cx'],e['cy']=point([e['cx'],e['cy']])
            e['rx'] *= scale; e['ry'] *= scale
    g['view_cues']['opposite_track_shift'] = [float(v*scale) for v in g['view_cues']['opposite_track_shift']]
    placement = {'scale':float(scale), 'translation':[float(tx),float(ty)],
                 'matrix':matrix.tolist(), 'canvas':[framing.width,framing.height],
                 'width_cap_fraction':framing.width_fraction,
                 'height_cap_fraction':framing.height_fraction,
                 'ground_fraction':framing.ground_fraction,
                 'rule':'Uniform contain transform; center vehicle bounds; align nearest estimated tire contact.'}
    return placed_rgb, placed_alpha, g, placement
