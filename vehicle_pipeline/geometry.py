"""Conservative, image-only estimates of vehicle support geometry.

Coordinates are in the original image. Wheel detections are ellipse hypotheses,
not semantic keypoints: confidence is deliberately lower for occluded wheels.
The footprint is an inferred ground-plane support quadrilateral, not a copied
vehicle silhouette. No view names or reference images enter the estimator.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


def _largest_component(alpha: np.ndarray) -> np.ndarray:
    binary = (alpha >= 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        raise ValueError("The foreground mask is empty.")
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def _bottom_profile(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    bottom = y1 - 1 - np.argmax(crop[::-1], axis=0)
    return gaussian_filter1d(median_filter(bottom.astype(float), size=5), 1.2)


def _sample(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(array, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _ellipse_score(gray: np.ndarray, mask: np.ndarray, bottom: np.ndarray,
                   bbox: list[int], cx: float, cy: float, rx: float,
                   ry: float) -> tuple[float, dict[str, float]]:
    x0, y0, x1, y1 = bbox
    bh, bw = y1 - y0, x1 - x0
    if not (x0 + .018 * bw < cx < x1 - .018 * bw and
            y0 + .51 * bh < cy < y1 - .035 * bh and
            .105 * bh < ry < .315 * bh):
        return 0., {}
    bx = int(np.clip(round(cx) - x0, 0, len(bottom) - 1))
    contact_error = abs(cy + ry - bottom[bx])
    if contact_error > .11 * bh:
        return 0., {}
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    levels = np.array([.81, .9, .98])[:, None]
    xx = cx + rx * levels * np.cos(angles)
    yy = cy + ry * levels * np.sin(angles)
    ring = _sample(gray, xx, yy)
    ring_fg = _sample(mask * 255, xx, yy) / 255
    if float(np.mean(ring_fg)) < .88:
        return 0., {}
    rubber = float(np.mean(ring < 86))
    ring_mean = float(np.mean(ring))
    if rubber < .57 or ring_mean > 96:
        return 0., {}
    inner_levels = np.linspace(.12, .68, 8)[:, None]
    inside = _sample(gray, cx + rx * inner_levels * np.cos(angles),
                     cy + ry * inner_levels * np.sin(angles))
    texture = float(np.clip((np.percentile(inside, 88) - np.percentile(inside, 12) - 15) / 75, 0, 1))
    bright_spokes = float(np.clip((np.percentile(inside, 85) - np.mean(ring) - 5) / 55, 0, 1))
    contact = float(np.exp(-.5 * (contact_error / max(3, .037 * bh)) ** 2))
    # Wheels are dark at their bottom and typically occupy a rounded protrusion
    # of the silhouette. This rejects rim-like circles in grilles and lamps.
    spread = max(2, int(rx * .85))
    local_x = np.clip([bx - spread, bx + spread], 0, len(bottom) - 1)
    protrusion = float(np.clip((bottom[bx] - np.mean(bottom[local_x])) / max(3, .34 * ry), 0, 1))
    score = .32 * rubber + .23 * contact + .14 * texture + .12 * bright_spokes + .19 * protrusion
    if contact < .14 or (texture < .12 and protrusion < .2):
        score *= .62
    return float(score), {"rubber": rubber, "rim_detail": texture,
                          "contact_agreement": contact, "protrusion": protrusion}


def _wheel_ellipses(rgb: np.ndarray, mask: np.ndarray, bbox: list[int],
                    bottom: np.ndarray, expanded: bool = False) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    crop_y = int(y0 + .48 * bh)
    roi = gray[crop_y:y1, x0:x1]
    roi = cv2.GaussianBlur(roi, (5, 5), 1)
    scale = min(1., 760. / bw)
    proposals: list[tuple[float, float, float, float]] = []
    # Silhouette protrusions supplement Hough: a dark oblique rim may yield a
    # misleading small circle in its lower half. Search tire-sized ellipses
    # above each rounded support and score them against the same RGB evidence.
    peaks, _ = find_peaks(bottom, prominence=max(4., .02 * bh), distance=max(12,int(.12*bw)))
    for peak in peaks:
        for xshift in (-.035, 0, .035):
            cx = x0 + peak + xshift*bh
            for rf in ((.11,.13,.15,.18,.21,.24) if expanded else (.15,.18,.21,.24)):
                ry=rf*bh
                for oval in (.48,.62,.76,.94):
                    proposals.append((cx, float(bottom[peak])-ry, ry*oval, ry))
    for ellipse_ratio in (.48, .62, .78, .96):
        sx, sy = scale / ellipse_ratio, scale
        stretched = cv2.resize(roi, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
        circles = cv2.HoughCircles(stretched, cv2.HOUGH_GRADIENT, dp=1.3,
                                  minDist=max(12, bh * sy * .14), param1=85, param2=24,
                                  minRadius=max(7, int(bh * sy * .10)),
                                  maxRadius=max(9, int(bh * sy * .28)))
        if circles is None:
            continue
        for cx, cy, radius in circles[0][:100]:
            for expansion in (1., 1.12, 1.25, 1.38):
                proposals.append((x0 + float(cx) / sx, crop_y + float(cy) / sy,
                                  float(radius) / sx * expansion, float(radius) / sy * expansion))
    ranked: list[dict[str, Any]] = []
    for cx, cy, rx, ry in proposals:
        score, evidence = _ellipse_score(gray, mask, bottom, bbox, cx, cy, rx, ry)
        # A tire is usually a substantial fraction of vehicle height. This
        # weak prior breaks ties against smaller circles nested inside a rim.
        score += .065 * float(np.clip((ry/bh-.12)/.10,0,1)) if score else 0
        if expanded and evidence.get('protrusion',0) < .15:
            score *= .62
        if score >= .63:
            ranked.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry,
                           "score": score, "evidence": evidence})
    ranked.sort(key=lambda c: c["score"], reverse=True)
    candidates: list[dict[str, Any]] = []
    for candidate in ranked:
        if not any(abs(candidate["cx"] - old["cx"]) < max(candidate["rx"], old["rx"]) * .18
                   and abs(candidate["cy"] - old["cy"]) < max(candidate["ry"], old["ry"]) * .18
                   and abs(candidate['rx']/candidate['ry']-old['rx']/old['ry']) < .08
                   for old in candidates):
            candidates.append(candidate)
        if len(candidates) >= 90:
            break
    if len(candidates) < 2:
        return candidates
    # A coherent pair is more reliable than the highest individual circle. The
    # tire ellipse's apparent widths and scale should change together in depth.
    pairs = []
    for a, b in combinations(candidates, 2):
        span = abs(a["cx"] - b["cx"])
        ratio = min(a["ry"], b["ry"]) / max(a["ry"], b["ry"])
        if span < .28 * bw or ratio < (.40 if expanded else .48):
            continue
        oval_a, oval_b = a["rx"] / a["ry"], b["rx"] / b["ry"]
        if abs(oval_a - oval_b) > .34:
            continue
        pair_score = a["score"] + b["score"] + .13 * min(span / (.65 * bw), 1) - .8 * abs(oval_a - oval_b)
        pairs.append((pair_score, [a, b]))
    selected = max(pairs, key=lambda pair: pair[0])[1] if pairs else candidates[:1]
    # Only widen the search when a selected "wheel" has no rounded contact
    # protrusion. High-camera views can expose a small distant wheel; a grille
    # can otherwise win the conventional-size search and reverse the footprint.
    if not expanded and (len(selected) < 2 or any(c['evidence']['protrusion'] < .15 for c in selected)):
        refined = _wheel_ellipses(rgb, mask, bbox, bottom, expanded=True)
        if len(refined) == 2:
            return refined
    return selected


def _fallback_contacts(rgb: np.ndarray, mask: np.ndarray, bbox: list[int],
                       bottom: np.ndarray) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contacts = []
    # For a head-on vehicle the wheel disks are hidden. Search the lateral
    # lower silhouette for the darkest deep supports, without fabricating rims.
    for lo, hi in ((.035, .26), (.74, .965)):
        xs = np.arange(int(x0 + lo * bw), int(x0 + hi * bw))
        if not len(xs):
            continue
        ys = bottom[np.clip(xs - x0, 0, len(bottom) - 1)]
        darkness = []
        for x, y in zip(xs, ys):
            yy = max(y0, int(y - .06 * bh))
            patch = gray[yy:int(y) + 1, max(x0, x - 2):min(x1, x + 3)]
            darkness.append(1. - float(np.mean(patch)) / 255. if patch.size else 0.)
        darkness = gaussian_filter1d(np.asarray(darkness), max(1, .008 * bw))
        depth = (ys - np.percentile(bottom, 45)) / max(1., .07 * bh)
        score = depth + .48 * darkness
        peak = int(np.argmax(score))
        threshold = score[peak] - .12
        good = np.where(score >= threshold)[0]
        near_peak = good[abs(good - peak) < .055 * bw]
        ix = int(np.median(near_peak)) if len(near_peak) else peak
        radius = float(np.clip(.145 * bh, .035 * bw, .11 * bw))
        contacts.append({"x": float(xs[ix]), "y": float(ys[ix]), "radius": radius,
                         "half_width": float(max(.025 * bw, .35 * radius)),
                         "confidence": float(.39 + .16 * darkness[ix]),
                         "source": "occluded_tire_lower_silhouette"})
    return contacts


def estimate_geometry(rgb: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """Estimate tire contacts and projected support footprint from RGB + alpha.

    ``bbox`` is [x0, y0, x1, y1] with exclusive upper bounds. ``footprint``
    contains the four ground-plane tire support corners in cyclic order.
    ``contacts`` includes measured/occluded supports; ``inferred_contacts``
    contains the hidden opposite-side pair, suitable for softer shading only.
    The normalization ground anchor is the lowest credible contact, not a
    bumper extreme. Radii describe the tire's vertical image radius.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != alpha.shape:
        raise ValueError("Expected matching RGB [H,W,3] and alpha [H,W] arrays.")
    mask = _largest_component(alpha)
    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    if min(bw, bh) < 24:
        raise ValueError("Vehicle foreground is too small for contact estimation.")
    bottom = _bottom_profile(mask, bbox)
    # Narrow projected bodies cannot expose a coherent side wheelbase. Avoid
    # interpreting large front grilles as a pair of wheel disks.
    wheels = sorted(_wheel_ellipses(rgb, mask, bbox, bottom), key=lambda wheel: wheel["cx"]) if bw/bh > 1.75 else []
    contacts = []
    for wheel in wheels:
        ix = int(np.clip(round(wheel["cx"]) - x0, 0, len(bottom) - 1))
        # The silhouette gives a less noisy contact than a Hough radius. Sample
        # a narrow region so a tilted tire's rounded bottom stays attached.
        half_patch = max(2, int(.24 * wheel["rx"]))
        lo, hi = max(0, ix - half_patch), min(len(bottom), ix + half_patch + 1)
        contact_y = float(np.percentile(bottom[lo:hi], 90))
        contacts.append({"x": float(wheel["cx"]), "y": contact_y,
                         "radius": float(wheel["ry"]), "half_width": float(wheel["rx"] * .32),
                         "confidence": float(min(.99,wheel["score"])), "source": "rubber_rim_ellipse",
                         "ellipse": {"cx": float(wheel["cx"]), "cy": float(wheel["cy"]),
                                     "rx": float(wheel["rx"]), "ry": float(wheel["ry"])},
                         "evidence": wheel["evidence"]})
    inferred = []
    if len(contacts) == 2:
        left, right = contacts
        left_overhang = left["x"] - x0 - wheels[0]["rx"]
        right_overhang = x1 - right["x"] - wheels[1]["rx"]
        overhang_asymmetry = (left_overhang - right_overhang) / bw
        wheel_oval = float(np.mean([w["rx"] / w["ry"] for w in wheels]))
        # A profile car's wheel disks are nearly circular and its track projects
        # almost vertically. Increasing visible end-face width supplies track.
        shift_x = float(np.clip(-.85 * (left_overhang - right_overhang), -.42 * bw, .42 * bw))
        if abs(overhang_asymmetry) < .065:
            shift_x *= .35
        shift_y = -float(.105 * bh + abs(shift_x) * .10)
        for contact in contacts:
            inferred.append({"x": float(contact["x"] + shift_x), "y": float(contact["y"] + shift_y),
                             "radius": float(contact["radius"] * .9), "half_width": float(contact["half_width"]),
                             "confidence": .34, "source": "inferred_opposite_track"})
        footprint = [[left["x"], left["y"]], [right["x"], right["y"]],
                     [inferred[1]["x"], inferred[1]["y"]], [inferred[0]["x"], inferred[0]["y"]]]
        perspective = float(np.clip(abs(shift_x) / (.3 * bw), 0, 1))
        side_on = float(np.clip((wheel_oval - .5) / .48, 0, 1) * (1. - .6 * perspective))
        support_mode = "visible_wheel_pair"
    else:
        contacts = _fallback_contacts(rgb, mask, bbox, bottom)
        left, right = contacts
        width = right["x"] - left["x"]
        depth = .12 * bh
        footprint = [[left["x"], left["y"]], [right["x"], right["y"]],
                     [right["x"] - .11 * width, right["y"] - depth],
                     [left["x"] + .11 * width, left["y"] - depth]]
        shift_x, shift_y, overhang_asymmetry, wheel_oval = 0., -depth, 0., 0.
        perspective, side_on, support_mode = 0., 0., "occluded_axle_pair"
    support = np.asarray(footprint,dtype=float)
    if support_mode == 'visible_wheel_pair':
        along = support[1]-support[0]
        support[[0,3]] -= .27*along
        support[[1,2]] += .27*along
    else:
        support[[0,3],0] -= .035*bw
        support[[1,2],0] += .035*bw
    support[:,0] = np.clip(support[:,0],x0+.025*bw,x1-.025*bw)
    ground_y = max(contact["y"] for contact in contacts)
    support[:2,1] = np.minimum(support[:2,1],ground_y+.06*np.median([c['radius'] for c in contacts]))
    confidence = float(np.mean([contact["confidence"] for contact in contacts]))
    return {"bbox": bbox, "contacts": contacts, "inferred_contacts": inferred,
            "lower_body_profile": [[float(x0+i),float(bottom[i])] for i in np.unique(np.linspace(0,len(bottom)-1,min(128,len(bottom))).astype(int))],
            "footprint": footprint, "chassis_footprint":support.tolist(), "ground_y": float(ground_y),
            "ground_anchor": [float(np.mean([c["x"] for c in contacts])), float(ground_y)],
            "confidence": confidence,
            "view_cues": {"support_mode": support_mode, "detected_wheel_count": len(wheels),
                          "aspect_ratio": float(bw / bh), "wheel_ellipse_ratio": wheel_oval,
                          "side_on_score": side_on, "perspective_score": perspective,
                          "overhang_asymmetry": float(overhang_asymmetry),
                          "opposite_track_shift": [float(shift_x), float(shift_y)],
                          "contact_slope": float((contacts[-1]["y"] - contacts[0]["y"]) /
                                                 max(1, contacts[-1]["x"] - contacts[0]["x"]))},
            "limitations": ["Support geometry is a monocular heuristic, not calibrated 3D reconstruction.",
                            "Occluded tire contacts and opposite track are uncertain estimates."]}

