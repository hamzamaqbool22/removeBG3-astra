"""Inspect supplied remove.bg targets; reference-only, never production inference.

The targets preserve the source framing at a lower resolution. Most added shadow
pixels are semitransparent black, so they can be separated approximately from
the vehicle. This is a useful diagnostic, not hand-labelled segmentation truth:
antialiased dark vehicle pixels and opaque black contacts remain ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def reference_layers(reference: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate vehicle alpha and observable black-shadow alpha (u8).

    Keep near-opaque black pixels on the vehicle side so black tires are not
    erased. Transparent colored fringe pixels belong to the vehicle. This
    decomposition cannot recover any shadow hidden behind the opaque vehicle.
    """
    rgba = np.asarray(reference.convert("RGBA"))
    alpha = rgba[..., 3]
    shadow = (rgba[..., :3].max(axis=2) <= 2) & (alpha < 250)
    vehicle_alpha = np.where(shadow, 0, alpha).astype(np.uint8)
    shadow_alpha = np.where(shadow, alpha, 0).astype(np.uint8)
    return vehicle_alpha, shadow_alpha


def bbox(mask: np.ndarray) -> list[int] | None:
    y, x = np.where(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def white_composite(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def run(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "Diagnostics of supplied references only; no production view labels or reference masks.",
        "caveats": [
            "Vehicle masks are approximate decompositions of reference RGBA, not manual ground truth.",
            "Semitransparent black vehicle edges can be mistaken for shadow; opaque black shadow can remain in the vehicle mask.",
            "Observable shadow masks omit ground shadow occluded by the vehicle.",
            "The reference image is downsampled; edge metrics below its resolution have limited meaning.",
            "All eight images depict one white sedan under similar outdoor conditions; these do not establish generalization.",
        ],
        "framing_recommendation": {
            "canvas_aspect_ratio": "4:3",
            "vehicle_width_cap_fraction": 0.88,
            "vehicle_height_target_fraction": 0.46,
            "nearest_ground_contact_y_fraction": 0.82,
            "scale_rule": "s = min(0.88 * canvas_width / vehicle_width, 0.46 * canvas_height / vehicle_height)",
            "placement_rule": "Center the robust vehicle bounds horizontally; translate the nearest reliable tire contact to the common ground baseline.",
            "explanation": "A common height target limits apparent zoom; a width cap fits long side profiles. Front/back views remain narrower without artificially filling the side-view width. Keep extra space below contacts for blur.",
            "limitation": "A 2D mask cannot establish metric vehicle size or remove perspective differences. Wheel diameter can refine this only when contacts/wheels are reliable.",
        },
        "images": [],
    }
    panel_width, panel_height = 350, 285
    columns = ["Raw", "Reference on white", "Approximate vehicle alpha", "Observable shadow"]
    paths = sorted((root / "images").glob("*.png"))
    collage = Image.new("RGB", (panel_width * 4, panel_height * len(paths)), "#ededed")
    draw = ImageDraw.Draw(collage)
    for row, path in enumerate(paths):
        reference_path = root / "remove.bg-outputs" / f"{path.stem}-removebg-preview.png"
        reference = Image.open(reference_path).convert("RGBA")
        source = Image.open(path).convert("RGB")
        raw_small = source.resize(reference.size, Image.Resampling.LANCZOS)
        rgb = np.asarray(raw_small).astype(np.float32)
        rgba = np.asarray(reference)
        vehicle, shadow = reference_layers(reference)
        b = bbox(vehicle > 127)
        assert b is not None
        x0, y0, x1, y1 = b
        width, height = reference.size
        interior = cv2.erode((vehicle == 255).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        error = np.abs(rgb - rgba[..., :3].astype(np.float32))[interior]
        shadow_support = shadow > 10
        shadow_values = shadow[shadow_support]
        ys, xs = np.where(shadow_support)
        weights = shadow[shadow_support].astype(float)
        metrics = {
            "name": path.stem,
            "source_size": list(source.size),
            "reference_size": list(reference.size),
            "source_to_reference_scale_xy": [width / source.width, height / source.height],
            "opaque_interior_rgb_mae_0_255": round(float(error.mean()), 4),
            "opaque_interior_rgb_p95_absolute_error": float(np.percentile(error, 95)),
            "vehicle_bbox_alpha_gt_127": b,
            "vehicle_width_fraction": round((x1-x0) / width, 5),
            "vehicle_height_fraction": round((y1-y0) / height, 5),
            "vehicle_bottom_fraction": round(y1 / height, 5),
            "vehicle_projected_aspect_ratio": round((x1-x0) / (y1-y0), 5),
            "shadow_bbox_alpha_gt_10": bbox(shadow_support),
            "shadow_alpha_support_quantiles": [float(x) for x in np.percentile(shadow_values, [10, 50, 90, 99])],
            "shadow_opacity_weighted_centroid": [round(float(np.average(xs, weights=weights)), 3), round(float(np.average(ys, weights=weights)), 3)],
            "visible_shadow_equivalent_opaque_area_pixels": round(float(shadow.sum()) / 255, 3),
            "shadow_pixels_gt_10": int(shadow_support.sum()),
        }
        report["images"].append(metrics)
        folder = out / path.stem
        folder.mkdir(exist_ok=True)
        Image.fromarray(vehicle).save(folder / "approximate_vehicle_alpha.png")
        Image.fromarray(shadow).save(folder / "observable_shadow_alpha.png")
        reference_white = white_composite(reference)
        reference_white.save(folder / "reference_white.png")
        panels = [raw_small, reference_white, Image.fromarray(vehicle).convert("RGB"), Image.fromarray(255-shadow).convert("RGB")]
        for col, panel in enumerate(panels):
            panel.thumbnail((panel_width-10, panel_height-30), Image.Resampling.LANCZOS)
            x = col * panel_width + (panel_width-panel.width)//2
            y = row * panel_height + 24
            collage.paste(panel, (x, y))
            draw.text((col*panel_width+7,row*panel_height+6),f"{path.stem} / {columns[col]}", fill="black")
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    collage.save(out / "reference_decomposition.jpg", quality=94)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.root, args.output or args.root / "outputs" / "reference_analysis")
    print(json.dumps(result, indent=2))
