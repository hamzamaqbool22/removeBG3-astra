"""Phase 1 pipeline; references are deliberately absent from this module."""
from dataclasses import dataclass
import time

import numpy as np
from PIL import Image, ImageOps

from .segmentation import Segmenter, refine_mask
from .geometry import estimate_geometry, draw_geometry_overlay
from .placement import Framing, normalize
from .shadows import render_shadows, composite_white


@dataclass
class ProcessedVehicle:
    final: np.ndarray
    stages: dict[str, np.ndarray]
    metadata: dict


class VehiclePipeline:
    def __init__(self, model="birefnet-general", threads=6, framing: Framing | None = None):
        self.model = model
        self.threads = threads
        self.framing = framing or Framing()
        self.segmenter = None

    def process(self, image: Image.Image, mask: np.ndarray | None = None) -> ProcessedVehicle:
        start = time.perf_counter()
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        timings = {}
        if mask is None:
            if self.segmenter is None:
                stage_start = time.perf_counter()
                self.segmenter = Segmenter(self.model, self.threads)
                timings["model_load_seconds"] = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            mask = self.segmenter.predict(rgb)
            timings["segmentation_seconds"] = time.perf_counter() - stage_start
            mask_origin = "model_inference"
            segmentation_info = self.segmenter.last_info.copy()
        else:
            if mask.shape != rgb.shape[:2]:
                raise ValueError("Cached mask dimensions do not match source")
            mask_origin = "cached_segmentation_for_development"
            segmentation_info = {"inference_passes": 0, "note": "cached mask"}

        stage_start = time.perf_counter()
        alpha = refine_mask(mask)
        timings["mask_refinement_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        geometry = estimate_geometry(rgb, alpha)
        timings["geometry_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        color, placed_alpha, placed_geometry, placement = normalize(
            rgb, alpha, geometry, self.framing
        )
        timings["placement_seconds"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        layers = render_shadows(placed_alpha.shape, placed_geometry)
        final = composite_white(color, placed_alpha, layers["combined"])
        timings["shadow_and_compositing_seconds"] = time.perf_counter() - stage_start
        timings["processing_seconds"] = time.perf_counter() - start

        normalized_white = composite_white(color, placed_alpha)
        stages = {
            "segmentation_mask": mask,
            "refined_mask": alpha,
            "geometry_contacts": draw_geometry_overlay(rgb, alpha, geometry),
            "normalized_placement": normalized_white,
            "normalized_alpha": placed_alpha,
            "normalized_cutout": np.dstack([color, placed_alpha]),
            "normalized_geometry": draw_geometry_overlay(
                normalized_white, placed_alpha, placed_geometry
            ),
        }
        for name, layer in layers.items():
            stages[f"shadow_{name}_alpha"] = np.uint8(np.rint(layer * 255))
            stages[f"shadow_{name}_white"] = np.uint8(np.rint((1 - layer) * 255))

        metadata = {
            "model": self.model,
            "mask_origin": mask_origin,
            "segmentation_info": segmentation_info,
            "source_size": [rgb.shape[1], rgb.shape[0]],
            "geometry": geometry,
            "normalized_geometry": placed_geometry,
            "placement": placement,
            "timings": timings,
        }
        return ProcessedVehicle(final, stages, metadata)
