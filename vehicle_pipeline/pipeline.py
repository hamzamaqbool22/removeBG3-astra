"""Raw vehicle to a cutout and refined shadow, using CPU inference."""
import logging
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from .segmentation import Segmenter, refine_mask
from .geometry import estimate_geometry
from .placement import Framing, normalize
from .original_shadow import shadow_from_photo
from .parking import composite_parking


class VehiclePipeline:
    def __init__(self):
        self.segmenter = None

    def process(self, image: Image.Image, background: Image.Image | None = None,
                enhancement: bool = False) -> Image.Image:
        started = time.perf_counter()

        def mark(stage):
            nonlocal started
            now = time.perf_counter()
            logging.getLogger("uvicorn.error").info("Timing %s: %.2fs", stage, now - started)
            started = now

        if self.segmenter is None:
            self.segmenter = Segmenter()
        mark("model ready")
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        mark("RGB conversion")
        predicted = self.segmenter.predict(rgb)
        mark("segmentation (%s passes)" % self.segmenter.last_info.get("inference_passes", "unknown"))
        alpha = refine_mask(predicted)
        mark("mask refinement")
        geometry = estimate_geometry(rgb, alpha)
        mark("geometry")
        source_shadow, shadow_mode = shadow_from_photo(rgb, alpha)
        logging.getLogger("uvicorn.error").info("Shadow mode: %s", shadow_mode)
        mark("source shadow recovery and contact completion")
        if background is not None:
            # Compose directly from source pixels. Normalizing onto a smaller
            # canvas first and then enlarging loses wheel/paint detail and
            # resamples the shadow edge twice. Parking placement is invariant
            # to the intermediate uniform scale/translation.
            result = composite_parking(np.dstack([rgb, alpha]),
                                     np.uint8(np.rint(np.clip(source_shadow,0,.97)*255)),
                                     geometry, background, enhancement)
            mark("background compositing and enhancement")
            return result
        color, alpha, geometry, placement = normalize(rgb, alpha, geometry, Framing())
        mark("placement")
        # Recover or synthesize in source coordinates, then transform the car
        # and shadow together. Weak shadows are replaced; strong ones stay intact.
        shadow = cv2.warpAffine(source_shadow, np.asarray(placement["matrix"], np.float32),
                               (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR)
        shadow = np.clip(shadow, 0, .97)
        mark("shadow rendering")
        if enhancement:
            color = np.asarray(ImageEnhance.Contrast(Image.fromarray(color)).enhance(1.02))
        # Car over black translucent shadow. Store straight RGBA so this PNG
        # can be placed on any background without a white matte or dark fringe.
        a = alpha.astype(np.float32)/255
        combined_alpha = a + shadow*(1-a)
        premult = color.astype(np.float32)*a[...,None]
        straight = np.divide(premult, combined_alpha[...,None],
                             out=np.zeros_like(premult), where=combined_alpha[...,None]>1e-6)
        result = Image.fromarray(np.dstack([np.uint8(np.clip(np.rint(straight),0,255)),
                                          np.uint8(np.rint(combined_alpha*255))]))
        mark("transparent compositing")
        return result
