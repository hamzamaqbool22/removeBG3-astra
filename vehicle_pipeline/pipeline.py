"""Raw vehicle to a cutout and refined shadow, using CPU inference."""
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from .segmentation import Segmenter, refine_mask
from .geometry import estimate_geometry
from .placement import Framing, normalize
from .shadows import render_shadows
from .parking import composite_parking
from .source_shadow import recover_shadow


class VehiclePipeline:
    def __init__(self):
        self.segmenter = None

    def process(self, image: Image.Image, background: Image.Image | None = None,
                enhancement: bool = False) -> Image.Image:
        if self.segmenter is None:
            self.segmenter = Segmenter()
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        alpha = refine_mask(self.segmenter.predict(rgb))
        geometry = estimate_geometry(rgb, alpha)
        source_shadow = recover_shadow(rgb, alpha)
        color, alpha, geometry, placement = normalize(rgb, alpha, geometry, Framing())
        if source_shadow is not None:
            # Preserve the photographed cast-shadow direction and extent. Do
            # not stack synthetic shading onto it (that doubles the darkness).
            shadow = cv2.warpAffine(source_shadow, np.asarray(placement["matrix"], np.float32),
                                    (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR)
            shadow = np.clip(shadow, 0, .97)
        else:
            shadow = render_shadows(alpha.shape, geometry)["combined"]
        cutout = np.dstack([color, alpha])
        if background is not None:
            return composite_parking(cutout, np.uint8(np.rint(shadow*255)),
                                     geometry, background, enhancement)
        if enhancement:
            color = np.asarray(ImageEnhance.Contrast(Image.fromarray(color)).enhance(1.02))
        # Car over black translucent shadow. Store straight RGBA so this PNG
        # can be placed on any background without a white matte or dark fringe.
        a = alpha.astype(np.float32)/255
        combined_alpha = a + shadow*(1-a)
        premult = color.astype(np.float32)*a[...,None]
        straight = np.divide(premult, combined_alpha[...,None],
                             out=np.zeros_like(premult), where=combined_alpha[...,None]>1e-6)
        return Image.fromarray(np.dstack([np.uint8(np.clip(np.rint(straight),0,255)),
                                          np.uint8(np.rint(combined_alpha*255))]))
