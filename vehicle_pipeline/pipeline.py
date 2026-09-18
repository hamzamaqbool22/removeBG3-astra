"""Raw vehicle to a cutout and refined shadow, using CPU inference."""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from .segmentation import Segmenter, refine_mask
from .geometry import estimate_geometry
from .placement import Framing, normalize
from .shadows import render_shadows
from .parking import composite_parking


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
        color, alpha, geometry, _ = normalize(rgb, alpha, geometry, Framing())
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
