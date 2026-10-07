"""Prepare aligned GPS inputs plus today's normal-shadow reference."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageOps
from vehicle_pipeline.segmentation import Segmenter, refine_mask
from vehicle_pipeline.geometry import estimate_geometry
from vehicle_pipeline.parking import composite_parking
from vehicle_pipeline.original_shadow import shadow_from_photo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--background', type=Path, required=True)
    p.add_argument('--mask', type=Path, help='Optional existing source-size grayscale vehicle mask')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Output folder already exists; choose a new folder to preserve previous tests')
    rgb = np.array(ImageOps.exif_transpose(Image.open(args.image)).convert('RGB'))
    bg = ImageOps.exif_transpose(Image.open(args.background)).convert('RGB')
    if args.mask:
        alpha = np.array(Image.open(args.mask).convert('L'))
        if alpha.shape != rgb.shape[:2]:
            p.error('Mask must match the EXIF-oriented source image dimensions')
    else:
        # Segmentation runs in its own process on CPU, leaving GPU memory for SDXL.
        os.environ['INFERENCE_DEVICE'] = 'cpu'
        alpha = refine_mask(Segmenter().predict(rgb))
    geometry = estimate_geometry(rgb, alpha)
    rgba = np.dstack([rgb, alpha])
    zero = np.zeros_like(alpha)
    composite = composite_parking(rgba, zero, geometry, bg)
    # Use the exact production placement/resampling for the aligned object mask.
    white = np.dstack([np.full_like(rgb, 255), alpha])
    mask = composite_parking(white, zero, geometry, Image.new('RGB', bg.size)).convert('L')
    shadow, mode = shadow_from_photo(rgb, alpha)
    normal = composite_parking(rgba, np.uint8(np.rint(np.clip(shadow,0,.97)*255)), geometry, bg)
    args.out.mkdir(parents=True)
    for name, img in [('composite', composite), ('mask', mask), ('background', bg), ('normal', normal)]:
        img.save(args.out / (name+'.png'))
    (args.out / 'input.json').write_text(json.dumps({'image':str(args.image.resolve()),
        'background':str(args.background.resolve()), 'normal_shadow_mode':mode,
        'enhancement':False}, indent=2))
    print('Prepared:', args.out.resolve())


if __name__ == '__main__':
    main()
