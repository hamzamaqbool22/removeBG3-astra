"""Separate preview runner; production code and API configuration are untouched."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_pipeline.segmentation import Segmenter, refine_mask
from vehicle_pipeline.geometry import estimate_geometry
from vehicle_pipeline.parking import composite_parking
from source_first import shadow_from_photo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--background', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expansion', default=.012, type=float,
                        help='Source shadow expansion: 0 to .03; default .012')
    args = parser.parse_args()
    rgb = np.asarray(ImageOps.exif_transpose(Image.open(args.image)).convert('RGB'))
    # Cache only segmentation to make shadow iterations inexpensive. Include
    # the segmentation implementation and input pixels in the cache identity.
    identity = hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()+
        (ROOT/'vehicle_pipeline/segmentation.py').read_bytes()).hexdigest()
    cache = ROOT/'.cache/shadow-lab'/f'{identity}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        alpha = np.load(cache, allow_pickle=False)
    else:
        alpha = refine_mask(Segmenter(memory_mode='lean').predict(rgb))
        np.save(cache, alpha)
    shadow, mode = shadow_from_photo(rgb, alpha, args.expansion)
    result = composite_parking(np.dstack([rgb,alpha]),np.uint8(np.rint(shadow*255)),
        estimate_geometry(rgb,alpha),Image.open(args.background))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.save(args.output)
    print(f'{mode}: {args.output}')


if __name__ == '__main__':
    main()
