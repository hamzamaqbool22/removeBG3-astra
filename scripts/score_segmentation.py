"""Reference-proxy segmentation diagnostics, not manually labelled ground truth."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image
from analyze_references import reference_layers

ROOT=Path(__file__).resolve().parents[1]


def score(model):
    rows=[]
    for p in sorted((ROOT/'images').glob('*.png')):
        ref=Image.open(ROOT/'remove.bg-outputs'/f'{p.stem}-removebg-preview.png')
        vehicle,shadow=reference_layers(ref)
        raw=Image.open(ROOT/'outputs/segmentation'/model/p.stem/'mask.png')
        pred=np.array(raw.resize(ref.size,Image.Resampling.LANCZOS))>127
        truth=vehicle>127
        inter=(pred&truth).sum(); union=(pred|truth).sum()
        kernel=np.ones((3,3),np.uint8)
        pb=pred & ~cv2.erode(pred.astype(np.uint8),kernel).astype(bool)
        tb=truth & ~cv2.erode(truth.astype(np.uint8),kernel).astype(bool)
        distance_to_t=cv2.distanceTransform((~tb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        distance_to_p=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        precision=float(np.mean(distance_to_t[pb]<=2))
        recall=float(np.mean(distance_to_p[tb]<=2))
        inner=cv2.erode(truth.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        outer=~cv2.dilate(truth.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        rows.append({'image':p.name,'mask_iou':float(inter/max(union,1)),
                     'boundary_f1_2px':2*precision*recall/max(precision+recall,1e-8),
                     'missing_interior_fraction':float((inner&~pred).sum()/max(inner.sum(),1)),
                     'excess_exterior_pixels':int((outer&pred).sum())})
    result={'model':model,'note':'Approximate vehicle alpha extracted from reference RGBA, downsampled to reference size. Black vehicle edges and shadow are ambiguous. These scores do not measure final shadow realism.',
            'mean_iou':float(np.mean([r['mask_iou'] for r in rows])),
            'mean_boundary_f1_2px':float(np.mean([r['boundary_f1_2px'] for r in rows])), 'images':rows}
    (ROOT/'outputs/segmentation'/model/'quality.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    models=[p.parent.name for p in sorted((ROOT/'outputs/segmentation').glob('*/metrics.json'))]
    results=[score(model) for model in models]
    (ROOT/'outputs/segmentation/quality_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:print(r['model'],'IoU',r['mean_iou'],'boundary F1',r['mean_boundary_f1_2px'])
