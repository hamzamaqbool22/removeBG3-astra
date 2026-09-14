"""Rerun real inference after making vehicles smaller or cropping the scene.

These are controlled variations of the same eight photographs, not unseen cars.
"""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from vehicle_pipeline.pipeline import VehiclePipeline


def main():
    pipe=VehiclePipeline()
    output=ROOT/'outputs/robustness'
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in sorted((ROOT/'images').glob('*.png')):
        rgb=np.asarray(Image.open(source).convert('RGB'))
        base=json.loads((ROOT/'outputs/phase1'/source.stem/'metadata.json').read_text())
        mask=np.asarray(Image.open(ROOT/'outputs/phase1'/source.stem/'refined_mask.png'))
        x0,y0,x1,y1=base['geometry']['bbox']
        margin=round(.06*max(x1-x0,y1-y0))
        crop=(max(0,x0-margin),max(0,y0-margin),min(rgb.shape[1],x1+margin),min(rgb.shape[0],y1+margin))
        transform=np.array([[.65,0,rgb.shape[1]*.175],[0,.65,rgb.shape[0]*.175]],np.float32)
        small=cv2.warpAffine(rgb,transform,(rgb.shape[1],rgb.shape[0]),borderValue=(170,170,170))
        small_mask=cv2.warpAffine(mask,transform,(rgb.shape[1],rgb.shape[0]))
        variants=[('smaller_in_frame',small,small_mask),('tight_scene_crop',rgb[crop[1]:crop[3],crop[0]:crop[2]],mask[crop[1]:crop[3],crop[0]:crop[2]])]
        for label,array,expected_mask in variants:
            result=pipe.process(Image.fromarray(array))
            dest=output/source.stem/label
            dest.mkdir(parents=True,exist_ok=True)
            Image.fromarray(array).save(dest/'input.png')
            Image.fromarray(result.final).save(dest/'final_white.png')
            Image.fromarray(result.stages['refined_mask']).save(dest/'mask.png')
            pred=result.stages['refined_mask']>127
            expected=expected_mask>127
            iou=float((pred&expected).sum()/max(1,(pred|expected).sum()))
            b=np.array(result.metadata['normalized_geometry']['bbox'])
            old=np.array(base['normalized_geometry']['bbox'])
            item={'image':source.name,'variation':label,'mask_consistency_iou':iou,
                  'normalized_bbox_max_delta_pixels':float(np.abs(b-old).max()),
                  'geometry_support_mode':result.metadata['geometry']['view_cues']['support_mode'],
                  'timings':result.metadata['timings']}
            records.append(item)
            (dest/'metadata.json').write_text(json.dumps(result.metadata,indent=2)+'\n')
            print(json.dumps(item),flush=True)
    (output/'metrics.json').write_text(json.dumps({'note':'Controlled changes of the same eight photos; no claim of generalization to new vehicles. Mask IoU compares to a transformed baseline model mask, not ground truth. Both variants rerun CPU segmentation.', 'images':records},indent=2)+'\n')


if __name__=='__main__':main()
