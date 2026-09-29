"""Offline validation only. Imports production modules read-only; writes here only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from vehicle_pipeline.segmentation import Segmenter, refine_mask
from vehicle_pipeline.geometry import estimate_geometry
from vehicle_pipeline.source_shadow import recover_shadow
from vehicle_pipeline.placement import normalize, Framing
from vehicle_pipeline.shadows import render_shadows
from vehicle_pipeline.parking import composite_parking

parser=argparse.ArgumentParser()
parser.add_argument('--image', type=Path)
args=parser.parse_args()
out=ROOT/'validation'
out.mkdir(exist_ok=True)
if args.image:
    rgb=np.array(ImageOps.exif_transpose(Image.open(args.image)).convert('RGB'))
    segmenter=Segmenter(threads=4,memory_mode='lean')
    alpha=refine_mask(segmenter.predict(rgb))
else:
    rgb=np.full((480,640,3),180,np.uint8)
    alpha=np.zeros((480,640),np.uint8)
    cv2.rectangle(alpha,(80,180),(550,320),255,-1)
    cv2.ellipse(alpha,(170,320),(38,48),0,0,360,255,-1)
    cv2.ellipse(alpha,(450,320),(38,48),0,0,360,255,-1)
    rgb[alpha>0]=230
    for x in (170,450):
        cv2.ellipse(rgb,(x,320),(38,48),0,0,360,(25,25,25),-1)
        cv2.ellipse(rgb,(x,320),(23,30),0,0,360,(150,150,150),-1)
    # Broad source cast shadow extending under and past the car.
    ground=np.zeros(alpha.shape,np.uint8)
    cv2.ellipse(ground,(315,345),(260,42),0,0,360,1,-1)
    rgb[(ground>0)&(alpha==0)]=95
geometry=estimate_geometry(rgb,alpha)
source=recover_shadow(rgb,alpha)
color,pa,g,placement=normalize(rgb,alpha,geometry,Framing())
layers=render_shadows(pa.shape,g)
shadow=layers['combined'] if source is None else np.clip(cv2.warpAffine(source,np.array(placement['matrix'],np.float32),(1024,768),flags=cv2.INTER_LINEAR),0,.97)
background=np.array(Image.open(ROOT/'backgrounds/parking-lots/1.png').convert('RGB'))
final=composite_parking(np.dstack([color,pa]),np.uint8(np.rint(shadow*255)),g,Image.fromarray(background),True)
final.save(out/'python-final.png')
arrays={'rgb':rgb,'alpha':alpha,'placed_rgb':color,'placed_alpha':pa,'shadow':shadow,'procedural':layers['combined'],'background':background,'final':np.array(final)}
if source is not None:arrays['source']=source
meta={'geometry':geometry,'placed_geometry':g,'source':source is not None,'arrays':{}}
for name,array in arrays.items():
    array.tofile(out/f'{name}.bin');meta['arrays'][name]={'shape':list(array.shape),'dtype':str(array.dtype)}
(out/'reference.json').write_text(json.dumps(meta))
print(json.dumps({'source_shadow':source is not None,'geometry':geometry['view_cues'],'output':str(out)}))
