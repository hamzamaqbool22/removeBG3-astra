"""Read-only production comparison. All results stay in extension/validation."""
import sys,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from vehicle_pipeline.pipeline import VehiclePipeline
from vehicle_pipeline.segmentation import Segmenter
source=Path(sys.argv[1])
out=ROOT/'validation/eight-python'
out.mkdir(parents=True,exist_ok=True)
pipeline=VehiclePipeline()
segmenter=Segmenter(threads=4,memory_mode='lean')
class Recorder:
    def predict(self,rgb):
        mask=segmenter.predict(rgb)
        rgb.tofile(out/f'{current}-rgb.bin');mask.tofile(out/f'{current}-mask.bin')
        records[current]={'width':rgb.shape[1],'height':rgb.shape[0]}
        return mask
pipeline.segmenter=Recorder()
background=Image.open(ROOT/'backgrounds/parking-lots/1.png').convert('RGB')
np.array(background).tofile(out/'background.bin')
records={}
for path in sorted(source.glob('*.png')):
    current=path.stem;start=time.perf_counter()
    with Image.open(path) as image:result=pipeline.process(image,background,True)
    result.save(out/f'{current}-local.png')
    records[current]['seconds']=time.perf_counter()-start
    print(current,round(records[current]['seconds'],2),'seconds',flush=True)
(out/'reference.json').write_text(json.dumps({'views':records,'background':{'width':background.width,'height':background.height}},indent=2))
