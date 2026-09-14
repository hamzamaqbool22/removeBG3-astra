"""Process the eight-view set and save every diagnostic stage; no API/UI."""
from pathlib import Path
import argparse
import json
import sys
import resource

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from vehicle_pipeline.pipeline import VehiclePipeline


def reference_on_canvas(path, source_size, matrix, canvas):
    """Align reference using the production transform, never the reverse."""
    ref=Image.open(path).convert('RGBA').resize(tuple(source_size),Image.Resampling.LANCZOS)
    arr=np.asarray(ref).astype(np.float32)
    alpha=arr[...,3]/255
    a=cv2.warpAffine(alpha,np.asarray(matrix,np.float32),tuple(canvas),flags=cv2.INTER_LINEAR)
    p=cv2.warpAffine(arr[...,:3]*alpha[...,None],np.asarray(matrix,np.float32),tuple(canvas),flags=cv2.INTER_LINEAR)
    return Image.fromarray(np.uint8(np.clip(p+255*(1-a[...,None]),0,255)))


def panels(items,cell=(512,410)):
    result=Image.new('RGB',(cell[0]*len(items),cell[1]),'#eeeeee')
    draw=ImageDraw.Draw(result)
    for i,(label,im) in enumerate(items):
        thumb=ImageOps.contain(im.convert('RGB'),(cell[0],cell[1]-26))
        result.paste(thumb,(i*cell[0]+(cell[0]-thumb.width)//2,26))
        draw.text((i*cell[0]+10,7),label,fill='black')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='birefnet-general')
    parser.add_argument('--cached-masks',action='store_true',help='Skip inference for shadow development; timings explicitly exclude segmentation')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/phase1')
    parser.add_argument('--input',type=Path,default=ROOT/'images')
    args=parser.parse_args()
    pipeline=VehiclePipeline(args.model)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];records=[]
    for path in sorted(args.input.glob('*.png')):
        source=Image.open(path)
        mask=None
        if args.cached_masks:
            mask=np.asarray(Image.open(ROOT/'outputs/segmentation'/args.model/path.stem/'mask.png'))
        result=pipeline.process(source,mask)
        out=args.output/path.stem;out.mkdir(exist_ok=True)
        Image.fromarray(result.final).save(out/'final_white.png')
        for name,arr in result.stages.items(): Image.fromarray(arr).save(out/f'{name}.png')
        (out/'metadata.json').write_text(json.dumps(result.metadata,indent=2)+'\n')
        refpath=ROOT/'remove.bg-outputs'/f'{path.stem}-removebg-preview.png'
        rowitems=[(f'{path.stem}: source',source),('Normalized cutout',Image.fromarray(result.stages['normalized_placement'])),
                  ('Generated shadow + vehicle',Image.fromarray(result.final))]
        if refpath.exists():
            ref=reference_on_canvas(refpath,result.metadata['source_size'],result.metadata['placement']['matrix'],result.metadata['placement']['canvas'])
            ref.save(out/'reference_aligned.png')
            rowitems.append(('remove.bg (same transform)',ref))
        row=panels(rowitems)
        row.save(out/'comparison.jpg',quality=95)
        rows.append(row)
        records.append({'image':path.name,**result.metadata})
        print(json.dumps({'image':path.name,'timings':result.metadata['timings'],'geometry_confidence':result.metadata['geometry']['confidence']}),flush=True)
    if not rows: raise SystemExit('No PNG input images')
    sheet=Image.new('RGB',(max(r.width for r in rows),sum(r.height for r in rows)),'white')
    y=0
    for row in rows: sheet.paste(row,(0,y));y+=row.height
    sheet.save(args.output/'comparison_all.jpg',quality=94)
    # A compact all-view presentation, separate from detailed side-by-side QA.
    grid=Image.new('RGB',(1024,4*410),'#eeeeee')
    for i,r in enumerate(records):
        im=Image.open(args.output/Path(r['image']).stem/'final_white.png')
        panel=panels([(Path(r['image']).stem,im)])
        grid.paste(panel,((i%2)*512,(i//2)*410))
    grid.save(args.output/'finals_all.jpg',quality=95)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)/2**20
    (args.output/'metrics.json').write_text(json.dumps({'model':args.model,'cached_masks':args.cached_masks,
        'process_peak_rss_mib':peak,'timing_note':'Processing excludes diagnostic rendering and disk I/O. Model load is separately reported on first image. Cached runs exclude inference.',
        'images':records},indent=2)+'\n')


if __name__=='__main__': main()
