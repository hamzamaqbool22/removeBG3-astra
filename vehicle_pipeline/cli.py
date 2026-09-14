"""Small file-based entry point for the Phase 1 pipeline."""
import argparse
import json
from pathlib import Path

from PIL import Image
from .pipeline import VehiclePipeline
from .placement import Framing


def main():
    parser=argparse.ArgumentParser(description='Remove a vehicle background, normalize framing, and render ground shadows on white.')
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model',default='birefnet-general')
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--width',type=int,default=1024)
    parser.add_argument('--height',type=int,default=768)
    parser.add_argument('--stages',type=Path,help='Optional directory for diagnostic masks, geometry, and individual shadow layers')
    args=parser.parse_args()
    pipe=VehiclePipeline(args.model,args.threads,Framing(width=args.width,height=args.height))
    with Image.open(args.input) as image:
        result=pipe.process(image)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(result.final).save(args.output)
    if args.stages:
        args.stages.mkdir(parents=True,exist_ok=True)
        for name,arr in result.stages.items():Image.fromarray(arr).save(args.stages/f'{name}.png')
        (args.stages/'metadata.json').write_text(json.dumps(result.metadata,indent=2)+'\n')
    print(json.dumps(result.metadata['timings'],indent=2))


if __name__=='__main__':main()
