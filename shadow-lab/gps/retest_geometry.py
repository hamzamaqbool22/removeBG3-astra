"""Rerun saved inputs with the fifth ControlNet channel zero; no segmentation."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

CASES=('001-1','002-15507230','003-183036695',
       '007-A355B808C0ACCDD03C1B44284C07BEB7','034-left','037-rearRight','038-right')


def select_cases(batch):
    selected=[]
    for name in CASES:
        item=batch/name
        required=[item/'input'/f'{f}.png' for f in ('composite','mask','background','normal')]
        required += [item/'result'/'protected-42.png',item/'result'/'raw-42.png',item/'result'/'report.json']
        missing=[str(p) for p in required if not p.is_file()]
        if missing:
            raise ValueError('Missing diagnostic files: '+', '.join(missing))
        report=json.loads((item/'result'/'report.json').read_text())
        if report['steps']!=50 or report['seeds']!=[42]:
            raise ValueError(f'{name}: baseline must have 50 steps and seed 42')
        selected.append(item)
    return selected


def comparison(item, result, destination):
    from PIL import Image, ImageDraw
    panels=[('Current normal shadows',item/'input'/'normal.png'),
            ('Previous AI: learned geometry',item/'result'/'protected-42.png'),
            ('Test AI: zero geometry',result/'protected-42.png'),
            ('Raw AI: zero geometry (diagnostic)',result/'raw-42.png')]
    sheet=Image.new('RGB',(1280,1040),'white')
    draw=ImageDraw.Draw(sheet)
    for i,(label,path) in enumerate(panels):
        x,y=(i%2)*640,(i//2)*520
        draw.text((x+8,y+8),label,fill='black')
        with Image.open(path) as source:
            image=source.convert('RGB')
        image.thumbnail((640,480))
        sheet.paste(image,(x+(640-image.width)//2,y+30))
    sheet.save(destination,quality=95)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:
        cases=select_cases(args.batch)
    except (ValueError,KeyError) as error:
        parser.error(str(error))
    if args.out.exists():
        parser.error('Choose a new output folder; existing results are preserved')
    args.out.mkdir(parents=True)
    comparisons=args.out/'comparisons'
    comparisons.mkdir()
    started=time.perf_counter()
    report={'started_at':datetime.now().astimezone().isoformat(),
            'geometry_mode':'zero','steps':50,'seed':42,'images':[]}
    from infer import GPSRunner
    print('Loading GPU models once; reusing saved inputs for seven cases.',flush=True)
    runner=GPSRunner()
    for index,item in enumerate(cases,1):
        tick=time.perf_counter()
        row={'name':item.name,'status':'running'}
        report['images'].append(row)
        print(f'[{index}/{len(cases)}] START {item.name}',flush=True)
        try:
            with (args.out/f'{item.name}.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                runner.run(item/'input',args.out/item.name,steps=50,seed=42,geometry_mode='zero')
            comparison(item,args.out/item.name,comparisons/f'{item.name}.jpg')
            row['status']='ok'
        except Exception:
            row['status']='failed'
            row['error']=traceback.format_exc()
            raise
        finally:
            row['seconds']=round(time.perf_counter()-tick,3)
            report['elapsed_seconds']=round(time.perf_counter()-started,3)
            (args.out/'summary.json').write_text(json.dumps(report,indent=2))
            print(f'{row["status"]}: {item.name} {row["seconds"]:.1f}s',flush=True)
    report['finished_at']=datetime.now().astimezone().isoformat()
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    print(f'Finished in {report["elapsed_seconds"]:.1f}s. Comparisons: {comparisons.resolve()}')


if __name__=='__main__':
    main()
