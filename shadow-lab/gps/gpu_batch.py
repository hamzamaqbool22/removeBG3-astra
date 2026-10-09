"""Two-stage GPU batch: one segmentation session, then one persistent GPS runner."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import gc
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from batch import ROOT, EXTENSIONS, FIELDS, now, duration, save_summary


def announce(folder, text):
    print(text, flush=True)
    with (folder/'batch.log').open('a') as log:
        log.write(text+'\n')


def persist(folder, rows, manifest, phase, startup, finished=None):
    # Wall time is distinct from the sum of active per-image processing times:
    # images wait between preparation and generation in this two-stage run.
    elapsed = (datetime.now().astimezone()-datetime.fromisoformat(manifest['started_at'])).total_seconds()
    save_summary(folder, rows, manifest['started_at'], max(0,elapsed), finished,
                 extra={'phase':phase, 'model_startup_seconds':startup,
                        'timing_note':'Per-image total_seconds = active preparation + inference/saving; excludes waiting and shared model startup.'})


def prepare_all(manifest_path, make_session=None, prepare_one=None):
    manifest=json.loads(manifest_path.read_text())
    folder=manifest_path.parent
    rows=manifest['rows']
    if make_session is None:
        from prepare import make_segmenter, prepare_image
        make_session=lambda: make_segmenter('cuda')
        prepare_one=prepare_image
    startup={}
    tick=time.perf_counter()
    announce(folder, 'STAGE 1: loading GPU background-removal model once')
    try:
        with (folder/'prepare-startup.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
            session=make_session()
        if 'CUDAExecutionProvider' not in session.providers:
            raise RuntimeError('GPU segmentation failed; refusing silent CPU fallback')
    except Exception:
        with (folder/'prepare-startup.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    startup['segmentation']=round(time.perf_counter()-tick,3)
    announce(folder, f'CUDA segmentation ready in {duration(startup["segmentation"])}; providers: {session.providers}')
    try:
        for index,row in enumerate(rows,1):
            item=Path(row['result_folder']).parent
            item.mkdir(exist_ok=True)
            row['started_at']=now()
            row['status']='preparing'
            persist(folder,rows,manifest,'preparing',startup)
            announce(folder,f'[{index}/{len(rows)}] PREPARE START {Path(row["image"]).name} | {row["started_at"]}')
            tick=time.perf_counter()
            try:
                with (item/'prepare.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                    prepare_one(Path(row['image']), Path(manifest['background']), item/'input', session)
                row['status']='prepared'
            except Exception as error:
                row['status']='failed'
                row['error']=f'prepare: {error}; see {item / "prepare.log"}'
                row['finished_at']=now()
                with (item/'prepare.log').open('a') as log:
                    traceback.print_exc(file=log)
            finally:
                row['prepare_seconds']=round(time.perf_counter()-tick,3)
                row['total_seconds']=row['prepare_seconds']
                persist(folder,rows,manifest,'preparing',startup)
            announce(folder,f'[{index}/{len(rows)}] {row["status"].upper()} {Path(row["image"]).name} | '
                     f'{now()} | preparation {duration(row["prepare_seconds"])}')
    finally:
        # Worker exits before diffusion loads, releasing ORT's CUDA arena/context.
        del session
    return rows, startup


def infer_all(folder, rows, manifest, startup, runner_factory=None):
    candidates=[r for r in rows if r['status']=='prepared']
    if not candidates:
        return
    if runner_factory is None:
        from infer import GPSRunner
        runner_factory=GPSRunner
    announce(folder,'STAGE 2: loading GPS models once (50 steps by default, same as before)')
    tick=time.perf_counter()
    with (folder/'inference-startup.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        runner=runner_factory()
    startup['gps']=round(time.perf_counter()-tick,3)
    persist(folder,rows,manifest,'inference',startup)
    announce(folder,f'GPS ready in {duration(startup["gps"])}; reusing models for {len(candidates)} images')
    for index,row in enumerate(rows,1):
        if row['status']!='prepared':
            continue
        item=Path(row['result_folder']).parent
        row['status']='generating'
        persist(folder,rows,manifest,'inference',startup)
        announce(folder,f'[{index}/{len(rows)}] AI START {Path(row["image"]).name} | {now()}')
        tick=time.perf_counter()
        fatal=False
        try:
            with (item/'inference.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                runner.run(item/'input',item/'result',manifest['samples'],manifest['steps'],manifest['seed'])
            row['status']='ok'
        except Exception as error:
            row['status']='failed'
            row['error']=f'inference: {error}; see {item / "inference.log"}'
            with (item/'inference.log').open('a') as log:
                traceback.print_exc(file=log)
            # A poisoned CUDA context or insufficient VRAM is batch-wide, not an image issue.
            fatal='CUDA' in str(error) or 'out of memory' in str(error).lower()
            gc.collect()
        finally:
            row['inference_seconds']=round(time.perf_counter()-tick,3)
            row['total_seconds']=round(row['prepare_seconds']+row['inference_seconds'],3)
            row['finished_at']=now()
            persist(folder,rows,manifest,'inference',startup)
        announce(folder,f'[{index}/{len(rows)}] {row["status"].upper()} {Path(row["image"]).name} | '
                 f'finished {row["finished_at"]} | active total {duration(row["total_seconds"])} | '
                 f'prepare {row["prepare_seconds"]:.1f}s | AI + saving {row["inference_seconds"]:.1f}s')
        if row['error']:
            announce(folder,row['error'])
        if fatal:
            raise RuntimeError('Stopped after GPU failure; see the last image inference.log. Completed results are saved.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path)
    p.add_argument('--background',type=Path,default=ROOT/'backgrounds/parking-lots/6.png')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/gps-batches')
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare-worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.prepare_worker:
        prepare_all(args.prepare_worker)
        return 0
    if args.images is None or not args.images.is_dir() or not args.background.is_file():
        p.error('Provide an images folder and an existing background')
    if not 1<=args.samples<=8 or not 1<=args.steps<=100:
        p.error('Use 1–8 samples and 1–100 steps')
    images=sorted((f.resolve() for f in args.images.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS),key=lambda f:f.name.casefold())
    if not images:
        p.error('No supported images found')
    folder=args.output_root.resolve()/datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f-gpu')
    folder.mkdir(parents=True)
    rows=[]
    for index,image in enumerate(images,1):
        safe=re.sub(r'[^\w.-]+','_',image.stem)[:80] or 'image'
        row=dict.fromkeys(FIELDS,'')
        row.update(image=str(image),status='pending',prepare_seconds=0.,inference_seconds=0.,total_seconds=0.,
                   result_folder=str(folder/f'{index:03d}-{safe}'/'result'))
        rows.append(row)
    manifest={'started_at':now(),'background':str(args.background.resolve()),'samples':args.samples,
              'steps':args.steps,'seed':args.seed,'rows':rows}
    manifest_path=folder/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2))
    startup={}
    persist(folder,rows,manifest,'starting',startup)
    announce(folder,f'BATCH START: {manifest["started_at"]} | {len(images)} images | GPU mode')
    announce(folder,f'Results: {folder}')
    announce(folder,'One-time model loading is reported separately. Per-image totals exclude waiting between stages.')
    interrupted=False
    try:
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--prepare-worker',str(manifest_path)],check=True,cwd=ROOT)
        report=json.loads((folder/'summary.json').read_text())
        rows,startup=report['images'],report['model_startup_seconds']
        infer_all(folder,rows,manifest,startup)
    except KeyboardInterrupt:
        interrupted=True
        announce(folder,'Batch interrupted; completed images are saved.')
    except Exception as error:
        announce(folder,f'BATCH ERROR: {error}')
        with (folder/'error.log').open('w') as log:
            traceback.print_exc(file=log)
    finally:
        # Read the most recent worker snapshot even if the child exited early.
        report=json.loads((folder/'summary.json').read_text())
        rows,startup=report['images'],report['model_startup_seconds']
        for row in rows:
            if row['status'] not in ('ok','failed'):
                row['status']='interrupted' if interrupted else 'unprocessed'
                row['error']=row['error'] or 'Batch stopped before completion; see batch.log and stage logs'
        persist(folder,rows,manifest,'finished',startup,now())
        report=json.loads((folder/'summary.json').read_text())
        announce(folder,f'BATCH END: {report["finished_at"]} | total {duration(report["elapsed_seconds"])} | '
                 f'{report["succeeded"]} succeeded, {report["failed"]} failed | report: {folder / "timings.csv"}')
    return 130 if interrupted else (0 if all(r['status']=='ok' for r in rows) else 1)


if __name__=='__main__':
    sys.exit(main())
