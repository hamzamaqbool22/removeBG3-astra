"""Run all images sequentially, keeping results, logs and wall-clock timings."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
FIELDS = ['image', 'status', 'started_at', 'finished_at', 'prepare_seconds',
          'inference_seconds', 'total_seconds', 'result_folder', 'error']


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def duration(seconds):
    return f'{seconds:.1f}s ({seconds / 60:.1f} min)'


def save_summary(folder, rows, started_at, elapsed, finished_at=None, extra=None):
    summary = {'started_at': started_at, 'finished_at': finished_at,
               'elapsed_seconds': round(elapsed, 3), 'completed': sum(r['status'] in ('ok','failed') for r in rows),
               'succeeded': sum(r['status'] == 'ok' for r in rows),
               'failed': sum(r['status'] == 'failed' for r in rows), 'images': rows}
    if extra:
        summary.update(extra)
    temp = folder / 'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2))
    temp.replace(folder / 'summary.json')
    temp = folder / 'timings.csv.tmp'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(folder / 'timings.csv')


def run_batch(images, background, folder, samples=1, steps=50, seed=42):
    folder.mkdir(parents=True, exist_ok=False)
    started_at, start = now(), time.perf_counter()
    rows = []
    interrupted = False
    with (folder / 'batch.log').open('w', buffering=1) as log:
        def announce(message):
            print(message, flush=True)
            log.write(message + '\n')

        announce(f'BATCH START: {started_at} | {len(images)} images')
        announce(f'Results: {folder}')
        announce('Times include preparation, model loading, generation and saving; models reload for each image.')
        (folder/'manifest.json').write_text(json.dumps({'images':[str(p) for p in images],
            'background':str(background), 'samples':samples,'steps':steps,'seed':seed},indent=2))
        save_summary(folder, rows, started_at, 0)
        for index, image in enumerate(images, 1):
            safe = re.sub(r'[^\w.-]+', '_', image.stem)[:80] or 'image'
            item = folder / f'{index:03d}-{safe}'
            item.mkdir()
            row = dict.fromkeys(FIELDS, '')
            row.update(image=str(image), status='ok', started_at=now(),
                       prepare_seconds=0., inference_seconds=0., result_folder=str(item/'result'))
            image_start = time.perf_counter()
            announce(f'[{index}/{len(images)}] START {image.name} | {row["started_at"]}')
            stages = [
                ('prepare', [sys.executable, '-u', str(SCRIPTS/'prepare.py'),
                             '--image', str(image), '--background', str(background), '--out', str(item/'input')]),
                ('inference', [sys.executable, '-u', str(SCRIPTS/'infer.py'),
                               '--input', str(item/'input'), '--out', str(item/'result'),
                               '--samples', str(samples), '--steps', str(steps), '--seed', str(seed)])]
            for stage, command in stages:
                stage_start = time.perf_counter()
                stage_log = item / f'{stage}.log'
                announce(f'  {stage}: running (log: {stage_log})')
                try:
                    with stage_log.open('w') as stream:
                        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                except KeyboardInterrupt:
                    row['status'], row['error'] = 'interrupted', f'Interrupted during {stage}'
                    interrupted = True
                except (subprocess.CalledProcessError, OSError) as error:
                    row['status'] = 'failed'
                    row['error'] = f'{stage}: {error}; see {stage_log}'
                finally:
                    row[f'{stage}_seconds'] = round(time.perf_counter()-stage_start, 3)
                if row['status'] != 'ok':
                    break
            row['finished_at'] = now()
            row['total_seconds'] = round(time.perf_counter()-image_start, 3)
            rows.append(row)
            save_summary(folder, rows, started_at, time.perf_counter()-start)
            announce(f'[{index}/{len(images)}] {row["status"].upper()} {image.name} | '
                     f'finished {row["finished_at"]} | total {duration(row["total_seconds"])} | '
                     f'prepare {row["prepare_seconds"]:.1f}s | inference {row["inference_seconds"]:.1f}s')
            if row['error']:
                announce('  ' + row['error'])
            if interrupted:
                break
        elapsed, finished_at = time.perf_counter()-start, now()
        save_summary(folder, rows, started_at, elapsed, finished_at)
        good = sum(row['status']=='ok' for row in rows)
        announce(f'BATCH END: {finished_at} | total {duration(elapsed)} | '
                 f'{good} succeeded, {len(rows)-good} failed/interrupted, {len(images)-len(rows)} unprocessed')
        announce(f'Timing report: {folder / "timings.csv"}')
    return 130 if interrupted else (1 if any(r['status']!='ok' for r in rows) else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--background', type=Path, default=ROOT/'backgrounds/parking-lots/6.png')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/gps-batches')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.images.is_dir() or not args.background.is_file():
        parser.error('Images folder and background file must exist')
    if not 1 <= args.samples <= 8 or not 1 <= args.steps <= 100:
        parser.error('Use 1–8 samples and 1–100 steps')
    images = sorted((p.resolve() for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                    key=lambda p:p.name.casefold())
    if not images:
        parser.error('No supported images in the folder')
    run = datetime.now().astimezone().strftime('%Y%m%d-%H%M%S-%f')
    return run_batch(images, args.background.resolve(), args.output_root.resolve()/run,
                     args.samples, args.steps, args.seed)


if __name__ == '__main__':
    sys.exit(main())
