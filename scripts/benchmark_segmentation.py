"""Run each model in a fresh process; save raw masks and measured CPU costs."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from PIL import Image, ImageOps, ImageDraw
from vehicle_pipeline.segmentation import Segmenter


def worker(args):
    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = threading.Event()
    def sample():
        while not done.wait(.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    session = Segmenter(args.worker, args.threads, memory_mode=args.memory_mode)
    load_seconds = time.perf_counter() - start
    model_rss = process.memory_info().rss
    paths = sorted((ROOT / "images").glob("*.png"))
    output = ROOT / "outputs/segmentation" / (args.worker + ('-lean' if args.memory_mode=='lean' else ''))
    output.mkdir(parents=True, exist_ok=True)
    first = np.asarray(Image.open(paths[0]).convert("RGB"))
    start = time.perf_counter()
    session.predict(first)
    warmup_seconds = time.perf_counter() - start
    records = []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        times = []
        cpu_times = []
        for _ in range(args.repeats):
            start, cpu_start = time.perf_counter(), time.process_time()
            mask = session.predict(rgb)
            times.append(time.perf_counter() - start)
            cpu_times.append(time.process_time() - cpu_start)
        dest = output / p.stem
        dest.mkdir(exist_ok=True)
        Image.fromarray(mask).save(dest / "mask.png")
        rgba = Image.fromarray(rgb).convert("RGBA")
        rgba.putalpha(Image.fromarray(mask))
        rgba.save(dest / "cutout.png")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        white.convert("RGB").save(dest / "white.png")
        record = {"image": p.name, "size": list(Image.open(p).size),
                  "wall_seconds": times, "cpu_seconds": cpu_times,
                  "median_wall_seconds": float(np.median(times)),
                  "rss_after_mib": process.memory_info().rss / 2**20}
        records.append(record)
        print(json.dumps({"model": args.worker, **record}), flush=True)
    done.set()
    thread.join()
    # Darwin reports ru_maxrss in bytes; Linux reports KiB.
    os_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    os_peak *= 1 if sys.platform == "darwin" else 1024
    result = {"model": args.worker, "memory_mode":args.memory_mode,"providers": session.providers,
              "threads": args.threads, "repeats": args.repeats,
              "load_seconds_including_download_if_missing": load_seconds,
              "warmup_seconds": warmup_seconds,
              "loaded_rss_mib": model_rss / 2**20,
              "sampled_peak_rss_mib": peak[0] / 2**20,
              "process_peak_rss_mib": os_peak / 2**20,
              "mean_median_wall_seconds": float(np.mean([r['median_wall_seconds'] for r in records])),
              "images": records,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "machine": platform.machine(), "ram_gib": psutil.virtual_memory().total / 2**30,
                              "packages": {p: importlib.metadata.version(p) for p in ['rembg','onnxruntime','numpy','Pillow']}}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2))


def overview(models):
    paths = sorted((ROOT / "images").glob("*.png"))
    cell_w, cell_h = 420, 335
    sheet = Image.new("RGB", (cell_w * (len(models)+1), cell_h * len(paths)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row,p in enumerate(paths):
        refs = [("reference", ROOT / "remove.bg-outputs" / f"{p.stem}-removebg-preview.png")]
        refs += [(m, ROOT / "outputs/segmentation" / m / p.stem / "white.png") for m in models]
        for col,(label,path) in enumerate(refs):
            if not path.exists(): continue
            im = Image.open(path).convert("RGBA")
            bg = Image.new("RGBA",im.size,"white"); bg.alpha_composite(im)
            thumb = ImageOps.contain(bg.convert("RGB"),(cell_w-8,cell_h-28))
            sheet.paste(thumb,(col*cell_w+(cell_w-thumb.width)//2,row*cell_h+25))
            draw.text((col*cell_w+8,row*cell_h+5), f"{p.stem} | {label}", fill="black")
    sheet.save(ROOT / "outputs/segmentation/comparison.jpg", quality=94)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['u2net','isnet-general-use','birefnet-general-lite','birefnet-general'])
    p.add_argument('--worker'); p.add_argument('--threads', type=int, default=6)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--memory-mode',choices=['balanced','lean'],default='balanced')
    p.add_argument('--overview-only', action='store_true')
    args = p.parse_args()
    if args.worker: worker(args); return
    if not args.overview_only:
        for model in args.models:
            subprocess.run([sys.executable,__file__,'--worker',model,'--threads',str(args.threads),'--repeats',str(args.repeats),'--memory-mode',args.memory_mode],check=True,cwd=ROOT)
    overview([m+('-lean' if args.memory_mode=='lean' else '') for m in args.models])


if __name__ == '__main__': main()
