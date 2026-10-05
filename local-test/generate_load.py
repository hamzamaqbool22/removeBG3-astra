"""Send simultaneous /generate uploads; save PNGs and report total wait times."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/generate')
    parser.add_argument('--input', type=Path, default=root / 'images')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--concurrency', type=int, help='Default: all requests, up to 100')
    parser.add_argument('--timeout', type=float, default=14400, help='Seconds per request, including queue wait')
    parser.add_argument('--background', type=int, default=1)
    parser.add_argument('--background-folder', choices=['parking-lots', 'backgrounds'], default='parking-lots')
    parser.add_argument('--no-enhancement', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Show image distribution without sending requests')
    args = parser.parse_args()
    concurrency = args.concurrency or min(args.count, 100)
    if args.count < 1 or not 1 <= concurrency <= 100 or args.timeout <= 0:
        parser.error('Use positive count/timeout and concurrency 1–100')
    concurrency = min(concurrency, args.count)
    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    if not paths:
        parser.error('Input folder has no images')
    planned = [paths[i % len(paths)] for i in range(args.count)]
    print(f'{args.count} requests, {concurrency} concurrent clients → {args.url}', flush=True)
    for path in paths:
        print(f'  {path.name}: {planned.count(path)} request(s)', flush=True)
    if args.dry_run:
        return
    # Prepare once per photo, before releasing the first wave together.
    bodies = {}
    boundary = 'vehicle-generate-load-boundary'
    for path in set(planned):
        raw = path.read_bytes()
        with Image.open(BytesIO(raw)) as im:
            im.verify()
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="vehicle{path.suffix}"\r\nContent-Type: application/octet-stream\r\n\r\n').encode() + raw
        fields = {'isBackgroundWant':'true','background':str(args.background),
                  'backgroundFolder':args.background_folder,'enhancment':str(not args.no_enhancement).lower()}
        for key, value in fields.items():
            body += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}'.encode()
        bodies[path] = body + f'\r\n--{boundary}--\r\n'.encode()
    output = args.output or root / 'outputs' / ('generate-load-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    if os.environ.get('API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['API_KEY']
    barrier = threading.Barrier(concurrency)
    started = time.perf_counter()

    def send(index):
        path = planned[index]
        if index < concurrency:
            barrier.wait(timeout=60)
        begin = time.perf_counter()
        print(f'SEND {index+1:03}/{args.count} {path.name}', flush=True)
        item = {'request':index+1,'image':path.name,'ok':False}
        try:
            with urlopen(Request(args.url, bodies[path], headers), timeout=args.timeout) as response:
                item['http_status'] = response.status
                item['worker'] = response.headers.get('X-Image-Worker', 'not-reported')
                result = response.read()
            with Image.open(BytesIO(result)) as image:
                if image.format != 'PNG':
                    raise ValueError('Response is not PNG')
                image.verify()
            name = f'{index+1:03}_{path.stem}.png'
            (output / name).write_bytes(result)
            item.update(ok=True, output=name)
        except HTTPError as exc:
            item.update(http_status=exc.code, error=exc.read(2048).decode(errors='replace'))
            exc.close()
        except Exception as exc:
            item['error'] = str(exc)
        item['seconds_including_queue'] = round(time.perf_counter()-begin, 2)
        print(f'{"OK" if item["ok"] else "FAIL"} {index+1:03}/{args.count} {path.name}: '
              f'{item["seconds_including_queue"]}s {item.get("error", "")}', flush=True)
        return item

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(send, i) for i in range(args.count)]):
            results.append(future.result())
    elapsed = time.perf_counter()-started
    success = sum(item['ok'] for item in results)
    report = {'requests':args.count,'concurrency':concurrency,'success':success,
              'elapsed_seconds':round(elapsed,2),'images_per_minute':round(success*60/elapsed,2),
              'worker_counts':{worker:sum(item.get('worker') == worker for item in results)
                               for worker in sorted({item.get('worker', 'not-reported') for item in results})},
              'results':sorted(results,key=lambda item:item['request'])}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(f'Finished: {success}/{args.count} succeeded in {elapsed:.1f}s. Outputs: {output}', flush=True)
    print('Requests by worker:', report['worker_counts'], flush=True)
    print('Times include queue waiting. No automatic retries. First request may include model loading.')
    raise SystemExit(0 if success == args.count else 1)


if __name__ == '__main__':
    main()
