"""Local sequential or simultaneous-client load test; no automatic retries."""
import argparse
import mimetypes
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
from queue import Queue, Empty
import time
from threading import Barrier
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', nargs='+', type=int, default=[8001])
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--concurrency', type=int, default=None,
                        help='Total simultaneous clients; defaults to one per server (maximum 50)')
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'images',
                        help='Vehicle photo or folder of photos (PNG, JPEG, WebP, BMP, TIFF)')
    args = parser.parse_args()
    if args.count < 1 or len(set(args.ports)) != len(args.ports):
        parser.error('Use a positive count and distinct server ports')
    concurrency = args.concurrency if args.concurrency is not None else len(args.ports)
    if not 1 <= concurrency <= 50:
        parser.error('concurrency must be between 1 and 50')
    concurrency = min(concurrency, args.count)
    source = args.input.expanduser()
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
    candidates = [source] if source.is_file() else sorted(source.iterdir()) if source.is_dir() else []
    images = [p for p in candidates if p.is_file() and p.suffix.lower() in extensions]
    if not images:
        parser.error(f'No vehicle photos found at {source}. Use --input "/full/path/to/car.jpg" '
                     'or --input "/full/path/to/photo-folder"')
    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read image {path}: {exc}')
    print(f'Testing {args.count} requests using {len(images)} photo(s); '
          f'{len(args.ports)} server(s), {concurrency} concurrent client(s).', flush=True)
    jobs = Queue()
    for i in range(args.count):
        jobs.put(images[i % len(images)])
    started = time.perf_counter()
    barrier = Barrier(concurrency)

    def worker(port):
        outcomes = Counter()
        first = True
        while True:
            try:
                path = jobs.get_nowait()
            except Empty:
                return outcomes
            boundary = 'vehicle-local-benchmark'
            mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                    f'filename="vehicle{path.suffix.lower()}"\r\nContent-Type: {mime}\r\n\r\n').encode()
            body += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
            headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
            if os.environ.get('API_KEY'):
                headers['Authorization'] = 'Bearer ' + os.environ['API_KEY']
            # Release the first wave together, after every client's upload is ready.
            if first:
                barrier.wait(timeout=30)
                first = False
            begin = time.perf_counter()
            try:
                req = Request(f'http://127.0.0.1:{port}/process', body, headers)
                with urlopen(req, timeout=300) as response:
                    result = response.read()
                    queued = response.status == 202
                if queued:
                    job = json.loads(result)
                    print(f'{port}: {path.name}: accepted job {job["jobId"]}', flush=True)
                    auth = {k: v for k, v in headers.items() if k == 'Authorization'}
                    base = f'http://127.0.0.1:{port}'
                    deadline = time.monotonic() + 900
                    while True:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Job still unfinished after 15 minutes')
                        with urlopen(Request(base + job['statusUrl'], headers=auth), timeout=30) as response:
                            status = json.load(response)
                        if status['status'] == 'failed':
                            raise RuntimeError(status['error'])
                        if status['status'] == 'completed':
                            break
                        time.sleep(2)
                    with urlopen(Request(base + job['resultUrl'], headers=auth), timeout=30) as response:
                        result = response.read()
                with Image.open(BytesIO(result)) as image:
                    assert image.format == 'PNG'
                    image.verify()
                outcomes['success'] += 1
                print(f'{port}: {path.name}: {time.perf_counter()-begin:.2f}s OK', flush=True)
            except HTTPError as exc:
                outcomes[f'HTTP {exc.code}'] += 1
                exc.close()
                print(f'{port}: {path.name}: {time.perf_counter()-begin:.2f}s HTTP {exc.code}', flush=True)
            except Exception as exc:
                outcomes['other errors'] += 1
                print(f'{port}: {path.name}: FAILED {exc}', flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = Counter()
        for result in pool.map(worker, [args.ports[i % len(args.ports)] for i in range(concurrency)]):
            outcomes.update(result)
    successes = outcomes['success']
    elapsed = time.perf_counter() - started
    print(f'{successes}/{args.count} successful in {elapsed:.1f}s; '
          f'{successes * 60 / elapsed:.2f} images/minute')
    print('Outcomes: ' + ', '.join(f'{key}={value}' for key, value in sorted(outcomes.items())))
    print('First requests may include model loading. No failed requests are retried.')
    if concurrency > 1:
        print('Burst throughput is not sustained capacity. HTTP 503 means busy, not a crash.')
    try:
        for port in args.ports:
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=10) as response:
                print(f'Server {port} health after test: HTTP {response.status}')
    except Exception as exc:
        print(f'Health check failed: {exc}')
    raise SystemExit(0 if successes == args.count else 1)


if __name__ == '__main__':
    main()
