"""HTTP integration checks. Add --real IMAGE to run the CPU model on an upload."""
import concurrent.futures
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn
from PIL import Image
from vehicle_pipeline import api
from vehicle_pipeline.jobs import JobFailure

png = io.BytesIO()
Image.new('RGBA', (4, 4), (20, 30, 40, 255)).save(png, format='PNG')
active = peak = 0
calls = []

def fake(data, options):
    global active, peak
    active += 1
    peak = max(active, peak)
    try:
        calls.append((data, options))
        time.sleep(.01)
        if (options.get('imageurl') or '').endswith('/fail'):
            raise JobFailure('Expected test failure')
        if options.get('output_format') == 'webp':
            output = io.BytesIO()
            Image.new('RGBA', (4,4), (20,30,40,128)).save(output, format='WEBP', quality=92)
            return output.getvalue()
        return png.getvalue()
    finally:
        active -= 1

real = len(sys.argv) == 3 and sys.argv[1] == '--real'
if not real:
    api.run_job = fake
api.API_KEY = 'test-only'
with tempfile.TemporaryDirectory() as directory:
    os.environ['JOB_DIR'] = directory
    os.environ['PRELOAD_MODEL'] = 'false'
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    base = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(api.app, log_level='error', limit_concurrency=256))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline
        time.sleep(.01)

    def post(body, content='application/json', key='test-only', accept='image/png'):
        req = urllib.request.Request(base + '/generate', body,
            {'Content-Type': content, 'Authorization': f'Bearer {key}', 'Accept': accept})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def url_post(i):
        return post(json.dumps({'imageurl': f'https://example.com/{i}'}).encode())

    def multipart(image):
        boundary = 'vehicle-test-boundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="car.png"\r\nContent-Type: image/png\r\n\r\n').encode() + image
        for key, value in {'isBackgroundWant': 'true', 'background': '1', 'backgroundFolder': 'parking-lots', 'enhancment': 'true'}.items():
            body += f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}'.encode()
        body += f'\r\n--{boundary}--\r\n'.encode()
        return post(body, f'multipart/form-data; boundary={boundary}')

    try:
        if real:
            status, headers, result = multipart(Path(sys.argv[2]).read_bytes())
            assert status == 200, (status, result[:500])
            image = Image.open(io.BytesIO(result))
            image.verify()
            Path('/tmp/generate-real-result.png').write_bytes(result)
            print(f'PASS: real CPU HTTP /generate returned {len(result)} bytes PNG; /tmp/generate-real-result.png')
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as pool:
                results = list(pool.map(url_post, range(100)))
            assert len(calls) == 100 and peak == 1
            assert all(s == 200 and h['Content-Type'] == 'image/png' and b == png.getvalue() for s,h,b in results)
            assert multipart(png.getvalue())[0] == 200
            assert calls[-1][0] == png.getvalue()
            assert calls[-1][1]['enhancment'] is True
            status, headers, result = post(b'{"imageurl":"https://example.com/webp"}', accept='image/webp')
            assert status == 200 and headers['Content-Type'] == 'image/webp'
            assert calls[-1][1]['output_format'] == 'webp'
            with Image.open(io.BytesIO(result)) as decoded:
                assert decoded.format == 'WEBP' and decoded.mode == 'RGBA'
            assert url_post('fail')[0] == 500
            assert post(b'{}')[0] == 422
            assert post(b'{}', key='wrong')[0] == 401
            assert post(b'bad', 'text/plain')[0] == 415
            api.app.state.jobs.capacity = 0
            status, headers, _ = url_post('full')
            assert status == 429 and headers['Retry-After'] == '10'
            print('PASS: 100 simultaneous HTTP requests, one processor, PNG responses, multipart, auth, validation, failure and queue-full handling')
    finally:
        server.should_exit = True
        thread.join(30)
        assert not thread.is_alive()
        sock.close()
