"""Fast queue checks with tiny fake image processing; no model loaded."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vehicle_pipeline.jobs import JobQueue
from vehicle_pipeline import api
from starlette.requests import Request
from fastapi import HTTPException


with tempfile.TemporaryDirectory() as directory:
    gate = threading.Event()
    active = 0
    peak = 0

    def process(data, options):
        global active, peak
        active += 1
        peak = max(peak, active)
        gate.wait(5)
        time.sleep(.005)
        active -= 1
        if options.get('fail'):
            raise ValueError('test failure')
        return b'png-' + (data or b'url')

    queue = JobQueue(directory, process, capacity=20)
    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            ids = list(pool.map(lambda i: queue.submit(str(i).encode(), {}), range(20)))
        assert len(set(ids)) == 20
        try:
            queue.submit(b'excess', {})
            raise AssertionError('Queue limit not enforced')
        except OverflowError:
            pass
        try:
            JobQueue(directory, process)
            raise AssertionError('Duplicate worker allowed')
        except RuntimeError:
            pass
        gate.set()
        deadline = time.monotonic() + 10
        while any(queue.get(i)['status'] != 'completed' for i in ids):
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert peak == 1
        for i, job_id in enumerate(ids):
            assert queue.get(job_id, True) == f'png-{i}'.encode()
        print('PASS: 20 concurrent submissions, all complete, one active processor, queue limit and owner lock')
    finally:
        gate.set()
        queue.close()
    # Simulate an interrupted job in the persisted database.
    with queue.connect() as db:
        db.execute("UPDATE jobs SET status='processing', input=?, result=NULL WHERE id=?", (b'retry', ids[0]))
    queue = JobQueue(directory, process)
    api.app.state.jobs = queue

    async def check_api():
        old_key = api.API_KEY
        api.API_KEY = 'test'
        try:
            async def receive():
                return {'type': 'http.request', 'body': json.dumps({'imageurl': 'https://example.com/car.jpg'}).encode(), 'more_body': False}
            request = Request({'type': 'http', 'method': 'POST', 'path': '/process', 'app': api.app,
                               'headers': [(b'content-type', b'application/json'), (b'authorization', b'Bearer test')]}, receive)
            job = await api.process(request)
            assert job['jobId'] and job['status'] == 'queued'
            while (await api.job_status(job['jobId'], request))['status'] != 'completed':
                await asyncio.sleep(.02)
            assert (await api.job_result(job['jobId'], request)).body == b'png-url'
            denied = Request({'type': 'http', 'headers': [], 'app': api.app})
            try:
                await api.job_status(job['jobId'], denied)
                raise AssertionError('Unauthenticated status allowed')
            except HTTPException as exc:
                assert exc.status_code == 401
        finally:
            api.API_KEY = old_key
    try:
        asyncio.run(check_api())
        assert queue.get(ids[0], True) == b'png-retry'
        failed = queue.submit(b'bad', {'fail': True})
        deadline = time.monotonic() + 5
        while queue.get(failed)['status'] != 'failed':
            assert time.monotonic() < deadline
            time.sleep(.02)
        queue.retention = -1
        assert queue.get(failed) is None
        print('PASS: restart recovery, API submission/status/result/auth, failed jobs, expiry')
    finally:
        queue.close()
