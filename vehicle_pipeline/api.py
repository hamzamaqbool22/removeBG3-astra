"""Run: uvicorn vehicle_pipeline.api:app --host 0.0.0.0 --port 8000"""
from io import BytesIO
from contextlib import asynccontextmanager
from pathlib import Path
import ipaddress
import json
import os
import secrets
import socket
import threading
import time
import logging
from urllib.parse import urlsplit, urljoin
from typing import Literal
import warnings
import asyncio

import urllib3
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request as BufferedRequest
from starlette.datastructures import UploadFile
from PIL import Image, UnidentifiedImageError

from .pipeline import VehiclePipeline
from .segmentation import Segmenter
from .jobs import JobQueue, JobFailure

ROOT = Path(__file__).resolve().parents[1]
BACKGROUNDS = ROOT / 'backgrounds'
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 20_000_000
pipeline = VehiclePipeline()
processing_lock = threading.Lock()  # One reusable CPU model, one inference at a time.
API_KEY = os.environ.get('API_KEY', '')


@asynccontextmanager
async def lifespan(app):
    if os.environ.get('PRELOAD_MODEL', 'false').lower() == 'true':
        pipeline.segmenter = await run_in_threadpool(Segmenter)
    app.state.jobs = JobQueue(os.environ.get('JOB_DIR', str(ROOT / '.cache/jobs')),
                             run_job, capacity=int(os.environ.get("QUEUE_CAPACITY", "1000")))
    try:
        yield
    finally:
        await run_in_threadpool(app.state.jobs.close)


app = FastAPI(title='Vehicle Image API', version='1.0.0', lifespan=lifespan)
upload_slots = asyncio.Semaphore(4)


def authenticate(request):
    if API_KEY and not secrets.compare_digest(
            request.headers.get('authorization', '').encode(), f'Bearer {API_KEY}'.encode()):
        raise HTTPException(401, 'Invalid API key', headers={'WWW-Authenticate': 'Bearer'})


@app.middleware('http')
async def bound_uploads(request, call_next):
    if request.method == 'POST' and request.url.path in ('/process', '/jobs'):
        async with upload_slots:
            return await call_next(request)
    return await call_next(request)


def run_job(data, options):
    options = Options.model_validate(options)
    started = time.perf_counter()
    remote = data is None
    if data is None:
        try:
            data = fetch_image(options.imageurl)
        except Exception as exc:
            raise JobFailure('Source image download failed. The image host may block the server or the URL may have expired.') from exc
    logging.getLogger("uvicorn.error").info("Timing input: source=%s bytes=%s download=%.2fs", "URL" if remote else "upload", len(data), time.perf_counter() - started)
    try:
        return process_image(data, options)
    except JobFailure:
        raise
    except Exception as exc:
        raise JobFailure('Vehicle image processing failed. Check server logs for this job ID (model, image, or background error).') from exc


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid')
    imageurl: str | None = None
    isBackgroundWant: bool = False
    background: int = Field(default=1, ge=1, le=9999)
    backgroundFolder: Literal["parking-lots", "backgrounds"] = "parking-lots"
    enhancment: bool = False  # Keep the spelling requested by the client.

    @field_validator('background', mode='before')
    @classmethod
    def background_number(cls, value):
        if isinstance(value, bool):
            raise ValueError('background must be a number')
        if isinstance(value, str) and value.endswith('.png'):
            value = value[:-4]
        return value


def fetch_image(url: str) -> bytes:
    """Fetch public HTTP(S) images; pin validated IPs to prevent DNS rebinding."""
    for _ in range(4):
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
            raise ValueError('imageurl must be a public HTTP or HTTPS URL')
        port = parts.port or (443 if parts.scheme == 'https' else 80)
        addresses = {item[4][0] for item in socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)}
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise ValueError('Private and local image URLs are not allowed')
        ip = sorted(addresses)[0]
        kwargs = dict(timeout=urllib3.Timeout(connect=10, read=20), retries=False)
        if parts.scheme == 'https':
            pool = urllib3.HTTPSConnectionPool(ip, port, server_hostname=parts.hostname,
                                              assert_hostname=parts.hostname, cert_reqs='CERT_REQUIRED', **kwargs)
        else:
            pool = urllib3.HTTPConnectionPool(ip, port, **kwargs)
        response = None
        try:
            target = parts.path or '/'
            if parts.query:
                target += '?' + parts.query
            response = pool.request('GET', target, headers={'Host': parts.netloc, 'Accept': 'image/*'},
                                    redirect=False, preload_content=False)
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if not location:
                    raise ValueError('Image URL redirect has no destination')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError('Image URL did not return a successful response')
            data = bytearray()
            for chunk in response.stream(65536):
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError('Image exceeds 25 MB')
            return bytes(data)
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError('Too many image URL redirects')


def process_image(data: bytes, options: Options) -> bytes:
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError('Image exceeds 20 megapixels')
                source.load()
                image = source.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Invalid or oversized image') from exc
    logging.getLogger("uvicorn.error").info("Timing decode: %sx%s %.2fs", image.width, image.height, time.perf_counter() - started)
    started = time.perf_counter()
    background = None
    if options.isBackgroundWant:
        path = BACKGROUNDS / options.backgroundFolder / f'{options.background}.png'
        if not path.is_file():
            raise JobFailure(f'Background {options.background} is not available in {options.backgroundFolder}')
        with Image.open(path) as im:
            background = im.convert('RGB')
    logging.getLogger("uvicorn.error").info("Timing background load: %.2fs", time.perf_counter() - started)
    started = time.perf_counter()
    with processing_lock:
        logging.getLogger("uvicorn.error").info("Timing model lock wait: %.2fs", time.perf_counter() - started)
        result = pipeline.process(image, background, options.enhancment)
    started = time.perf_counter()
    output = BytesIO()
    result.save(output, format='PNG')
    logging.getLogger("uvicorn.error").info("Timing PNG encoding: %.2fs bytes=%s", time.perf_counter() - started, output.tell())
    return output.getvalue()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/jobs', status_code=202)
@app.post('/process', status_code=202,
          openapi_extra={"requestBody": {"required": True, "content": {
              "application/json": {"schema": Options.model_json_schema()},
              "multipart/form-data": {"schema": {"type": "object", "properties": {
                  "image": {"type": "string", "format": "binary"},
                  "imageurl": {"type": "string"},
                  "isBackgroundWant": {"type": "boolean", "default": False},
                  "background": {"type": "integer", "default": 1},
                  "backgroundFolder": {"type": "string", "enum": ["parking-lots", "backgrounds"], "default": "parking-lots"},
                  "enhancment": {"type": "boolean", "default": False}
              }}}
          }}})
async def process(request: Request):
    """JSON: imageurl, isBackgroundWant, background, backgroundFolder, enhancment.

    Multipart: image (file/blob) or imageurl (file/URL), plus the same fields.
    Booleans in multipart are strings such as true/false. Returns a queued job.
    """
    authenticate(request)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise HTTPException(413, 'Request exceeds 25 MB')
    upload_data = None
    try:
        content_type = request.headers.get('content-type', '')
        if content_type.startswith('application/json'):
            fields = json.loads(body)
            options = Options.model_validate(fields)
        elif content_type.startswith('multipart/form-data'):
            async def receive():
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            buffered = BufferedRequest(request.scope, receive)
            async with buffered.form(max_files=1, max_fields=6, max_part_size=MAX_BYTES) as form:
                fields = dict(form)
                upload = fields.pop('image', None)
                if isinstance(fields.get('imageurl'), UploadFile):
                    upload = fields.pop('imageurl')
                if upload is not None:
                    if not isinstance(upload, UploadFile):
                        raise ValueError('image must be a file/blob upload')
                    upload_data = await upload.read()
                options = Options.model_validate(fields)
        else:
            raise HTTPException(415, 'Use application/json or multipart/form-data')
        if (upload_data is not None) == bool(options.imageurl):
            raise ValueError('Provide exactly one image upload or imageurl')
        if options.isBackgroundWant and not (BACKGROUNDS / options.backgroundFolder / f'{options.background}.png').is_file():
            raise ValueError('Selected background is not available')
        job_id = await run_in_threadpool(request.app.state.jobs.submit, upload_data, options.model_dump())
    except OverflowError as exc:
        raise HTTPException(429, str(exc), headers={'Retry-After': '10'}) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {'jobId': job_id, 'status': 'queued', 'statusUrl': f'/jobs/{job_id}',
            'resultUrl': f'/jobs/{job_id}/result'}


@app.get('/jobs/{job_id}')
async def job_status(job_id: str, request: Request):
    authenticate(request)
    job = await run_in_threadpool(request.app.state.jobs.get, job_id)
    if job is None:
        raise HTTPException(404, 'Job not found or expired')
    return job


@app.get('/jobs/{job_id}/result')
async def job_result(job_id: str, request: Request):
    authenticate(request)
    job = await run_in_threadpool(request.app.state.jobs.get, job_id)
    if job is None:
        raise HTTPException(404, 'Job not found or expired')
    if job['status'] != 'completed':
        raise HTTPException(409, 'Result not ready; check job status')
    result = await run_in_threadpool(request.app.state.jobs.get, job_id, True)
    if result is None:
        raise HTTPException(404, 'Result expired')
    return Response(result, media_type='image/png')


@app.post('/generate', response_class=Response,
          responses={200: {"content": {"image/png": {}}}})
async def generate(request: Request):
    """Same inputs as /process; wait in the shared FIFO and return the PNG.

    No client polling or automatic resubmission. Disconnecting discards delivery,
    though already accepted work may still finish in the existing queue.
    """
    # Bound uploads only, never hold an upload slot while waiting for inference.
    async with upload_slots:
        submitted = await process(request)
    job_id = submitted['jobId']
    while True:
        if await request.is_disconnected():
            return Response(status_code=499)
        job = await run_in_threadpool(request.app.state.jobs.get, job_id)
        if job is None:
            raise HTTPException(410, 'Image job expired')
        if job['status'] == 'failed':
            raise HTTPException(500, job['error'] or 'Image processing failed')
        if job['status'] == 'completed':
            result = await run_in_threadpool(request.app.state.jobs.get, job_id, True)
            if result is None:
                raise HTTPException(410, 'Image result expired')
            return Response(result, media_type='image/png',
                            headers={'Cache-Control': 'no-store'})
        await asyncio.sleep(0.5)
