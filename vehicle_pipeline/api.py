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
from urllib.parse import urlsplit, urljoin
from typing import Literal
import warnings

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
    yield


app = FastAPI(title='Vehicle Image API', version='1.0.0', lifespan=lifespan)


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
    background = None
    if options.isBackgroundWant:
        path = BACKGROUNDS / options.backgroundFolder / f'{options.background}.png'
        if not path.is_file():
            raise ValueError(f'Background {options.background} is not available in {options.backgroundFolder}')
        with Image.open(path) as im:
            background = im.convert('RGB')
    with processing_lock:
        result = pipeline.process(image, background, options.enhancment)
    output = BytesIO()
    result.save(output, format='PNG')
    return output.getvalue()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/process', response_class=Response,
          responses={200: {'content': {'image/png': {}}}},
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
    Booleans in multipart are strings such as true/false. Returns PNG bytes.
    """
    if API_KEY and not secrets.compare_digest(
            request.headers.get('authorization', '').encode(), f'Bearer {API_KEY}'.encode()):
        raise HTTPException(401, 'Invalid API key', headers={'WWW-Authenticate': 'Bearer'})
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
        if upload_data is None:
            try:
                upload_data = await run_in_threadpool(fetch_image, options.imageurl)
            except (OSError, urllib3.exceptions.HTTPError) as exc:
                raise ValueError('Could not fetch image URL') from exc
        result = await run_in_threadpool(process_image, upload_data, options)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(result, media_type='image/png', headers={'Content-Disposition': 'inline; filename="vehicle.png"'})
