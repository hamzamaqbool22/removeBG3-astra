# Vehicle Image API

> **Queue update: the API contract changed.** POST `/process` (also `/jobs`)
> now returns **202 JSON with a job ID**, not PNG. See [QUEUE.md](QUEUE.md) for
> submission, polling, download and deployment changes. The direct-PNG client
> examples below describe the previous API and must be migrated before use.

Raw vehicle photo → background removal → refined tire/chassis shadows → PNG.
CPU only. One endpoint; no UI, output folders, benchmark scripts, or mode switches.

## Start

Python 3.11–3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn vehicle_pipeline.api:app --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs
Health check: `GET /health`.

The BiRefNet model lives in `.cache/models/birefnet-general.onnx`. Keep this file;
if absent, rembg downloads it on the first request. The model uses about 8 GB RAM
in the measured CPU setup; use one server worker. Inference is serialized and the
loaded model is reused. The first request takes longer to load the model.

## POST /process

JSON for an image URL:

```json
{
  "imageurl": "https://example.com/car.jpg",
  "isBackgroundWant": true,
  "background": 1,
  "backgroundFolder": "parking-lots",
  "enhancment": true
}
```

- `imageurl`: public HTTP(S) image URL. For a blob, use multipart upload instead.
- `isBackgroundWant`: default `false`. False returns transparent RGBA PNG **with
  refined shadows**. True places the car on a parking background.
- `background`: optional number, default `1`; selects `backgrounds/parking-lots/1.png`.
  The string `"1.png"` is also accepted. Ignored when no background is requested.
- `backgroundFolder`: optional, default `"parking-lots"`. Allowed values are
  `"parking-lots"` (25 images) and `"backgrounds"` (40 images). Selects the folder
  under `backgrounds/`; arbitrary paths are rejected. For example, folder
  `"backgrounds"` with background `12` selects `backgrounds/backgrounds/12.png`.
  An unavailable number returns 422; it does not silently select a different image.
- `enhancment`: optional, default `false` (spelling matches the requested API).
  True applies the vehicle color/exposure correction and subtle final contrast.
  Without a background, true applies subtle vehicle contrast only.

The response is the image itself (`Content-Type: image/png`), not JSON or base64.
Transparent output is 1024 × 768. Background output uses that background's size.
A single aspect-preserving framing rule retains the example's appearance. There
is no auto/Org selector, parking-line detection, or scene placement model.
Lighting is a photo correction; original reflections and perspective remain.

### Upload a file/blob

```bash
curl --fail-with-body http://localhost:8000/process \
  -F 'image=@images/left.png' \
  -F 'isBackgroundWant=true' \
  -F 'backgroundFolder=parking-lots' \
  -F 'background=15' \
  -F 'enhancment=true' \
  --output result.png
```

```javascript
const form = new FormData();
form.append('image', blob, 'vehicle.jpg');
form.append('isBackgroundWant', 'true');
form.append('background', '1');
form.append('backgroundFolder', 'parking-lots');
form.append('enhancment', 'true');
const response = await fetch('/process', { method: 'POST', body: form });
if (!response.ok) throw new Error(await response.text());
const result = await response.blob();
```

The file field may also be named `imageurl`. Provide either one upload or one URL.
JSON cannot contain a binary blob. Uploads/downloads are limited to 25 MB and
input images to 20 megapixels. Invalid inputs return 422; oversized requests 413;
unsupported request types 415. Private/local URLs are rejected, including redirects.
The API does not save submitted images or generated results.

## Files

- `vehicle_pipeline/api.py`: request validation, URL download, PNG response.
- `pipeline.py`: segmentation, geometry, refined shadows, transparent export.
- `segmentation.py`, `geometry.py`, `placement.py`, `shadows.py`: production processing.
- `parking.py`: simple background replacement and optional enhancement.
- `images/`: original input examples (excluded from Docker).
- `backgrounds/parking-lots/`, `backgrounds/backgrounds/`: numbered runtime PNGs.

## Source-aware shadows (CPU only)

The API now first attempts to recover the **visible original cast shadow** from
pavement below the vehicle. It estimates smooth ground brightness, separates
connected dark regions from the cutout, and exports their attenuation as shadow
alpha. The same transform moves the car and its shadow. This preserves the
photographed direction, extent, and camera-height appearance without diffusion
or GPU processing. BiRefNet remains only for background removal, on CPU.

When the ground estimate or shadow evidence is insufficient, procedural contact
shading is used instead. Dark ground markings can be ambiguous; a cropped or
absent shadow cannot be recovered exactly. Recovered shadows also retain the
source lighting direction, which may differ from a replacement background.
No algorithm here guarantees exact reconstruction for arbitrary scenes.

## Ambient blending

With `isBackgroundWant=true` and `enhancment=true`, enhancement now also applies
subtle spatial ambient color and ground bounce to the vehicle. Broad upper-scene
color affects the upper body gently; three pavement samples near the placed car
provide a smoothly varying lower-body tint. Deep blacks are preserved and
saturated colors receive less adaptation. The vehicle alpha, placement, and
shadow opacity are unchanged. Existing final contrast enhancement still applies.

This is CPU-only photo blending, not generated reflections or physical relighting.
It adds no API fields. With enhancement disabled, results are unchanged. Without
a selected background, the existing transparent-output behavior is unchanged.

## Hosting

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker, custom-server, Vast, Railway,
and Laravel integration instructions. Set `API_KEY` for Bearer authentication on
`/process`; unset preserves local unauthenticated use. `CPU_THREADS` defaults to 6
locally and 4 in Docker. Docker preloads the model before accepting traffic.
Both background folders are included in Docker. Only the selected PNG is loaded
for a request; the full collection is not preloaded into RAM.
