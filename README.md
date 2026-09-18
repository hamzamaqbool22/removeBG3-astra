# Vehicle Image API

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
  "enhancment": true
}
```

- `imageurl`: public HTTP(S) image URL. For a blob, use multipart upload instead.
- `isBackgroundWant`: default `false`. False returns transparent RGBA PNG **with
  refined shadows**. True places the car on a parking background.
- `background`: optional number, default `1`; selects `backgrounds/ParkingLots/1.png`.
  The string `"1.png"` is also accepted. Ignored when no background is requested.
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
  -F 'background=15' \
  -F 'enhancment=true' \
  --output result.png
```

```javascript
const form = new FormData();
form.append('image', blob, 'vehicle.jpg');
form.append('isBackgroundWant', 'true');
form.append('background', '1');
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
- `images/`, `remove.bg-outputs/`, `backgrounds/`: original assets retained.
