# Vehicle Image Express API

Standalone Node/Express backend derived from chrome-extension v0.4.0. The
original extension and Python backend are unchanged. This project contains its
own snapshot of their JavaScript geometry, shadows, normalization and lighting.
It uses the same BiRefNet Lite 512 FP32 model with native ONNX Runtime CPU;
Sharp handles decoding/encoding. No browser, GPU, Python, Redis or external
image-processing service is required. Small decoder/runtime differences can
produce differences from browser outputs; no promise of pixel-identical results.

## Run locally

Requires Node.js 22.13+ (Node 22 LTS recommended), npm, and sufficient disk/RAM.
From this folder:

```sh
npm ci
cp .env.example .env
npm run setup
npm start
```

API: http://127.0.0.1:8002. `GET /health` is liveness; `GET /ready` returns 200
once the model is loaded (503 during startup/recovery). The first setup downloads
and verifies the ~183 MiB CPU model. Restart reuses it. Numbered backgrounds come
from your Cloudinary account and are cached with checksum verification. Each
folder includes a default if Cloudinary fails; the job status exposes `warning`.

## Frontend contract

One image per request. An uploaded browser Blob/File uses multipart form field
`image`. A remote URL uses JSON `imageurl` instead; never send both. Any public
HTTPS host is accepted for `imageurl`. Redirects, local-network/private IPs,
custom ports, and embedded credentials are rejected.

POST `/jobs` (alias `/process`) returns **202 JSON**, not image bytes:

```json
{
  "jobId": "a-unique-UUID",
  "status": "queued",
  "statusUrl": "/jobs/a-unique-UUID",
  "resultUrl": "/jobs/a-unique-UUID/result",
  "pollAfterMs": 2000
}
```

Options (same names for JSON and form fields):

- `isBackgroundWant`: true/false, default false. False returns transparent PNG
  including shadows; true composites onto the selected background.
- `folder`: `parking-lots` (default) or `backgrounds`. `backgroundFolder` is an alias.
- `background`: number or filename, e.g. `3` or `3.png`; defaults to `1.png`.
- `enhancement`: true/false, default false. Historical `enhancment` also accepted.

`GET /backgrounds` lists available numbered images.

```sh
curl -F 'image=@/absolute/path/car.jpg' \
  -F 'isBackgroundWant=true' -F 'folder=parking-lots' \
  -F 'background=3' -F 'enhancement=true' \
  http://127.0.0.1:8002/jobs
```

JSON URL request:

```json
{
  "imageurl": "https://your-image-host.example/car.jpg",
  "isBackgroundWant": true,
  "folder": "parking-lots",
  "background": 3,
  "enhancement": true
}
```

Poll `GET /jobs/{jobId}` about every 2 seconds (add random jitter for many users).
Statuses: `queued`, `running`, `done`, `failed`. Queue position is informational,
not a time guarantee. Status includes stage, timestamps, error, and an optional
background-fallback warning. On done, GET `/jobs/{jobId}/result` returns PNG.
On failed, show `error`. Not-ready result requests return 409; unknown/expired
jobs return 404. Download results within RESULT_TTL_HOURS (24 by default).

```js
// Use from your trusted application backend when API_KEY is enabled.
const form = new FormData();
form.append('image', file); // browser File/Blob
form.append('isBackgroundWant', 'true');
form.append('folder', 'parking-lots');
form.append('background', '3');
const submitted = await fetch(`${base}/jobs`, {method:'POST', body:form, headers});
if (!submitted.ok) throw new Error((await submitted.json()).error);
const job = await submitted.json();
for (;;) {
  await new Promise(resolve => setTimeout(resolve, 2000 + Math.random()*500));
  const response = await fetch(`${base}${job.statusUrl}`, {headers});
  if (!response.ok) throw new Error('Job status unavailable');
  const state = await response.json();
  if (state.status === 'failed') throw new Error(state.error);
  if (state.status === 'done') {
    const response = await fetch(`${base}${job.resultUrl}`, {headers});
    if (!response.ok) throw new Error('Result unavailable');
    const photo = await response.blob();
    // Display/download photo. Also display state.warning if present.
    break;
  }
}
```

## What happens if 100 people upload at once?

The default queue admits up to 200 unfinished jobs (including uploads in progress).
Uploads stream to disk with a 20 MiB limit each. Waiting images are not decoded
or held in RAM. SQLite stores queue state; a separate child process runs exactly
ONE image at a time. The API remains available for uploads/status requests while
that process does CPU work. This bounds inference concurrency, not waiting time.
At 5 seconds/image, the last of a 100-image burst waits roughly 8 minutes, plus
uploads/startup. Actual time depends on hardware and image complexity.

- Queue full: HTTP 429 and Retry-After: 10. Retry later with jitter.
- Low disk space: HTTP 503. Keep several GB free: 200 x 20 MiB inputs can approach
  4 GiB, plus results, model and cache. 1,000 retained jobs is also an admission cap.
- Bad/oversized input: HTTP 400/413, or a failed job when decoding fails.
- Decoding is capped at 12 megapixels by default (up to 20 configurable).
- Worker timeout (180 seconds by default) or crash: current job fails, worker is
  replaced, remaining waiting jobs continue. It does not endlessly retry bad images.
- Server restart: queued jobs survive; interrupted running jobs are requeued.
- Completed inputs are deleted. Finished/failed jobs and outputs expire after 24h.
- The worker exits if its API parent disappears. Temporary orphan uploads are
  cleaned after an hour. Keep the DATA_DIR on persistent local storage.

Run **one API process with one data directory**. Do not use PM2 cluster mode,
multiple Uvicorn-style workers, or mount one SQLite directory across servers.
The PID lock prevents two live local processes from owning the same queue.
For multiple machines, use a shared queue such as Redis and shared object storage
in a future scaling change. This project intentionally serves one machine.

## Deploy

Start with one processor on a CPU server, then measure real RAM and throughput.
A conservative starting test allocation is 4 CPU cores / 8 GB RAM; this is not a
verified minimum or a guarantee against all host OOM conditions. Never increase
processing concurrency without measurements. CPU_THREADS controls internal
math threads, NOT simultaneous images.

Set API_KEY to a random secret before exposing the service. The API requires
`Authorization: Bearer YOUR_KEY` for all job/background endpoints. Keep that
secret in your Laravel/backend server; do NOT ship it in a public extension.
This is service authentication, not per-customer authorization. Your application
must enforce customer login, ownership, rate limits and quotas before proxying
requests. Job IDs are random, but do not replace your application's access checks.
Set CORS_ORIGINS only for explicitly permitted browser origins if needed.

Place behind HTTPS via your reverse proxy. Restrict direct access to port 8002.
Forward Authorization headers; allow 21 MiB HTTP bodies for 20 MiB uploads and
up to 120 seconds upload time. Job requests return quickly after upload; image
processing does not keep the original request open. Avoid logging image URLs
or image contents. Model/Cloudinary downloads require outbound HTTPS.

Docker (build context is this express-api folder):

```sh
# Put API_KEY in .env first.
docker compose up --build -d
docker compose logs -f
curl http://127.0.0.1:8002/health
```

compose.yaml caps the whole service at 8 GB and 4 CPUs, and persists data in a
volume. A whole-container OOM/restart is still possible if limits are too small;
queued work persists, and interrupted work is retried on restart. Test the target
Linux server before production. Do not delete the volume to update the app.

## Test and measure

`npm test` exercises 100 concurrent HTTP uploads using a tiny fake image worker
(to test admission/queue behavior independently of model speed), full-queue
rejection, persistence/recovery, worker crash and timeout replacement.

For REAL image load, start the API, then in a second terminal:

```sh
npm run load -- --input ../chrome-extension/images --count 100
```

That submits 100 requests concurrently, cycling through available images. It
polls results, downloads every successful PNG and writes validation/load/report.json.
Start with --count 8, watch RAM, then try 100. This is a finite burst test, not a
claim about 24-hour throughput. For authenticated tests, export API_KEY to the
load-test process too. The output directory can contain customer photos; keep
it private and delete it after testing.

## Source and licenses

Pipeline files in src/pipeline are an independent snapshot from chrome-extension.
BiRefNet Lite model source/revision/hashes and Cloudinary URLs are pinned in
assets/catalog.json. Third-party notices are in licenses. JS geometry/shadows
are unchanged; a failed segmentation (e.g. missing roof) is not fixed by this
API conversion. npm lockfile pins the tested dependency versions.
