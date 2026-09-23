# Vehicle API deployment handoff

## Before deployment

The deployment includes 25 PNGs in `backgrounds/parking-lots/` and 40 in
`backgrounds/backgrounds/`. The optional `backgroundFolder` field selects
`parking-lots` (default) or `backgrounds`; `background` selects a number
(default 1). Missing backgrounds return 422; transparent output does not need
them. Images stay on disk and only the selected background is loaded. Original
examples, local caches, virtualenvs and review outputs are excluded from Docker.

Start with **4 CPU cores and 12–16 GB RAM per replica**, one Uvicorn worker.
The prior local CPU run used approximately 8 GB RAM; Linux peak memory and speed
must be measured on your server with representative images. This is a starting
allocation, not a capacity guarantee. Each additional process loads its own model.
No CUDA, GPU or diffusion is used. CPU BiRefNet removes backgrounds; shadows and
ambient blending use ordinary image processing.

## Custom server (recommended starting point)

Install Docker Engine and the Compose plugin. Copy this project, or clone its
repository. Keep both background folders in the deployment. From the project directory:

```bash
cp .env.example .env
openssl rand -hex 32
# Put the generated value after API_KEY= in .env; keep it secret.
docker compose up -d --build
docker compose logs -f vehicle-api
curl --fail http://127.0.0.1:8000/health
```

The image build downloads and checksum-verifies about 928 MiB of ONNX weights;
internet access is needed during build. Dependencies are installed from
pyproject.toml's version ranges, so retain the validated image tag/digest for
rollbacks rather than rebuilding it for every restart. The model is baked into
the image; do not mount an empty volume over `/app/.cache/models`.

Startup loads the model before accepting HTTP traffic. `/health` reports server
liveness after startup, not background availability or an inference warmup.
Run a real upload after deployment and measure latency and peak memory. Try both
transparent and background output, with enhancement on/off, then concurrent load.

Compose exposes only loopback port 8000. Put an HTTPS reverse proxy in front of
it, or call over localhost from Laravel on the same server. For an existing
Nginx HTTPS virtual host, use:

```nginx
location / {
    client_max_body_size 25m;
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header Authorization $http_authorization;
    proxy_connect_timeout 10s;
    proxy_read_timeout 180s;
    proxy_send_timeout 180s;
}
```

Use your normal TLS/domain setup. Keep API_KEY only in the Laravel backend and
server environment, never in browser JavaScript. `/process` requires
`Authorization: Bearer <API_KEY>` when configured. `/health` stays public.
The service does not save user uploads or results. Logs go to stdout/stderr.

One image is inferred at a time per process. Docker's Uvicorn connection limit
is 4, which bounds concurrent waiting connections and returns 503 under excess
load (health probes count too). This is not a durable job queue or a guarantee
of four simultaneous images. Clients should retry 503 with bounded backoff or
route to another healthy replica. Do not increase worker count without providing
RAM for every model. Tune CPU allocation and CPU_THREADS together.

To run without Docker, use Python 3.11–3.13, `pip install .`, set API_KEY,
CPU_THREADS=4 and PRELOAD_MODEL=true, then supervise this command with systemd:

```bash
uvicorn vehicle_pipeline.api:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 4
```

Run from the project directory with a writable `.cache/`, or set `U2NET_HOME`
and `NUMBA_CACHE_DIR` to writable absolute paths. Copy the existing
`.cache/models/birefnet-general.onnx` there to avoid the first-start download.

## Laravel integration

Set `services.vehicle.url` and `services.vehicle.key` in your Laravel config,
backed by private environment variables. A controller can upload and return PNG:

```php
$file = $request->file('image'); // Validate required image and size first.
$result = Http::withToken(config('services.vehicle.key'))
    ->connectTimeout(10)->timeout(180)
    ->attach('image', fopen($file->getRealPath(), 'r'), $file->getClientOriginalName())
    ->post(rtrim(config('services.vehicle.url'), '/') . '/process', [
        'isBackgroundWant' => 'true',
        'background' => '1',
        'backgroundFolder' => 'parking-lots',
        'enhancment' => 'true',
    ]);
$result->throw(); // Map upstream errors/timeouts in your application's handler.
return response($result->body(), 200)->header('Content-Type', 'image/png');
```

Import `Illuminate\Support\Facades\Http`. For image URLs use a JSON POST with
`imageurl` and the same fields (JSON booleans). No base64 decoding is needed.
See README for the full existing API contract. Do not blindly retry every 422;
fix the input. When retrying uploads after 503, reopen the file stream.

## Sharing a Vast instance with Flux

A busy GPU does **not** inherently block CPU inference. Start this API as a
separate supervised process on a separate port, using a separate virtualenv if
inside the existing Flux container. Follow the non-Docker command above and
expose the port through the instance's configured networking/reverse proxy.
Avoid changing Flux's Python dependencies or replacing its running container.

Send vehicle requests to this service, not the Flux generation queue. Existing
Vast Serverless worker routing may still queue or cold-start: its behavior
depends on your worker/template, not on the fact this code uses CPU. An additional
route alone does not prove queue isolation. Inspect and load-test your Flux
worker before integrating; its code has not been reviewed here.

Reserve CPU/RAM for both services and test while Flux is busy. Flux also uses
host CPU/RAM, so competition can increase latency. The screenshot's host figures
are not proof of free peak-time capacity. If instances run only 12 hours daily,
this API also disappears when they stop unless another always-on service or
independent lifecycle handles it. For predictable availability, a separate CPU
server is simpler. Official worker queue/concurrency reference:
https://github.com/vast-ai/pyworker

## Railway testing

Deploy the root Dockerfile from a repository containing both background folders.
Set API_KEY, CPU_THREADS=4 and PRELOAD_MODEL=true. The Docker command honors
Railway's PORT. Use one replica initially, a healthcheck path of `/health`, and
allow a startup healthcheck timeout of 300 seconds for model loading. The check
does not succeed until FastAPI startup completes. Leave the Docker start command
in place and generate an HTTPS service domain.

Choose a plan/resource allocation that can supply the recommended RAM and CPU;
do not assume a free/trial allocation can fit this model. Railway does not use
compose.yaml's resource limits, so configure limits in its service settings.
Watch memory and CPU after a full-size upload; the container's idle healthcheck
alone does not verify inference capacity. Avoid service sleeping if low latency
matters. Current plan and deployment documentation:
https://docs.railway.com/pricing/plans
https://docs.railway.com/deployments/healthchecks

## Validation status

Local API/configuration checks are separate from a Linux container deployment.
Docker Engine was not running on the development machine when this handoff was
prepared, so the image has not been built or load-tested here. No cloud service
has been created or uploaded as part of this preparation.
