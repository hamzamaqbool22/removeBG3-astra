# Two-instance /generate trial (Linux server)

This optional add-on leaves the current API on 127.0.0.1:8000 untouched. A second
API listens on 8001 with its own model and persistent queue. A loopback-only Nginx
load balancer on 8002 sends /generate requests to the instance with fewer active
connections. Each waiting synchronous request stays connected, so this usually
spreads the queues. It is not a global FIFO, a global queue limit, or an exact
measure of remaining work; unequal image processing times can still cause imbalance.

Other paths (/process, /jobs, /jobs/<id>, /health) always go to the original API.
This preserves existing job lookup. /health via 8002 checks only the original
API; check both APIs separately. Both instances require the same API_KEY.
No public endpoint, extension change, or public Nginx change is required for the trial.

## Capacity and expectations

Start with the original API at its current 8 CPU quota / 8 threads, and the extra
API at 6 CPU quota / 6 threads with a 24 GB memory limit. On the reported 16-core,
64 GB machine this leaves CPU/memory headroom, but inspect actual Laravel and
model memory usage before load testing. Limits are ceilings, not reservations.
If the primary still has a 4 CPU quota, the total is 10 CPUs instead of 14.
Do not add ten model processes or change Uvicorn --workers; each queue has a
single owner enforced by a file lock. A second model increases memory use.

Parallelism aims to improve throughput and queue wait for multiple clients. It
does not promise 3–4 seconds per image or fix extension storage/download delays.
Both CPUs compete for memory bandwidth/cache; compare measured results before
increasing further. The extra queue defaults to 500, independent of the original
1000: total potential acceptance is 1500, not a shared 1000-job queue. Capacity
is a limit, not a latency guarantee, and payload disk limits can reject work sooner.
Proxy failures and full queues are returned without retrying image generation.
Long waits can still exceed the public Nginx/PHP/Laravel/client timeout.

## Upload from the Mac

Run locally, not inside SSH. Keep the existing server .env and compose.yaml.
Upload the latest backend separately if it is not already deployed.

```bash
cd /Users/hamzamaqbool/Documents/python/removeBG3
ssh social_auto_image@img.gemquery.com 'mkdir -p /opt/removebg/deploy /opt/removebg/local-test'
scp compose.parallel.yaml PARALLEL.md social_auto_image@img.gemquery.com:/opt/removebg/
scp deploy/generate-balancer.conf social_auto_image@img.gemquery.com:/opt/removebg/deploy/
scp local-test/generate_load.py social_auto_image@img.gemquery.com:/opt/removebg/local-test/
```

## Start on the server

Check free memory and current resource limits first. Schedule the trial away from
customer load. The original API continues running; building/loading the new model
still consumes resources and can temporarily slow it.

```bash
cd /opt/removebg
free -h
sudo docker stats --no-stream
sudo docker inspect "$(sudo docker compose ps -q vehicle-api)" --format 'CPU quota={{.HostConfig.NanoCpus}} Memory limit={{.HostConfig.Memory}}'
sudo docker compose -p removebg-parallel -f compose.parallel.yaml config --quiet
sudo docker compose -p removebg-parallel -f compose.parallel.yaml run --rm --no-deps generate-balancer nginx -t
sudo docker compose -p removebg-parallel -f compose.parallel.yaml up -d --build
sudo docker compose -p removebg-parallel -f compose.parallel.yaml ps
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8002/health
```

The second model can take several minutes to start. The balancer waits for its
healthcheck. Do not switch Laravel if any check fails. For diagnostics:

```bash
sudo docker compose -p removebg-parallel -f compose.parallel.yaml logs --since=5m --timestamps -f
```

Ctrl+C only leaves the logs. On future updates restart the balancer after API
updates as needed; upstream addresses are fixed host-loopback ports.

## Benchmark without changing production routing

Ensure /opt/removebg/images contains the eight test photos (upload them if absent).
Run the benchmark in a temporary client container using the extra API's built
image. It runs only the client script, not another model. Host networking lets it
reach both test ports. The existing .env supplies the API key without printing it;
Docker env-file values must be plain KEY=value (do not wrap the key in quotes).

```bash
IMAGE_ID=$(sudo docker compose -p removebg-parallel -f compose.parallel.yaml images -q vehicle-api-extra)
sudo docker run --rm --network host --env-file .env -v "$PWD/local-test/generate_load.py:/tmp/generate_load.py:ro" -v "$PWD/images:/tmp/test-images:ro" "$IMAGE_ID" python /tmp/generate_load.py --input /tmp/test-images --url http://127.0.0.1:8000/generate --count 8 --concurrency 2 --output /tmp/baseline
sudo docker run --rm --network host --env-file .env -v "$PWD/local-test/generate_load.py:/tmp/generate_load.py:ro" -v "$PWD/images:/tmp/test-images:ro" "$IMAGE_ID" python /tmp/generate_load.py --input /tmp/test-images --url http://127.0.0.1:8002/generate --count 8 --concurrency 2 --output /tmp/parallel
```

These temporary client containers exit and discard test PNGs/reports; console
shows total batch duration and per-request timing. Repeat with --concurrency 1
and 4, using identical photos/settings. In another terminal watch `docker stats`
and the balancer logs: requests should appear on both :8000 and :8001, and the two
API logs should show overlapping processing periods. Stop if memory pressure,
OOM restarts, failures, or single-image latency worsens substantially.

## Enable after benchmarking

In Laravel, change only its Python upstream base URL from http://127.0.0.1:8000
to http://127.0.0.1:8002, retaining the API key and existing auth/user checks.
The exact environment variable must be read from the deployed Laravel config;
do not guess its name. Refresh Laravel config and long-lived queue workers using
the deployment's normal procedure. Public https://img.gemquery.com/generate and
the extension stay unchanged. Do not bypass Laravel by exposing port 8002 publicly.

## Roll back

Point Laravel back to 8000 first. Let active requests and both queues drain; the
extra API volume is preserved but pending requests may be interrupted if stopped.
Then stop only this add-on (never use down -v):

```bash
cd /opt/removebg
sudo docker compose -p removebg-parallel -f compose.parallel.yaml stop
```

The original `sudo docker compose ... vehicle-api` commands remain unchanged.
To customize the add-on later, set PARALLEL_CPU_THREADS, PARALLEL_CPUS and
PARALLEL_QUEUE_CAPACITY in .env. Leave the existing CPU_THREADS setting intact.

## Local validation

Compose rendering can be checked without a Docker daemon. Container Nginx syntax,
real CPU throughput, Linux host networking and peak model memory must be tested
on the server: Docker Engine was unavailable on the development Mac. No server
changes have been applied by preparing these files.

## Existing queue volume: permission denied on worker.lock

An image built before the queue directory was created/owned by `vehicle` can
leave a fresh mounted volume unwritable to the application. The Dockerfile now
pre-creates the directory before assigning ownership. Rebuilding alone does not
repair an already existing volume. Stop the extra instance and repair only its
queue volume, then verify access as its normal user:

```bash
cd /opt/removebg
sudo docker compose -p removebg-parallel -f compose.parallel.yaml stop vehicle-api-extra
sudo docker compose -p removebg-parallel -f compose.parallel.yaml run --rm --no-deps --user root vehicle-api-extra chown -R vehicle:vehicle /app/.cache/jobs
sudo docker compose -p removebg-parallel -f compose.parallel.yaml run --rm --no-deps vehicle-api-extra python -c 'from pathlib import Path; p=Path("/app/.cache/jobs/.permission-check"); p.write_text("ok"); p.unlink(); print("Queue directory is writable")'
sudo docker compose -p removebg-parallel -f compose.parallel.yaml up -d
sudo docker compose -p removebg-parallel -f compose.parallel.yaml ps
```

This does not delete queued jobs, alter the original queue, or run the API as
root. Upload the corrected Dockerfile for future builds as well.

## Single-pass segmentation experiment

`SEGMENTATION_MODE=auto` (default) preserves adaptive crop refinement for small
vehicles. `single` makes exactly one model call, keeping the first mask. This
can lose fine details on distant vehicles and must be compared visually; it
cannot eliminate queue waiting, result download, or extension storage time.
Set the extra instance through `PARALLEL_SEGMENTATION_MODE=single` in Compose.

After uploading the updated segmentation.py, compose.parallel.yaml, Dockerfile
and local-test/compare_segmentation.sh, run as the normal SSH user:

```bash
cd /opt/removebg
bash local-test/compare_segmentation.sh
```

The script rebuilds only the extra API, runs both modes with six threads directly
on 8001 with concurrency 1, warms each mode once, then processes eight photos. It
saves PNGs, JSON reports and per-mode server logs in a dated outputs directory.
Avoid other generation requests during the experiment. Existing primary API is
untouched. The extra API ends in single mode; restarting with no override returns
to the default auto mode unless PARALLEL_SEGMENTATION_MODE is saved in .env.
Compare `Timing segmentation (1 passes)` / `(2 passes)` for the same image in
each mode. A queue-inclusive 15-second HTTP request alone does not prove two
inference passes. The eight-image set may not contain any small-car two-pass
cases: include the actual slow photo in images/ for a follow-up comparison.
Docker's --env-file parser expects an unquoted API_KEY value.

## Optional third-instance comparison

Upload compose.third.yaml, deploy/generate-balancer-three.conf and
local-test/compare_instances.sh alongside the current files. Run
`bash local-test/compare_instances.sh` as the normal SSH user in /opt/removebg.
Avoid other generation traffic during this experiment. The original instance
must already use single-pass mode; the script sets the second to single/6 threads.
It compares two repeated 16-image batches at concurrency 3 on two versus three
instances. Both tests use the same image distribution and warmups. The third gets
4 CPU threads, a 4 CPU quota, its own queue on port 8003 and 20 GiB memory ceiling;
the trial balancer listens only on host loopback 8004. This tests adding capacity
to the existing 8+6 allocation: 18 CPU quota units can contend on the reported
16-physical-core host. It is not a claim of optimal CPU partitioning.

Expected retained model RAM from measurements is ~42 GiB for three, but peaks can
be larger. The script requires 22 GiB MemAvailable before starting the third;
this is only a preflight check, not protection against other services growing.
Monitor `sudo docker stats` and `free -h` in another SSH terminal. The 20 GiB
third-instance cap may expose an OOM on larger inputs; do not raise it blindly.
The script stops the third project on completion/failure after startup. It does
not modify Laravel or route public traffic to the trial. Single-pass remains
active in the existing second instance after the test. Outputs persist in a dated
outputs/instances-* directory. A shell killed forcibly may skip cleanup; then run
`sudo docker compose -p removebg-third -f compose.third.yaml stop` manually.

Capacity remains per-instance (no shared FIFO); three queues can accept up to
2000 jobs with current defaults, subject to payload limits and HTTP timeouts.
