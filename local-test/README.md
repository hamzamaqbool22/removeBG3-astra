# Direct /generate burst test

Restart the backend to load the updated capacity and queue logging:
```sh
CPU_THREADS=4 QUEUE_CAPACITY=1000 PRELOAD_MODEL=true .venv/bin/python -m uvicorn vehicle_pipeline.api:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 1200
```
Wait for “Application startup complete”. In another terminal, from this project:
```sh
.venv/bin/python local-test/generate_load.py --count 16
```
With the eight photos in `images/`, each is sent twice. All 16 requests start
together. The server processes one image at a time; look for Queue accepted,
Queue processing, and Queue completed in the server terminal. The client prints
SEND/OK/FAIL and saves PNGs plus report.json in a unique outputs/generate-load-*
folder. Each request includes the selected background and enhancement, matching
the extension. Defaults: parking-lots/1.png, enhancement enabled.

For 100 simultaneous requests (cycles through the eight images):
```sh
.venv/bin/python local-test/generate_load.py --count 100
```
For exactly 100 copies of EACH image, use --count 800 --concurrency 100 (100
in flight at once, not 800). No submissions are retried. Client timeout is four
hours per request. Ctrl-C does not cancel already accepted server work.

CPU_THREADS controls ONNX intra-operation threads for ONE image, not parallel
images. Compare 4 and 8 by restarting the server with each setting and repeating
the same test after model preload. Look at server processing times and total
throughput; individual client times include queue waiting. More threads may be
slower. PRELOAD_MODEL moves model loading to startup, not faster inference.
The optional adaptive crop can perform a second inference for distant cars.

Queue capacity is now 1,000 active/waiting jobs, configurable using QUEUE_CAPACITY.
The existing 2 GB job storage limit remains: large uploads or retained outputs
can cause HTTP 429 earlier. HTTP 503 means the Uvicorn connection limit was hit.
1,000 jobs at 8–12 seconds each means approximately 2.2–3.3 hours to drain.
The extension testing panel currently has a two-hour batch timeout, so it is not
suitable for the tail of that worst-case queue without a timeout change.

---

# Local Mac capacity test

To simulate 20 users submitting together to one running server:
```sh
.venv/bin/python local-test/load.py --ports 8001 --count 20 --concurrency 20 --input "/full/path/to/photos"
```
This starts 20 client requests, not 20 model processes. The first wave is released
together. Keep the server's connection limit enabled: busy responses (503) are
expected. The script counts successes and HTTP errors, does not retry, and checks
server health afterward. A nonzero exit with busy responses is expected. When
count exceeds concurrency, each client sends its next job after its previous one
finishes. Default behavior remains one client per server.

Run commands from the project root. This Mac has 24 GB RAM; start with ONE
server after closing heavy apps. At setup time it had about 22 GB swap in use.
Check Activity Monitor's Memory Pressure during inference and stop the test if
pressure stays yellow/red or the computer becomes unresponsive. Ctrl-C stops
the client; stop its server too to end any inference already running.

Terminal 1 — server:
```sh
sh local-test/server.sh 8001
```

Terminal 2 — test client:
```sh
.venv/bin/python local-test/load.py --ports 8001 --count 8
```

If `images/` is missing, choose an existing raw vehicle photo or folder:
```sh
.venv/bin/python local-test/load.py --ports 8001 --count 8 --input "/full/path/to/car.jpg"
```
Replace the example path with your photo's actual path (you can drag the photo
from Finder into Terminal after `--input `). A single photo is repeated for the
requested count. Folders support PNG, JPEG, WebP, BMP and TIFF, including uppercase
extensions. Only files directly inside the selected folder are used.

Repeat the client command for warm-model timing. The first run includes model
loading. It cycles through the selected photos and checks returned PNG validity,
saving no outputs. This measures transparent output, not URL download or parking
enhancement. Requests are sent sequentially to each server; waiting jobs remain
in this client's memory. This is NOT a production durable queue.

Only if memory pressure remains healthy and there is room for another model,
open another terminal and run `sh local-test/server.sh 8002`. Then test with:
```sh
.venv/bin/python local-test/load.py --ports 8001 8002 --count 16
```

Both servers share the same CPU and RAM. More processes can be slower. Do not
start three or more on this 24 GB Mac without measured evidence of headroom.
Set the same API_KEY in server and client terminals if using authentication.
These servers bind only to localhost, not the public internet. Mac results are
useful for comparison but do not establish Linux hosting performance.
