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
