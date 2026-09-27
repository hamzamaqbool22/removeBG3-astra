# Persistent queued processing

Restart the API to apply these changes. One model worker processes images in
order; accepted uploads wait on disk. This does not start more model copies.

## Client migration

POST `/process` or `/jobs` with the existing JSON URL or multipart upload fields.
Options, background folders, API_KEY authentication and image rendering are
unchanged. The response is now **202 JSON**, not a finished PNG:

```json
{
  "jobId": "unique-id",
  "status": "queued",
  "statusUrl": "/jobs/unique-id",
  "resultUrl": "/jobs/unique-id/result"
}
```

GET statusUrl every two seconds. Status is queued, processing, completed or
failed. When completed, GET resultUrl for image/png. Failed jobs have an error
message: stop polling and show it. Results not ready return 409; missing/expired
jobs return 404. Send the same Bearer API_KEY on submission, status and result.
Keep that key in Laravel, not browser JavaScript. Laravel must associate job IDs
with the submitting user and enforce ownership: Python has shared-key access,
not individual user accounts. Do not resubmit an accepted job while polling.
A lost submission response can cause duplicate jobs if the client resubmits.

## Storage and limits

Default JOB_DIR is `.cache/jobs`, containing SQLite uploads/options/results.
One process owns this directory; starting another on it fails explicitly. Use
exactly one Uvicorn worker. Interrupted jobs retry on restart with the same ID.
Queued work persists only if the directory survives the restart/redeployment.

At most 100 queued/processing jobs; full queue returns 429 with Retry-After=10.
Four request bodies are buffered simultaneously; other connections wait.
Uvicorn's connection limit is now 128; extreme overload may still return 503.
Request limit stays 25 MiB, decoded image limit 20 megapixels. Invalid image
contents or URL failures produce failed jobs asynchronously; malformed request
options return 422 immediately. The queue does not increase inference speed.

Results expire one hour after completion. Inputs and source URLs are cleared
after completion/failure. New submissions stop when stored payloads reach 2 GiB;
in-flight results and database journals can exceed this threshold. Allow at least
10 GB disk headroom, monitor disk space, and save finished results elsewhere if
needed longer. SQLite reuses deleted space rather than immediately shrinking its
file; deleted content can remain in database pages, journals or backups.

The worker processes one image at a time. A hung inference stalls the queue until
the process is restarted; /health only confirms HTTP liveness, not job progress.
Monitor job ages and failed jobs. This is a single-host queue, not a distributed
multi-server queue or a promise to accept unlimited uploads.

## Deployment changes

Compose includes a named volume at `/app/.cache/jobs` and a 24 GB memory limit.
Recent Mac measurements showed Python around 15 GB: benchmark Linux peak memory
before selecting hosting limits. Four CPU threads remain the default in launchers.
For manual launch use `--workers 1 --limit-concurrency 128`.

Railway needs a persistent volume mounted at JOB_DIR=/app/.cache/jobs with write
permission for container UID 10001. Use one replica. Independent replicas cannot
see each other's SQLite jobs. Without persistent storage a redeploy loses jobs.
Vast instances must similarly preserve JOB_DIR, and cannot process work while off.
Graceful shutdown waits for the active job; queued jobs resume at next startup.

## Verification

`python local-test/check_queue.py` checks 20 concurrent submissions with a tiny
fake processor, serialized execution, restart recovery, limits, authentication,
status/result handlers, failures and expiration. It does not benchmark the model.

After restarting the real server, run the existing test with `--count 20
--concurrency 20`. It now polls accepted jobs and downloads/checks completed PNGs.
It waits up to 15 minutes per job. Do not start a second model process on this Mac.
