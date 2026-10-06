#!/usr/bin/env bash
# Linux server only. Run as the normal SSH user; script uses sudo where needed.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d images && -f .env ]] || { echo 'Missing images/ or .env'; exit 1; }
primary_mode=$(sudo docker compose exec -T vehicle-api python -c 'import os; print(os.getenv("SEGMENTATION_MODE", "auto"))')
[[ "$primary_mode" == single ]] || { echo 'Primary API must already be in single mode for this comparison.'; exit 1; }
third=(docker compose -p removebg-third -f compose.third.yaml)
# Leave the original API unchanged. Match the second instance to single mode.
sudo env PARALLEL_SEGMENTATION_MODE=single PARALLEL_CPU_THREADS=6 docker compose \
  -p removebg-parallel -f compose.parallel.yaml up -d --wait vehicle-api-extra
sudo "${third[@]}" stop
results="$PWD/outputs/instances-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$results"
image_id=$(sudo docker compose -p removebg-parallel -f compose.parallel.yaml images -q vehicle-api-extra)
bench() {
  local port="$1" count="$2" concurrency="$3" label="$4"
  sudo docker run --rm --network host --env-file .env --user "$(id -u):$(id -g)" \
    -v "$PWD/local-test/generate_load.py:/tmp/generate_load.py:ro" \
    -v "$PWD/images:/tmp/test-images:ro" -v "$results:/results" \
    "$image_id" python /tmp/generate_load.py --input /tmp/test-images \
    --url "http://127.0.0.1:$port/generate" --count "$count" \
    --concurrency "$concurrency" --timeout 180 --output "/results/$label"
}
echo 'Warm up both existing instances'
bench 8000 1 1 warmup-primary
bench 8001 1 1 warmup-second
for run in 1 2; do
  echo "TWO instances: measured run $run"
  bench 8002 16 3 "two-$run"
done
# Observed models retain ~14 GiB each. Require headroom before starting the third.
available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kb < 22 * 1024 * 1024 )); then
  echo 'Less than 22 GiB available RAM. Third instance not started.'
  free -h
  exit 1
fi
sudo "${third[@]}" build vehicle-api-third
sudo "${third[@]}" run --rm --no-deps generate-balancer-three nginx -t
# Ensure ownership even if an earlier attempt left a root-owned queue volume.
sudo "${third[@]}" run --rm --no-deps --user root vehicle-api-third chown -R vehicle:vehicle /app/.cache/jobs
cleanup() { sudo "${third[@]}" stop; }
trap cleanup EXIT
sudo "${third[@]}" up -d --wait
bench 8003 1 1 warmup-third
for run in 1 2; do
  echo "THREE instances: measured run $run"
  bench 8004 16 3 "three-$run"
  sudo docker stats --no-stream
done
sudo "${third[@]}" logs --no-color --since=30m > "$results/third-server.log"
python3 - "$results" <<'REPORT'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for label in ('two-1','two-2','three-1','three-2'):
    r = json.loads((root/label/'report.json').read_text())
    times = sorted(x['seconds_including_queue'] for x in r['results'])
    print(label, 'success=', r['success'], 'total_seconds=', r['elapsed_seconds'],
          'mean_request_seconds=', round(sum(times)/len(times),2),
          'max_request_seconds=', max(times), 'workers=', r['worker_counts'])
REPORT
printf '\nSaved PNGs and reports: %s\n' "$results"
echo 'Stopping the third trial instance; existing two-instance setup stays running.'
