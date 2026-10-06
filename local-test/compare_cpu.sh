#!/usr/bin/env bash
# Run in /opt/removebg as the normal SSH user. No other image requests during this test.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d images && -f .env ]] || { echo 'Missing images/ or .env'; exit 1; }
compose=(docker compose -p removebg-parallel -f compose.parallel.yaml)
container=$(sudo "${compose[@]}" ps -q vehicle-api-extra)
[[ -n "$container" ]] || { echo 'Start the existing second instance first'; exit 1; }
# Save only tuning settings, never the API key or full container environment.
old_threads=$(sudo "${compose[@]}" exec -T vehicle-api-extra python -c 'import os; print(os.getenv("CPU_THREADS", "6"))')
old_mode=$(sudo "${compose[@]}" exec -T vehicle-api-extra python -c 'import os; print(os.getenv("SEGMENTATION_MODE", "auto"))')
old_nano=$(sudo docker inspect "$container" --format '{{.HostConfig.NanoCpus}}')
old_cpus=$(python3 -c 'import sys; print(int(sys.argv[1])/1e9)' "$old_nano")
[[ "$old_nano" != 0 ]] || { echo 'Expected a CPU quota on the second instance; no changes made'; exit 1; }
results="$PWD/outputs/cpu-trial-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$results"
lscpu > "$results/cpu.txt"
free -h > "$results/memory-before.txt"
restore() {
  echo "Restoring second instance: $old_threads threads, $old_cpus CPU allowance, mode $old_mode"
  sudo env PARALLEL_CPU_THREADS="$old_threads" PARALLEL_CPUS="$old_cpus" PARALLEL_SEGMENTATION_MODE="$old_mode" \
    "${compose[@]}" up -d --wait vehicle-api-extra
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for threads in 6 16; do
  echo "TEST: one instance, $threads CPU allowance / $threads model threads, single pass"
  sudo env PARALLEL_CPU_THREADS="$threads" PARALLEL_CPUS="$threads" PARALLEL_SEGMENTATION_MODE=single \
    "${compose[@]}" up -d --wait vehicle-api-extra
  image_id=$(sudo "${compose[@]}" images -q vehicle-api-extra)
  container=$(sudo "${compose[@]}" ps -q vehicle-api-extra)
  sudo docker inspect "$container" --format 'Applied CPU quota={{.HostConfig.NanoCpus}}'
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for phase in warmup measured-1 measured-2; do
    count=8
    [[ "$phase" != warmup ]] || count=1
    sudo docker run --rm --network host --env-file .env --user "$(id -u):$(id -g)" \
      -v "$PWD/local-test/generate_load.py:/tmp/generate_load.py:ro" \
      -v "$PWD/images:/tmp/test-images:ro" -v "$results:/results" \
      "$image_id" python /tmp/generate_load.py --input /tmp/test-images \
      --url http://127.0.0.1:8001/generate --count "$count" --concurrency 1 \
      --timeout 180 --output "/results/threads-$threads-$phase"
  done
  sudo "${compose[@]}" logs --since "$started" --timestamps --no-color vehicle-api-extra > "$results/threads-$threads.log"
done
python3 - "$results" <<'REPORT'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for threads in (6,16):
    for run in (1,2):
        r=json.loads((root/f'threads-{threads}-measured-{run}'/'report.json').read_text())
        times=[x['seconds_including_queue'] for x in r['results']]
        print(f'{threads} threads run {run}: success={r["success"]}/{r["requests"]}, '
              f'total={r["elapsed_seconds"]}s, average={sum(times)/len(times):.2f}s, max={max(times):.2f}s')
REPORT
printf '\nResults saved: %s\n' "$results"
