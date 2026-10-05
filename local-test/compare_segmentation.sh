#!/usr/bin/env bash
# Run on the Linux server as the SSH user (not with sudo bash).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d images && -f .env ]] || { echo "Missing images/ or .env in $(pwd)"; exit 1; }
compose=(docker compose -p removebg-parallel -f compose.parallel.yaml)
sudo "${compose[@]}" build vehicle-api-extra
results="$PWD/outputs/segmentation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$results"
for mode in auto single; do
  echo "Testing segmentation mode: $mode (six model threads, one request at a time)"
  sudo env PARALLEL_SEGMENTATION_MODE="$mode" PARALLEL_CPU_THREADS=6 \
    "${compose[@]}" up -d --wait vehicle-api-extra
  image_id=$(sudo "${compose[@]}" images -q vehicle-api-extra)
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for phase in warmup measured; do
    count=1
    [[ "$phase" != measured ]] || count=8
    sudo docker run --rm --network host --env-file .env \
      --user "$(id -u):$(id -g)" \
      -v "$PWD/local-test/generate_load.py:/tmp/generate_load.py:ro" \
      -v "$PWD/images:/tmp/test-images:ro" -v "$results:/results" \
      "$image_id" python /tmp/generate_load.py \
      --input /tmp/test-images --url http://127.0.0.1:8001/generate \
      --count "$count" --concurrency 1 --output "/results/$mode-$phase"
  done
  sudo "${compose[@]}" logs --since "$started" --timestamps --no-color vehicle-api-extra > "$results/$mode-server.log"
done
printf '\nResults and PNG comparisons saved in: %s\n' "$results"
echo 'The extra instance is now in single-pass mode. Original instance unchanged.'
echo 'To restore automatic mode:'
echo 'sudo env PARALLEL_SEGMENTATION_MODE=auto PARALLEL_CPU_THREADS=6 docker compose -p removebg-parallel -f compose.parallel.yaml up -d --wait vehicle-api-extra'
