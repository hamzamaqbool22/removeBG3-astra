# GPSDiffusion-SDXL shadow experiment (RTX 4090)

This is an offline test, **not a production deployment**. Normal shadows, API routes,
Dockerfiles and existing Python environments are unchanged. The local repository is
`removeBG3`; the GitHub/server checkout is `removeBG3-astra`.

## 1. Install on the GPU instance

Use Python 3.11 or 3.12. Check `python3 --version` first. If your default Python is
another version but `python3.11` is installed, prefix setup with `GPS_PYTHON=python3.11`.
Do not install the upstream repository's old requirements into the existing environment.

```bash
cd /workspace/removeBG3-astra
git pull --ff-only
nvidia-smi
python3 --version
bash shadow-lab/gps/setup.sh
```

Setup creates `.venv-gps`, installs PyTorch 2.6 / CUDA 12.4 wheels and the inference
dependencies, downloads pinned source + model weights into `.cache/gps-sdxl`, checks
the shadow archive SHA256 and extracts only the required files. It downloads **no
training dataset**. Internet access to GitHub, PyPI, PyTorch and Hugging Face is needed.
Downloads can take a while: the GPS archive alone is 6.7 GB, plus SDXL and packages.
Allow at least 22 GiB free **after** installing packages. The archive is removed after
successful extraction. Re-running setup reuses completed files/downloads.

The GPU must be available to this process. This does not start or stop any existing
server. Do not run concurrently with Flux on the same 24GB GPU for the first test.

## 2. Upload one original vehicle photo

Upload a source photo to `/workspace/car.jpg` with the instance file browser. Use an
original photo, not an already-composited image with shadows. Choose any background
from `backgrounds/parking-lots/`.

```bash
.venv-gps/bin/python shadow-lab/gps/prepare.py \
  --image /workspace/car.jpg \
  --background backgrounds/parking-lots/6.png \
  --out outputs/gps-input-01
```

This runs the existing vehicle segmentation/placement and normal-shadow method on
CPU, and saves `composite.png` (no added shadow), aligned `mask.png`, `background.png`,
and `normal.png`. BiRefNet downloads automatically on first use. CPU preparation
finishes before GPS starts, so it does not compete for GPU memory. An optional
`--mask /workspace/car-mask.png` accepts an already-created source-size vehicle mask.

## 3. Run the AI test

```bash
.venv-gps/bin/python shadow-lab/gps/infer.py \
  --input outputs/gps-input-01 \
  --out outputs/gps-result-01 \
  --samples 1 --steps 50 --seed 42
```

Open `outputs/gps-result-01/comparison.jpg` in the instance file browser. The first
run is the GPU validation: local image-processing tests do not prove CUDA inference
or vehicle shadow quality. If it fails, retain the full traceback.

Outputs:
- `comparison.jpg`: shadow-free / normal / AI comparison.
- `protected-42.png`: full-resolution result with original opaque vehicle pixels preserved.
- `raw-42.png`: unprotected 512px AI image, for diagnosis (not a customer output).
- `model-input.png`, `model-mask.png`, `geometry-region.png`: exact model inputs/prior.
- `shadow-mask-42.png`, `opacity-42.png`: learned support and transferred darkness.
- `report.json`: GPU name, model revisions, loading/generation/postprocessing times,
  peak allocated GPU memory and opaque-vehicle pixel-change check. Allocated memory
  is not total device usage; check `nvidia-smi` too.

For three candidates, use a **new output folder**:

```bash
.venv-gps/bin/python shadow-lab/gps/infer.py \
  --input outputs/gps-input-01 \
  --out outputs/gps-result-01-three \
  --samples 3 --steps 50 --seed 42
```

Outputs are intentionally not overwritten. Use a different input/output folder for
each car or background. Repeat on Jeep front/front-left/front-right/rear and other
cars. Inspect tyre contact, road detail, shadow direction, colour and detached patches.
There is no automatic quality selection or silent fallback in this experiment.

## What differs from the upstream evaluation script

- Explicit one-image input, no dataset or ground-truth images required.
- Letterbox to 512 without stretching; keep full-size car and background.
- One candidate by default; explicit seed/step controls.
- Native PyTorch attention; no xformers/MMDetection dependency.
- All networks in evaluation mode; text encoders released before ControlNet loading.
- VAE stays float32 to avoid half-precision overflow. Diffusion uses float16.
- Decode bounding-box geometry into the actual mask, avoiding the upstream temporary
  `astype` array fill and tensor-swap aliasing. No random box perturbation.
- Calculate centroid conditioning once; don't accumulate it over diffusion steps.
- Preserve the released adapter's four-token attention setting for compatibility.
- Keep diffusion, geometry and postprocessing networks loaded across images in the GPU batch.
- Transfer only neutral shadow darkening onto the original background. This transfer
  is **our experimental integration**, not an official GPS quality claim. It can
  under-represent coloured shadows and can still produce bad shadow shapes.
- Opaque vehicle pixels are unchanged from the prepared composite. Soft edges retain
  the original car contribution while the background contribution is darkened.

## Sources and model provenance

- [BCMI source](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL),
  pinned to `1b6879642d8eecc079dd32c94eadf32545788c34` (MIT; see downloaded LICENSE).
- [SDXL checkpoint mirror](https://huggingface.co/GrigoriiU/GPSDiffusion-SDXL),
  pinned to `41cfe67d120021339fef39092dae235e792eb722`.
  This is a **community re-upload**, linked in [upstream issue #31](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL/issues/31),
  not an author-controlled Hugging Face release. SHA256 verifies the pinned mirror
  archive, not equivalence to the original Baidu archive. The mirror contains legacy
  PyTorch files and a centroid pickle; do not substitute arbitrary untrusted files.
- [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), pinned
  to `462165984030d82259a11f4367a4eed129e94a7b`, separate Open RAIL++ terms.

No commercial licence clearance is implied by this test setup.

## Local checks (no GPU/models needed)

```bash
.venv-gps/bin/python shadow-lab/gps/test_imaging.py
```

## Batch test a folder

```bash
cd /workspace/removeBG3-astra
git pull --ff-only
.venv-gps/bin/python -u shadow-lab/gps/batch.py \
  --images /workspace/images-GPS \
  --background backgrounds/parking-lots/6.png \
  --samples 1 --steps 50 --seed 42
```

All supported images directly inside that folder are attempted in sorted order
(JPG/JPEG/PNG/WebP/BMP/TIF/TIFF, including filenames with spaces). Background-only
or already-processed photos are not filtered out automatically; use original
vehicle photos for a meaningful quality comparison.

Each run creates `outputs/gps-batches/<timestamp>/`. Each numbered image folder
contains `input/`, `result/`, `prepare.log` and `inference.log`. Open the result's
`comparison.jpg` to compare no shadow, normal shadows and AI shadows.

The terminal and `batch.log` show start/end timestamps (server timezone with UTC
offset), completion time and duration for each image, and total batch duration.
`timings.csv` and `summary.json` are refreshed after every image. Total per-image
time includes preparation, model loading, generation and saving. Models reload
per image in separate processes; these are end-to-end timings, not warm GPU
throughput measurements. Detailed GPU generation times remain in each result's
`report.json`. Stage output is captured in its log instead of flooding the console.

A failed image does not stop the remaining images. Ctrl+C stops the batch and
writes a summary; completed results stay saved. Exit code is 1 if any image fails,
130 if interrupted, and 0 if every image succeeds. Re-running starts a fresh batch
rather than overwriting or resuming previous results. No server processes are changed.

## Faster GPU batch (recommended)

Stop the old batch with Ctrl+C before upgrading. Existing installations only need
this runtime upgrade; cached model weights are reused:

```bash
cd /workspace/removeBG3-astra
git pull --ff-only
bash shadow-lab/gps/enable_gpu.sh
.venv-gps/bin/python -u shadow-lab/gps/gpu_batch.py \
  --images /workspace/images-GPS \
  --background backgrounds/parking-lots/6.png \
  --samples 1 --steps 50 --seed 42
```

Stage 1 prepares all images using one CUDA BiRefNet session. Its subprocess exits
before Stage 2, freeing segmentation GPU memory. Stage 2 loads GPS models once and
reuses them for every prepared image. It retains the same 50 steps, seed, float32
VAE, conditioning and vehicle-preservation rules. CPU image processing and disk
I/O still occur; GPU utilization need not stay at 100%. Actual 4090 speed and peak
memory must be measured on the instance; local tests cannot establish them.

Results use `outputs/gps-batches/<timestamp>-gpu/`. Start/completion timestamps and
active preparation/inference durations are printed for each image. Shared model
loading is recorded separately in `model_startup_seconds` in `summary.json` and is
included in total batch wall time. Per-image totals exclude both shared loading
and waiting between the two stages; do not compare these directly with the old
runner's model-reloading totals. `prepare-startup.log` and `inference-startup.log`
contain model initialization output; per-image logs and comparisons remain in
each numbered folder. The summary and CSV are refreshed throughout both stages.

CUDA preparation is required: a missing CUDA execution provider stops the batch
with an error. Ordinary image failures continue; CUDA/context/out-of-memory errors
stop model reuse and preserve finished results. Incomplete rows are marked
`unprocessed` or `interrupted`. Ctrl+C stops the run; a fresh command creates a new
batch, not a resume. The old isolated `batch.py` remains available.

For GPU preparation of one image, add `--device cuda` to `prepare.py` after
installing the GPU runtime. No production service configuration changes are needed.
