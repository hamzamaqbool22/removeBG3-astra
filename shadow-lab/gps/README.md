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
- Run the released 256px postprocessing network after releasing diffusion networks.
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
