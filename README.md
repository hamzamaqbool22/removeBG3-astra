# Vehicle background removal and grounding — Phase 1

A CPU-only Python pipeline that removes a vehicle background, estimates ground
contacts, preserves the vehicle's aspect ratio, and renders a white-background
image with separate tire, underbody, and diffuse ground shadows.

The current deliverable is the image-processing benchmark. FastAPI, the testing
UI, and scene backgrounds have not been added.

## Review the results

- `outputs/phase1/finals_all.jpg`: all eight final images on a common canvas.
- `outputs/phase1/comparison_all.jpg`: sources, normalized cutouts, generated
  shadows, and matching remove.bg references.
- `outputs/phase1/<image>/final_white.png`: full-size 1024 × 768 result.
- `outputs/phase1/<image>/comparison.jpg`: individual comparison.
- `outputs/phase1/<image>/metadata.json`: contacts, geometry evidence, affine
  placement, and per-stage timing.
- `outputs/segmentation/`: raw masks, cutouts, CPU timings, memory, and reference
  proxy scores for each model.
- `outputs/robustness/`: actual inference on smaller-in-frame and tightly cropped
  versions of all eight source photographs.

The reference is resized back to source coordinates and receives **the same
transform as our output** in comparisons. It does not determine the production
vehicle size, angle, contacts, or shadow. Reference front/rear shadows reach the
edge of their original canvases; that missing extent cannot be recovered when
aligning the references.

`outputs/phase1_iteration*` contains earlier exploratory passes, including
deliberate diagnostics of unsuccessful shadow shapes. Use `outputs/phase1`
for the current result.

## Run

Python 3.11–3.13 is supported by the pinned rembg version. This benchmark used
Python 3.13.4 on macOS arm64, with 24 GiB system RAM.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m vehicle_pipeline.cli images/frontLeft.png outputs/example.png --stages outputs/example-stages
python scripts/process_benchmark.py
python -m unittest discover -s tests -v
```

Model weights are downloaded on first use into `.cache/models/`. The selected
full BiRefNet weight file is approximately 928 MiB. Internet access is needed
for that first download; processing then runs locally with ONNX Runtime's
`CPUExecutionProvider`. No GPU, paid background-removal API, or generative image
service is used. The supplied checkout reuses already-installed U2-Net/IS-Net
weights through local cache symlinks; those symlinks are not package dependencies.

The default is six ONNX intra-operation threads, one inter-operation thread,
and one reusable model session. Keep image processing sequential on this
24 GiB machine; these quality-oriented models have substantial peak memory.
`requirements-benchmark.txt` records the exact direct-package versions used.

## Architecture

1. **Segmentation**: full `birefnet-general` at its native 1024 × 1024 network
   input, projected back to the original resolution. Small disconnected mask
   detections are removed while preserving the largest foreground and its soft
   edge. For a small subject (both mask dimensions below 60% of the scene), a
   second inference on a padded original-resolution crop recovers small details
   such as mirrors. This costs an extra inference only for small subjects.
   Original opaque vehicle pixels remain unchanged apart from the final
   uniform resampling.
2. **Contact estimation**: lower silhouette evidence, dark rubber annuli,
   rim detail, and elliptical wheel hypotheses. A coherent pair is selected
   jointly. Narrow projected vehicles use lateral tire-strip estimates because
   grilles must not be treated as exposed wheel faces. These are image-derived
   cues, not filename or manually provided angle labels.
3. **Framing**: one uniform scale, horizontal centering, and translation to
   place the nearest estimated tire contact at `0.80 × canvas height`.
   The vehicle fits within `0.88 × canvas width` and `0.46 × canvas height`.
   This uses projected geometry rather than trying to force a narrow front
   view to occupy the same width as a side view. It normalizes apparent framing,
   not calibrated real-world vehicle dimensions.
4. **Ground shadows**: a rounded projected chassis footprint, a lower-chassis
   occlusion band constrained by the contact plane, tight tire contacts, and a
   broader diffuse component. The components combine through light transmission.
   Blur scales with estimated wheel size. No duplicated whole-car silhouette
   or generic drop-shadow operation is used.
5. **Composite**: premultiplied-alpha resampling and white-ground compositing.
   Vehicle color and perspective are preserved; no vehicle pixels are generated.

The production package never opens `remove.bg-outputs/`. Reference handling is
confined to scripts for evaluation and comparison.

## Saved stages for every benchmark view

- Raw segmentation mask and refined mask at source resolution.
- Source geometry/contact overlay and normalized geometry overlay.
- Normalized placement on white, normalized alpha, and transparent cutout.
- Tire-contact, underbody, diffuse-ground, and combined shadow alpha images.
- White previews of each shadow component.
- Final white-background image, aligned reference, comparison, and metadata.

Shadow alpha PNGs use white for opaque shadow. Their `_white.png` counterparts
show what that shadow looks like on a white surface.

## Reproduce the benchmarks

```sh
# Each model gets a fresh process, one warmup, and two measured runs per image.
python scripts/benchmark_segmentation.py
python scripts/analyze_references.py
python scripts/score_segmentation.py

# Faster iteration after masks exist; this explicitly excludes model inference.
python scripts/process_benchmark.py --cached-masks --output outputs/development

# Actual inference on controlled size/framing variations.
python scripts/check_framing_robustness.py
```

Warm inference timing includes model preprocessing, inference, and mask
resampling. Model load/warmup are reported separately. Process peak RSS includes
the Python runtime, model session, and working memory. Final-pipeline timing
excludes diagnostic drawing and disk I/O; its first result includes model load.
Timing varies with other desktop activity and memory pressure.

## Model decision

The full BiRefNet model is the quality default after inspecting all eight views.
It preserves the front mirrors better than the smaller candidates and scored
highest on the approximate reference-mask comparison:

- U2-Net: 0.186 s mean warmed inference; 2.14 GiB peak RSS; 0.9834 mask IoU;
  0.9169 boundary F1.
- IS-Net: 0.493 s; 2.31 GiB; 0.9863 IoU; 0.9327 boundary F1.
- BiRefNet Lite: 4.153 s; 7.51 GiB; 0.9883 IoU; 0.9546 boundary F1.
- Full BiRefNet: 6.938 s; 8.21 GiB; 0.9912 IoU; 0.9734 boundary F1.

Disabling ONNX memory arenas/patterns was also measured. It produced the same
masks but was slower (7.657 s) and peaked higher (9.88 GiB) on this system, so
the default retains the balanced runtime settings. Raw results are preserved in
`outputs/segmentation/birefnet-general-lean/`.

The final raw-image pipeline averaged **7.59 seconds per image after the first
image**, with a **7.73 GiB process peak** during that run. Geometry, framing,
shadow rendering, and compositing averaged approximately **0.14 seconds**;
segmentation dominates the runtime. The first image took 9.99 seconds including
session loading and the first inference.

The eight originals did not require adaptive crop refinement. A more distant
vehicle can require two inference passes and therefore take approximately twice
as long; `segmentation_info` records this in per-image metadata. An initial
smaller-front-view test lost both mirrors in the first mask. The automatic crop
pass recovered them, using the input pixels rather than reconstructing them.

## Validation and limits

Five regression tests cover all-view contacts, exact preservation of opaque
vehicle color during shadow compositing, canvas clipping, front/rear grille
rejection, proportion-preserving framing, and normalization after shrinking and
translating a source. The fast tests use cached masks. The separate robustness
script reruns segmentation on changed image framing.

All 16 controlled size/crop variants completed with a mean mask-consistency IoU
of 0.9963 (minimum 0.9946). The largest normalized bounding-box change was
7.18 pixels on the 1024 × 768 canvas. These comparisons use transformed baseline
model masks, so they test stability rather than absolute segmentation accuracy.
The visual comparison is in `outputs/robustness/comparison_all.jpg`.

The final raw-image rerun produced byte-identical PNGs for all eight original
views after adding adaptive crop refinement. `outputs/benchmark_summary.json`
combines the model decision, measurements, robustness results, and artifact
verification. Regenerate the summary with `python scripts/summarize_benchmark.py`.

The reference masks are approximate decompositions of transparent PNGs, not
manual ground truth. Semitransparent black vehicle edges can be confused with
shadow; the references also have lower resolution. IoU and boundary F1 describe
segmentation agreement, **not proof of shadow realism or remove.bg superiority**.

All supplied photographs depict one white sedan. Wheel/ground geometry is a
monocular heuristic; its confidence is an evidence score, not a calibrated
probability. Hidden tires and the opposite track are estimates. Dark wheels,
occlusion, multiple vehicles, unusual camera elevations, non-level ground,
cropped vehicles, SUVs, and trucks need additional evaluation. Cropped-away
vehicle parts cannot be restored while preserving the original appearance.
The shadow aims for diffuse studio grounding on white, not reconstruction of
the original outdoor sun direction. Phase 2 would need scene-specific ground
perspective and illumination.

Model/tool sources: [rembg](https://github.com/danielgatis/rembg) and
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
