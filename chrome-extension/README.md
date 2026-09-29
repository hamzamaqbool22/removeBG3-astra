# Vehicle Studio — local Chrome extension

Version **0.4.0** processes photos entirely on the user's computer. No Python server, helper
app, API key, or image upload service is used. The existing Python project is
unchanged.

## Test it now

1. Open **Google Chrome** and go to `chrome://extensions`.
2. Enable **Developer mode**.
3. If already installed, click **Reload** on Vehicle Studio. Otherwise choose
   **Load unpacked** and select this `chrome-extension` folder.
4. Close the old side panel, then click the extension icon to reopen it.
5. Confirm the log says **v0.4.0, BiRefNet Lite 512**.
6. Upload one or more vehicle photos. Choose transparent, white, a numbered
   background, or your own background. Enable lighting if wanted.
7. Choose **Local GPU** or **Local CPU**, then **Generate** and **Download PNG**.

Do not open `panel.html` using a `file://` URL. Keep the side panel open during
processing. Photos run sequentially to limit memory. **Stop / release memory**
terminates the processing worker; Generate can start it again.

The timestamped live log shows model loading, inference, shadow stages, timings,
completion and errors. **Copy log** copies it for debugging. Developer Console
messages use the `[Vehicle Local]` prefix.

## First-use downloads

The extension downloads only the selected CPU or GPU model from its pinned
Hugging Face URL on first use. Progress appears in the panel. Downloads are
checked against a bundled SHA-256 checksum before saving in extension Cache
Storage. Subsequent runs reuse that model, including after reopening the panel
and while offline. Switching CPU/GPU downloads the other model once.

All 65 numbered backgrounds download from Cloudinary when selected and are
cached locally with checksum verification. Only `parking-lots/default.png` and
`backgrounds/default.png` are bundled (the original number 1 from each folder).
An uncached background that fails to download within 15 seconds falls back to
its folder default. The panel visibly reports substitutions. Saved backgrounds
work offline; an uncached model still needs internet before processing.
No photo is included in any network request. The model host receives normal
download connection information (such as IP address), not photos. JavaScript
and WASM are bundled; no executable code is downloaded remotely.

Keep the panel open for the initial download. Stop terminates the worker and
cancels ongoing work. Incomplete downloads are not saved and restart on retry.
**Clear saved model/background downloads** releases the worker and clears saved assets.
Uninstalling the extension or clearing its storage may require downloading again.
The `unlimitedStorage` permission supports retaining the large local models;
available disk space still matters.

## Quality and device differences

The browser uses **BiRefNet Lite at 512 × 512**, replacing the original full-size
1024 model that exceeded the browser's runtime limits. GPU uses approximately
94 MiB FP16 weights; CPU uses approximately 183 MiB FP32 weights. GPU requires
WebGPU and `shader-f16`; choose CPU if the GPU is unsupported.

The geometry, contact shadows, source-shadow recovery, placement and lighting
are JavaScript ports of the Python implementation. Segmentation is different,
so fine edges, mirrors and resulting contact estimates may differ. This is
**not pixel-identical to the Python server**, and CPU/GPU precision can produce
small differences too. The original appearance is not generated or repainted.

## Checks completed on this Mac

- Version 0.4.0: all 65 Cloudinary images verified byte-for-byte. Packaged GPU
  processing passed with a downloaded background, offline cached background,
  and a default fallback. Both folders passed cache and offline-fallback checks;
  the fallback notice and clearing saved downloads were verified. The ZIP has
  exactly two PNG files, both named `default.png`, and no model weights.

- Version 0.3.0 packaged extension: CPU and GPU each downloaded the pinned
  public model, verified and cached it, processed two photos and downloaded PNGs.
  Reloading the panel with the browser offline processed a third photo using
  cached weights and bundled parking background 1. Clearing saved models passed.
- Seven automated checks cover checksum/size verification, corrupt-cache
  recovery, rejected/partial downloads not being cached, and error reporting.

Earlier image-quality checks (same models and processing algorithms):

- All eight vehicle views completed on GPU and CPU, with parking background 1
  and lighting enabled; all 16 PNGs downloaded successfully.
- Mean processing time: approximately **3.4 seconds GPU**, **7 seconds CPU** per
  photo in that test, excluding model loading. Other devices/images will differ.
- Actual unpacked-extension checks passed on both devices using two additional
  photos, white output, PNG downloads, and Stop/release.
- With the original Python masks supplied to both implementations, final RGB
  mean absolute differences were below **0.002 out of 255** across all eight views.
  This isolates the ported image-processing logic from the smaller model.
- Lite CPU masks had **97.6–99.1% intersection-over-union** against the original
  masks at alpha >= 128. Mask overlap is not a guarantee of visual quality.

Review `validation/eight-comparison.jpg` for the Python/browser comparison,
`validation/eight-gpu/` and `validation/eight-cpu/` for images, and
`validation/comparison-metrics.json` for measurements. These are local test
artifacts, ignored by Git. This has not been validated across Windows PCs,
low-memory machines, or all possible photographs.

## Developer setup

### Build a shareable ZIP with first-use model downloads

Run `npm run bundle` after runtime assets and backgrounds are prepared. It writes
`release/vehicle-studio-0.4.0-download.zip` and a matching unpacked directory.
Send only that ZIP, not this development folder. The package includes
two default backgrounds, runtime code, licenses, checksums and
`INSTALL.txt`. It excludes development dependencies, tools, sample photos and
validation results. The original Python application is untouched.

Recipients extract the ZIP and use Chrome's **Load unpacked** on its folder.
This package is for testing; creating it does not publish or approve a Web Store
listing. The ZIP excludes both models. It is approximately 15.9 MiB (16.6 MB). Models and the 65 numbered backgrounds are
not included. The runtime-only ZIP was approximately 10 MiB.

Assets are already prepared locally. To rebuild after cloning:

```sh
npm ci
npm run prepare:assets
npm test
npm run bundle
```

Run these from this directory. `prepare:assets` copies local runtime libraries
and two defaults from the parent project's backgrounds into this folder only. `assets.json` pins
model/background URLs, sizes and hashes. The Cloudinary URLs use the
`mssoxcbv` cloud and the `backgrounds` / `parking-lots` public-ID prefixes.
All 65 versionless URLs were downloaded and verified byte-for-byte against
the original PNGs. If you replace a Cloudinary image, update its hash and size
in this manifest too; otherwise the app safely uses its default. `npm run download:models` remains an
optional developer utility; its output is never included in this release.

`tools/extension-check.mjs --cache-check` tests a real unpacked extension,
first-use model download, two local photos and PNG downloads, then reloads the
panel offline and clears cached assets. Add `--cloud-check` to exercise
Cloudinary downloads, offline cache reuse, and default fallback in both folders.
Add `--gpu` to test GPU. Set `EXTENSION_ROOT` to the packaged directory to test
the distribution instead of this development folder.

`node tools/browser-check.mjs --infer --cpu` runs a temporary Chrome test.
The scripts under `tools/reference*` import the production Python code read-only
for offline comparisons; they are not part of extension execution.

Models, vendor libraries, backgrounds, test browsers and validation outputs are
ignored by Git. Include the required runtime assets when sharing an unpacked
copy. This folder is a testing build, not a published Chrome Web Store package.

## Model provenance

Browser export: https://huggingface.co/studioludens/birefnet-lite-512

Pinned revision: `4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7`.
Upstream model: ZhengPeng7/BiRefNet_lite, MIT license. Runtime and model license
notices are in `licenses/`.
