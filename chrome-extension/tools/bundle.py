"""Bundle two default backgrounds and runtime; download models and numbered backgrounds."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'manifest.json').read_text())
config = json.loads((ROOT / 'assets.json').read_text())
catalog = json.loads((ROOT / 'backgrounds/catalog.json').read_text())
assert set(config['backgrounds']) == {f'{folder}/{name}' for folder, names in catalog.items() for name in names}, 'Run npm run prepare:assets to refresh background metadata'
for spec in config['fallbacks'].values():
    relative=spec['file']
    image = (ROOT / 'backgrounds' / relative).read_bytes()
    assert len(image) == spec['bytes'] and hashlib.sha256(image).hexdigest() == spec['sha256'], f'Background changed: {relative}; run npm run prepare:assets'
release = ROOT / 'release'
release.mkdir(exist_ok=True)
name = f"vehicle-studio-{manifest['version']}-download"
# ORT 1.30's WebGPU bundle uses asyncify for BOTH CPU and GPU. Other builds
# (plain, JSPI, JSEP) are not used by this entry point and must not be shipped.
paths = ['manifest.json', 'assets.json', 'background.js', 'panel.html', 'panel.css', 'panel.js', 'worker.js', 'backgrounds/catalog.json',
         'vendor/opencv.js', 'vendor/ort.webgpu.min.js', 'vendor/ort-wasm-simd-threaded.asyncify.mjs', 'vendor/ort-wasm-simd-threaded.asyncify.wasm']
paths += ['backgrounds/' + spec['file'] for spec in config['fallbacks'].values()]
for folder in ('src', 'licenses'):
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*') if p.is_file() and not p.name.startswith('.')]
with tempfile.TemporaryDirectory(dir=release) as temporary:
    stage = Path(temporary) / name
    stage.mkdir()
    for relative in paths:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (stage / 'assets.json').write_text(json.dumps(config, indent=2) + '\n')
    (stage / 'INSTALL.txt').write_text('''VEHICLE STUDIO — DOWNLOAD-ON-FIRST-USE EDITION

Extract this ZIP. In Chrome, open chrome://extensions, enable Developer mode,
choose Load unpacked, and select the extracted folder containing manifest.json.
Click the extension icon to open the panel. Upload a photo and Generate.

The selected model downloads on first use: GPU ~94 MiB or CPU ~183 MiB.
It is saved locally. Switching devices downloads the other model once.
Keep the panel open during downloads and processing. Stop cancels a download;
an interrupted download restarts next time. Clear saved downloads removes it.
Removing the extension or clearing its storage can also remove saved files.

Photos stay on your computer. Internet is needed for uncached assets only.
Numbered backgrounds download from Cloudinary and are saved locally.
Each folder includes default.png, used if the selected background download
fails. A visible notice tells you when a default was used. Cached selections
work offline. A model must already be downloaded for offline processing.
This is an unpacked testing build, not a Web Store approved listing.
''')
    sums = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    archive = Path(temporary) / f'{name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(stage))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP verification failed'
        assert not any(p.startswith(('models/', 'node_modules/', 'tools/', 'validation/')) for p in z.namelist())
    destination = release / name
    if destination.exists(): shutil.rmtree(destination)
    shutil.move(str(stage), destination)
    archive.replace(release / archive.name)
    print(f'Extension ZIP: {release / archive.name} ({(release / archive.name).stat().st_size / 1024**2:.2f} MiB)')

print('Included two default.png backgrounds. Models and numbered backgrounds download and cache on first use.')
