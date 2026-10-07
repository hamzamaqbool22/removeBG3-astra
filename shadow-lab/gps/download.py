"""Pinned public source + checkpoints. Never downloads the training dataset."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/gps-sdxl'
SOURCE_REV = '1b6879642d8eecc079dd32c94eadf32545788c34'
WEIGHTS_REV = '41cfe67d120021339fef39092dae235e792eb722'
BASE_REV = '462165984030d82259a11f4367a4eed129e94a7b'
ARCHIVE_SHA = '37e4876705e254b1f465c9deb577b44838752c358e9eaf044e746674caaa5810'
REQUIRED = ['controlnet/config.json', 'controlnet/diffusion_pytorch_model.safetensors',
            'ip_adapter.ckpt', 'Shadow_cls.pth', 'Shadow_reg.pth',
            'Shadow_cls_label.pkl', 'Shadow_ppp.ckpt']


def main():
    from huggingface_hub import hf_hub_download, snapshot_download
    CACHE.mkdir(parents=True, exist_ok=True)
    # 6.7GB archive + extracted weights + SDXL + download overhead.
    if not (CACHE / 'ready.json').exists() and shutil.disk_usage(CACHE).free < 22 * 1024**3:
        raise RuntimeError('Need at least 22 GiB free for model downloads/extraction, after installing packages.')
    source = CACHE / 'upstream'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', SOURCE_REV], check=True)
    # Keep inference on native PyTorch attention; no compiled xformers required.
    path = source / 'attention_processor.py'
    text = path.read_text()
    text = text.replace('\nimport xformers\n', '\ntry:\n    import xformers\nexcept ImportError:\n    xformers = None\n')
    path.write_text(text)
    path = source / 'base_network.py'
    path.write_text(path.read_text().replace('models.resnet50(pretrained=True)', 'models.resnet50(weights=None)'))
    weights = CACHE / 'weights'
    if not all((weights / name).is_file() for name in REQUIRED):
        print('Downloading SDXL checkpoints from the community mirror linked in upstream issue #31.', flush=True)
        archive = Path(hf_hub_download('GrigoriiU/GPSDiffusion-SDXL', 'pretrained_models_sdxl.zip',
                                     revision=WEIGHTS_REV, cache_dir=CACHE / 'hub'))
        digest = hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != ARCHIVE_SHA:
            raise RuntimeError('Checkpoint archive checksum mismatch')
        weights.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in REQUIRED:
                matches = [n for n in z.namelist() if n == name or n.endswith('/' + name)]
                if len(matches) != 1:
                    raise RuntimeError(f'Expected exactly one {name} in SDXL archive; found {matches}')
                target = weights / name
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + '.part')
                with z.open(matches[0]) as src, temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                temp.replace(target)
        # Drop archive cache only after successful extraction to save the 50GB disk.
        shutil.rmtree(CACHE / 'hub')
    snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', revision=BASE_REV,
                      local_dir=CACHE / 'base', allow_patterns=[
                          'model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*',
                          'text_encoder/config.json', 'text_encoder/model.fp16.safetensors',
                          'text_encoder_2/config.json', 'text_encoder_2/model.fp16.safetensors',
                          'unet/config.json', 'unet/diffusion_pytorch_model.fp16.safetensors',
                          'vae/config.json', 'vae/diffusion_pytorch_model.fp16.safetensors'])
    (CACHE / 'ready.json').write_text(json.dumps({'source': SOURCE_REV, 'weights': WEIGHTS_REV,
                                                'base': BASE_REV, 'archive_sha256': ARCHIVE_SHA}, indent=2))
    print('GPS-SDXL models ready:', CACHE)


if __name__ == '__main__':
    main()
