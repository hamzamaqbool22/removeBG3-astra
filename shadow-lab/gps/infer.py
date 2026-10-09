"""Isolated GPSDiffusion-SDXL experiment using pinned upstream networks.

Adapted inference flow from BCMI GPSDiffusion-SDXL (MIT; upstream downloaded
by download.py). The pretrained adapter's four-token setting is retained.
No server route or production shadow code is changed.
"""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw
from imaging import letterbox, restore_map, shadow_opacity, apply_shadow, geometry_region

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/gps-sdxl'


class GPSRunner:
    """Load CUDA models once; process images sequentially at the original quality settings."""

    def __init__(self):
        if not (CACHE / 'ready.json').exists():
            raise RuntimeError('Run shadow-lab/gps/setup.sh first')
        import torch
        from torch import nn
        from diffusers import StableDiffusionXLPipeline, ControlNetModel, DDPMScheduler
        if not torch.cuda.is_available():
            raise RuntimeError('This experiment requires an NVIDIA CUDA GPU')
        sys.path.insert(0, str(CACHE / 'upstream'))
        from base_network import MaskCls, RegNetwork
        from attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
        weights = CACHE / 'weights'
        dtype = torch.float16
        started = time.perf_counter()
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        print('Loading SDXL; the first run also initializes CUDA.', flush=True)

        from checkpoint import load_checkpoint as load

        with torch.inference_mode():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(CACHE / 'base'), variant='fp16', torch_dtype=dtype,
                local_files_only=True, add_watermarker=False).to('cuda')
            prompt, _, pooled, _ = pipe.encode_prompt(
                prompt='foreground object with shadow', device=torch.device('cuda'),
                do_classifier_free_guidance=False)
            # Release text encoders before loading ControlNet and geometry models.
            pipe.text_encoder = None
            pipe.text_encoder_2 = None
            pipe.tokenizer = None
            pipe.tokenizer_2 = None
            gc.collect()
            torch.cuda.empty_cache()
            unet, vae = pipe.unet, pipe.vae
            vae.to(dtype=torch.float32)
            vae.enable_slicing()
            scheduler = DDPMScheduler.from_pretrained(str(CACHE / 'base'), subfolder='scheduler', local_files_only=True)
            controlnet = ControlNetModel.from_pretrained(str(weights / 'controlnet'), torch_dtype=dtype).to('cuda').eval()
            if controlnet.config.conditioning_channels != 5:
                raise RuntimeError('Wrong checkpoint: GPS-SDXL requires five conditioning channels')

            processors = {}
            for name in unet.attn_processors:
                if name.startswith('mid_block'):
                    hidden = unet.config.block_out_channels[-1]
                elif name.startswith('up_blocks'):
                    hidden = list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
                else:
                    hidden = unet.config.block_out_channels[int(name.split('.')[1])]
                processors[name] = (AttnProcessor2_0() if name.endswith('attn1.processor') else
                                   IPAttnProcessor2_0(hidden, unet.config.cross_attention_dim, num_tokens=4))
            unet.set_attn_processor(processors)
            adapter = load(weights / 'ip_adapter.ckpt')
            nn.ModuleList(unet.attn_processors.values()).load_state_dict(adapter['ip_adapter'], strict=True)
            # These names/shapes match the released image_proj checkpoint.
            projection = nn.Module()
            projection.proj = nn.Linear(2048, unet.config.cross_attention_dim)
            projection.norm = nn.LayerNorm(unet.config.cross_attention_dim)
            projection.load_state_dict(adapter['image_proj'], strict=True)
            projection.to(device='cuda', dtype=dtype).eval()
            del adapter
            unet.to(device='cuda', dtype=dtype).eval()
            classifier, regressor = MaskCls(256), RegNetwork()
            classifier.load_state_dict(load(weights/'Shadow_cls.pth')['net'], strict=True)
            regressor.load_state_dict(load(weights/'Shadow_reg.pth')['net'], strict=True)
            classifier.cuda().eval()
            regressor.cuda().eval()
            from train_post_process_predictor import Post_Process_Net
            post = Post_Process_Net(image_size=256,in_channels=7,out_channels=4,model_channels=96,
                attention_resolutions=[],num_res_blocks=2,channel_mult=[1,2,2,4],num_head_channels=64,
                use_spatial_transformer=False,use_linear_in_transformer=True,transformer_depth=1,
                context_dim=320,legacy=False,use_checkpoint=False)
            state = load(weights/'Shadow_ppp.ckpt')
            state = state.get('state_dict',state)
            state = {k.removeprefix('post_process_net.'):v for k,v in state.items() if k.startswith('post_process_net.')}
            post.load_state_dict(state,strict=True)
            post.cuda().eval()
            del state
            # Immutable geometry prototypes are shared across all images.
            with (weights/'Shadow_cls_label.pkl').open('rb') as f:
                centroids = pickle.load(f)
            torch.cuda.synchronize()
            self.load_seconds = time.perf_counter()-started
            self.calls = 0
            self.prompt, self.pooled = prompt, pooled
            self.unet, self.vae, self.scheduler = unet, vae, scheduler
            self.controlnet, self.projection = controlnet, projection
            self.classifier, self.regressor = classifier, regressor
            self.post, self.centroids = post, centroids
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            print(f'GPS models loaded once in {self.load_seconds:.1f}s', flush=True)

    def run(self, input_folder, output_folder, samples=1, steps=50, seed=42):
        import torch
        args = SimpleNamespace(input=Path(input_folder), out=Path(output_folder), samples=samples, steps=steps, seed=seed)
        if args.out.exists():
            raise ValueError('Output folder exists; choose a new folder')
        if not 1 <= samples <= 8 or not 1 <= steps <= 100:
            raise ValueError('Use 1–8 samples and 1–100 steps')
        composite = np.array(Image.open(args.input / 'composite.png').convert('RGB'))
        alpha = np.array(Image.open(args.input / 'mask.png').convert('L'))
        background = np.array(Image.open(args.input / 'background.png').convert('RGB'))
        if composite.shape != background.shape:
            raise ValueError('Composite and background dimensions do not match')
        small, mask, box = letterbox(composite, alpha)
        args.out.mkdir(parents=True)
        Image.fromarray(small).save(args.out / 'model-input.png')
        Image.fromarray(mask).save(args.out / 'model-mask.png')

        prompt, pooled = self.prompt, self.pooled
        unet, vae, scheduler = self.unet, self.vae, self.scheduler
        controlnet, projection = self.controlnet, self.projection
        classifier, regressor = self.classifier, self.regressor
        post, centroids = self.post, self.centroids
        dtype = torch.float16
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            condition = torch.from_numpy(np.concatenate([small,mask[...,None]],axis=-1).astype(np.float32)/255).permute(2,0,1)[None].cuda()
            pred = regressor(condition)[0].cpu().numpy()
            contours, _ = cv2.findContours((mask>127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            (x,y),(w,h),angle = cv2.minAreaRect(np.concatenate(contours))
            if w < h:
                w,h = h,w
                angle += 90
            fg = np.array([x,y,w+1,h+1,angle]).astype(int)
            region = geometry_region(pred, fg)
            Image.fromarray(region*255).save(args.out/'geometry-region.png')
            control_image = torch.cat([condition,torch.from_numpy(region.astype(np.float32))[None,None].cuda()],dim=1).to(dtype)
            labels = classifier(condition).topk(64,largest=True,sorted=False).indices[0].cpu().tolist()
            # This pickle comes only from the SHA256-verified pinned checkpoint archive.
            embeddings = np.stack([np.asarray(centroids[label]).reshape(2048) for label in labels])
            embeddings = torch.from_numpy(embeddings).to(device='cuda',dtype=dtype)[None]
            tokens = projection.norm(projection.proj(embeddings))
            encoder_states = torch.cat([prompt,tokens],dim=1)
            added = {'text_embeds':pooled, 'time_ids':torch.tensor([[512,512,0,0,512,512]],device='cuda',dtype=dtype)}
            del condition,embeddings,tokens
            source = torch.from_numpy(small.astype(np.float32)/127.5-1).permute(2,0,1)[None].cuda()
            load_seconds = self.load_seconds if self.calls == 0 else 0.0
            generated = []
            timings = []
            for index in range(args.samples):
                seed = args.seed + index
                generator = torch.Generator(device='cuda').manual_seed(seed)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                latents = vae.encode(source).latent_dist.sample(generator=generator).to(dtype) * vae.config.scaling_factor
                noise = torch.randn(latents.shape, generator=generator, device='cuda', dtype=dtype)
                latents = scheduler.add_noise(latents, noise, torch.tensor([999],device='cuda'))
                scheduler.set_timesteps(args.steps, device='cuda')
                for step,t in enumerate(scheduler.timesteps):
                    down, mid = controlnet(latents,t,encoder_hidden_states=prompt,added_cond_kwargs=added,
                                           controlnet_cond=control_image,return_dict=False)
                    predicted = unet(latents,t,encoder_hidden_states=encoder_states,added_cond_kwargs=added,
                                     down_block_additional_residuals=down,mid_block_additional_residual=mid).sample
                    latents = scheduler.step(predicted,t,latents,generator=generator).prev_sample
                    if step % 10 == 0:
                        print(f'Sample {index+1}/{args.samples}, step {step+1}/{args.steps}',flush=True)
                decoded = vae.decode((latents / vae.config.scaling_factor).float()).sample
                if not torch.isfinite(decoded).all():
                    raise RuntimeError('Model produced non-finite pixels; no result accepted')
                result = np.uint8(np.rint((decoded[0].float().clamp(-1,1).permute(1,2,0).cpu().numpy()+1)*127.5))
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-tick)
                generated.append(result)
                Image.fromarray(result).save(args.out/f'raw-{seed}.png')
            peak_generation = torch.cuda.max_memory_allocated()/1024**3
            # Release image-specific tensors, retaining all model weights for the batch.
            del source,latents,noise,decoded,down,mid,predicted,control_image,encoder_states
            torch.cuda.reset_peak_memory_stats()
            panels = [('Without shadow',Image.fromarray(composite))]
            normal = args.input/'normal.png'
            if normal.exists():
                panels.append(('Current normal shadows',Image.open(normal).convert('RGB')))
            protection = []
            post_times = []
            for index,result in enumerate(generated):
                tick = time.perf_counter()
                seed = args.seed+index
                comp256 = cv2.resize(small,(256,256)).astype(np.float32)/127.5-1
                gen256 = cv2.resize(result,(256,256),interpolation=cv2.INTER_NEAREST).astype(np.float32)/127.5-1
                mask256 = cv2.resize(mask,(256,256)).astype(np.float32)/255
                tensor = torch.from_numpy(np.concatenate([gen256,comp256,mask256[...,None]],axis=-1)).permute(2,0,1)[None].cuda()
                logits = post(tensor,timesteps=torch.zeros(1,device='cuda'))[0,3].cpu().numpy()
                support = cv2.resize((logits>=0).astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
                Image.fromarray(np.uint8(support*255)).save(args.out/f'shadow-mask-{seed}.png')
                opacity = restore_map(shadow_opacity(small,result,support),box,alpha.shape)
                final = apply_shadow(composite,background,alpha,opacity)
                delta = np.abs(final.astype(int)-composite.astype(int))
                max_vehicle_change = int(delta[alpha==255].max()) if np.any(alpha==255) else 0
                if max_vehicle_change:
                    raise RuntimeError('Vehicle preservation check failed')
                protection.append(max_vehicle_change)
                Image.fromarray(final).save(args.out/f'protected-{seed}.png')
                Image.fromarray(np.uint8(opacity*255)).save(args.out/f'opacity-{seed}.png')
                panels.append((f'AI shadow, seed {seed}',Image.fromarray(final)))
                post_times.append(time.perf_counter()-tick)
            sheet = Image.new('RGB',(640*2, (len(panels)+1)//2*510),'white')
            draw = ImageDraw.Draw(sheet)
            for i,(label,img) in enumerate(panels):
                x,y = (i%2)*640,(i//2)*510
                draw.text((x+10,y+8),label,fill='black')
                img.thumbnail((640,480))
                sheet.paste(img,(x+(640-img.width)//2,y+30))
            sheet.save(args.out/'comparison.jpg',quality=95)
            report = {'gpu':torch.cuda.get_device_name(0),'steps':args.steps,'samples':args.samples,
                'seeds':list(range(args.seed,args.seed+args.samples)), 'load_seconds':load_seconds,
                'generation_seconds':timings,'postprocess_seconds':post_times,
                'peak_generation_allocated_gib':peak_generation,
                'peak_postprocess_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                'max_opaque_vehicle_pixel_change':protection,
                'versions':json.loads((CACHE/'ready.json').read_text()),
                'notes':'Experimental neutral shadow transfer. Raw model outputs are also saved. No automatic quality acceptance.'}
            (args.out/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
            print('Comparison:',(args.out/'comparison.jpg').resolve())
            self.calls += 1
            return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Folder produced by prepare.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.samples <= 8 or not 1 <= args.steps <= 100:
        parser.error('Use 1–8 samples and 1–100 steps')
    if args.out.exists():
        parser.error('Output folder exists; use a new name to preserve earlier results')
    if not (CACHE / 'ready.json').exists():
        parser.error('Run shadow-lab/gps/setup.sh first')
    runner = GPSRunner()
    runner.run(args.input, args.out, args.samples, args.steps, args.seed)


if __name__ == '__main__':
    main()
