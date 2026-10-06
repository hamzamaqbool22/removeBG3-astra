"""Reusable segmentation session; CPU by default, CUDA only by explicit opt-in."""
from pathlib import Path
import os
import logging

import cv2
import numpy as np
from PIL import Image


class Segmenter:
    def __init__(self, model: str = "birefnet-general", threads: int | None = None,
                 model_dir: str | Path | None = None, memory_mode: str = 'balanced'):
        self.segmentation_mode = os.environ.get("SEGMENTATION_MODE", "auto").strip().lower()
        if self.segmentation_mode not in ("auto", "single"):
            raise ValueError("SEGMENTATION_MODE must be auto or single")
        root = Path(__file__).resolve().parents[1]
        threads = threads if threads is not None else int(os.environ.get("CPU_THREADS", "6"))
        if threads < 1:
            raise ValueError("CPU_THREADS must be positive")
        cache = Path(model_dir or os.environ.get("U2NET_HOME") or root / ".cache/models")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["U2NET_HOME"] = str(cache.resolve())
        os.environ.setdefault("NUMBA_CACHE_DIR", str(root / ".cache/numba"))
        import onnxruntime as ort
        device = os.environ.get("INFERENCE_DEVICE", "cpu").strip().lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("INFERENCE_DEVICE must be cpu or cuda")
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if not hasattr(ort, "preload_dlls"):
                raise RuntimeError("GPU mode requires onnxruntime-gpu with preload_dlls")
            ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDA provider is unavailable")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        from rembg.sessions import sessions_class
        cls = next((c for c in sessions_class if c.name() == model), None)
        if cls is None:
            raise ValueError(f"Unsupported segmentation model: {model}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if memory_mode not in ('balanced','lean'):
            raise ValueError('memory_mode must be balanced or lean')
        if memory_mode == 'lean':
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
        self.session = cls(model, opts, providers=providers)
        self.model = model
        self.providers = self.session.inner_session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError("CUDA initialization failed; refusing CPU fallback")
        logging.getLogger("uvicorn.error").info(
            "Inference device: %s; providers: %s", device, self.providers)
        self.last_info = {}
        logging.getLogger("uvicorn.error").info("Segmentation mode: %s", self.segmentation_mode)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image")
        mask = np.asarray(self.session.predict(Image.fromarray(rgb))[0], dtype=np.uint8)
        self.last_info = {'inference_passes': 1, 'adaptive_crop': None}
        if self.segmentation_mode == 'single':
            return mask
        # Distant vehicles lose small parts (particularly mirrors) when the
        # entire scene is resized for the network. Use the first mask only to
        # localize a generously padded crop, then rerun on original RGB pixels.
        count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
        if count > 1:
            main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, w, h, area = stats[main]
            ih, iw = mask.shape
            if area > mask.size * .01 and max(w/iw, h/ih) < .60:
                pad = max(16, round(.20 * max(w,h)))
                x0, y0 = max(0,x-pad), max(0,y-pad)
                x1, y1 = min(iw,x+w+pad), min(ih,y+h+pad)
                crop = rgb[y0:y1,x0:x1]
                detail = np.asarray(self.session.predict(Image.fromarray(crop))[0], dtype=np.uint8)
                mask = np.zeros((ih,iw),np.uint8)
                mask[y0:y1,x0:x1] = detail
                self.last_info = {'inference_passes':2,'adaptive_crop':[int(x0),int(y0),int(x1),int(y1)]}
        return mask


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Discard disconnected background detections without eroding body details."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    if count <= 1:
        raise ValueError("No foreground vehicle found")
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[main, cv2.CC_STAT_AREA] < mask.size * 0.01:
        raise ValueError("Foreground is too small to process reliably")
    support = (labels == main).astype(np.uint8)
    support = cv2.dilate(support, np.ones((9, 9), np.uint8))
    result = mask.copy()
    result[support == 0] = 0
    result[result < 3] = 0
    result[result > 252] = 255
    return result
