import io
import base64
from typing import Any
import numpy as np
from PIL import Image

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

def tensor_to_pil(tensor: Any) -> Image.Image:
    """
    将 ComfyUI 图像张量（NumPy 或 Torch）转换为 PIL Image。
    支持形状：[B,H,W,C]、[H,W,C]、[B,C,H,W]、[C,H,W]；数值范围 [0,1] 或 [0,255]。
    """
    if torch is not None and isinstance(tensor, torch.Tensor):
        t = tensor.detach().to("cpu")
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and (t.shape[0] in (1, 3, 4)) and (t.shape[-1] not in (1, 3, 4)):
            t = t.permute(1, 2, 0)
        if t.dtype.is_floating_point:
            t = t.clamp(0, 1).mul(255).to(torch.uint8)
        else:
            t = t.clamp(0, 255).to(torch.uint8)
        arr = t.numpy()
    else:
        arr = tensor
        if isinstance(arr, Image.Image):
            img = arr
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        if hasattr(arr, "ndim") and arr.ndim == 4:
            arr = arr[0]
        if hasattr(arr, "shape") and arr.ndim == 3 and (arr.shape[0] in (1, 3, 4)) and (arr.shape[-1] not in (1, 3, 4)):
            arr = np.transpose(arr, (1, 2, 0))
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "f":
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        elif isinstance(arr, np.ndarray):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(arr)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and (arr.shape[0] in (1, 3, 4)) and (arr.shape[-1] not in (1, 3, 4)):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype.kind == "f":
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """PIL Image 转 base64 字符串"""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_pil(base64_str: str) -> Image.Image:
    """base64 字符串转 PIL Image"""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))

def pil_to_tensor(image: Image.Image) -> np.ndarray:
    """
    PIL 转 ComfyUI 张量格式：
    返回形状 [1, H, W, C]，值域 0-1（NumPy float32）
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    np_image = np.array(image).astype(np.float32) / 255.0
    return np_image[np.newaxis, ...]