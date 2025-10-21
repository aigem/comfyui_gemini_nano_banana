import json
import os
import time
import hashlib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from .image_utils import base64_to_pil, pil_to_tensor
from .api_client import call_gemini_api_text, normalize_generation_config
from .doc_utils import load_node_description
from .response_utils import extract_usage_metadata, extract_image_from_response
from .constants import (
    INSTRUCTION_PRESET_DIR,
    SYSTEM_PROMPT_DIR,
    DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

class GeminiTextToImage:
    """
    文生图：使用 Gemini 2.5 Flash Image API（兼容 API）从文本生成图像
    """

    CATEGORY = "AI/Gemini"
    DEFAULT_MODEL = DEFAULT_MODEL
    DESCRIPTION = load_node_description("GeminiTextToImage")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        preset_choices = cls._instruction_choices()
        preset_choices = tuple(preset_choices) if preset_choices else ("(none)",)
        default_preset = preset_choices[0] if preset_choices else "(none)"

        system_prompt_choices = tuple(cls._system_prompt_choices())
        default_system_prompt = system_prompt_choices[0] if system_prompt_choices else "(none)"

        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "用中文详细描述你想要生成的图片内容、风格与细节。",
                        "tooltip": "文生图提示词（最大长度受环境变量限制）。",
                    },
                ),
                "num_outputs": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "parallel": ("BOOLEAN", {"default": True, "tooltip": "并行请求多张生成。"}),
                "api_key": ("STRING", {"default": "", "password": True}),
                "timeout": ("INT", {"default": 60, "min": 5, "max": 120, "step": 5, "display": "slider"}),
            },
            "optional": {
                "instruction_preset": (preset_choices, {"default": default_preset}),
                "system_prompt": (system_prompt_choices, {"default": default_system_prompt}),
                "max_output_tokens": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 64}),
                "minimal_response": ("BOOLEAN", {"default": True, "tooltip": "提示模型仅返回 PNG 图片，不要文字。"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483646, "step": 1, "tooltip": "生成种子；-1 表示随机，每次运行都会变化。"}),
                "base_url": ("STRING", {"default": "https://apis.kuai.host/"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metrics_json")
    FUNCTION = "run"
    OUTPUT_NODE = False

    @classmethod
    def _instruction_choices(cls) -> List[str]:
        choices: List[str] = ["(none)"]
        try:
            root = INSTRUCTION_PRESET_DIR
            if root.exists():
                for p in sorted(root.rglob("*.json")):
                    try:
                        rel = p.relative_to(root).as_posix()
                    except Exception:
                        continue
                    choices.append(rel)
        except Exception as exc:
            logger.warning("枚举指令预设失败: %s", exc)
        return choices

    @classmethod
    def _system_prompt_choices(cls) -> List[str]:
        choices: List[str] = ["(none)"]
        try:
            root = SYSTEM_PROMPT_DIR
            if root.exists():
                for p in sorted(root.rglob("*.md")):
                    try:
                        rel = p.relative_to(root).as_posix()
                    except Exception:
                        continue
                    choices.append(rel)
        except Exception as exc:
            logger.warning("枚举系统提示失败: %s", exc)
        return choices

    def _load_system_prompt(self, name: str) -> Optional[str]:
        if not name or name.strip().lower() in {"(none)", "none"}:
            return None
        candidate = (SYSTEM_PROMPT_DIR / name).resolve()
        try:
            candidate.relative_to(SYSTEM_PROMPT_DIR.resolve())
        except Exception:
            return None
        if not candidate.is_file():
            return None
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None

    def _load_instruction_preset(self, name: str) -> Optional[Dict[str, Any]]:
        if not name or name.strip().lower() in {"(none)", "none"}:
            return None
        candidate = (INSTRUCTION_PRESET_DIR / name).resolve()
        try:
            candidate.relative_to(INSTRUCTION_PRESET_DIR.resolve())
        except Exception:
            return None
        if not candidate.is_file():
            return None
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _apply_prompt_preset(self, base_prompt: str, preset: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
        if not isinstance(preset, dict):
            return base_prompt.strip(), []
        prompt_core = base_prompt.strip()
        template = preset.get("prompt_template")
        if isinstance(template, str) and "{prompt}" in template:
            prompt_core = template.replace("{prompt}", prompt_core)
        segments: List[str] = []
        prefix = preset.get("prompt_prefix")
        if isinstance(prefix, str) and prefix.strip():
            segments.append(prefix.strip())
        if prompt_core:
            segments.append(prompt_core)
        directives = preset.get("directives")
        if isinstance(directives, list):
            for item in directives:
                if isinstance(item, str) and item.strip():
                    segments.append(item.strip())
        suffix = preset.get("prompt_suffix")
        if isinstance(suffix, str) and suffix.strip():
            segments.append(suffix.strip())
        final_prompt = "\n\n".join(seg for seg in segments if seg)
        extra_parts: List[str] = []
        extra = preset.get("extra_parts")
        if isinstance(extra, list):
            for part in extra:
                if isinstance(part, str) and part.strip():
                    extra_parts.append(part.strip())
        return (final_prompt or prompt_core, extra_parts)

    def _validate(self, api_key: str, prompt: str, max_chars: int = 1000) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("API key 不能为空")
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空")
        p = prompt.strip()
        if len(p) > max_chars:
            raise ValueError(f"提示词过长（>{max_chars} 字符）")
        pl = p.lower()
        forbidden = ("<script", "</script", "<iframe", "</iframe", "javascript:")
        for tag in forbidden:
            if tag in pl:
                raise ValueError("提示词包含禁止的标记/脚本片段")

    def run(
        self,
        prompt: str,
        api_key: str,
        timeout: int,
        instruction_preset: str = "(none)",
        system_prompt: str = "(none)",
        max_output_tokens: int = 0,
        minimal_response: bool = True,
        temperature: float = 0.0,
        top_p: float = 0.0,
        num_outputs: int = 1,
        parallel: bool = True,
        base_url: str = "https://apis.kuai.host/",
        model: Optional[str] = None,
        seed: int = -1,
    ) -> Tuple[Any, str]:
        try:
            self._validate(api_key, prompt)
            overall_start = time.time()

            selected_preset = (instruction_preset or "").strip()
            preset_data = self._load_instruction_preset(selected_preset)

            system_prompt_used = ""
            if isinstance(system_prompt, str):
                candidate = system_prompt.strip()
                loaded_system = self._load_system_prompt(candidate) if candidate else None
                if loaded_system:
                    system_prompt_used = loaded_system
                elif candidate.lower() not in {"", "(none)", "none"}:
                    system_prompt_used = candidate

            extra_user_parts: List[str] = []
            final_prompt = prompt.strip()
            if preset_data:
                preset_system = preset_data.get("system_prompt")
                if isinstance(preset_system, str) and preset_system.strip():
                    if system_prompt_used:
                        system_prompt_used = f"{system_prompt_used}\n\n{preset_system.strip()}"
                    else:
                        system_prompt_used = preset_system.strip()
                final_prompt, extra_user_parts = self._apply_prompt_preset(final_prompt, preset_data)

            if bool(minimal_response):
                extra_user_parts.append("请仅返回生成的图像（PNG），不要返回文字说明。")

            override_src: Dict[str, Any] = {}
            if isinstance(max_output_tokens, (int, float)) and int(max_output_tokens) > 0:
                override_src["max_output_tokens"] = int(max_output_tokens)
            if isinstance(temperature, (int, float)) and float(temperature) > 0:
                override_src["temperature"] = max(0.0, min(2.0, float(temperature)))
            if isinstance(top_p, (int, float)) and float(top_p) > 0:
                override_src["top_p"] = max(0.0, min(1.0, float(top_p)))
            # 处理种子：-1 表示随机；非负表示固定基准种子，按 idx 偏移复现多图
            try:
                if isinstance(seed, (int, float)):
                    seed_int = int(seed)
                    if seed_int >= 0:
                        override_src["seed"] = seed_int % 2147483647
                # 注意：具体每张的 idx 偏移在 _one_call 内实现
            except Exception:
                pass

            override_cfg = normalize_generation_config(override_src)

            effective_model = self.DEFAULT_MODEL if model is None else model

            import numpy as _np
            results_np: List[_np.ndarray] = []
            aggregated_tokens: Dict[str, int] = {}
            items: List[Dict[str, Any]] = []

            def _one_call(idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                # 文生图并发多图时注入 per-request seed 与零宽字符扰动（在 temperature==0 下）
                metrics_ref: Dict[str, Any] = {}
                per_request_cfg: Dict[str, Any] = dict(override_cfg or {})
                try:
                    if int(num_outputs) > 1:
                        import time as _t
                        if isinstance(seed, int) and seed >= 0:
                            seed_base = seed % 2147483647
                        else:
                            seed_base = int((_t.time() * 1000)) & 0x7FFFFFFF
                        per_request_cfg["seed"] = (seed_base + int(idx)) % 2147483647
                    else:
                        # 单图时也要根据 seed 决定：固定或随机
                        if isinstance(seed, int) and seed >= 0:
                            per_request_cfg["seed"] = seed % 2147483647
                        else:
                            import time as _t2
                            per_request_cfg["seed"] = (int((_t2.time() * 1000)) & 0x7FFFFFFF)
                except Exception:
                    pass
                per_extra_parts = list(extra_user_parts or [])
                try:
                    temp_val = override_src.get("temperature", 0.0)
                    if int(num_outputs) > 1 and (not isinstance(temp_val, (int, float)) or float(temp_val) <= 0.0):
                        invisible = "\u200b" * (int(idx) + 1)
                        per_extra_parts.append(invisible)
                except Exception:
                    pass
                resp = call_gemini_api_text(
                    prompt=final_prompt,
                    api_key=api_key,
                    timeout=timeout,
                    model=effective_model,
                    system_prompt=system_prompt_used or None,
                    extra_text_parts=per_extra_parts or None,
                    generation_config=per_request_cfg or None,
                    base_url=base_url,
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "2")),
                    metrics_ref=metrics_ref,
                )
                return resp, metrics_ref

            if int(num_outputs) <= 1:
                resp, mref = _one_call(0)
                ut = extract_usage_metadata(resp.get("usageMetadata") or resp.get("usage_metadata"))
                if ut:
                    for k, v in ut.items():
                        aggregated_tokens[k] = aggregated_tokens.get(k, 0) + int(v)
                items.append({"index": 0, "status": mref.get("status"), "latency_ms": mref.get("latency_ms")})
                img_b64 = extract_image_from_response(resp)
                img = base64_to_pil(img_b64)
                results_np.append(pil_to_tensor(img))
            else:
                if bool(parallel):
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    max_workers = max(1, min(int(num_outputs), 4))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = {ex.submit(_one_call, i): i for i in range(int(num_outputs))}
                        for fut in as_completed(futs):
                            i = futs[fut]
                            try:
                                resp, mref = fut.result()
                                ut = extract_usage_metadata(resp.get("usageMetadata") or resp.get("usage_metadata"))
                                if ut:
                                    for k, v in ut.items():
                                        aggregated_tokens[k] = aggregated_tokens.get(k, 0) + int(v)
                                items.append({"index": i, "status": mref.get("status"), "latency_ms": mref.get("latency_ms")})
                                img_b64 = extract_image_from_response(resp)
                                img = base64_to_pil(img_b64)
                                results_np.append(pil_to_tensor(img))
                            except Exception as e:
                                logger.warning("第 %s 张生成失败: %s", i, e)
                else:
                    for i in range(int(num_outputs)):
                        try:
                            resp, mref = _one_call(i)
                            ut = extract_usage_metadata(resp.get("usageMetadata") or resp.get("usage_metadata"))
                            if ut:
                                for k, v in ut.items():
                                    aggregated_tokens[k] = aggregated_tokens.get(k, 0) + int(v)
                            items.append({"index": i, "status": mref.get("status"), "latency_ms": mref.get("latency_ms")})
                            img_b64 = extract_image_from_response(resp)
                            img = base64_to_pil(img_b64)
                            results_np.append(pil_to_tensor(img))
                        except Exception as e:
                            logger.warning("第 %s 张生成失败: %s", i, e)

            if not results_np:
                raise RuntimeError("所有生成请求均失败，未获取到有效图像。")
            result_np = _np.concatenate(results_np, axis=0)  # [B,H,W,C]

            if torch is not None:
                result_tensor = torch.from_numpy(result_np.copy()).to("cpu").to(dtype=torch.float32)
            else:
                result_tensor = result_np

            latency_ms = int((time.time() - overall_start) * 1000)
            metrics_payload: Dict[str, Any] = {
                "model": effective_model,
                "latency_ms": latency_ms,
                "status": None,
                "tokens": aggregated_tokens or None,
                "seed": int(seed) if isinstance(seed, int) else None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # 写入简易历史（独立文件夹可与现有复用）
            enable_history = os.getenv("GEMINI_HISTORY", "0").lower() in {"1", "true", "yes", "on"}
            history_dir = os.getenv("GEMINI_HISTORY_DIR", os.path.join("output", "gemini_history"))
            if enable_history:
                try:
                    os.makedirs(history_dir, exist_ok=True)
                    prompt_hash = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest() if final_prompt else None
                    ts = datetime.utcnow().isoformat() + "Z"
                    history_obj = {
                        "timestamp": ts,
                        "node": "GeminiTextToImage",
                        "model": effective_model,
                        "params": {
                            "timeout": timeout,
                            "prompt": final_prompt,
                            "system_prompt": system_prompt_used,
                            "instruction_preset": selected_preset,
                            "minimal_response": bool(minimal_response),
                            "max_output_tokens": override_src.get("max_output_tokens"),
                            "temperature": override_src.get("temperature"),
                            "top_p": override_src.get("top_p"),
                            "base_url": base_url,
                        },
                        "hashes": {"prompt": prompt_hash},
                        "metrics": {
                            "latency_ms": latency_ms,
                            "tokens": aggregated_tokens or None,
                        },
                    }
                    fname = f"history_text_{ts.replace(':', '-').replace('.', '-')}.json"
                    fpath = os.path.join(history_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(history_obj, f, ensure_ascii=False, indent=2)
                    logger.info("已写入文生图历史记录: %s", fpath)
                except Exception as exc:
                    logger.warning("写入历史记录失败: %s", exc)

            metrics_json = json.dumps({k: v for k, v in metrics_payload.items() if v is not None}, ensure_ascii=False)
            return result_tensor, metrics_json

        except Exception as exc:
            safe_msg = str(exc).replace(api_key if api_key else "", "***")
            logger.error("Gemini 文生图节点错误: %s", safe_msg)
            raise RuntimeError(f"Gemini 文生图失败: {safe_msg}")

# 节点注册
NODE_CLASS_MAPPINGS = {"GeminiTextToImage": GeminiTextToImage}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTextToImage": "Gemini Nano Banana T2I文生图助手 [kuai.host]"}