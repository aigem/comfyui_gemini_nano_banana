"""
ComfyUI 自定义节点：Gemini Nano Banana Edit改图助手 [kuai.host]
使用 Gemini 2.5 Flash Image API（兼容 API）编辑图像
"""

import json
import os
import time
import hashlib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# 尝试导入 torch（ComfyUI 环境通常可用）
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# 外部模块（已拆分）
from .image_utils import tensor_to_pil, pil_to_base64, base64_to_pil, pil_to_tensor
from .api_client import call_gemini_api, normalize_generation_config
from .response_utils import extract_usage_metadata, extract_image_from_response
from .doc_utils import load_node_description

logger = logging.getLogger(__name__)

# 预设目录
INSTRUCTION_PRESET_DIR = Path(__file__).resolve().parent / "instruction_presets"
SYSTEM_PROMPT_DIR = INSTRUCTION_PRESET_DIR / "system_prompt"


class GeminiEditImage:
    """
    使用 Gemini 2.5 Flash Image API 编辑图像的 ComfyUI 节点
    接收图像张量与文本提示，返回编辑后的图像与指标 JSON
    """

    CATEGORY = "AI/Gemini"
    DEFAULT_MODEL = "gemini-2.5-flash-image"
    DESCRIPTION = load_node_description("GeminiEditImage")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        raw_choices = cls._instruction_choices()
        preset_choices = tuple(raw_choices) if raw_choices else ("(none)",)
        default_preset = preset_choices[0] if preset_choices else "(none)"
        system_prompt_choices = tuple(cls._system_prompt_choices())
        default_system_prompt = system_prompt_choices[0] if system_prompt_choices else "(none)"
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "请根据提示修改这张图片",
                        "tooltip": "用自然语言描述你希望如何修改输入图像。",
                    },
                ),
                "num_outputs": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "生成张数（批量返回）。",
                    },
                ),
                "parallel": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "并行请求生成多图（可能更快，注意速率限制）。",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "password": True,
                        "tooltip": "Gemini API 密钥（不会以明文写入磁盘）。",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 5,
                        "max": 120,
                        "step": 5,
                        "display": "slider",
                        "tooltip": "API 请求超时时间（秒）。",
                    },
                ),
            },
            "optional": {
                "instruction_preset": (
                    preset_choices,
                    {
                        "default": default_preset,
                        "tooltip": "从 instruction_presets/ 选择 JSON 预设以扩展提示与配置（可选）。",
                    },
                ),
                "system_prompt": (
                    system_prompt_choices,
                    {
                        "default": default_system_prompt,
                        "tooltip": "从 instruction_presets/system_prompt/ 选择系统提示（可选）。",
                    },
                ),
                "image_2": ("IMAGE", {"tooltip": "可选参考图像（作为额外上下文）。"}),
                "image_3": ("IMAGE", {"tooltip": "可选参考图像（作为额外上下文）。"}),
                "image_4": ("IMAGE", {"tooltip": "可选参考图像（作为额外上下文）。"}),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "响应 token 数上限（0 使用服务默认值）。",
                    },
                ),
                "minimal_response": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "在提示中附加“仅返回编辑后的 PNG 图像，不要文字说明”。",
                    },
                ),
                "edit_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                        "display": "slider",
                        "tooltip": "编辑强度提示：0 保守，1 强烈。",
                    },
                ),
                "color_lock": ("BOOLEAN", {"default": False, "tooltip": "尽量保留原始配色。"}),
                "style_lock": ("BOOLEAN", {"default": False, "tooltip": "尽量保持原始画风/笔触。"}),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": 0.01,
                        "display": "slider",
                        "tooltip": "采样温度覆写（0 使用服务默认值）。",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                        "display": "slider",
                        "tooltip": "核采样上限（0 使用服务默认值）。",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483646,
                        "step": 1,
                        "tooltip": "生成种子；-1 表示随机，每次运行都会变化。",
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://apis.kuai.host/",
                        "tooltip": "Gemini 兼容服务的基础地址（如：https://apis.kuai.host/）。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metrics_json")
    FUNCTION = "run"
    OUTPUT_NODE = False

    def __init__(self):
        self.model_name = self.DEFAULT_MODEL
        # 环境变量配置
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
        self.enable_history = os.getenv("GEMINI_HISTORY", "0").lower() in {"1", "true", "yes", "on"}
        self.history_dir = os.getenv("GEMINI_HISTORY_DIR", os.path.join("output", "gemini_history"))
        self.max_prompt_chars = int(os.getenv("GEMINI_MAX_PROMPT_CHARS", "1000"))
        env_forbidden = os.getenv("GEMINI_FORBIDDEN_TAGS")
        if env_forbidden:
            self.forbidden_tags = tuple(x.strip().lower() for x in env_forbidden.split(",") if x.strip())
        else:
            self.forbidden_tags = ("<script", "</script", "<iframe", "</iframe", "javascript:")
        # 最新请求指标
        self._last_metrics: Dict[str, Any] = {}

    @classmethod
    def _instruction_choices(cls) -> List[str]:
        choices: List[str] = ["(none)"]
        try:
            preset_root = INSTRUCTION_PRESET_DIR
            if preset_root.exists():
                for preset_path in sorted(preset_root.rglob("*.json")):
                    try:
                        rel_path = preset_path.relative_to(preset_root)
                    except ValueError:
                        continue
                    choices.append(rel_path.as_posix())
        except Exception as exc:
            logger.warning("枚举指令预设失败: %s", exc)
        return choices

    @classmethod
    def _system_prompt_choices(cls) -> List[str]:
        choices: List[str] = ["(none)"]
        try:
            prompt_root = SYSTEM_PROMPT_DIR
            if prompt_root.exists():
                for prompt_path in sorted(prompt_root.rglob("*.md")):
                    try:
                        rel_path = prompt_path.relative_to(prompt_root)
                    except ValueError:
                        continue
                    choices.append(rel_path.as_posix())
        except Exception as exc:
            logger.warning("枚举系统提示失败: %s", exc)
        return choices

    def _load_instruction_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        if not preset_name or preset_name.strip().lower() in {"(none)", "none"}:
            return None
        preset_root = INSTRUCTION_PRESET_DIR.resolve()
        candidate_path = (INSTRUCTION_PRESET_DIR / preset_name).resolve()
        try:
            candidate_path.relative_to(preset_root)
        except ValueError:
            logger.warning("指令预设路径越界: %s", preset_name)
            return None
        if not candidate_path.is_file():
            logger.warning("指令预设不存在: %s", candidate_path)
            return None
        try:
            with candidate_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning("加载指令预设失败 %s: %s", candidate_path, exc)
            return None

    def _load_system_prompt(self, prompt_name: str) -> Optional[str]:
        if not prompt_name or prompt_name.strip().lower() in {"(none)", "none"}:
            return None
        prompt_root = SYSTEM_PROMPT_DIR.resolve()
        candidate_path = (SYSTEM_PROMPT_DIR / prompt_name).resolve()
        try:
            candidate_path.relative_to(prompt_root)
        except ValueError:
            logger.warning("系统提示路径越界: %s", prompt_name)
            return None
        if not candidate_path.is_file():
            logger.warning("系统提示不存在: %s", candidate_path)
            return None
        try:
            with candidate_path.open("r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as exc:
            logger.warning("加载系统提示失败 %s: %s", candidate_path, exc)
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
        final_prompt = "\n\n".join(seg for seg in segments if isinstance(seg, str) and seg)
        extra_parts: List[str] = []
        extra = preset.get("extra_parts")
        if isinstance(extra, list):
            for part in extra:
                if isinstance(part, str) and part.strip():
                    extra_parts.append(part.strip())
        return (final_prompt or prompt_core, extra_parts)

    def validate_inputs(self, api_key: str, prompt: str) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("API key 不能为空")
        if not prompt or not prompt.strip():
            raise ValueError("提示词不能为空")
        p = prompt.strip()
        if len(p) > self.max_prompt_chars:
            raise ValueError(f"提示词过长（>{self.max_prompt_chars} 字符）")
        pl = p.lower()
        for tag in self.forbidden_tags:
            if tag and tag in pl:
                raise ValueError("提示词包含禁止的标记/脚本片段")

    def redact_key(self, key: str) -> str:
        if not key:
            return "EMPTY"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def run(
        self,
        image,
        prompt: str,
        api_key: str,
        timeout: int,
        instruction_preset: str = "(none)",
        system_prompt: str = "(none)",
        image_2=None,
        image_3=None,
        image_4=None,
        max_output_tokens: int = 0,
        minimal_response: bool = True,
        edit_strength: float = 0.5,
        color_lock: bool = False,
        style_lock: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.0,
        num_outputs: int = 1,
        parallel: bool = True,
        base_url: str = "https://apis.kuai.host/",
        model: Optional[str] = None,
        seed: int = -1,
    ) -> Tuple[Any, str]:
        try:
            self.validate_inputs(api_key, prompt)

            is_torch_input = (torch is not None) and isinstance(image, torch.Tensor)
            orig_device = image.device if is_torch_input else None
            orig_shape = tuple(image.shape) if hasattr(image, "shape") else None
            orig_is_bchw = False
            if orig_shape and len(orig_shape) == 4:
                if orig_shape[1] in (1, 3, 4) and not (orig_shape[-1] in (1, 3, 4)):
                    orig_is_bchw = True

            logger.info("开始图像编辑，timeout=%ss", timeout)

            pil_image = tensor_to_pil(image)
            image_base64_list: List[str] = [pil_to_base64(pil_image, format="PNG")]
            for extra in (image_2, image_3, image_4):
                if extra is None:
                    continue
                try:
                    extra_pil = tensor_to_pil(extra)
                    image_base64_list.append(pil_to_base64(extra_pil, format="PNG"))
                except Exception as exc:
                    logger.warning("参考图处理失败: %s", exc)

            overall_start = time.time()

            aug_parts: List[str] = []
            try:
                es = max(0.0, min(1.0, float(edit_strength)))
                if abs(es - 0.5) > 1e-6:
                    aug_parts.append(f"应用编辑强度 {es:.2f}（0=保守，1=强烈）。")
            except Exception:
                pass
            if bool(color_lock):
                aug_parts.append("尽量保留原始配色。")
            if bool(style_lock):
                aug_parts.append("尽量保持原始画风/笔触。")
            final_prompt = prompt.strip()
            if aug_parts:
                final_prompt = f"{prompt.strip()}\n\n" + " ".join(aug_parts)

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
            if preset_data:
                preset_system = preset_data.get("system_prompt")
                if isinstance(preset_system, str) and preset_system.strip():
                    if system_prompt_used:
                        system_prompt_used = f"{system_prompt_used}\n\n{preset_system.strip()}"
                    else:
                        system_prompt_used = preset_system.strip()
                final_prompt, extra_user_parts = self._apply_prompt_preset(final_prompt, preset_data)

            # 生成配置
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

            if bool(minimal_response):
                extra_user_parts.append("请仅返回编辑后的图像（PNG），不要返回文字说明。")

            # 模型与 API 调用
            effective_model = self.model_name if model is None else model
            # 支持一次生成多张；当 num_outputs>1 时，发起多次请求（可并行）
            import numpy as _np  # 局部导入，避免顶层类型检查依赖
            results_np: List[_np.ndarray] = []
            aggregated_tokens: Dict[str, int] = {}
            items: List[Dict[str, Any]] = []

            def _one_call(idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                # 为并发多图注入轻量随机性：每请求不同 seed；temperature==0 时再加零宽字符片段打破缓存
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
                    # 当 temperature 未显式提升且多图时，增加零宽字符做为无害扰动
                    temp_val = override_src.get("temperature", 0.0)
                    if int(num_outputs) > 1 and (not isinstance(temp_val, (int, float)) or float(temp_val) <= 0.0):
                        invisible = "\u200b" * (int(idx) + 1)
                        per_extra_parts.append(invisible)
                except Exception:
                    pass
                resp = call_gemini_api(
                    images_base64=image_base64_list,
                    prompt=final_prompt,
                    api_key=api_key,
                    timeout=timeout,
                    model=effective_model,
                    system_prompt=system_prompt_used or None,
                    extra_text_parts=per_extra_parts or None,
                    generation_config=per_request_cfg or None,
                    base_url=base_url,
                    max_retries=self.max_retries,
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

            # 若有成功结果则拼接为批次输出
            if not results_np:
                raise RuntimeError("所有生成请求均失败，未获取到有效图像。")
            result_np = _np.concatenate(results_np, axis=0)  # [B,H,W,C]

            # 原输入是 BCHW 时，转换维度
            if orig_is_bchw:
                result_np = _np.transpose(result_np, (0, 3, 1, 2))

            # 聚合 token 指标
            if aggregated_tokens:
                tokens0 = self._last_metrics.get("tokens", {}) or {}
                for k, v in aggregated_tokens.items():
                    tokens0[k] = int(tokens0.get(k, 0)) + int(v)
                self._last_metrics["tokens"] = tokens0

            # 额外记录每张的子项指标
            self._last_metrics["items"] = items

            # 输出张量类型与设备
            if is_torch_input:
                assert torch is not None
                result_tensor = torch.from_numpy(result_np.copy()).to(
                    orig_device if orig_device is not None else "cpu"
                ).to(dtype=torch.float32)
            else:
                result_tensor = result_np

            latency_ms = int((time.time() - overall_start) * 1000)
            retry_count = int(self._last_metrics.get("retry_count", 0) or 0)
            request_count = int(self._last_metrics.get("request_count", retry_count + 1) or (retry_count + 1))

            metrics_payload: Dict[str, Any] = {
                "model": effective_model,
                "latency_ms": latency_ms,
                "retry_count": retry_count,
                "request_count": request_count,
                "status": self._last_metrics.get("status"),
                "tokens": self._last_metrics.get("tokens"),
                "applied_preset": {
                    "path": selected_preset,
                    "name": preset_data.get("name") if isinstance(preset_data, dict) else None,
                    "description": preset_data.get("description") if isinstance(preset_data, dict) else None,
                }
                if preset_data
                else None,
                "max_output_tokens": override_src.get("max_output_tokens"),
                "minimal_response": bool(minimal_response),
                "temperature": override_src.get("temperature"),
                "top_p": override_src.get("top_p"),
                "seed": int(seed) if isinstance(seed, int) else None,
                "reference_image_count": len(image_base64_list),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # 历史记录
            if self.enable_history:
                try:
                    os.makedirs(self.history_dir, exist_ok=True)
                    input_hashes = [hashlib.sha256(img.encode("utf-8")).hexdigest() for img in image_base64_list]
                    prompt_hash = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest() if final_prompt else None
                    ts = datetime.utcnow().isoformat() + "Z"
                    history_obj = {
                        "timestamp": ts,
                        "node": "GeminiEditImage",
                        "model": effective_model,
                        "params": {
                            "timeout": timeout,
                            "prompt": final_prompt,
                            "system_prompt": system_prompt_used,
                            "instruction_preset": selected_preset,
                            "edit_strength": float(edit_strength),
                            "color_lock": bool(color_lock),
                            "style_lock": bool(style_lock),
                            "minimal_response": bool(minimal_response),
                            "max_output_tokens": override_src.get("max_output_tokens"),
                            "temperature": override_src.get("temperature"),
                            "top_p": override_src.get("top_p"),
                            "base_url": base_url,
                        },
                        "hashes": {"input_images": input_hashes, "prompt": prompt_hash},
                        "metrics": {
                            "latency_ms": latency_ms,
                            "retry_count": retry_count,
                            "request_count": request_count,
                            "status": self._last_metrics.get("status"),
                            "tokens": self._last_metrics.get("tokens"),
                        },
                    }
                    fname = f"history_{ts.replace(':', '-').replace('.', '-')}.json"
                    fpath = os.path.join(self.history_dir, fname)
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(history_obj, f, ensure_ascii=False, indent=2)
                    logger.info("已写入历史记录: %s", fpath)
                except Exception as exc:
                    logger.warning("写入历史记录失败: %s", exc)

            metrics_json = json.dumps({k: v for k, v in metrics_payload.items() if v is not None}, ensure_ascii=False)
            return result_tensor, metrics_json

        except Exception as exc:
            safe_msg = str(exc).replace(api_key if api_key else "", "***")
            logger.error("Gemini Nano Banana Edit改图助手 [kuai.host]错误: %s", safe_msg)
            raise RuntimeError(f"Gemini Nano Banana Edit改图助手 [kuai.host]失败: {safe_msg}")


# 节点注册
NODE_CLASS_MAPPINGS = {"GeminiEditImage": GeminiEditImage}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiEditImage": "Gemini Nano Banana Edit改图助手 [kuai.host]"}

__version__ = "1.10"
__author__ = "ComfyUI Gemini Integration (modular Chinese-friendly)"