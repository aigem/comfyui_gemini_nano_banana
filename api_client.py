import time
import random
from typing import Any, Dict, List, Optional, cast

# 惰性安全导入 requests，避免在 ComfyUI 导入阶段直接失败
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


def normalize_generation_config(raw_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """规范化生成配置键到 Gemini API 期望的命名"""
    if not isinstance(raw_config, dict):
        return {}
    key_map = {
        "max_output_tokens": "maxOutputTokens",
        "max_tokens": "maxOutputTokens",
        "top_p": "topP",
        "stop_sequences": "stopSequences",
        "response_mime_type": "responseMimeType",
        "candidate_count": "candidateCount",
    }
    normalized: Dict[str, Any] = {}
    for key, value in raw_config.items():
        if value is None:
            continue
        k = str(key)
        nk = key_map.get(k) or key_map.get(k.replace("-", "_")) or key_map.get(k.replace("-", "_").lower()) or k
        normalized[nk] = value
    return normalized


def _ensure_requests():
    if requests is None:
        raise RuntimeError("缺少依赖 requests。请在 ComfyUI 环境安装：pip install requests")


def _post_with_retry(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    metrics_ref: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """统一重试与指标记录"""
    _ensure_requests()
    start_time = time.time()
    attempts = 0
    last_status: Optional[int] = None
    if metrics_ref is not None:
        metrics_ref.update({"retry_count": 0, "latency_ms": None, "status": None, "request_count": 0})

    while True:
        try:
            attempts += 1
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)  # type: ignore
            last_status = resp.status_code  # type: ignore

            if resp.status_code == 200:  # type: ignore
                if metrics_ref is not None:
                    latency_ms = int((time.time() - start_time) * 1000)
                    metrics_ref.update(
                        {
                            "retry_count": attempts - 1,
                            "latency_ms": latency_ms,
                            "status": resp.status_code,  # type: ignore
                            "request_count": attempts,
                        }
                    )
                return resp.json()  # type: ignore

            if resp.status_code == 429 or (500 <= resp.status_code < 600):  # type: ignore
                if attempts - 1 >= max_retries:
                    detail = resp.text[:200] if getattr(resp, "text", None) else ""  # type: ignore
                    raise RuntimeError(f"API request failed with status {resp.status_code}: {detail}")  # type: ignore
                retry_after = 0.0
                try:
                    ra_header = resp.headers.get("Retry-After")  # type: ignore
                    if ra_header:
                        retry_after = float(ra_header)
                except Exception:
                    retry_after = 0.0
                base_delay = (2 ** (attempts - 1)) * 0.5
                jitter = random.uniform(0, 0.5)
                delay = max(retry_after, base_delay + jitter)
                time.sleep(delay)
                continue

            msg = f"API request failed with status {resp.status_code}"  # type: ignore
            if getattr(resp, "text", None):  # type: ignore
                msg += f": {resp.text[:200]}"  # type: ignore
            raise RuntimeError(msg)

        except Exception as exc:
            # 细分 requests 异常类型：Timeout / RequestException
            is_timeout = getattr(type(exc), "__name__", "") == "Timeout"
            is_req_exc = getattr(type(exc), "__name__", "") in {"RequestException", "ConnectionError", "HTTPError"}
            if attempts - 1 >= max_retries or not (is_timeout or is_req_exc):
                # 超过重试或非网络超时错误，直接抛出
                raise
            base_delay = (2 ** (attempts - 1)) * 0.5
            jitter = random.uniform(0, 0.5)
            time.sleep(base_delay + jitter)
            continue
        finally:
            if metrics_ref is not None:
                metrics_ref["latency_ms"] = int((time.time() - start_time) * 1000)
                if last_status is not None:
                    metrics_ref["status"] = last_status
                metrics_ref["retry_count"] = max(metrics_ref.get("retry_count", 0), attempts - 1)
                metrics_ref["request_count"] = max(metrics_ref.get("request_count", 0), attempts)


def call_gemini_api_text(
    *,
    prompt: str,
    api_key: str,
    timeout: int,
    model: str,
    system_prompt: Optional[str] = None,
    extra_text_parts: Optional[List[str]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    max_retries: int = 2,
    metrics_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    文生图调用：不需要上传任何图像，仅以文本触发图像生成。
    与 call_gemini_api 一致的重试/指标语义。
    """
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    text_parts: List[str] = []
    if isinstance(prompt, str) and prompt.strip():
        text_parts.append(prompt.strip())
    if extra_text_parts:
        for t in extra_text_parts:
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
    if not text_parts:
        raise ValueError("Prompt is required for Gemini API call")

    # 仅文本 parts
    user_parts: List[Dict[str, Any]] = [{"text": tp} for tp in text_parts]
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": user_parts}],
        "generationConfig": {"candidateCount": 1},
    }

    if isinstance(system_prompt, str) and system_prompt.strip():
        payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_prompt.strip()}]}

    if isinstance(generation_config, dict) and generation_config:
        gen_cfg = payload["generationConfig"]
        gen_cfg.update(generation_config)

    endpoint_url = f"{base}/v1beta/models/{model}:generateContent"

    return _post_with_retry(
        url=endpoint_url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
        metrics_ref=metrics_ref,
    )


def call_gemini_api(
    *,
    images_base64: List[str],
    prompt: str,
    api_key: str,
    timeout: int,
    model: str,
    system_prompt: Optional[str] = None,
    extra_text_parts: Optional[List[str]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    max_retries: int = 2,
    metrics_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    以可配置 base_url 调用 Gemini 兼容 API。将运行指标写入 metrics_ref：latency_ms、retry_count、status、request_count。
    """
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    text_parts: List[str] = []
    if isinstance(prompt, str) and prompt.strip():
        text_parts.append(prompt.strip())
    if extra_text_parts:
        for t in extra_text_parts:
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
    if not text_parts:
        raise ValueError("Prompt is required for Gemini API call")
    if not images_base64:
        raise ValueError("At least one image is required for Gemini API call")

    parts: List[Dict[str, Any]] = []
    for encoded in images_base64:
        s = encoded.strip()
        if s:
            parts.append({"inline_data": {"mime_type": "image/png", "data": s}})
    if not parts:
        raise ValueError("No valid images provided for Gemini API call")

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"candidateCount": 1},
    }

    contents_list = cast(List[Dict[str, Any]], payload["contents"])
    user_parts = cast(List[Dict[str, Any]], contents_list[0]["parts"])
    for tp in text_parts:
        user_parts.append({"text": tp})

    if isinstance(system_prompt, str) and system_prompt.strip():
        payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_prompt.strip()}]}

    if isinstance(generation_config, dict) and generation_config:
        gen_cfg = cast(Dict[str, Any], payload["generationConfig"])
        gen_cfg.update(generation_config)

    endpoint_url = f"{base}/v1beta/models/{model}:generateContent"

    return _post_with_retry(
        url=endpoint_url,
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
        metrics_ref=metrics_ref,
    )