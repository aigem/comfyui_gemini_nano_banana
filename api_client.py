import time
import random
from typing import Any, Dict, List, Optional, cast
import requests

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

    start_time = time.time()
    attempts = 0
    last_status: Optional[int] = None
    if metrics_ref is not None:
        metrics_ref.update({"retry_count": 0, "latency_ms": None, "status": None, "request_count": 0})

    while True:
        try:
            attempts += 1
            resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout)
            last_status = resp.status_code

            if resp.status_code == 200:
                if metrics_ref is not None:
                    latency_ms = int((time.time() - start_time) * 1000)
                    metrics_ref.update({
                        "retry_count": attempts - 1,
                        "latency_ms": latency_ms,
                        "status": resp.status_code,
                        "request_count": attempts,
                    })
                return resp.json()

            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                if attempts - 1 >= max_retries:
                    detail = resp.text[:200] if resp.text else ""
                    raise RuntimeError(f"API request failed with status {resp.status_code}: {detail}")
                retry_after = 0.0
                try:
                    ra_header = resp.headers.get("Retry-After")
                    if ra_header:
                        retry_after = float(ra_header)
                except Exception:
                    retry_after = 0.0
                base_delay = (2 ** (attempts - 1)) * 0.5
                jitter = random.uniform(0, 0.5)
                delay = max(retry_after, base_delay + jitter)
                time.sleep(delay)
                continue

            msg = f"API request failed with status {resp.status_code}"
            if resp.text:
                msg += f": {resp.text[:200]}"
            raise RuntimeError(msg)

        except requests.Timeout as exc:
            if attempts - 1 >= max_retries:
                raise RuntimeError(f"Request timed out after {timeout} seconds.") from exc
            base_delay = (2 ** (attempts - 1)) * 0.5
            jitter = random.uniform(0, 0.5)
            time.sleep(base_delay + jitter)
            continue
        except requests.RequestException as exc:
            if attempts - 1 >= max_retries:
                raise RuntimeError(f"Network error occurred: {exc}") from exc
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