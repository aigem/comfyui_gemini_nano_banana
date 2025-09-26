from typing import Any, Dict, List, Optional

def extract_usage_metadata(usage_meta: Any) -> Optional[Dict[str, int]]:
    """解析 usage 元数据为标准化的 token 指标。"""
    if not isinstance(usage_meta, dict):
        return None
    tokens: Dict[str, int] = {}
    for key, value in usage_meta.items():
        if not isinstance(value, (int, float)):
            continue
        nk = str(key).replace("_", "").replace("-", "").lower()
        iv = int(value)
        if "prompt" in nk and "image" not in nk:
            tokens["prompt_tokens"] = iv
        elif "candidat" in nk or "response" in nk:
            tokens["response_tokens"] = iv
        elif "total" in nk:
            tokens["total_tokens"] = iv
        elif "image" in nk:
            tokens["image_tokens"] = iv
        elif "billable" in nk:
            tokens["billable_tokens"] = iv
    return tokens or None

def extract_image_from_response(response: Dict[str, Any]) -> str:
    """
    从 Gemini API 响应中提取 base64 图像。
    优先 candidates[0].content.parts 内的 inline_data/inlineData.data。
    """
    if "candidates" not in response or not response["candidates"]:
        pf = response.get("promptFeedback") or response.get("prompt_feedback")
        if isinstance(pf, dict) and pf.get("blockReason"):
            raise ValueError(f"Response blocked by safety: {pf.get('blockReason')}")
        raise ValueError("No candidates in response")

    cand = response["candidates"][0]
    finish_reason = cand.get("finishReason") or cand.get("finish_reason")
    if "content" not in cand:
        raise ValueError("No content in candidate")
    content = cand["content"]
    if "parts" not in content or not content["parts"]:
        raise ValueError("No parts in content")

    text_snippets: List[str] = []
    for part in content["parts"]:
        if isinstance(part, dict):
            txt = part.get("text")
            if isinstance(txt, str) and txt.strip():
                text_snippets.append(txt.strip())
            inline_obj = None
            if "inline_data" in part and isinstance(part["inline_data"], dict):
                inline_obj = part["inline_data"]
            elif "inlineData" in part and isinstance(part["inlineData"], dict):
                inline_obj = part["inlineData"]
            if inline_obj and "data" in inline_obj:
                return inline_obj["data"]

    diags = []
    if finish_reason:
        diags.append(f"finishReason={finish_reason}")
    pf = response.get("promptFeedback") or response.get("prompt_feedback")
    if isinstance(pf, dict):
        br = pf.get("blockReason") or pf.get("block_reason")
        if br:
            diags.append(f"blockReason={br}")
    if text_snippets:
        diags.append(f"model_text='{text_snippets[0][:160]}'")
    msg = "No image part found in response."
    if diags:
        msg += " Details: " + "; ".join(diags)
    msg += " Try refining your prompt to explicitly request an edited image output only (e.g., 'Return only the edited image as PNG.')."
    raise ValueError(msg)