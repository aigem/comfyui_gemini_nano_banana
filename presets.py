import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .constants import INSTRUCTION_PRESET_DIR, SYSTEM_PROMPT_DIR

logger = logging.getLogger(__name__)

def instruction_choices() -> List[str]:
    choices: List[str] = ['(none)']
    try:
        preset_root = INSTRUCTION_PRESET_DIR
        if preset_root.exists():
            for preset_path in sorted(preset_root.rglob('*.json')):
                try:
                    rel_path = preset_path.relative_to(preset_root)
                except ValueError:
                    continue
                choices.append(rel_path.as_posix())
    except Exception as exc:
        logger.warning('枚举指令预设失败: %s', exc)
    return choices

def system_prompt_choices() -> List[str]:
    choices: List[str] = ['(none)']
    try:
        prompt_root = SYSTEM_PROMPT_DIR
        if prompt_root.exists():
            for prompt_path in sorted(prompt_root.rglob('*.md')):
                try:
                    rel_path = prompt_path.relative_to(prompt_root)
                except ValueError:
                    continue
                choices.append(rel_path.as_posix())
    except Exception as exc:
        logger.warning('枚举系统提示失败: %s', exc)
    return choices

def load_instruction_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    if not preset_name or preset_name.strip() in {'(none)', 'none'}:
        return None
    preset_root = INSTRUCTION_PRESET_DIR.resolve()
    candidate_path = (INSTRUCTION_PRESET_DIR / preset_name).resolve()
    try:
        candidate_path.relative_to(preset_root)
    except ValueError:
        logger.warning('拒绝加载预设(越界路径): %s', preset_name)
        return None
    if not candidate_path.is_file():
        logger.warning('未找到指令预设: %s', candidate_path)
        return None
    try:
        with candidate_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning('加载指令预设失败 %s: %s', candidate_path, exc)
        return None

def load_system_prompt(prompt_name: str) -> Optional[str]:
    if not prompt_name or prompt_name.strip() in {'(none)', 'none'}:
        return None
    prompt_root = SYSTEM_PROMPT_DIR.resolve()
    candidate_path = (SYSTEM_PROMPT_DIR / prompt_name).resolve()
    try:
        candidate_path.relative_to(prompt_root)
    except ValueError:
        logger.warning('拒绝加载系统提示(越界路径): %s', prompt_name)
        return None
    if not candidate_path.is_file():
        logger.warning('未找到系统提示: %s', candidate_path)
        return None
    try:
        with candidate_path.open('r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as exc:
        logger.warning('加载系统提示失败 %s: %s', candidate_path, exc)
        return None

def apply_prompt_preset(base_prompt: str, preset: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not isinstance(preset, dict):
        return base_prompt.strip(), []
    prompt_core = base_prompt.strip()
    template = preset.get('prompt_template')
    if isinstance(template, str) and '{prompt}' in template:
        prompt_core = template.replace('{prompt}', prompt_core)
    segments: List[str] = []
    prefix = preset.get('prompt_prefix')
    if isinstance(prefix, str) and prefix.strip():
        segments.append(prefix.strip())
    if prompt_core:
        segments.append(prompt_core)
    directives = preset.get('directives')
    if isinstance(directives, list):
        for item in directives:
            if isinstance(item, str) and item.strip():
                segments.append(item.strip())
    suffix = preset.get('prompt_suffix')
    if isinstance(suffix, str) and suffix.strip():
        segments.append(suffix.strip())
    final_prompt = "\n\n".join(seg for seg in segments if isinstance(seg, str) and seg)
    extra_parts: List[str] = []
    extra = preset.get('extra_parts')
    if isinstance(extra, list):
        for part in extra:
            if isinstance(part, str) and part.strip():
                extra_parts.append(part.strip())
    return (final_prompt or prompt_core, extra_parts)