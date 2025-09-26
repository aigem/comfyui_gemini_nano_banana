from pathlib import Path

# 常量与默认值
DEFAULT_SYSTEM_PROMPT = ""
INSTRUCTION_PRESET_DIR = Path(__file__).resolve().parent / "instruction_presets"
SYSTEM_PROMPT_DIR = INSTRUCTION_PRESET_DIR / "system_prompt"

DEFAULT_MODEL = "gemini-2.5-flash-image-preview"
DEFAULT_BASE_URL = "https://apis.kuai.host/"

# 历史记录相关环境变量键名
ENV_MAX_RETRIES = "GEMINI_MAX_RETRIES"
ENV_ENABLE_HISTORY = "GEMINI_HISTORY"
ENV_HISTORY_DIR = "GEMINI_HISTORY_DIR"
ENV_MAX_PROMPT_CHARS = "GEMINI_MAX_PROMPT_CHARS"
ENV_FORBIDDEN_TAGS = "GEMINI_FORBIDDEN_TAGS"