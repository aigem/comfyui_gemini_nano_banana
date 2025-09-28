from pathlib import Path

def load_node_description(name: str) -> str:
    """
    从 descriptions/{name}.md 加载节点说明。
    若文件不存在或读取失败，返回空字符串，以避免影响节点加载。
    """
    try:
        base = Path(__file__).resolve().parent
        desc_dir = base / "descriptions"
        md_path = desc_dir / f"{name}.md"
        if md_path.is_file():
            return md_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""