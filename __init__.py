"""
ComfyUI Gemini Edit Image Node
A custom node for editing images using Google Gemini 2.5 Flash Image API
"""

from .nodes import NODE_CLASS_MAPPINGS as _e_img, NODE_DISPLAY_NAME_MAPPINGS as _e_img_dn
_t2i = {}
_t2i_dn = {}
try:
    from .nodes_text import NODE_CLASS_MAPPINGS as _t2i, NODE_DISPLAY_NAME_MAPPINGS as _t2i_dn  # type: ignore
except Exception:
    pass

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(_e_img)
NODE_CLASS_MAPPINGS.update(_t2i)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(_e_img_dn)
NODE_DISPLAY_NAME_MAPPINGS.update(_t2i_dn)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "1.10"