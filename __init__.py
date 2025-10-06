# Morfeo Nodes - Unified init for all nodes in this repo

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

# Safe imports for each node
try:
    from .save_binary_video_node import NODE_CLASS_MAPPINGS as _m1, NODE_DISPLAY_NAME_MAPPINGS as _d1
    NODE_CLASS_MAPPINGS.update(_m1)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d1)
except Exception:
    pass

try:
    from .base64_to_video_file_node import NODE_CLASS_MAPPINGS as _m2, NODE_DISPLAY_NAME_MAPPINGS as _d2
    NODE_CLASS_MAPPINGS.update(_m2)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d2)
except Exception:
    pass

try:
    from .sora_poller_node import NODE_CLASS_MAPPINGS as _m3, NODE_DISPLAY_NAME_MAPPINGS as _d3
    NODE_CLASS_MAPPINGS.update(_m3)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d3)
except Exception:
    pass

try:
    from .sora_to_save_video_bridge import NODE_CLASS_MAPPINGS as _m4, NODE_DISPLAY_NAME_MAPPINGS as _d4
    NODE_CLASS_MAPPINGS.update(_m4)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d4)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
