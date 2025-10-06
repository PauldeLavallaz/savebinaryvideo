# Morfeo Nodes pack init
# Agrega en el mismo repo:
#   - save_binary_video_node.py
#   - base64_to_video_file_node.py
#   - sora_poller_node.py
#   - sora_to_save_video_bridge.py  (este archivo)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

# Save Binary Video
try:
    from .save_binary_video_node import NODE_CLASS_MAPPINGS as _m1, NODE_DISPLAY_NAME_MAPPINGS as _d1
    NODE_CLASS_MAPPINGS.update(_m1)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d1)
except Exception:
    pass

# Base64 / Bytes → Video File
try:
    from .base64_to_video_file_node import NODE_CLASS_MAPPINGS as _m2, NODE_DISPLAY_NAME_MAPPINGS as _d2
    NODE_CLASS_MAPPINGS.update(_m2)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d2)
except Exception:
    pass

# Sora Poll & Download (bytes)
try:
    from .sora_poller_node import NODE_CLASS_MAPPINGS as _m3, NODE_DISPLAY_NAME_MAPPINGS as _d3
    NODE_CLASS_MAPPINGS.update(_m3)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d3)
except Exception:
    pass

# 🎬 Sora → Video (for Save Video)
try:
    from .sora_to_save_video_bridge import NODE_CLASS_MAPPINGS as _m4, NODE_DISPLAY_NAME_MAPPINGS as _d4
    NODE_CLASS_MAPPINGS.update(_m4)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d4)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
