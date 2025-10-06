import os
from typing import Any

try:
    from folder_paths import get_output_directory
except Exception:
    def get_output_directory():
        return os.path.join(os.getcwd(), "output")


class SaveBinaryVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_data": ("ANY",),
                "filename": ("STRING", {"default": "sora_output.mp4"}),
                "subfolder": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_video"
    CATEGORY = "Morfeo/IO"

    def _coerce_to_bytes(self, data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)

        if isinstance(data, dict):
            for key in ("bytes", "data", "content", "binary", "body"):
                if key in data:
                    return self._coerce_to_bytes(data[key])
            data = str(data)

        if isinstance(data, str):
            s = data.strip()
            if s.startswith("b'") or s.startswith('b"'):
                try:
                    return eval(s)
                except Exception:
                    return s.encode("utf-8", errors="ignore")
            return s.encode("utf-8", errors="ignore")

        return bytes(data)

    def save_video(self, binary_data, filename, subfolder, overwrite):
        base_out = get_output_directory()
        out_dir = os.path.join(base_out, subfolder) if subfolder else base_out
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.splitext(filename)[1]:
            filename += ".mp4"

        file_path = os.path.join(out_dir, filename)

        if (not overwrite) and os.path.exists(file_path):
            root, ext = os.path.splitext(file_path)
            i = 1
            while os.path.exists(f"{root}_{i}{ext}"):
                i += 1
            file_path = f"{root}_{i}{ext}"

        payload = self._coerce_to_bytes(binary_data)

        with open(file_path, "wb") as f:
            f.write(payload)

        print(f"[SaveBinaryVideo] Saved: {file_path}")
        return (file_path,)


NODE_CLASS_MAPPINGS = {"SaveBinaryVideo": SaveBinaryVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveBinaryVideo": "💾 Save Binary Video"}
