import os

class SaveBinaryVideo:
    """
    Guarda un binario (bytes) como archivo de video MP4.
    Ideal para la salida del endpoint /videos/{id}/content de OpenAI.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "binary_data": ("BYTEBUFFER",),
                "filename": ("STRING", {"default": "sora_output.mp4"}),
                "folder": ("STRING", {"default": "output"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_video"
    CATEGORY = "Morfeo/IO"

    def save_video(self, binary_data, filename, folder):
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)

        # Si es un string tipo b'...', convertirlo a bytes reales
        if isinstance(binary_data, str) and binary_data.startswith("b'"):
            try:
                binary_data = eval(binary_data)
            except Exception:
                binary_data = binary_data.encode()

        with open(file_path, "wb") as f:
            f.write(binary_data)

        print(f"✅ Video guardado en {file_path}")
        return (file_path,)


# Registro del nodo
NODE_CLASS_MAPPINGS = {
    "SaveBinaryVideo": SaveBinaryVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveBinaryVideo": "💾 Save Binary Video"
}
