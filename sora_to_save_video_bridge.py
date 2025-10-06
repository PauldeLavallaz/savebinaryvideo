# 🎬 Sora → Video (for Save Video)  — FINAL
import os, time, json
from typing import Any, List
import numpy as np

try:
    import requests
except Exception:
    requests = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from folder_paths import get_output_directory
except Exception:
    def get_output_directory():
        return os.path.join(os.getcwd(), "output")

def _json_to_dict(obj: Any):
    if isinstance(obj, (bytes, bytearray, memoryview)):
        s = bytes(obj).decode("utf-8", errors="ignore")
        return json.loads(s)
    if isinstance(obj, str):
        s = obj.strip()
        if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
            try:
                s_eval = eval(s)
                if isinstance(s_eval, (bytes, bytearray)):
                    s = s_eval.decode("utf-8", errors="ignore")
            except Exception:
                pass
        return json.loads(s)
    if isinstance(obj, dict):
        return obj
    return json.loads(str(obj))

def _http_get_json(url, headers):
    if requests is None:
        import urllib.request
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

def _http_get_bytes(url, headers):
    if requests is None:
        import urllib.request
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    r = requests.get(url, headers=headers, timeout=300, stream=True)
    r.raise_for_status()
    return r.content

def _mp4_bytes_to_frames_list(mp4_bytes: bytes, fps_override: int | None = None):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no disponible. Instalar opencv-python-headless.")
    out_dir = get_output_directory()
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "_tmp_sora_decode.mp4")
    with open(tmp_path, "wb") as f:
        f.write(mp4_bytes)
    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir MP4 temporal.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps_override and fps_override > 0:
        fps = fps_override
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    try: os.remove(tmp_path)
    except: pass
    if not frames:
        raise RuntimeError("Sin frames decodificados del MP4.")
    return frames, int(round(fps))

class _SimpleVideo:
    def __init__(self, frames_rgb_uint8: List[np.ndarray], fps: int):
        self._frames = frames_rgb_uint8
        self._fps = int(fps)

    def get_dimensions(self):
        h, w, _ = self._frames[0].shape
        return (w, h)

    def get_fps(self):
        return self._fps

    def get_frame_count(self):
        return len(self._frames)

    def frame_generator(self):
        for f in self._frames:
            yield f

    # Acepta kwargs que pasa SaveVideo: format, codec, fps, etc.
    def save_to(self, path: str, *, format: str | None = None, codec: str | None = None,
                fps: float | None = None, **kwargs):
        if cv2 is None:
            raise RuntimeError("OpenCV requerido para save_to().")
        w, h = self.get_dimensions()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Mapeo básico de codec -> fourcc para OpenCV
        # (SaveVideo puede pasar 'auto', 'h264', 'hevc', 'mp4v', etc.)
        codec = (codec or "mp4v").lower()
        mapping = {
            "auto": "mp4v",
            "mp4v": "mp4v",
            "h264": "avc1",   # depende de build ffmpeg/opencv
            "hevc": "hevc",   # idem
            "mpeg4": "mp4v",
            "vp9": "vp90",
            "av1": "av01",
        }
        fourcc_str = mapping.get(codec, "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        out_fps = float(fps) if fps and fps > 0 else float(self._fps)
        vw = cv2.VideoWriter(path, fourcc, out_fps, (w, h))
        for f in self._frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[SimpleVideo] Saved to {path} (format={format}, codec={codec}, fps={out_fps})")

class SoraPollDownloadToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth_header": ("STRING", {"default": "Bearer sk-..."}),
                "video_id": ("STRING", {"default": ""}),
                "create_response": ("ANY", {"default": ""}),
                "poll_interval_sec": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_attempts": ("INT", {"default": 60, "min": 1, "max": 2000}),
                "variant": ("STRING", {"default": ""}),
                "fps_override": ("INT", {"default": 0, "min": 0, "max": 120}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "status_json", "file_path")
    FUNCTION = "run"
    CATEGORY = "Morfeo/Sora"

    def run(self, auth_header, video_id, create_response, poll_interval_sec, max_attempts, variant, fps_override):
        vid = (video_id or "").strip()
        if not vid and create_response:
            try:
                payload = _json_to_dict(create_response)
                vid = (payload.get("id") or "").strip()
            except Exception:
                pass
        if not vid:
            raise ValueError("Falta 'video_id' o 'create_response' con 'id'.")

        base = "https://api.openai.com/v1/videos"
        status_url = f"{base}/{vid}"
        headers = {"Authorization": auth_header}

        attempts = 0
        last_json = None
        while attempts < max_attempts:
            attempts += 1
            last_json = _http_get_json(status_url, headers)
            s = last_json.get("status")
            if s == "completed":
                break
            if s == "failed" or last_json.get("error"):
                empty = _SimpleVideo([np.zeros((1,1,3), dtype=np.uint8)], 1)
                return (empty, json.dumps(last_json), "")
            time.sleep(poll_interval_sec)

        if not last_json or last_json.get("status") != "completed":
            empty = _SimpleVideo([np.zeros((1,1,3), dtype=np.uint8)], 1)
            return (empty, json.dumps(last_json or {}), "")

        content_url = f"{status_url}/content"
        if variant:
            content_url += f"?variant={variant}"
        mp4_bytes = _http_get_bytes(content_url, headers)

        out_dir = get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{vid}.mp4")
        with open(file_path, "wb") as f:
            f.write(mp4_bytes)

        frames, fps = _mp4_bytes_to_frames_list(mp4_bytes, fps_override if fps_override > 0 else None)
        video_obj = _SimpleVideo(frames, fps)
        return (video_obj, json.dumps(last_json), file_path)

NODE_CLASS_MAPPINGS = {"SoraPollDownloadToVideo": SoraPollDownloadToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SoraPollDownloadToVideo": "🎬 Sora → Video (for Save Video)"}
