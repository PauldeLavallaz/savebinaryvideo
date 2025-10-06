# 🎬 Sora → Video (for Save Video)
# End-to-end: poll OpenAI Videos API, download MP4, decode to VIDEO object
# compatible with the native ComfyUI "Save Video" node.
# Category: Morfeo/Sora

import os, time, json
from typing import Any, List

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


# ---- Helpers ---------------------------------------------------------------

def _json_to_dict(obj: Any):
    """Accepts bytes/str/dict and returns a dict."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        s = bytes(obj).decode("utf-8", errors="ignore")
        return json.loads(s)
    if isinstance(obj, str):
        s = obj.strip()
        # Allow strings like b'{"id": ...}'
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
    """
    Decode MP4 bytes into a list of RGB uint8 frames + fps.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available. Install opencv-python-headless.")

    out_dir = get_output_directory()
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, "_tmp_sora_decode.mp4")
    with open(tmp_path, "wb") as f:
        f.write(mp4_bytes)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open temp MP4 for decoding.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps_override and fps_override > 0:
        fps = fps_override

    frames: List = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # uint8 RGB
        frames.append(frame_rgb)

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not frames:
        raise RuntimeError("No frames decoded from MP4.")

    return frames, int(round(fps))


# ---- VIDEO adapter for native SaveVideo -----------------------------------

class _SimpleVideo:
    """
    Minimal adapter that implements the interface expected by ComfyUI SaveVideo:
      - get_dimensions() -> (w, h)
      - get_fps() -> int
      - get_frame_count() -> int
      - frame_generator() -> yields uint8 RGB frames
    """
    def __init__(self, frames_rgb_uint8: List, fps: int):
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


# ---- Node ------------------------------------------------------------------

class SoraPollDownloadToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth_header": ("STRING", {"default": "Bearer sk-..."}),
                "video_id": ("STRING", {"default": ""}),            # optional if create_response is given
                "create_response": ("ANY", {"default": ""}),        # ANY: bytes/str/dict with {"id": ...}
                "poll_interval_sec": ("INT", {"default": 5, "min": 1, "max": 60}),
                "max_attempts": ("INT", {"default": 60, "min": 1, "max": 2000}),
                "variant": ("STRING", {"default": ""}),             # optional ?variant=
                "fps_override": ("INT", {"default": 0, "min": 0, "max": 120}),  # 0 = use file FPS
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "status_json", "file_path")
    FUNCTION = "run"
    CATEGORY = "Morfeo/Sora"

    def run(self, auth_header, video_id, create_response, poll_interval_sec, max_attempts, variant, fps_override):
        # Derive video id
        vid = (video_id or "").strip()
        if not vid and create_response:
            try:
                payload = _json_to_dict(create_response)
                vid = (payload.get("id") or "").strip()
            except Exception:
                pass
        if not vid:
            raise ValueError("SoraPollDownloadToVideo: Provide 'video_id' or pass 'create_response' from POST /v1/videos (must contain 'id').")

        base = "https://api.openai.com/v1/videos"
        status_url = f"{base}/{vid}"
        headers = {"Authorization": auth_header}

        # Polling until completed/failed
        attempts = 0
        last_json = None
        while attempts < max_attempts:
            attempts += 1
            last_json = _http_get_json(status_url, headers)
            status = last_json.get("status")
            if status == "completed":
                break
            if status == "failed" or last_json.get("error"):
                # Emit empty 1x1 video object + status for debugging
                empty = _SimpleVideo([__import__("numpy").zeros((1,1,3), dtype="uint8")], fps=1)
                return (empty, json.dumps(last_json), "")
            time.sleep(poll_interval_sec)

        if not last_json or last_json.get("status") != "completed":
            empty = _SimpleVideo([__import__("numpy").zeros((1,1,3), dtype="uint8")], fps=1)
            return (empty, json.dumps(last_json or {}), "")

        # Download content
        content_url = f"{status_url}/content"
        if variant:
            content_url += f"?variant={variant}"
        mp4_bytes = _http_get_bytes(content_url, headers)

        # Save reference MP4 to /output
        out_dir = get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{vid}.mp4")
        with open(file_path, "wb") as f:
            f.write(mp4_bytes)

        # Decode to frames + fps and wrap into VIDEO object
        frames, fps = _mp4_bytes_to_frames_list(mp4_bytes, fps_override if fps_override > 0 else None)
        video_obj = _SimpleVideo(frames, fps)

        return (video_obj, json.dumps(last_json), file_path)


NODE_CLASS_MAPPINGS = {"SoraPollDownloadToVideo": SoraPollDownloadToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SoraPollDownloadToVideo": "🎬 Sora → Video (for Save Video)"}
