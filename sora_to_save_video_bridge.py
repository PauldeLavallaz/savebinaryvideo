# Sora → Save Video Bridge (ComfyUI custom node)
# End-to-end: poll OpenAI Videos API, download MP4, decode to VIDEO tensor for native "Save Video".
# Category: Morfeo/Sora

import os, time, json, io

from typing import Any, List

try:
    import requests
except Exception:
    requests = None

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np
import torch

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
        if s.startswith("b'") or s.startswith('b"'):
            try:
                s = eval(s)
                if isinstance(s, (bytes, bytearray)):
                    s = s.decode("utf-8", errors="ignore")
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


def _mp4_bytes_to_video_tensor(mp4_bytes: bytes, fps_override: int | None = None):
    """
    Decode MP4 bytes into ComfyUI VIDEO type:
    returns dict: {"frames": torch.FloatTensor [N,H,W,3] in 0..1, "fps": int}
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not available in this environment. Install opencv-python-headless.")

    # Save to temp to let OpenCV read it
    out_dir = get_output_directory()
    tmp_path = os.path.join(out_dir, "_tmp_sora_decode.mp4")
    with open(tmp_path, "wb") as f:
        f.write(mp4_bytes)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open temp MP4 for decoding.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0
    if fps_override:
        fps = fps_override

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if not frames:
        raise RuntimeError("No frames decoded from MP4.")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [N,H,W,3]
    tensor = torch.from_numpy(arr)  # FloatTensor [N,H,W,3]

    return {"frames": tensor, "fps": int(round(fps))}


class SoraPollDownloadToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth_header": ("STRING", {"default": "Bearer sk-..."}),
                "video_id": ("STRING", {"default": ""}),          # opcional si pasás create_response
                "create_response": ("ANY", {"default": ""}),      # ANY: bytes/str/dict del POST
                "poll_interval_sec": ("INT", {"default": 5, "min":1, "max":60}),
                "max_attempts": ("INT", {"default": 60, "min":1, "max":2000}),
                "variant": ("STRING", {"default": ""}),           # ?variant=
                "fps_override": ("INT", {"default": 0, "min":0, "max":120}), # 0 = usa FPS del archivo
            }
        }

    RETURN_TYPES = ("VIDEO","STRING","STRING")
    RETURN_NAMES = ("video","status_json","file_path")
    FUNCTION = "run"
    CATEGORY = "Morfeo/Sora"

    def run(self, auth_header, video_id, create_response, poll_interval_sec, max_attempts, variant, fps_override):
        # Derivar ID
        vid = (video_id or "").strip()
        if not vid and create_response:
            try:
                payload = _json_to_dict(create_response)
                vid = payload.get("id","").strip()
            except Exception:
                pass
        if not vid:
            raise ValueError("SoraPollDownloadToVideo: Provide 'video_id' or 'create_response' with an 'id'.")

        base = "https://api.openai.com/v1/videos"
        status_url = f"{base}/{vid}"
        headers = {"Authorization": auth_header}

        # Polling
        attempts = 0
        last_json = None
        while attempts < max_attempts:
            attempts += 1
            last_json = _http_get_json(status_url, headers)
            if last_json.get("status") == "completed":
                break
            if last_json.get("status") == "failed" or last_json.get("error"):
                # Propagar el estado para debug
                return ({"frames": torch.zeros((1,1,1,3), dtype=torch.float32), "fps": 1}, json.dumps(last_json), "")
            time.sleep(poll_interval_sec)

        if not last_json or last_json.get("status") != "completed":
            return ({"frames": torch.zeros((1,1,1,3), dtype=torch.float32), "fps": 1}, json.dumps(last_json or {}), "")

        # Download content
        content_url = f"{status_url}/content"
        if variant:
            content_url += f"?variant={variant}"
        mp4_bytes = _http_get_bytes(content_url, headers)

        # Save a copy to /output for referencia
        out_dir = get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{vid}.mp4")
        with open(file_path, "wb") as f:
            f.write(mp4_bytes)

        # Decode to VIDEO tensor
        video_tensor = _mp4_bytes_to_video_tensor(mp4_bytes, fps_override if fps_override>0 else None)

        return (video_tensor, json.dumps(last_json), file_path)


NODE_CLASS_MAPPINGS = {"SoraPollDownloadToVideo": SoraPollDownloadToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SoraPollDownloadToVideo": "🎬 Sora → Video (for Save Video)"}
