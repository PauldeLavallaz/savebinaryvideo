# 🎬 Sora → Video (for Save Video) — AUDIO + faststart + ID tolerant
import os, time, json, subprocess, re
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
        # doble parse por si viene escapado
        try:
            d = json.loads(s)
            if isinstance(d, str):
                return json.loads(d)
            return d
        except Exception:
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
        # doble parse si viene con comillas escapadas
        d = json.loads(s)
        if isinstance(d, str):
            return json.loads(d)
        return d
    if isinstance(obj, dict):
        return obj
    return json.loads(str(obj))

def _extract_video_id(source: Any) -> str:
    """Devuelve un video_id aunque venga JSON, bytes, texto con logs, etc."""
    # 1) intentar como dict JSON
    try:
        d = _json_to_dict(source)
        vid = (d.get("id") or d.get("video_id") or "").strip()
        if vid:
            return vid
    except Exception:
        pass
    # 2) raw string/bytes con regex
    s = ""
    if isinstance(source, (bytes, bytearray, memoryview)):
        s = bytes(source).decode("utf-8", errors="ignore")
    else:
        s = str(source)
    m = re.search(r"(video_[0-9a-fA-F]+)", s)
    return m.group(1) if m else ""

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

def _run_ffmpeg(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode("utf-8", errors="ignore"), p.stderr.decode("utf-8", errors="ignore")

class _SimpleVideo:
    def __init__(self, frames_rgb_uint8: List[np.ndarray], fps: int, audio_path: str | None = None):
        self._frames = frames_rgb_uint8
        self._fps = int(fps)
        self._audio_path = audio_path

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

    def _write_silent_video(self, path: str, codec: str | None, fps: float | None):
        if cv2 is None:
            raise RuntimeError("OpenCV requerido para save_to().")
        w, h = self.get_dimensions()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        c = (codec or "mp4v").lower()
        mapping = {"auto":"mp4v","mp4v":"mp4v","h264":"avc1","hevc":"hevc","mpeg4":"mp4v","vp9":"vp90","av1":"av01"}
        fourcc = cv2.VideoWriter_fourcc(*mapping.get(c, "mp4v"))
        out_fps = float(fps) if fps and fps > 0 else float(self._fps)
        vw = cv2.VideoWriter(path, fourcc, out_fps, (w, h))
        for f in self._frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return out_fps

    def save_to(self, path: str, *, format: str | None = None, codec: str | None = None,
                fps: float | None = None, **kwargs):
        temp_silent = os.path.splitext(path)[0] + ".__silent__.mp4"
        out_fps = self._write_silent_video(temp_silent, codec, fps)

        audio_src = self._audio_path
        ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")

        if audio_src and os.path.exists(audio_src):
            cmd_copy = [
                ffmpeg_bin, "-y",
                "-i", temp_silent,
                "-i", audio_src,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-c:v", "copy",
                "-c:a", "copy",
                "-movflags", "+faststart",
                path
            ]
            code, _, _ = _run_ffmpeg(cmd_copy)
            if code != 0 or (not os.path.exists(path)) or os.path.getsize(path) == 0:
                cmd_enc = [
                    ffmpeg_bin, "-y",
                    "-i", temp_silent,
                    "-i", audio_src,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "baseline",
                    "-level", "3.1",
                    "-r", str(out_fps),
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "48000",
                    "-movflags", "+faststart",
                    path
                ]
                code2, _, _ = _run_ffmpeg(cmd_enc)
                if code2 != 0 or (not os.path.exists(path)) or os.path.getsize(path) == 0:
                    os.replace(temp_silent, path)
                else:
                    try: os.remove(temp_silent)
                    except: pass
            else:
                try: os.remove(temp_silent)
                except: pass
        else:
            os.replace(temp_silent, path)

        print(f"[SimpleVideo] Saved to {path} (codec={codec}, fps={out_fps}, audio={'yes' if audio_src else 'no'})")

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
        if not vid and create_response not in ("", None):
            vid = _extract_video_id(create_response)

        if not vid:
            raise ValueError("Falta 'video_id' o 'create_response' con 'id'. Conectá POST.any → create_response, o pasá el ID en video_id.")

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
                empty = _SimpleVideo([np.zeros((1,1,3), dtype=np.uint8)], 1, audio_path=None)
                return (empty, json.dumps(last_json), "")
            time.sleep(poll_interval_sec)

        if not last_json or last_json.get("status") != "completed":
            empty = _SimpleVideo([np.zeros((1,1,3), dtype=np.uint8)], 1, audio_path=None)
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
        video_obj = _SimpleVideo(frames, fps, audio_path=file_path)
        return (video_obj, json.dumps(last_json), file_path)

NODE_CLASS_MAPPINGS = {"SoraPollDownloadToVideo": SoraPollDownloadToVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"SoraPollDownloadToVideo": "🎬 Sora → Video (for Save Video)"}
