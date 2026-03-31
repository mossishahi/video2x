#!/usr/bin/env python3
"""Video2X for Mac — native macOS GUI for video upscaling and frame interpolation."""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import webview
from flask import Flask, jsonify, request

from remote_gpu import RemoteGPU, load_gpu_configs, save_gpu_configs, DEFAULT_CONFIG_PATH

app = Flask(__name__, static_folder=None)

HOME = Path.home()
V2X_DIR = HOME / "Desktop" / "code" / "video2x-build"
V2X_BIN = V2X_DIR / "build" / "video2x-install" / "bin" / "video2x"
V2X_ENV = {
    **os.environ,
    "VK_ICD_FILENAMES": "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
    "DYLD_LIBRARY_PATH": f"{V2X_DIR}/build:{V2X_DIR}/build/video2x-install/lib:/opt/homebrew/lib",
}

state = {
    "status": "idle",
    "frame": 0,
    "total": 0,
    "fps": 0.0,
    "elapsed": "00:00:00",
    "remaining": "--:--:--",
    "progress": 0.0,
    "log": "",
    "devices": [],
    "gpu_util": 0,
    "gpu_render": 0,
    "gpu_tiler": 0,
}
process_handle = None
gpu_poll_active = False
remote_gpu = RemoteGPU(state)
PROGRESS_RE = re.compile(
    r"frame=(\d+)/(\d+)\s+\(([^)]+)\);\s+fps=([^;]+);\s+elapsed=([^;]+);\s+remaining=(.+)"
)


def poll_gpu_utilization():
    """Poll Apple Silicon GPU utilization via ioreg while processing."""
    global gpu_poll_active
    gpu_poll_active = True
    while gpu_poll_active and state["status"] in ("running", "cancelling"):
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=3
            )
            text = result.stdout
            for key, field in [("gpu_util", "Device Utilization"), ("gpu_render", "Renderer Utilization"), ("gpu_tiler", "Tiler Utilization")]:
                m = re.search(rf'"{field} %"=(\d+)', text)
                if m:
                    state[key] = int(m.group(1))
        except Exception:
            pass
        time.sleep(1.5)
    state["gpu_util"] = 0
    state["gpu_render"] = 0
    state["gpu_tiler"] = 0
    gpu_poll_active = False


def detect_devices():
    try:
        result = subprocess.run(
            [str(V2X_BIN), "--list-devices"],
            capture_output=True, text=True, env=V2X_ENV, timeout=10
        )
        devices = []
        lines = result.stdout.strip().split("\n")
        idx, name, dtype = -1, "", ""
        for line in lines:
            m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
            if m:
                if idx >= 0:
                    devices.append({"id": idx, "name": name, "type": dtype})
                idx, name, dtype = int(m.group(1)), m.group(2), ""
            elif line.strip().startswith("Type:"):
                dtype = line.strip()[5:].strip()
        if idx >= 0:
            devices.append({"id": idx, "name": name, "type": dtype})
        state["devices"] = devices
    except Exception as e:
        state["log"] += f"Device detection failed: {e}\n"


def probe_video(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=10, env=V2X_ENV
        )
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return {
                    "width": s.get("width", 0),
                    "height": s.get("height", 0),
                    "fps": s.get("r_frame_rate", "?"),
                    "codec": s.get("codec_name", "?"),
                    "duration": round(float(fmt.get("duration", 0)), 1),
                    "size_mb": round(int(fmt.get("size", 0)) / 1024 / 1024, 1),
                    "frames": int(s.get("nb_frames", 0)),
                }
    except Exception:
        pass
    return {}


def run_processing(args):
    global process_handle
    state["status"] = "running"
    state["frame"] = 0
    state["total"] = 0
    state["fps"] = 0.0
    state["progress"] = 0.0
    state["elapsed"] = "00:00:00"
    state["remaining"] = "--:--:--"
    state["log"] = f"$ video2x {' '.join(args)}\n"

    # video2x looks for models/ relative to cwd
    models_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if not (models_dir / "models").is_dir():
        models_dir = V2X_DIR / "build" / "video2x-install" / "share" / "video2x"

    try:
        proc = subprocess.Popen(
            [str(V2X_BIN)] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=V2X_ENV, bufsize=1, universal_newlines=True,
            cwd=str(models_dir),
        )
        process_handle = proc

        for line in proc.stdout:
            clean = line.replace("\x1b[K", "").replace("\r", "").strip()
            if not clean:
                continue
            m = PROGRESS_RE.search(clean)
            if m:
                state["frame"] = int(m.group(1))
                state["total"] = int(m.group(2))
                state["fps"] = float(m.group(4))
                state["elapsed"] = m.group(5)
                state["remaining"] = m.group(6).strip()
                if state["total"] > 0:
                    state["progress"] = state["frame"] / state["total"]
            else:
                state["log"] += clean + "\n"
                if len(state["log"]) > 50000:
                    state["log"] = state["log"][-40000:]

        proc.wait()
        if proc.returncode == 0:
            state["status"] = "finished"
            state["progress"] = 1.0
            state["log"] += "\nProcessing completed successfully!\n"
        else:
            state["status"] = "failed"
            state["log"] += f"\nProcess exited with code {proc.returncode}\n"
    except Exception as e:
        state["status"] = "failed"
        state["log"] += f"\nError: {e}\n"
    finally:
        process_handle = None


class Api:
    """Exposed to JavaScript via pywebview's JS bridge."""

    def browse_input(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video Files (*.mp4;*.mkv;*.avi;*.mov;*.webm)",),
        )
        if result and len(result) > 0:
            path = result[0]
            info = probe_video(path)
            return json.dumps({"path": path, "info": info})
        return json.dumps({"path": "", "info": {}})

    def browse_output(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename="output.mp4",
            file_types=("MP4 Video (*.mp4)",),
        )
        if result:
            return json.dumps({"path": result})
        return json.dumps({"path": ""})

    def browse_config(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("YAML Config (*.yaml;*.yml)",),
        )
        if result and len(result) > 0:
            path = result[0]
            configs = load_gpu_configs(path)
            return json.dumps({"path": path, "configs": configs})
        return json.dumps({"path": "", "configs": []})


js_api = Api()


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/state")
def get_state():
    return jsonify(state)


@app.route("/api/devices")
def get_devices():
    return jsonify(state["devices"])


@app.route("/api/probe", methods=["POST"])
def probe():
    path = request.json.get("path", "")
    info = probe_video(path)
    return jsonify(info)


@app.route("/api/start", methods=["POST"])
def start():
    if state["status"] == "running":
        return jsonify({"error": "Already running"}), 400

    data = request.json
    args = [
        "-i", data["input"],
        "-o", data["output"],
        "-p", data["processor"],
        "-d", str(data.get("device", 0)),
        "-c", data.get("codec", "libx264"),
        "--log-level", "info",
    ]

    proc = data["processor"]
    if proc in ("realesrgan", "realcugan"):
        args += ["-s", str(data.get("scale", 4))]
        if proc == "realesrgan":
            args += ["--realesrgan-model", data.get("model", "realesr-animevideov3")]
        else:
            args += ["--realcugan-model", data.get("model", "models-se")]
    elif proc == "libplacebo":
        args += ["-w", str(data.get("width", 3840)), "-h", str(data.get("height", 2160))]
        args += ["--libplacebo-shader", data.get("model", "anime4k-v4-a")]
    elif proc == "rife":
        args += ["-m", str(data.get("multiplier", 2))]
        args += ["--rife-model", data.get("model", "rife-v4.6")]

    threading.Thread(target=run_processing, args=(args,), daemon=True).start()
    threading.Thread(target=poll_gpu_utilization, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/cancel", methods=["POST"])
def cancel():
    global process_handle
    global gpu_poll_active
    if process_handle:
        gpu_poll_active = False
        state["status"] = "cancelling"
        state["log"] += "\nCancelling... please wait.\n"
        process_handle.terminate()
        try:
            process_handle.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process_handle.kill()
        state["status"] = "idle"
        state["log"] += "Cancelled.\n"
        process_handle = None
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    state["status"] = "idle"
    return jsonify({"ok": True})


@app.route("/api/remote/configs")
def remote_configs():
    configs = load_gpu_configs()
    return jsonify({"configs": configs})


@app.route("/api/remote/configs/save", methods=["POST"])
def remote_config_save():
    gpu = request.json
    configs = load_gpu_configs()
    configs.append(gpu)
    save_gpu_configs(configs)
    return jsonify({"ok": True, "configs": load_gpu_configs()})


@app.route("/api/remote/configs/delete", methods=["POST"])
def remote_config_delete():
    idx = request.json.get("index", -1)
    configs = load_gpu_configs()
    if 0 <= idx < len(configs):
        configs.pop(idx)
        save_gpu_configs(configs)
    return jsonify({"ok": True, "configs": load_gpu_configs()})


@app.route("/api/remote/connect", methods=["POST"])
def remote_connect():
    data = request.json
    remote_gpu.slurm_opts = data.get("slurm", {})
    scheduler = data.get("scheduler", "auto")
    ok = remote_gpu.connect(
        host=data["host"],
        username=data.get("user") or data.get("username", ""),
        password=data.get("password"),
        key_path=data.get("key_path"),
        port=int(data.get("port", 22)),
        proxy=data.get("proxy"),
        auth=data.get("auth", "key"),
    )
    if ok and scheduler == "slurm":
        state["remote_has_slurm"] = True
    elif ok and scheduler == "direct":
        state["remote_has_slurm"] = False
    return jsonify({"ok": ok, "gpu": state.get("remote_gpu_name", ""), "slurm": state.get("remote_has_slurm", False)})


@app.route("/api/remote/disconnect", methods=["POST"])
def remote_disconnect():
    remote_gpu.disconnect()
    state["log"] += "Disconnected from remote.\n"
    return jsonify({"ok": True})


@app.route("/api/remote/start", methods=["POST"])
def remote_start():
    if state["status"] == "running":
        return jsonify({"error": "Already running"}), 400

    data = request.json
    input_path = data["input"]
    use_slurm = data.get("use_slurm", False)

    proc_type = data["processor"]
    args_parts = [f"-p {proc_type}", "-d 0", f"-c {data.get('codec', 'libx264')}", "--log-level info"]

    if proc_type in ("realesrgan", "realcugan"):
        args_parts.append(f"-s {data.get('scale', 4)}")
        if proc_type == "realesrgan":
            args_parts.append(f"--realesrgan-model {data.get('model', 'realesr-animevideov3')}")
        else:
            args_parts.append(f"--realcugan-model {data.get('model', 'models-se')}")
    elif proc_type == "libplacebo":
        args_parts.append(f"-w {data.get('width', 3840)} -h {data.get('height', 2160)}")
        args_parts.append(f"--libplacebo-shader {data.get('model', 'anime4k-v4-a')}")
    elif proc_type == "rife":
        args_parts.append(f"-m {data.get('multiplier', 2)}")
        args_parts.append(f"--rife-model {data.get('model', 'rife-v4.6')}")

    args_str = " ".join(args_parts)

    def run_remote():
        state["status"] = "running"
        state["frame"] = 0
        state["total"] = 0
        state["fps"] = 0.0
        state["progress"] = 0.0
        state["elapsed"] = "00:00:00"
        state["remaining"] = "--:--:--"
        state["log"] = ""
        remote_gpu._cancel = False

        try:
            if not remote_gpu.install_video2x():
                state["status"] = "failed"
                return

            remote_path = remote_gpu.upload_video(input_path)
            state["log"] += "\n"

            result_path = remote_gpu.process_video(remote_path, args_str, use_slurm=use_slurm)
            if not result_path:
                state["status"] = "failed"
                return

            output_dir = os.path.dirname(input_path)
            local_result = remote_gpu.download_result(result_path, output_dir)

            state["status"] = "finished"
            state["progress"] = 1.0
            state["log"] += f"\nDone! Saved to: {local_result}\n"
        except Exception as e:
            state["status"] = "failed"
            state["log"] += f"\nRemote error: {e}\n"

    threading.Thread(target=run_remote, daemon=True).start()
    return jsonify({"ok": True})


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video2X for Mac</title>
<style>
:root {
  --bg: #1a1a2e;
  --surface: #16213e;
  --surface2: #0f3460;
  --accent: #e94560;
  --accent2: #533483;
  --text: #eee;
  --text2: #aab;
  --success: #4ade80;
  --radius: 12px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  height: 100vh; display: flex; flex-direction: column;
  user-select: none; overflow: hidden;
}
.header {
  display: flex; align-items: center; gap: 12px;
  padding: 16px 24px;
  background: linear-gradient(135deg, var(--surface), var(--surface2));
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.header h1 { font-size: 18px; font-weight: 600; }
.header .sub { font-size: 12px; color: var(--text2); }
.header .gpu {
  margin-left: auto; font-size: 12px;
  background: rgba(255,255,255,.08); padding: 6px 14px;
  border-radius: 20px; color: var(--text2);
}
.main { display: flex; flex: 1; overflow: hidden; }
.sidebar {
  width: 360px; min-width: 320px; overflow-y: auto; padding: 20px;
  background: var(--surface);
  border-right: 1px solid rgba(255,255,255,.06);
  display: flex; flex-direction: column; gap: 20px;
}
.content { flex: 1; display: flex; flex-direction: column; }
.section-title {
  font-size: 13px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .5px; color: var(--text2); margin-bottom: 8px;
}
.dropzone {
  border: 2px dashed rgba(255,255,255,.15); border-radius: var(--radius);
  padding: 24px; text-align: center; cursor: pointer; transition: all .2s;
}
.dropzone:hover, .dropzone.over {
  border-color: var(--accent); background: rgba(233,69,96,.05);
}
.dropzone .icon { font-size: 28px; margin-bottom: 8px; }
.dropzone .filename { font-size: 14px; font-weight: 500; color: var(--text); }
.dropzone .info { font-size: 12px; color: var(--text2); margin-top: 4px; }
.proc-card {
  padding: 10px 14px; border-radius: 8px; cursor: pointer; transition: all .15s;
  border: 2px solid transparent; background: rgba(255,255,255,.03);
}
.proc-card:hover { background: rgba(255,255,255,.06); }
.proc-card.active { border-color: var(--accent); background: rgba(233,69,96,.08); }
.proc-card .name { font-size: 14px; font-weight: 600; }
.proc-card .desc { font-size: 11px; color: var(--text2); margin-top: 2px; }
.setting-row {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.setting-row label { font-size: 13px; min-width: 80px; color: var(--text2); }
.setting-row select, .setting-row input[type="number"], .setting-row input[type="text"] {
  flex: 1; padding: 6px 10px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.1);
  border-radius: 6px; color: var(--text); font-size: 13px;
}
.setting-row select { appearance: auto; }
.seg-control {
  display: flex; gap: 0; border-radius: 8px; overflow: hidden;
  border: 1px solid rgba(255,255,255,.1);
}
.seg-control button {
  flex: 1; padding: 8px; border: none;
  background: rgba(255,255,255,.05); color: var(--text2);
  font-size: 13px; font-weight: 500; cursor: pointer; transition: all .15s;
}
.seg-control button.active { background: var(--accent); color: #fff; }
.btn-primary {
  width: 100%; padding: 14px; border: none; border-radius: var(--radius);
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: opacity .2s;
}
.btn-primary:hover { opacity: .9; }
.btn-primary:disabled { opacity: .4; cursor: not-allowed; }
.btn-cancel {
  width: 100%; padding: 14px; border: 2px solid var(--accent);
  border-radius: var(--radius); background: transparent;
  color: var(--accent); font-size: 15px; font-weight: 600; cursor: pointer;
}
.btn-cancel:hover { background: rgba(233,69,96,.1); }
.progress-area {
  padding: 24px; background: var(--surface);
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.progress-bar-wrap {
  height: 8px; background: rgba(255,255,255,.08);
  border-radius: 4px; overflow: hidden; margin: 12px 0;
}
.progress-bar {
  height: 100%; border-radius: 4px; transition: width .3s;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stats { display: flex; gap: 0; justify-content: space-around; }
.stat { text-align: center; }
.stat .val { font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat .lbl { font-size: 11px; color: var(--text2); margin-top: 2px; }
.log-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.log-header {
  display: flex; align-items: center; padding: 12px 20px;
  font-size: 13px; font-weight: 600; color: var(--text2);
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.log-content {
  flex: 1; overflow-y: auto; padding: 12px 20px;
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 12px; line-height: 1.6; color: var(--text2);
  white-space: pre-wrap; word-break: break-all;
}
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 600;
}
.badge.success { background: rgba(74,222,128,.1); color: var(--success); }
.badge.error { background: rgba(233,69,96,.1); color: var(--accent); }
.badge.warning { background: rgba(255,200,50,.1); color: #fbbf24; }
@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
  display: inline-block; width: 16px; height: 16px;
  border: 2px solid rgba(255,200,50,.3); border-top-color: #fbbf24;
  border-radius: 50%; animation: spin .6s linear infinite;
  vertical-align: middle; margin-right: 8px;
}
.path-input {
  width: 100%; padding: 8px 10px; margin-top: 6px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
  border-radius: 6px; color: var(--text); font-size: 12px;
  font-family: 'SF Mono', 'Menlo', monospace;
}
.path-input::placeholder { color: var(--text2); }
.cloud-icon {
  font-size: 22px; cursor: pointer; padding: 4px 10px;
  border-radius: 8px; transition: all .2s; margin-left: 8px;
  color: var(--text2);
}
.cloud-icon:hover { background: rgba(255,255,255,.1); color: var(--accent); }
.cloud-icon.connected { color: var(--success); }
.cloud-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,.3); z-index: 99;
}
.cloud-panel {
  position: fixed; top: 56px; right: 16px; width: 360px;
  background: var(--surface); border: 1px solid rgba(255,255,255,.1);
  border-radius: 12px; padding: 20px; z-index: 100;
  box-shadow: 0 12px 40px rgba(0,0,0,.5);
}
.cloud-panel-title {
  font-size: 15px; font-weight: 700; margin-bottom: 6px;
}
.cloud-add-card {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  padding: 16px; border: 2px dashed rgba(255,255,255,.12);
  border-radius: 10px; cursor: pointer; color: var(--text2);
  font-size: 13px; transition: all .2s; margin-top: 8px;
}
.cloud-add-card:hover { border-color: var(--accent); color: var(--accent); background: rgba(233,69,96,.04); }
.cloud-gpu-item {
  padding: 10px 14px; border-radius: 8px; cursor: pointer;
  border: 1px solid rgba(255,255,255,.08); margin-bottom: 6px;
  background: rgba(255,255,255,.03); transition: all .15s;
  position: relative;
}
.cloud-gpu-item:hover { border-color: var(--accent); background: rgba(233,69,96,.05); }
.cloud-gpu-item .cg-name { font-size: 14px; font-weight: 600; }
.cloud-gpu-item .cg-host { font-size: 11px; color: var(--text2); margin-top: 2px; }
.cloud-gpu-item .cg-tag {
  display: inline-block; font-size: 10px; padding: 2px 8px;
  background: rgba(255,255,255,.08); border-radius: 4px;
  color: var(--text2); margin-top: 4px;
}
.cloud-gpu-item .cg-delete {
  position: absolute; top: 8px; right: 10px; font-size: 14px;
  color: var(--text2); cursor: pointer; opacity: 0; transition: opacity .15s;
  padding: 2px 6px; border-radius: 4px;
}
.cloud-gpu-item:hover .cg-delete { opacity: 1; }
.cloud-gpu-item .cg-delete:hover { color: var(--accent); background: rgba(233,69,96,.15); }
.cloud-type-card {
  display: flex; align-items: center; gap: 14px;
  padding: 14px; border: 1px solid rgba(255,255,255,.08);
  border-radius: 10px; cursor: pointer; margin-bottom: 8px;
  background: rgba(255,255,255,.03); transition: all .15s;
}
.cloud-type-card:hover { border-color: var(--accent); background: rgba(233,69,96,.05); }
.ct-icon { font-size: 26px; }
.ct-name { font-size: 14px; font-weight: 600; }
.ct-desc { font-size: 11px; color: var(--text2); margin-top: 2px; }
.cp-back {
  font-size: 18px; cursor: pointer; padding: 2px 8px;
  border-radius: 6px; color: var(--text2); transition: all .15s;
}
.cp-back:hover { color: var(--text); background: rgba(255,255,255,.08); }
.cp-field { margin-bottom: 10px; }
.cp-field label { display: block; font-size: 11px; color: var(--text2); margin-bottom: 4px; font-weight: 600; }
.cp-field input, .cp-field select {
  width: 100%; padding: 8px 10px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.1);
  border-radius: 6px; color: var(--text); font-size: 13px;
}
.cp-field select { appearance: auto; }
.cp-field-row { display: flex; gap: 10px; }
.cp-field-row .cp-field { flex: 1; }
.gpu-gauge-wrap {
  margin-top: 16px; padding: 14px; border-radius: 10px;
  background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.06);
  display: none;
}
.gpu-gauge-wrap.active { display: block; }
.gpu-gauge-title {
  font-size: 12px; font-weight: 600; color: var(--text2);
  text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px;
}
.gpu-gauge-row {
  display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.gpu-gauge-row:last-child { margin-bottom: 0; }
.gpu-gauge-label { font-size: 12px; color: var(--text2); min-width: 70px; }
.gpu-gauge-bar {
  flex: 1; height: 10px; background: rgba(255,255,255,.08);
  border-radius: 5px; overflow: hidden;
}
.gpu-gauge-fill {
  height: 100%; border-radius: 5px; transition: width .8s, background .5s;
}
.gpu-gauge-val {
  font-size: 13px; font-weight: 700; min-width: 42px; text-align: right;
  font-variant-numeric: tabular-nums;
}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Video2X for Mac</h1>
    <div class="sub">ML-powered video upscaling &amp; frame interpolation</div>
  </div>
  <div class="gpu" id="gpuBadge">Detecting GPU...</div>
  <div class="cloud-icon" id="cloudIcon" onclick="toggleCloudPanel()" title="Connect to external GPU">&#9729;</div>
</div>

<div class="cloud-overlay" id="cloudOverlay" style="display:none" onclick="toggleCloudPanel()"></div>
<div class="cloud-panel" id="cloudPanel" style="display:none">
  <!-- Main list view -->
  <div id="cpList">
    <div class="cloud-panel-title">External GPUs</div>
    <div id="cloudCards"></div>
    <div class="cloud-add-card" onclick="showAddStep1()">
      <span style="font-size:22px">+</span>
      <span>Add External GPU</span>
    </div>
  </div>

  <!-- Step 1: Pick type -->
  <div id="cpStep1" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="showList()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0">Select GPU Type</div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('slurm')">
      <div class="ct-icon">&#128421;</div>
      <div>
        <div class="ct-name">SLURM Cluster</div>
        <div class="ct-desc">HPC cluster with job scheduler (sbatch/srun)</div>
      </div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('direct')">
      <div class="ct-icon">&#128187;</div>
      <div>
        <div class="ct-name">SSH Direct</div>
        <div class="ct-desc">GPU workstation or server with direct access</div>
      </div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('cloud')">
      <div class="ct-icon">&#9729;</div>
      <div>
        <div class="ct-name">Cloud Provider</div>
        <div class="ct-desc">RunPod, Vast.ai, Lambda, or any cloud GPU</div>
      </div>
    </div>
  </div>

  <!-- Step 2: Fill fields -->
  <div id="cpStep2" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="showAddStep1()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0" id="cpStep2Title">Configure</div>
    </div>
    <div id="cpFields"></div>
    <button class="btn-primary" style="font-size:13px;padding:10px;margin-top:8px" onclick="saveNewGpu()">Add GPU</button>
  </div>

  <!-- Connected view -->
  <div id="cpConnected" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="disconnectRemote()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0">Connected</div>
    </div>
    <div class="badge success" id="cloudBadge" style="font-size:12px;margin-bottom:10px"></div>
    <button class="btn-cancel" style="font-size:12px;padding:8px" onclick="disconnectRemote()">Disconnect</button>
  </div>

  <!-- Password prompt -->
  <div id="cpPassword" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="showList()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0">Enter Password</div>
    </div>
    <div class="cp-field">
      <label>Password</label>
      <input id="sshPass" type="password" placeholder="SSH password">
    </div>
    <button class="btn-primary" style="font-size:13px;padding:10px;margin-top:8px" onclick="submitPassword()">Connect</button>
  </div>

  <!-- Connecting spinner -->
  <div id="cpConnecting" style="display:none;text-align:center;padding:30px 0">
    <div class="spinner" style="width:28px;height:28px;border-width:3px;margin:0 auto 12px"></div>
    <div style="font-size:13px;color:var(--text2)" id="cpConnectMsg">Connecting...</div>
  </div>
</div>

<div class="main">
  <div class="sidebar">

    <div>
      <div class="section-title">Input Video</div>
      <div class="dropzone" id="dropzone" onclick="browseInput()">
        <div class="icon">&#127916;</div>
        <div class="filename" id="inputName">Click to browse for a video</div>
        <div class="info" id="inputInfo"></div>
      </div>
      <input class="path-input" id="pathInput" type="text"
             placeholder="Or paste full file path here and press Enter"
             onkeydown="if(event.key==='Enter') loadPathInput()">
    </div>

    <div>
      <div class="section-title">Processor</div>
      <div id="procCards" style="display:flex;flex-direction:column;gap:6px"></div>
    </div>

    <div>
      <div class="section-title">Settings</div>
      <div id="settingsArea"></div>
    </div>

    <div>
      <div class="section-title">Output</div>
      <div class="setting-row">
        <label>Codec</label>
        <select id="codec">
          <option value="libx264">H.264</option>
          <option value="libx265">H.265 (HEVC)</option>
        </select>
      </div>
      <div class="setting-row">
        <label>Output</label>
        <input id="outputPath" type="text" placeholder="Auto-generated"
               style="cursor:text">
      </div>
    </div>

    <div id="actionArea"></div>

  </div>

  <div class="content">
    <div class="progress-area">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <span style="font-weight:600">Progress</span>
        <span id="pctText" style="font-size:22px;font-weight:700;color:var(--accent)">0%</span>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-bar" id="progressBar" style="width:0%"></div>
      </div>
      <div class="stats">
        <div class="stat"><div class="val" id="statFrame">--</div><div class="lbl">Frame</div></div>
        <div class="stat"><div class="val" id="statFps">--</div><div class="lbl">FPS</div></div>
        <div class="stat"><div class="val" id="statElapsed">--</div><div class="lbl">Elapsed</div></div>
        <div class="stat"><div class="val" id="statRemaining">--</div><div class="lbl">Remaining</div></div>
      </div>
      <div class="gpu-gauge-wrap" id="gpuGauge">
        <div class="gpu-gauge-title">GPU Utilization</div>
        <div class="gpu-gauge-row">
          <span class="gpu-gauge-label">Device</span>
          <div class="gpu-gauge-bar"><div class="gpu-gauge-fill" id="gpuBarDevice" style="width:0%"></div></div>
          <span class="gpu-gauge-val" id="gpuValDevice">0%</span>
        </div>
        <div class="gpu-gauge-row">
          <span class="gpu-gauge-label">Renderer</span>
          <div class="gpu-gauge-bar"><div class="gpu-gauge-fill" id="gpuBarRender" style="width:0%"></div></div>
          <span class="gpu-gauge-val" id="gpuValRender">0%</span>
        </div>
        <div class="gpu-gauge-row">
          <span class="gpu-gauge-label">Tiler</span>
          <div class="gpu-gauge-bar"><div class="gpu-gauge-fill" id="gpuBarTiler" style="width:0%"></div></div>
          <span class="gpu-gauge-val" id="gpuValTiler">0%</span>
        </div>
      </div>
    </div>
    <div class="log-area">
      <div class="log-header">Log Output</div>
      <div class="log-content" id="logContent">Ready. Select a video and click Start Processing.</div>
    </div>
  </div>
</div>

<script>
const PROCESSORS = [
  {id:'realesrgan', name:'Real-ESRGAN', desc:'Best for general video & anime upscaling',
   models:['realesr-animevideov3','realesrgan-plus-anime','realesrgan-plus','realesr-generalv3']},
  {id:'realcugan', name:'Real-CUGAN', desc:'Optimized for anime upscaling',
   models:['models-se','models-pro','models-nose']},
  {id:'libplacebo', name:'Anime4K (libplacebo)', desc:'Shader-based upscaling',
   models:['anime4k-v4-a','anime4k-v4-a+a','anime4k-v4-b','anime4k-v4-b+b','anime4k-v4-c','anime4k-v4-c+a']},
  {id:'rife', name:'RIFE', desc:'Increases frame rate for smoother motion',
   models:['rife-v4.26','rife-v4.25','rife-v4.25-lite','rife-v4.6','rife-v4']},
];

var selectedProc = 'realesrgan';
var inputPath = '';
var scale = 4;
var multiplier = 2;
var computeMode = 'local';
var remoteConnected = false;
var remoteSlurm = false;

function init() {
  renderProcessors();
  renderSettings();
  renderAction();
  fetch('/api/devices').then(function(r){return r.json()}).then(function(devs) {
    if (devs.length > 0) {
      document.getElementById('gpuBadge').textContent = '\u{1F7E2} ' + devs[0].name;
    } else {
      document.getElementById('gpuBadge').textContent = 'No GPU found';
    }
  }).catch(function(){});
  startPolling();
}

function renderProcessors() {
  var el = document.getElementById('procCards');
  var html = '';
  for (var i = 0; i < PROCESSORS.length; i++) {
    var p = PROCESSORS[i];
    var cls = p.id === selectedProc ? 'proc-card active' : 'proc-card';
    html += '<div class="' + cls + '" onclick="selectProc(\'' + p.id + '\')">';
    html += '<div class="name">' + p.name + '</div>';
    html += '<div class="desc">' + p.desc + '</div></div>';
  }
  el.innerHTML = html;
}

function selectProc(id) {
  selectedProc = id;
  renderProcessors();
  renderSettings();
}

function renderSettings() {
  var proc = null;
  for (var i = 0; i < PROCESSORS.length; i++) {
    if (PROCESSORS[i].id === selectedProc) proc = PROCESSORS[i];
  }
  if (!proc) return;

  var isUpscaler = selectedProc !== 'rife';
  var html = '';

  if (isUpscaler && selectedProc !== 'libplacebo') {
    html += '<div style="margin-bottom:10px"><div style="font-size:12px;color:var(--text2);margin-bottom:6px">Scale Factor</div>';
    html += '<div class="seg-control">';
    var scales = [2, 3, 4];
    for (var i = 0; i < scales.length; i++) {
      var s = scales[i];
      var ac = scale === s ? ' active' : '';
      html += '<button class="' + ac + '" onclick="scale=' + s + ';renderSettings()">' + s + 'x</button>';
    }
    html += '</div></div>';
  } else if (selectedProc === 'libplacebo') {
    html += '<div class="setting-row"><label>Width</label><input id="outW" type="number" value="3840"></div>';
    html += '<div class="setting-row"><label>Height</label><input id="outH" type="number" value="2160"></div>';
  } else {
    html += '<div style="margin-bottom:10px"><div style="font-size:12px;color:var(--text2);margin-bottom:6px">Frame Rate Multiplier</div>';
    html += '<div class="seg-control">';
    var muls = [2, 3, 4];
    for (var i = 0; i < muls.length; i++) {
      var m = muls[i];
      var ac = multiplier === m ? ' active' : '';
      html += '<button class="' + ac + '" onclick="multiplier=' + m + ';renderSettings()">' + m + 'x</button>';
    }
    html += '</div></div>';
  }

  html += '<div class="setting-row"><label>Model</label><select id="model">';
  for (var i = 0; i < proc.models.length; i++) {
    html += '<option value="' + proc.models[i] + '">' + proc.models[i] + '</option>';
  }
  html += '</select></div>';

  document.getElementById('settingsArea').innerHTML = html;
}

function renderAction() {
  var el = document.getElementById('actionArea');
  fetch('/api/state').then(function(r){return r.json()}).then(function(s) {
    if (s.status === 'cancelling') {
      el.innerHTML = '<div class="badge warning"><span class="spinner"></span>Cancelling...</div>';
    } else if (s.status === 'running') {
      el.innerHTML = '<button class="btn-cancel" onclick="cancelProcessing()">Cancel Processing</button>';
    } else if (s.status === 'finished') {
      el.innerHTML = '<div class="badge success">\u2713 Processing Complete!</div>' +
        '<button class="btn-primary" style="margin-top:10px" onclick="resetState()">Process Another</button>';
    } else if (s.status === 'failed') {
      el.innerHTML = '<div class="badge error">\u2717 Processing Failed</div>' +
        '<button class="btn-primary" style="margin-top:10px" onclick="resetState()">Try Again</button>';
    } else {
      if (computeMode === 'cloud' && remoteConnected) {
        var dis = inputPath ? '' : ' disabled';
        el.innerHTML = '<button class="btn-primary" onclick="startRemoteProcessing()"' + dis + '>\u{2601}\uFE0F Start on Cloud GPU</button>';
      } else {
        var dis = inputPath ? '' : ' disabled';
        el.innerHTML = '<button class="btn-primary" onclick="startProcessing()"' + dis + '>Start Processing</button>';
      }
    }
  }).catch(function(){});
}

function browseInput() {
  if (window.pywebview) {
    window.pywebview.api.browse_input().then(function(result) {
      var d = JSON.parse(result);
      if (d.path) {
        setInput(d.path, d.info);
      }
    });
  } else {
    alert('File browser not available. Paste a file path in the box below instead.');
  }
}

function loadPathInput() {
  var path = document.getElementById('pathInput').value.trim();
  if (!path) return;
  fetch('/api/probe', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: path})
  }).then(function(r){return r.json()}).then(function(info) {
    if (info.width) {
      setInput(path, info);
    } else {
      alert('Could not read video at: ' + path);
    }
  });
}

function setInput(path, info) {
  inputPath = path;
  var parts = path.replace(/\\/g, '/').split('/');
  document.getElementById('inputName').textContent = parts[parts.length - 1];
  document.getElementById('pathInput').value = path;
  if (info && info.width) {
    document.getElementById('inputInfo').textContent =
      info.width + 'x' + info.height + ' | ' + info.fps + ' fps | ' + info.duration + 's | ' + info.size_mb + ' MB';
  }
  renderAction();
}

function browseOutput() {
  if (window.pywebview) {
    window.pywebview.api.browse_output().then(function(result) {
      var d = JSON.parse(result);
      if (d.path) document.getElementById('outputPath').value = d.path;
    });
  }
}

function startProcessing() {
  if (!inputPath) return;
  var output = document.getElementById('outputPath').value;
  if (!output) {
    var parts = inputPath.split('.');
    var ext = parts.pop();
    output = parts.join('.') + '_upscaled.' + ext;
  }

  var modelEl = document.getElementById('model');
  var body = {
    input: inputPath,
    output: output,
    processor: selectedProc,
    scale: scale,
    multiplier: multiplier,
    model: modelEl ? modelEl.value : '',
    codec: document.getElementById('codec').value,
    device: 0
  };

  if (selectedProc === 'libplacebo') {
    var wEl = document.getElementById('outW');
    var hEl = document.getElementById('outH');
    body.width = parseInt(wEl ? wEl.value : '3840');
    body.height = parseInt(hEl ? hEl.value : '2160');
  }

  fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  }).then(function() { renderAction(); });
}

function cancelProcessing() {
  fetch('/api/cancel', {method:'POST'}).then(function() { renderAction(); });
}

function resetState() {
  fetch('/api/reset', {method:'POST'}).then(function() { renderAction(); });
}

function startPolling() {
  setInterval(function() {
    fetch('/api/state').then(function(r){return r.json()}).then(function(s) {
      var pct = Math.round(s.progress * 100);
      document.getElementById('pctText').textContent = pct + '%';
      document.getElementById('progressBar').style.width = pct + '%';
      document.getElementById('statFrame').textContent = s.total > 0 ? s.frame + '/' + s.total : '--';
      document.getElementById('statFps').textContent = s.fps > 0 ? s.fps.toFixed(1) : '--';
      document.getElementById('statElapsed').textContent = s.elapsed;
      document.getElementById('statRemaining').textContent = s.remaining;
      var logEl = document.getElementById('logContent');
      if (s.log) { logEl.textContent = s.log; logEl.scrollTop = logEl.scrollHeight; }

      var isActive = s.status === 'running' || s.status === 'cancelling';
      var gaugeEl = document.getElementById('gpuGauge');
      if (isActive) {
        gaugeEl.classList.add('active');
        updateGpuBar('Device', s.gpu_util);
        updateGpuBar('Render', s.gpu_render);
        updateGpuBar('Tiler', s.gpu_tiler);
        var badge = document.getElementById('gpuBadge');
        badge.textContent = '\u{1F7E2} Apple M3 \u2022 ' + s.gpu_util + '%';
      } else {
        gaugeEl.classList.remove('active');
      }

      renderAction();
    }).catch(function(){});
  }, 500);
}

function updateGpuBar(name, val) {
  var bar = document.getElementById('gpuBar' + name);
  var label = document.getElementById('gpuVal' + name);
  if (!bar || !label) return;
  bar.style.width = val + '%';
  label.textContent = val + '%';
  if (val > 80) {
    bar.style.background = '#e94560';
    label.style.color = '#e94560';
  } else if (val > 40) {
    bar.style.background = '#fbbf24';
    label.style.color = '#fbbf24';
  } else {
    bar.style.background = '#4ade80';
    label.style.color = '#4ade80';
  }
}

var gpuConfigs = [];
var pendingConfig = null;
var addingType = '';

function cpShowOnly(id) {
  var ids = ['cpList','cpStep1','cpStep2','cpConnected','cpPassword','cpConnecting'];
  for (var i = 0; i < ids.length; i++) {
    document.getElementById(ids[i]).style.display = ids[i] === id ? 'block' : 'none';
  }
}

function toggleCloudPanel() {
  var panel = document.getElementById('cloudPanel');
  var overlay = document.getElementById('cloudOverlay');
  var visible = panel.style.display !== 'none';
  if (visible) {
    panel.style.display = 'none';
    overlay.style.display = 'none';
  } else {
    panel.style.display = 'block';
    overlay.style.display = 'block';
    if (remoteConnected) {
      cpShowOnly('cpConnected');
    } else {
      showList();
    }
  }
}

function showList() {
  cpShowOnly('cpList');
  fetch('/api/remote/configs').then(function(r){return r.json()}).then(function(d) {
    gpuConfigs = d.configs;
    renderCards();
  });
}

function renderCards() {
  var el = document.getElementById('cloudCards');
  if (gpuConfigs.length === 0) {
    el.innerHTML = '<div style="padding:20px;text-align:center;border:2px dashed rgba(255,255,255,.1);border-radius:10px;color:var(--text2);font-size:13px">' +
      'No external GPUs added yet.<br><span style="font-size:11px">Click below to add one.</span></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < gpuConfigs.length; i++) {
    var g = gpuConfigs[i];
    var type = g.type === 'slurm' ? 'SLURM' : g.type === 'cloud' ? 'Cloud' : 'SSH';
    var proxy = g.proxy ? ' \u2192 ' + g.proxy : '';
    html += '<div class="cloud-gpu-item" onclick="connectToGpu(' + i + ')">';
    html += '<span class="cg-delete" onclick="event.stopPropagation();deleteGpu(' + i + ')">\u00d7</span>';
    html += '<div class="cg-name">' + g.name + '</div>';
    html += '<div class="cg-host">' + g.user + '@' + g.host + ':' + g.port + proxy + '</div>';
    html += '<span class="cg-tag">' + type + '</span>';
    if (g.auth === 'password') html += ' <span class="cg-tag">\u{1F511} password</span>';
    html += '</div>';
  }
  el.innerHTML = html;
}

function deleteGpu(idx) {
  if (!confirm('Remove this GPU?')) return;
  fetch('/api/remote/configs/delete', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: idx})
  }).then(function(r){return r.json()}).then(function(d) {
    gpuConfigs = d.configs;
    renderCards();
  });
}

function showAddStep1() { cpShowOnly('cpStep1'); }

function showAddStep2(type) {
  addingType = type;
  cpShowOnly('cpStep2');
  var titles = {slurm: 'Add SLURM Cluster', direct: 'Add SSH GPU', cloud: 'Add Cloud GPU'};
  document.getElementById('cpStep2Title').textContent = titles[type] || 'Configure';

  var html = '';
  html += '<div class="cp-field"><label>Name</label><input id="af_name" placeholder="e.g. Lab A100"></div>';
  html += '<div class="cp-field-row">';
  html += '<div class="cp-field"><label>Host</label><input id="af_host" placeholder="cluster.uni.edu"></div>';
  html += '<div class="cp-field" style="max-width:80px"><label>Port</label><input id="af_port" type="number" value="22"></div>';
  html += '</div>';
  html += '<div class="cp-field"><label>Username</label><input id="af_user" placeholder="jdoe"></div>';
  html += '<div class="cp-field"><label>Authentication</label><select id="af_auth">';
  html += '<option value="key">SSH Key (~/.ssh/id_rsa)</option>';
  html += '<option value="password">Password (prompt on connect)</option>';
  html += '<option value="agent">SSH Agent</option>';
  html += '</select></div>';

  if (type === 'slurm') {
    html += '<div style="margin:14px 0 8px;font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px">SLURM Settings</div>';
    html += '<div class="cp-field"><label>Partition <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_partition" placeholder="gpu"></div>';
    html += '<div class="cp-field-row">';
    html += '<div class="cp-field"><label>GPU Resource</label><input id="af_gres" value="gpu:1"></div>';
    html += '<div class="cp-field"><label>Memory</label><input id="af_mem" value="32G"></div>';
    html += '</div>';
    html += '<div class="cp-field"><label>Time Limit</label><input id="af_time" value="02:00:00"></div>';
  }

  html += '<div style="margin:14px 0 8px;font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px">Advanced</div>';
  html += '<div class="cp-field"><label>Jump Host / Proxy <span style="color:var(--text2);font-weight:400">(optional, e.g. user@login-node)</span></label><input id="af_proxy" placeholder="jdoe@login.hpc.university.edu"></div>';
  html += '<div class="cp-field"><label>SSH Key Path <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_keypath" placeholder="~/.ssh/id_rsa"></div>';

  document.getElementById('cpFields').innerHTML = html;
}

function saveNewGpu() {
  var name = document.getElementById('af_name').value.trim();
  var host = document.getElementById('af_host').value.trim();
  var user = document.getElementById('af_user').value.trim();
  if (!name || !host || !user) { alert('Fill in name, host, and username.'); return; }

  var gpu = {
    name: name,
    type: addingType,
    host: host,
    port: parseInt(document.getElementById('af_port').value) || 22,
    user: user,
    auth: document.getElementById('af_auth').value,
  };

  var proxyEl = document.getElementById('af_proxy');
  var keyEl = document.getElementById('af_keypath');
  if (proxyEl && proxyEl.value.trim()) gpu.proxy = proxyEl.value.trim();
  if (keyEl && keyEl.value.trim()) gpu.key_path = keyEl.value.trim();

  if (addingType === 'slurm') {
    var p = document.getElementById('af_partition');
    if (p && p.value.trim()) gpu.partition = p.value.trim();
    gpu.gres = document.getElementById('af_gres').value.trim() || 'gpu:1';
    gpu.mem = document.getElementById('af_mem').value.trim() || '32G';
    gpu.time_limit = document.getElementById('af_time').value.trim() || '02:00:00';
  }

  fetch('/api/remote/configs/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(gpu)
  }).then(function(r){return r.json()}).then(function(d) {
    gpuConfigs = d.configs;
    showList();
  });
}

function connectToGpu(idx) {
  var cfg = gpuConfigs[idx];
  if (cfg.auth === 'password') {
    pendingConfig = cfg;
    cpShowOnly('cpPassword');
    return;
  }
  doConnect(cfg);
}

function submitPassword() {
  if (!pendingConfig) return;
  pendingConfig.password = document.getElementById('sshPass').value;
  doConnect(pendingConfig);
  pendingConfig = null;
}

function doConnect(cfg) {
  cpShowOnly('cpConnecting');
  document.getElementById('cpConnectMsg').textContent = 'Connecting to ' + cfg.name + '...';

  fetch('/api/remote/connect', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(cfg)})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.ok) {
        remoteConnected = true;
        computeMode = 'cloud';
        remoteSlurm = d.slurm;
        var gpu = d.gpu || 'GPU';
        var sched = d.slurm ? ' | SLURM' : ' | Direct';
        document.getElementById('cloudBadge').innerHTML = '\u{1F7E2} ' + cfg.name + ' \u2014 ' + gpu + sched;
        document.getElementById('cloudIcon').classList.add('connected');
        document.getElementById('gpuBadge').textContent = '\u{2601}\uFE0F ' + gpu;
        cpShowOnly('cpConnected');
        renderAction();
      } else {
        alert('Connection failed. Check the log for details.');
        showList();
      }
    });
}

function disconnectRemote() {
  fetch('/api/remote/disconnect', {method:'POST'});
  remoteConnected = false;
  computeMode = 'local';
  document.getElementById('cloudIcon').classList.remove('connected');
  document.getElementById('gpuBadge').textContent = '\u{1F7E2} Apple M3';
  showList();
  renderAction();
}

function startRemoteProcessing() {
  if (!inputPath || !remoteConnected) return;

  var modelEl = document.getElementById('model');
  var body = {
    input: inputPath,
    processor: selectedProc,
    scale: scale,
    multiplier: multiplier,
    model: modelEl ? modelEl.value : '',
    codec: document.getElementById('codec').value,
    use_slurm: remoteSlurm,
  };
  if (selectedProc === 'libplacebo') {
    body.width = parseInt(document.getElementById('outW') ? document.getElementById('outW').value : '3840');
    body.height = parseInt(document.getElementById('outH') ? document.getElementById('outH').value : '2160');
  }
  fetch('/api/remote/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)})
    .then(function() { renderAction(); });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    detect_devices()
    port = 52845

    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    time.sleep(0.5)

    window = webview.create_window(
        "Video2X for Mac",
        f"http://127.0.0.1:{port}",
        width=1000, height=720, min_size=(800, 600),
        confirm_close=True,
        js_api=js_api,
    )
    webview.start()
