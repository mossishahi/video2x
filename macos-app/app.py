#!/usr/bin/env python3
"""Video2X for Mac — native macOS GUI for video upscaling and frame interpolation."""

import json
import os
import re
import subprocess
import threading
import time
import uuid
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
ANSI_RE = re.compile(r'(\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K|\[K)')
SPLIT_RE = re.compile(r'[\r\n]+')
PROGRESS_RE = re.compile(
    r"frame=(\d+)/(\d+)\s+\(([^)]+)\);\s+fps=([^;]+);\s+elapsed=([^;]+);\s+remaining=(\S+)"
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
jobs = {}               # job_id -> job dict
output_dir = ""         # global output directory
devices = []            # detected local GPU devices
gpu_util_data = {"gpu_util": 0, "gpu_render": 0, "gpu_tiler": 0}
active_processes = {}   # job_id -> {"type": "local"/"remote", "proc"/"rgpu": handle}
gpu_poll_active = False
gpu_passwords = {}      # "user@host:port" -> password (in-memory only)


def _gpu_key(cfg):
    return f"{cfg.get('user', '')}@{cfg.get('host', '')}:{cfg.get('port', 22)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def probe_video(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=10, env=V2X_ENV,
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


def format_info(info):
    if not info or not info.get("width"):
        return ""
    fps = info.get("fps", "?")
    if isinstance(fps, str) and "/" in fps:
        try:
            n, d = fps.split("/")
            fps = round(int(n) / int(d), 1)
        except (ValueError, ZeroDivisionError):
            pass
    return f"{info['width']}x{info['height']} | {fps}fps | {info.get('duration', '?')}s"


def detect_devices():
    global devices
    try:
        result = subprocess.run(
            [str(V2X_BIN), "--list-devices"],
            capture_output=True, text=True, env=V2X_ENV, timeout=10,
        )
        devs = []
        lines = result.stdout.strip().split("\n")
        idx, name, dtype = -1, "", ""
        for line in lines:
            m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
            if m:
                if idx >= 0:
                    devs.append({"id": idx, "name": name, "type": dtype})
                idx, name, dtype = int(m.group(1)), m.group(2), ""
            elif line.strip().startswith("Type:"):
                dtype = line.strip()[5:].strip()
        if idx >= 0:
            devs.append({"id": idx, "name": name, "type": dtype})
        devices = devs
    except Exception:
        pass


def poll_gpu_utilization():
    global gpu_poll_active
    gpu_poll_active = True
    while gpu_poll_active:
        has_local = any(
            j["status"] == "running" and j["gpu"] == "local"
            for j in jobs.values()
        )
        if not has_local:
            break
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=3,
            )
            text = result.stdout
            for key, field in [
                ("gpu_util", "Device Utilization"),
                ("gpu_render", "Renderer Utilization"),
                ("gpu_tiler", "Tiler Utilization"),
            ]:
                m = re.search(rf'"{field} %"=(\d+)', text)
                if m:
                    gpu_util_data[key] = int(m.group(1))
        except Exception:
            pass
        time.sleep(1.5)
    gpu_util_data.update(gpu_util=0, gpu_render=0, gpu_tiler=0)
    gpu_poll_active = False


def generate_output_path(input_path, out_dir):
    name, ext = os.path.splitext(os.path.basename(input_path))
    directory = out_dir if out_dir else os.path.dirname(input_path)
    return os.path.join(directory, name + "_upscaled" + ext)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def build_v2x_args(job):
    args = [
        "-i", job["input"], "-o", job["output"],
        "-p", job["processor"], "-d", "0",
        "-c", job["codec"], "--log-level", "info",
    ]
    p = job["processor"]
    if p in ("realesrgan", "realcugan"):
        args += ["-s", str(job.get("scale", 4))]
        flag = "--realesrgan-model" if p == "realesrgan" else "--realcugan-model"
        args += [flag, job.get("model", "realesr-animevideov3")]
    elif p == "libplacebo":
        args += ["-w", str(job.get("width", 3840)), "-h", str(job.get("height", 2160))]
        args += ["--libplacebo-shader", job.get("model", "anime4k-v4-a")]
    elif p == "rife":
        args += ["-m", str(job.get("multiplier", 2))]
        args += ["--rife-model", job.get("model", "rife-v4.6")]
    return args


def build_remote_args_str(job):
    p = job["processor"]
    parts = [f"-p {p}", "-d 0", f"-c {job.get('codec', 'libx264')}", "--log-level info"]
    if p in ("realesrgan", "realcugan"):
        parts.append(f"-s {job.get('scale', 4)}")
        flag = "--realesrgan-model" if p == "realesrgan" else "--realcugan-model"
        parts.append(f"{flag} {job.get('model', 'realesr-animevideov3')}")
    elif p == "libplacebo":
        parts.append(f"-w {job.get('width', 3840)} -h {job.get('height', 2160)}")
        parts.append(f"--libplacebo-shader {job.get('model', 'anime4k-v4-a')}")
    elif p == "rife":
        parts.append(f"-m {job.get('multiplier', 2)}")
        parts.append(f"--rife-model {job.get('model', 'rife-v4.6')}")
    return " ".join(parts)


def _reset_job_progress(job):
    job.update(frame=0, total=0, fps=0.0, progress=0.0,
               elapsed="00:00:00", remaining="--:--:--")


def run_local_job(job):
    global gpu_poll_active
    job["status"] = "running"
    _reset_job_progress(job)

    args = build_v2x_args(job)
    job["log"] = f"$ video2x {' '.join(args)}\n"

    models_dir = V2X_DIR / "build" / "video2x-install" / "share" / "video2x"
    os.makedirs(os.path.dirname(job["output"]), exist_ok=True)

    try:
        proc = subprocess.Popen(
            [str(V2X_BIN)] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=V2X_ENV, bufsize=1, universal_newlines=True,
            cwd=str(models_dir),
        )
        active_processes[job["id"]] = {"type": "local", "proc": proc}

        if not gpu_poll_active:
            threading.Thread(target=poll_gpu_utilization, daemon=True).start()

        for raw_line in proc.stdout:
            for line in SPLIT_RE.split(raw_line):
                clean = ANSI_RE.sub("", line).strip()
                if not clean:
                    continue
                m = PROGRESS_RE.search(clean)
                if m:
                    job["frame"] = int(m.group(1))
                    job["total"] = int(m.group(2))
                    job["fps"] = float(m.group(4))
                    job["elapsed"] = m.group(5)
                    job["remaining"] = m.group(6).strip()
                    if job["total"] > 0:
                        job["progress"] = job["frame"] / job["total"]
                    continue
                if "frame=" in clean:
                    continue
                job["log"] += clean + "\n"
                if len(job["log"]) > 50000:
                    job["log"] = job["log"][-40000:]

        proc.wait()
        if proc.returncode == 0:
            job["status"] = "finished"
            job["progress"] = 1.0
            job["log"] += "\nProcessing completed successfully!\n"
        else:
            job["status"] = "failed"
            job["log"] += f"\nProcess exited with code {proc.returncode}\n"
    except Exception as e:
        job["status"] = "failed"
        job["log"] += f"\nError: {e}\n"
    finally:
        active_processes.pop(job["id"], None)


def run_remote_job(job, gpu_config):
    job["status"] = "running"
    _reset_job_progress(job)
    job["log"] = ""

    rgpu = RemoteGPU(job)
    active_processes[job["id"]] = {"type": "remote", "rgpu": rgpu}

    slurm = gpu_config.get("slurm", {})
    rgpu.slurm_opts = {
        "partition": gpu_config.get("partition", slurm.get("partition", "")),
        "gres": gpu_config.get("gres", slurm.get("gres", "gpu:1")),
        "mem": gpu_config.get("mem", slurm.get("mem", "32G")),
        "time": gpu_config.get("time_limit", slurm.get("time", "")),
        "qos": gpu_config.get("qos", slurm.get("qos", "")),
        "nice": gpu_config.get("nice", slurm.get("nice", "")),
        "extra_sbatch": gpu_config.get("extra_sbatch", slurm.get("extra_sbatch", "")),
    }
    use_slurm = gpu_config.get("type", "direct") == "slurm"

    gpu_key = _gpu_key(gpu_config)
    pw = gpu_passwords.get(gpu_key)

    try:
        job["stage"] = "Connecting..."
        ok = rgpu.connect(
            host=gpu_config["host"],
            username=gpu_config.get("user", ""),
            password=pw,
            key_path=gpu_config.get("key_path"),
            port=int(gpu_config.get("port", 22)),
            proxy=gpu_config.get("proxy"),
            auth=gpu_config.get("auth", "key"),
        )
        if not ok:
            job["status"] = "failed"
            job["stage"] = "Connection failed"
            return

        job["stage"] = "Setting up..."
        if not rgpu.install_video2x():
            job["status"] = "failed"
            job["stage"] = "Setup failed"
            rgpu.disconnect()
            return

        job["stage"] = "Uploading video..."
        remote_path = rgpu.upload_video(job["input"])

        job["stage"] = "Submitting job..." if use_slurm else "Processing..."
        args_str = build_remote_args_str(job)
        result_path = rgpu.process_video(remote_path, args_str, use_slurm=use_slurm)
        if not result_path:
            job["status"] = "failed"
            job["stage"] = "Processing failed"
            rgpu.disconnect()
            return

        job["stage"] = "Downloading result..."
        os.makedirs(job["output_dir"], exist_ok=True)
        local_result = rgpu.download_result(result_path, job["output_dir"])

        job["status"] = "finished"
        job["progress"] = 1.0
        job["stage"] = "Complete"
        job["output"] = local_result
    except Exception as e:
        job["status"] = "failed"
        job["stage"] = "Error"
        job["log"] += f"\nRemote error: {e}\n"
    finally:
        try:
            rgpu.disconnect()
        except Exception:
            pass
        active_processes.pop(job["id"], None)


def _probe_job(job):
    """Background probe for a newly added job."""
    info = probe_video(job["input"])
    job["input_info"] = format_info(info)


# ---------------------------------------------------------------------------
# pywebview JS API
# ---------------------------------------------------------------------------

class Api:
    def browse_files(self):
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Video Files (*.mp4;*.mkv;*.avi;*.mov;*.webm)",),
        )
        if result:
            return json.dumps({"paths": list(result)})
        return json.dumps({"paths": []})

    def browse_output_dir(self):
        window = webview.windows[0]
        result = window.create_file_dialog(webview.FOLDER_DIALOG)
        if result and len(result) > 0:
            return json.dumps({"path": result[0]})
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


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/jobs")
def get_jobs():
    local_name = devices[0]["name"] if devices else "Apple M3"
    gpu_list = [{"id": "local", "name": "Local GPU (" + local_name + ")"}]
    for i, c in enumerate(load_gpu_configs()):
        gpu_list.append({"id": str(i), "name": c.get("name", "Remote GPU " + str(i))})
    return jsonify(
        jobs=list(jobs.values()),
        output_dir=output_dir,
        gpu_util=gpu_util_data,
        gpus=gpu_list,
    )


@app.route("/api/jobs/add", methods=["POST"])
def add_jobs():
    data = request.json
    paths = data.get("paths", [])
    if "input" in data and not paths:
        paths = [data["input"]]

    added = []
    for path in paths:
        job_id = str(uuid.uuid4())[:8]
        out_dir = output_dir or os.path.dirname(path)
        job = {
            "id": job_id,
            "input": path,
            "input_name": os.path.basename(path),
            "input_info": "Analyzing\u2026",
            "output_dir": out_dir,
            "output": generate_output_path(path, out_dir),
            "gpu": data.get("gpu", "local"),
            "processor": data.get("processor", "realesrgan"),
            "scale": int(data.get("scale", 4)),
            "multiplier": int(data.get("multiplier", 2)),
            "model": data.get("model", "realesr-animevideov3"),
            "codec": data.get("codec", "libx264"),
            "width": int(data.get("width", 3840)),
            "height": int(data.get("height", 2160)),
            "status": "queued",
            "progress": 0.0,
            "frame": 0,
            "total": 0,
            "fps": 0.0,
            "elapsed": "00:00:00",
            "remaining": "--:--:--",
            "log": "",
            "stage": "",
            "remote_gpu_util": 0,
        }
        jobs[job_id] = job
        added.append(job)
        threading.Thread(target=_probe_job, args=(job,), daemon=True).start()

    return jsonify(ok=True, jobs=added)


@app.route("/api/jobs/<job_id>/start", methods=["POST"])
def start_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404
    if job["status"] not in ("queued", "failed", "cancelled"):
        return jsonify(error="Job not startable in current state"), 400

    data = request.json or {}
    if "gpu" in data:
        job["gpu"] = data["gpu"]

    # Store password for this session if provided
    if data.get("password"):
        configs = load_gpu_configs()
        try:
            gpu_idx = int(data.get("gpu", ""))
            if 0 <= gpu_idx < len(configs):
                gpu_passwords[_gpu_key(configs[gpu_idx])] = data["password"]
        except (ValueError, IndexError):
            pass

    job["output_dir"] = output_dir or os.path.dirname(job["input"])
    job["output"] = generate_output_path(job["input"], job["output_dir"])

    gpu = job["gpu"]
    if gpu == "local":
        threading.Thread(target=run_local_job, args=(job,), daemon=True).start()
    else:
        configs = load_gpu_configs()
        try:
            gpu_idx = int(gpu)
            if 0 <= gpu_idx < len(configs):
                threading.Thread(
                    target=run_remote_job,
                    args=(job, configs[gpu_idx]),
                    daemon=True,
                ).start()
            else:
                return jsonify(error="Invalid GPU index"), 400
        except (ValueError, IndexError):
            return jsonify(error="Invalid GPU selection"), 400

    return jsonify(ok=True)


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="Job not found"), 404

    try:
        p = active_processes.pop(job_id, None)
        if p:
            if p["type"] == "local":
                try:
                    p["proc"].terminate()
                    p["proc"].wait(timeout=3)
                except Exception:
                    try:
                        p["proc"].kill()
                    except Exception:
                        pass
            elif p["type"] == "remote":
                try:
                    p["rgpu"].cancel()
                except Exception:
                    pass
    except Exception:
        pass

    job["status"] = "cancelled"
    job["log"] += "\nCancelled by user.\n"
    return jsonify(ok=True)


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    if job_id in active_processes:
        return jsonify(error="Cannot delete a running job"), 400
    jobs.pop(job_id, None)
    return jsonify(ok=True)


@app.route("/api/gpus")
def get_gpus():
    local_name = devices[0]["name"] if devices else "Apple M3"
    gpu_list = [{"id": "local", "name": "Local GPU (" + local_name + ")"}]
    for i, c in enumerate(load_gpu_configs()):
        gpu_list.append({"id": str(i), "name": c.get("name", "Remote GPU " + str(i))})
    return jsonify(gpus=gpu_list)


@app.route("/api/jobs/<job_id>/log")
def get_job_log(job_id):
    job = jobs.get(job_id)
    log_text = job["log"] if job else "Job not found"
    name = job["input_name"] if job else "Unknown"
    return f"""<!DOCTYPE html><html><head><title>Log: {name}</title>
<style>body{{background:#1a1a2e;color:#aab;font-family:SF Mono,Menlo,monospace;font-size:12px;padding:16px;white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0}}
.bar{{position:fixed;top:0;left:0;right:0;padding:8px 16px;background:#16213e;border-bottom:1px solid rgba(255,255,255,.1);display:flex;gap:10px;align-items:center;z-index:10}}
.bar button{{background:#e94560;color:#fff;border:none;padding:5px 14px;border-radius:6px;cursor:pointer;font-size:12px}}
.bar .back{{background:rgba(255,255,255,.1);color:#eee}}
.bar span{{font-size:13px;font-weight:600;color:#eee}}
pre{{margin-top:44px;user-select:text;-webkit-user-select:text}}</style></head><body>
<div class="bar"><button class="back" onclick="window.location='/'">&#8592; Back</button><span>Log: {name}</span><button onclick="navigator.clipboard.writeText(document.getElementById('l').textContent)">Copy</button></div>
<pre id="l">{log_text}</pre></body></html>"""


@app.route("/api/open-folder", methods=["POST"])
def open_folder():
    path = (request.json or {}).get("path", "")
    if path:
        if os.path.isfile(path):
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["open", path])
    return jsonify(ok=True)


@app.route("/api/output-dir", methods=["POST"])
def set_output_dir():
    global output_dir
    output_dir = (request.json or {}).get("path", "")
    for job in jobs.values():
        if job["status"] == "queued":
            job["output_dir"] = output_dir or os.path.dirname(job["input"])
            job["output"] = generate_output_path(job["input"], job["output_dir"])
    return jsonify(ok=True, path=output_dir)


@app.route("/api/gpu-util")
def get_gpu_util():
    return jsonify(gpu_util_data)


@app.route("/api/devices")
def get_devices():
    return jsonify(devices)


@app.route("/api/probe", methods=["POST"])
def probe():
    path = (request.json or {}).get("path", "")
    return jsonify(probe_video(path))


@app.route("/api/remote/configs")
def remote_configs():
    return jsonify(configs=load_gpu_configs())


@app.route("/api/remote/configs/save", methods=["POST"])
def remote_config_save():
    gpu = dict(request.json)
    pw = gpu.pop("password", None)
    configs = load_gpu_configs()
    configs.append(gpu)
    save_gpu_configs(configs)
    if pw:
        gpu_passwords[_gpu_key(gpu)] = pw
    return jsonify(ok=True, configs=load_gpu_configs())


@app.route("/api/remote/configs/update", methods=["POST"])
def remote_config_update():
    data = request.json or {}
    idx = data.get("index", -1)
    gpu = data.get("gpu", {})
    configs = load_gpu_configs()
    if 0 <= idx < len(configs):
        configs[idx] = gpu
        save_gpu_configs(configs)
    return jsonify(ok=True, configs=load_gpu_configs())


@app.route("/api/remote/configs/delete", methods=["POST"])
def remote_config_delete():
    idx = (request.json or {}).get("index", -1)
    configs = load_gpu_configs()
    if 0 <= idx < len(configs):
        configs.pop(idx)
        save_gpu_configs(configs)
        for job in jobs.values():
            if job["status"] == "queued" and job["gpu"] != "local":
                try:
                    gi = int(job["gpu"])
                    if gi == idx:
                        job["gpu"] = "local"
                    elif gi > idx:
                        job["gpu"] = str(gi - 1)
                except ValueError:
                    pass
    return jsonify(ok=True, configs=load_gpu_configs())


# ---------------------------------------------------------------------------
# HTML page  (all CSS / HTML / JS inline)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video2X for Mac</title>
<style>
:root{
  --bg:#1a1a2e;--surface:#16213e;--surface2:#0f3460;
  --accent:#e94560;--accent2:#533483;
  --text:#eee;--text2:#aab;--success:#4ade80;
  --radius:12px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro',system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  height:100vh;display:flex;flex-direction:column;
  user-select:none;overflow:hidden;
}

/* ---- header ---- */
.header{
  display:flex;align-items:center;gap:12px;padding:14px 24px;
  background:linear-gradient(135deg,var(--surface),var(--surface2));
  border-bottom:1px solid rgba(255,255,255,.06);
}
.header h1{font-size:18px;font-weight:600}
.header .sub{font-size:12px;color:var(--text2)}
.header .gpu{
  margin-left:auto;font-size:12px;
  background:rgba(255,255,255,.08);padding:6px 14px;
  border-radius:20px;color:var(--text2);
}
.cloud-icon{
  font-size:22px;cursor:pointer;padding:4px 10px;
  border-radius:8px;transition:all .2s;margin-left:8px;color:var(--text2);
}
.cloud-icon:hover{background:rgba(255,255,255,.1);color:var(--accent)}
.cloud-icon.connected{color:var(--success);text-shadow:0 0 8px rgba(74,222,128,.4)}

/* ---- layout ---- */
.main{display:flex;flex:1;overflow:hidden}
.sidebar{
  width:350px;min-width:310px;overflow-y:auto;padding:20px;
  background:var(--surface);
  border-right:1px solid rgba(255,255,255,.06);
  display:flex;flex-direction:column;gap:20px;
}
.content{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* ---- sidebar widgets ---- */
.section-title{
  font-size:13px;font-weight:600;text-transform:uppercase;
  letter-spacing:.5px;color:var(--text2);margin-bottom:8px;
}
.output-dir-picker{
  display:flex;align-items:center;gap:10px;
  padding:10px 14px;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08);border-radius:8px;
  cursor:pointer;transition:all .2s;
}
.output-dir-picker:hover{border-color:var(--accent);background:rgba(233,69,96,.04)}
.output-dir-icon{font-size:20px;flex-shrink:0}
.output-dir-path{
  flex:1;font-size:12px;color:var(--text2);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.proc-card{
  padding:10px 14px;border-radius:8px;cursor:pointer;transition:all .15s;
  border:2px solid transparent;background:rgba(255,255,255,.03);
}
.proc-card:hover{background:rgba(255,255,255,.06)}
.proc-card.active{border-color:var(--accent);background:rgba(233,69,96,.08)}
.proc-card .name{font-size:14px;font-weight:600}
.proc-card .desc{font-size:11px;color:var(--text2);margin-top:2px}
.setting-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.setting-row label{font-size:13px;min-width:80px;color:var(--text2)}
.setting-row select,.setting-row input[type="number"]{
  flex:1;padding:6px 10px;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);
  border-radius:6px;color:var(--text);font-size:13px;
}
.setting-row select{appearance:auto}
.seg-control{
  display:flex;gap:0;border-radius:8px;overflow:hidden;
  border:1px solid rgba(255,255,255,.1);
}
.seg-control button{
  flex:1;padding:8px;border:none;
  background:rgba(255,255,255,.05);color:var(--text2);
  font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;
}
.seg-control button.active{background:var(--accent);color:#fff}
.add-videos-btn{
  display:flex;align-items:center;justify-content:center;gap:10px;
  padding:18px;border:2px dashed rgba(255,255,255,.15);
  border-radius:var(--radius);cursor:pointer;
  color:var(--text2);font-size:15px;font-weight:600;transition:all .2s;
}
.add-videos-btn:hover{
  border-color:var(--accent);color:var(--accent);
  background:rgba(233,69,96,.05);
}

/* ---- jobs area ---- */
.tabs-bar{
  display:flex;align-items:center;gap:0;
  padding:0 20px;
  border-bottom:1px solid rgba(255,255,255,.06);
  background:var(--surface);
}
.tab{
  padding:12px 18px;font-size:13px;font-weight:600;color:var(--text2);
  cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;
}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-count{
  font-size:11px;background:rgba(255,255,255,.08);padding:1px 7px;
  border-radius:10px;margin-left:6px;
}
.btn-start-all{
  padding:6px 14px;border:none;border-radius:6px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;font-size:12px;font-weight:600;cursor:pointer;
}
.btn-start-all:hover{opacity:.9}
.jobs-container{flex:1;overflow-y:auto;padding:14px 20px 20px}

/* ---- job card ---- */
.job-card{
  background:var(--surface);border:1px solid rgba(255,255,255,.08);
  border-radius:var(--radius);padding:16px;margin-bottom:12px;
  transition:border-color .3s;
}
@keyframes pulse-border{
  0%,100%{border-color:rgba(233,69,96,.15)}
  50%{border-color:rgba(233,69,96,.35)}
}
.job-card.status-running{animation:pulse-border 2s ease-in-out infinite}
.job-card.status-finished{border-color:rgba(74,222,128,.2)}
.job-card.status-failed{border-color:rgba(233,69,96,.2)}
.job-card.status-cancelled{border-color:rgba(255,200,50,.15)}
.job-header{display:flex;justify-content:space-between;align-items:flex-start}
.job-info{flex:1;min-width:0}
.job-filename{font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.job-meta{font-size:12px;color:var(--text2);margin-top:2px}
.job-proc-tag{
  display:inline-block;font-size:11px;padding:2px 8px;margin-top:4px;
  background:rgba(255,255,255,.06);border-radius:4px;color:var(--text2);
}
.job-delete{
  font-size:18px;cursor:pointer;color:var(--text2);
  padding:0 6px;border-radius:4px;flex-shrink:0;transition:all .15s;
}
.job-delete:hover{color:var(--accent);background:rgba(233,69,96,.1)}
.job-gpu-row{display:flex;align-items:center;gap:10px;margin-top:10px}
.job-gpu-row label{font-size:12px;color:var(--text2);min-width:30px}
.job-gpu-select{
  flex:1;padding:5px 8px;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);
  border-radius:6px;color:var(--text);font-size:12px;appearance:auto;
}
.job-gpu-select:disabled{opacity:.5}
.job-progress-row{display:flex;align-items:center;gap:10px;margin-top:10px}
.job-progress-bar-wrap{
  flex:1;height:6px;background:rgba(255,255,255,.08);
  border-radius:3px;overflow:hidden;
}
.job-progress-bar{
  height:100%;border-radius:3px;transition:width .3s;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
}
.status-finished .job-progress-bar{background:var(--success)}
.status-failed .job-progress-bar{background:var(--accent)}
.status-cancelled .job-progress-bar{background:#fbbf24}
.job-pct{
  font-size:13px;font-weight:700;min-width:38px;text-align:right;
  color:var(--accent);font-variant-numeric:tabular-nums;
}
.status-finished .job-pct{color:var(--success)}
.job-stats{
  display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;align-items:center;
  font-size:12px;color:var(--text2);font-variant-numeric:tabular-nums;
}
.job-stage{
  color:var(--accent);font-weight:600;font-style:italic;
}
.job-actions{display:flex;gap:8px;margin-top:10px}
.job-btn{
  padding:6px 16px;border-radius:6px;font-size:12px;
  font-weight:600;cursor:pointer;transition:all .15s;
}
.job-btn.start{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border:none;color:#fff;
}
.job-btn.cancel{
  background:transparent;border:1px solid var(--accent);color:var(--accent);
}
.job-btn.cancel:hover{background:rgba(233,69,96,.1)}
.job-btn.retry{
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
  color:var(--text);
}
.job-btn.retry:hover{border-color:var(--accent);color:var(--accent)}
.job-btn.remove{
  background:transparent;border:1px solid rgba(255,255,255,.08);
  color:var(--text2);
}
.job-btn.remove:hover{border-color:var(--accent);color:var(--accent)}
.job-log-toggle{
  font-size:11px;color:var(--text2);cursor:pointer;
  padding:2px 8px;border-radius:4px;background:rgba(255,255,255,.06);
  margin-left:auto;transition:all .15s;
}
.job-log-toggle:hover{color:var(--text);background:rgba(255,255,255,.1)}
.job-log{
  margin-top:8px;max-height:120px;overflow-y:auto;padding:8px 10px;
  background:rgba(0,0,0,.25);border-radius:6px;
  font-family:'SF Mono','Menlo',monospace;font-size:11px;
  line-height:1.5;color:var(--text2);white-space:pre-wrap;word-break:break-all;
  user-select:text;-webkit-user-select:text;
}
.badge{
  display:inline-flex;align-items:center;gap:6px;
  padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;
}
.badge.success{background:rgba(74,222,128,.1);color:var(--success)}
.badge.error{background:rgba(233,69,96,.1);color:var(--accent)}
.badge.warning{background:rgba(255,200,50,.1);color:#fbbf24}
.empty-state{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;height:100%;color:var(--text2);
}
.empty-icon{font-size:48px;margin-bottom:16px;opacity:.6}
.empty-text{font-size:18px;font-weight:600;margin-bottom:6px}
.empty-sub{font-size:13px}

/* ---- cloud panel ---- */
.pw-overlay{
  position:fixed;top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,.5);z-index:200;
  display:flex;align-items:center;justify-content:center;
}
.pw-dialog{
  background:var(--surface);border:1px solid rgba(255,255,255,.1);
  border-radius:14px;padding:24px;width:360px;
  box-shadow:0 16px 48px rgba(0,0,0,.6);
}
.pw-title{font-size:15px;font-weight:700;margin-bottom:16px}
.pw-input-wrap{position:relative;margin-bottom:16px}
.pw-input{
  width:100%;padding:10px 40px 10px 12px;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
  border-radius:8px;color:var(--text);font-size:14px;
  letter-spacing:2px;
}
.pw-input:focus{outline:none;border-color:var(--accent)}
.pw-eye{
  position:absolute;right:10px;top:50%;transform:translateY(-50%);
  cursor:pointer;font-size:18px;opacity:.5;user-select:none;
}
.pw-eye:hover{opacity:1}
.pw-buttons{display:flex;gap:10px;justify-content:flex-end}
.pw-btn{
  padding:8px 20px;border-radius:8px;font-size:13px;font-weight:600;
  cursor:pointer;border:none;
}
.pw-cancel{background:rgba(255,255,255,.08);color:var(--text2)}
.pw-cancel:hover{background:rgba(255,255,255,.12)}
.pw-ok{background:var(--accent);color:#fff}
.pw-ok:hover{opacity:.9}
.cloud-overlay{
  position:fixed;top:0;left:0;right:0;bottom:0;
  background:rgba(0,0,0,.3);z-index:99;
}
.cloud-panel{
  position:fixed;top:56px;right:16px;width:360px;
  background:var(--surface);border:1px solid rgba(255,255,255,.1);
  border-radius:12px;padding:20px;z-index:100;
  box-shadow:0 12px 40px rgba(0,0,0,.5);max-height:80vh;overflow-y:auto;
}
.cloud-panel-title{font-size:15px;font-weight:700;margin-bottom:6px}
.cloud-add-card{
  display:flex;align-items:center;justify-content:center;gap:8px;
  padding:16px;border:2px dashed rgba(255,255,255,.12);
  border-radius:10px;cursor:pointer;color:var(--text2);
  font-size:13px;transition:all .2s;margin-top:8px;
}
.cloud-add-card:hover{border-color:var(--accent);color:var(--accent);background:rgba(233,69,96,.04)}
.cloud-gpu-item{
  padding:10px 14px;border-radius:8px;
  border:1px solid rgba(255,255,255,.08);margin-bottom:6px;
  background:rgba(255,255,255,.03);transition:all .15s;position:relative;
}
.cloud-gpu-item:hover{border-color:var(--accent);background:rgba(233,69,96,.05)}
.cloud-gpu-item .cg-name{font-size:14px;font-weight:600}
.cloud-gpu-item .cg-host{font-size:11px;color:var(--text2);margin-top:2px}
.cloud-gpu-item .cg-tag{
  display:inline-block;font-size:10px;padding:2px 8px;
  background:rgba(255,255,255,.08);border-radius:4px;
  color:var(--text2);margin-top:4px;
}
.cloud-gpu-item .cg-delete{
  position:absolute;top:8px;right:10px;font-size:16px;
  color:var(--text2);cursor:pointer;opacity:1;padding:4px 8px;border-radius:4px;
  padding:2px 6px;border-radius:4px;
}
.cloud-gpu-item:hover .cg-delete{opacity:1}
.cloud-gpu-item .cg-delete:hover{color:var(--accent);background:rgba(233,69,96,.15)}
.cloud-gpu-item .cg-edit{
  position:absolute;top:8px;right:36px;font-size:14px;
  color:var(--text2);cursor:pointer;opacity:1;padding:4px 8px;border-radius:4px;
}
.cloud-gpu-item .cg-edit:hover{color:var(--accent);background:rgba(233,69,96,.15)}
.cloud-type-card{
  display:flex;align-items:center;gap:14px;
  padding:14px;border:1px solid rgba(255,255,255,.08);
  border-radius:10px;cursor:pointer;margin-bottom:8px;
  background:rgba(255,255,255,.03);transition:all .15s;
}
.cloud-type-card:hover{border-color:var(--accent);background:rgba(233,69,96,.05)}
.ct-icon{font-size:26px}
.ct-name{font-size:14px;font-weight:600}
.ct-desc{font-size:11px;color:var(--text2);margin-top:2px}
.cp-back{
  font-size:18px;cursor:pointer;padding:2px 8px;
  border-radius:6px;color:var(--text2);transition:all .15s;
}
.cp-back:hover{color:var(--text);background:rgba(255,255,255,.08)}
.cp-field{margin-bottom:10px}
.cp-field label{display:block;font-size:11px;color:var(--text2);margin-bottom:4px;font-weight:600}
.cp-field input,.cp-field select{
  width:100%;padding:8px 10px;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);
  border-radius:6px;color:var(--text);font-size:13px;
}
.cp-field select{appearance:auto}
.cp-field-row{display:flex;gap:10px}
.cp-field-row .cp-field{flex:1}
.btn-primary{
  width:100%;padding:14px;border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;font-size:15px;font-weight:600;cursor:pointer;transition:opacity .2s;
}
.btn-primary:hover{opacity:.9}
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{
  display:inline-block;width:16px;height:16px;
  border:2px solid rgba(255,200,50,.3);border-top-color:#fbbf24;
  border-radius:50%;animation:spin .6s linear infinite;
  vertical-align:middle;margin-right:8px;
}
</style>
</head>
<body>

<!-- ==================== HEADER ==================== -->
<div class="header">
  <div>
    <h1>Video2X for Mac</h1>
    <div class="sub">ML-powered video upscaling &amp; frame interpolation</div>
  </div>
  <div class="gpu" id="gpuBadge">Detecting GPU...</div>
  <div class="cloud-icon" id="cloudIcon" onclick="toggleCloudPanel()" title="Manage remote GPUs">&#9729;</div>
</div>

<!-- ==================== CLOUD PANEL ==================== -->
<div class="cloud-overlay" id="cloudOverlay" style="display:none" onclick="toggleCloudPanel()"></div>

<!-- Password dialog -->
<div class="pw-overlay" id="pwOverlay" style="display:none">
  <div class="pw-dialog">
    <div class="pw-title" id="pwTitle">Enter SSH Password</div>
    <div class="pw-input-wrap">
      <input type="password" id="pwInput" class="pw-input" placeholder="Password" onkeydown="if(event.key==='Enter')submitPwDialog()">
      <span class="pw-eye" onmousedown="document.getElementById('pwInput').type='text'" onmouseup="document.getElementById('pwInput').type='password'" onmouseleave="document.getElementById('pwInput').type='password'">&#128065;</span>
    </div>
    <div class="pw-buttons">
      <button class="pw-btn pw-cancel" onclick="cancelPwDialog()">Cancel</button>
      <button class="pw-btn pw-ok" onclick="submitPwDialog()">Connect</button>
    </div>
  </div>
</div>
<div class="cloud-panel" id="cloudPanel" style="display:none">

  <div id="cpList">
    <div class="cloud-panel-title">External GPUs</div>
    <div style="font-size:12px;color:var(--text2);margin-bottom:10px">Manage remote GPUs available for per-file assignment.</div>
    <div id="cloudCards"></div>
    <div class="cloud-add-card" onclick="showAddStep1()">
      <span style="font-size:22px">+</span><span>Add External GPU</span>
    </div>
  </div>

  <div id="cpStep1" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="showList()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0">Select GPU Type</div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('slurm')">
      <div class="ct-icon">&#128421;</div>
      <div><div class="ct-name">SLURM Cluster</div><div class="ct-desc">HPC cluster with job scheduler</div></div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('direct')">
      <div class="ct-icon">&#128187;</div>
      <div><div class="ct-name">SSH Direct</div><div class="ct-desc">GPU workstation or server with direct access</div></div>
    </div>
    <div class="cloud-type-card" onclick="showAddStep2('cloud')">
      <div class="ct-icon">&#9729;</div>
      <div><div class="ct-name">Cloud Provider</div><div class="ct-desc">RunPod, Vast.ai, Lambda, or any cloud GPU</div></div>
    </div>
  </div>

  <div id="cpStep2" style="display:none">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <span class="cp-back" onclick="showAddStep1()">&larr;</span>
      <div class="cloud-panel-title" style="margin:0" id="cpStep2Title">Configure</div>
    </div>
    <div id="cpFields"></div>
    <button class="btn-primary" style="font-size:13px;padding:10px;margin-top:8px" id="cpSaveBtn" onclick="saveNewGpu()">Add GPU</button>
  </div>
</div>

<!-- ==================== MAIN LAYOUT ==================== -->
<div class="main">

  <!-- ---- sidebar ---- -->
  <div class="sidebar">

    <div>
      <div class="section-title">Output Directory</div>
      <div class="output-dir-picker" onclick="browseOutputDir()">
        <span class="output-dir-icon">&#128193;</span>
        <span class="output-dir-path" id="outputDirDisplay">Click to select output folder</span>
      </div>
    </div>

    <div>
      <div class="section-title">Processor</div>
      <div id="procCards" style="display:flex;flex-direction:column;gap:6px"></div>
    </div>

    <div>
      <div class="section-title">Settings</div>
      <div id="settingsArea"></div>
      <div class="setting-row" style="margin-top:10px">
        <label>Codec</label>
        <select id="codec">
          <option value="libx264">H.264</option>
          <option value="libx265">H.265 (HEVC)</option>
        </select>
      </div>
    </div>

    <div class="add-videos-btn" onclick="browseFiles()">
      <span style="font-size:22px">+</span><span>Add Videos</span>
    </div>

  </div>

  <!-- ---- content ---- -->
  <div class="content">
    <div class="tabs-bar">
      <div class="tab active" id="tabActive" onclick="switchTab('active')">Active <span class="tab-count" id="activeCount"></span></div>
      <div class="tab" id="tabHistory" onclick="switchTab('history')">History <span class="tab-count" id="historyCount"></span></div>
      <div style="flex:1"></div>
      <button class="btn-start-all" id="btnStartAll" onclick="startAllQueued()" style="display:none">&#9654; Start All</button>
    </div>
    <div class="jobs-container" id="jobList">
      <div class="empty-state" id="emptyState">
        <div class="empty-icon">&#127916;</div>
        <div class="empty-text">Add videos to get started</div>
        <div class="empty-sub">Select video files to add them to the processing queue</div>
      </div>
    </div>
  </div>

</div>

<script>
/* ================================================================
   ES5 JavaScript — no const/let, no arrow functions, no templates
   ================================================================ */

var PROCESSORS = [
  {id:'realesrgan',name:'Real-ESRGAN',desc:'Best for general video & anime upscaling',
   models:['realesr-animevideov3','realesrgan-plus-anime','realesrgan-plus','realesr-generalv3']},
  {id:'realcugan',name:'Real-CUGAN',desc:'Optimized for anime upscaling',
   models:['models-se','models-pro','models-nose']},
  {id:'libplacebo',name:'Anime4K (libplacebo)',desc:'Shader-based upscaling',
   models:['anime4k-v4-a','anime4k-v4-a+a','anime4k-v4-b','anime4k-v4-b+b','anime4k-v4-c','anime4k-v4-c+a']},
  {id:'rife',name:'RIFE',desc:'Increases frame rate for smoother motion',
   models:['rife-v4.26','rife-v4.25','rife-v4.25-lite','rife-v4.6','rife-v4']}
];

var selectedProc = 'realesrgan';
var scale = 4;
var multiplier = 2;
var outputDir = '';
var jobsData = [];
var gpuList = [];
var gpuConfigs = [];
var addingType = '';
var editingIndex = -1;
var sessionPasswords = {};
var pwCallback = null;
var lastJobIds = '';
var lastJobStates = {};
var expandedLogs = {};

/* ---------- init ---------- */

function init() {
  renderProcessors();
  renderSettings();
  fetch('/api/devices').then(function(r){return r.json()}).then(function(devs) {
    if (devs.length > 0) {
      document.getElementById('gpuBadge').textContent = '\u{1F7E2} ' + devs[0].name;
    } else {
      document.getElementById('gpuBadge').textContent = 'No GPU found';
    }
  }).catch(function(){});
  startPolling();
}

/* ---------- processors & settings ---------- */

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

/* ---------- file & dir browsing ---------- */

function browseOutputDir() {
  if (window.pywebview) {
    window.pywebview.api.browse_output_dir().then(function(result) {
      var d = JSON.parse(result);
      if (d.path) {
        outputDir = d.path;
        document.getElementById('outputDirDisplay').textContent = d.path;
        document.getElementById('outputDirDisplay').title = d.path;
        fetch('/api/output-dir', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({path: d.path})
        });
      }
    });
  } else {
    var path = prompt('Enter output directory path:');
    if (path) {
      outputDir = path;
      document.getElementById('outputDirDisplay').textContent = path;
      fetch('/api/output-dir', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({path: path})
      });
    }
  }
}

function browseFiles() {
  if (window.pywebview) {
    window.pywebview.api.browse_files().then(function(result) {
      var d = JSON.parse(result);
      if (d.paths && d.paths.length > 0) {
        addFiles(d.paths);
      }
    });
  } else {
    var path = prompt('Enter video file path:');
    if (path) addFiles([path]);
  }
}

function addFiles(paths) {
  var modelEl = document.getElementById('model');
  var codecEl = document.getElementById('codec');
  var body = {
    paths: paths,
    processor: selectedProc,
    scale: scale,
    multiplier: multiplier,
    model: modelEl ? modelEl.value : '',
    codec: codecEl ? codecEl.value : 'libx264'
  };
  if (selectedProc === 'libplacebo') {
    var wEl = document.getElementById('outW');
    var hEl = document.getElementById('outH');
    body.width = parseInt(wEl ? wEl.value : '3840');
    body.height = parseInt(hEl ? hEl.value : '2160');
  }
  fetch('/api/jobs/add', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  }).then(function(r){return r.json()}).then(function() {
    switchTab('active');
    lastJobIds = '';
  });
}

/* ---------- polling ---------- */

function startPolling() {
  setInterval(function() {
    fetch('/api/jobs').then(function(r){return r.json()}).then(function(d) {
      jobsData = d.jobs;
      gpuList = d.gpus || [];
      if (d.output_dir && !outputDir) {
        outputDir = d.output_dir;
        document.getElementById('outputDirDisplay').textContent = d.output_dir;
      }
      updateJobsUI();
      updateGpuBadge(d.gpu_util);
    }).catch(function(){});
  }, 500);
}

function updateGpuBadge(gu) {
  var badge = document.getElementById('gpuBadge');
  var cloudIcon = document.getElementById('cloudIcon');
  var hasLocalRunning = false;
  var hasRemoteRunning = false;
  for (var i = 0; i < jobsData.length; i++) {
    if (jobsData[i].status === 'running') {
      if (jobsData[i].gpu === 'local') hasLocalRunning = true;
      else hasRemoteRunning = true;
    }
  }

  if (hasLocalRunning && gu && gu.gpu_util > 0) {
    badge.textContent = '\u{1F7E2} Apple M3 \u2022 ' + gu.gpu_util + '%';
  } else {
    badge.textContent = '\u{1F7E2} Apple M3';
  }

  if (hasRemoteRunning) {
    cloudIcon.classList.add('connected');
  } else {
    cloudIcon.classList.remove('connected');
  }
}

/* ---------- job list rendering ---------- */

function updateJobsUI() {
  var needFullRender = false;
  var currentIds = [];
  for (var i = 0; i < jobsData.length; i++) {
    var j = jobsData[i];
    currentIds.push(j.id + ':' + j.status);
    if (lastJobStates[j.id] !== j.status) needFullRender = true;
  }
  var idsStr = currentIds.join(',');
  if (idsStr !== lastJobIds) needFullRender = true;

  if (needFullRender) {
    renderJobList();
    lastJobIds = idsStr;
    lastJobStates = {};
    for (var i = 0; i < jobsData.length; i++) {
      lastJobStates[jobsData[i].id] = jobsData[i].status;
    }
  } else {
    for (var i = 0; i < jobsData.length; i++) {
      updateJobCard(jobsData[i]);
    }
  }
}

function renderJobList() {
  var container = document.getElementById('jobList');
  var activeJobs = [];
  var historyJobs = [];
  for (var i = 0; i < jobsData.length; i++) {
    var s = jobsData[i].status;
    if (s === 'queued' || s === 'running' || s === 'cancelling') {
      activeJobs.push(jobsData[i]);
    } else {
      historyJobs.push(jobsData[i]);
    }
  }

  document.getElementById('activeCount').textContent = activeJobs.length || '';
  document.getElementById('historyCount').textContent = historyJobs.length || '';

  var visible = currentTab === 'active' ? activeJobs : historyJobs;
  var startAll = document.getElementById('btnStartAll');
  var hasQueued = false;
  for (var i = 0; i < activeJobs.length; i++) {
    if (activeJobs[i].status === 'queued') { hasQueued = true; break; }
  }
  startAll.style.display = hasQueued && currentTab === 'active' ? '' : 'none';

  if (visible.length === 0) {
    var msg = currentTab === 'active' ? 'No active jobs. Add videos to get started.' : 'No completed jobs yet.';
    container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#127916;</div><div class="empty-text">' + msg + '</div></div>';
    return;
  }

  var html = '';
  for (var i = 0; i < visible.length; i++) {
    html += buildJobCardHTML(visible[i]);
  }
  container.innerHTML = html;
}

function formatProcTag(job) {
  var names = {realesrgan:'Real-ESRGAN',realcugan:'Real-CUGAN',libplacebo:'Anime4K',rife:'RIFE'};
  var n = names[job.processor] || job.processor;
  if (job.processor === 'rife') return n + ' \u00b7 ' + job.multiplier + 'x \u00b7 ' + job.model;
  if (job.processor === 'libplacebo') return n + ' \u00b7 ' + job.width + 'x' + job.height + ' \u00b7 ' + job.model;
  return n + ' \u00b7 ' + job.scale + 'x \u00b7 ' + job.model;
}

function escH(t) {
  var d = document.createElement('div');
  d.appendChild(document.createTextNode(t || ''));
  return d.innerHTML;
}

function buildJobCardHTML(job) {
  var sc = 'status-' + job.status;
  var h = '<div class="job-card ' + sc + '" id="job-' + job.id + '">';

  /* header */
  h += '<div class="job-header"><div class="job-info">';
  h += '<div class="job-filename">' + escH(job.input_name) + '</div>';
  h += '<div class="job-meta">' + escH(job.input_info) + '</div>';
  h += '<div class="job-proc-tag">' + escH(formatProcTag(job)) + '</div>';
  h += '</div>';
  if (job.status !== 'running') {
    h += '<span class="job-delete" onclick="deleteJob(\'' + job.id + '\')">&times;</span>';
  }
  h += '</div>';

  /* gpu selector */
  var gpuDisabled = (job.status !== 'queued' && job.status !== 'failed' && job.status !== 'cancelled') ? ' disabled' : '';
  h += '<div class="job-gpu-row"><label>GPU</label>';
  h += '<select class="job-gpu-select" id="gpu-' + job.id + '" onchange="setJobGpu(\'' + job.id + '\',this.value)"' + gpuDisabled + '>';
  for (var g = 0; g < gpuList.length; g++) {
    var sel = gpuList[g].id === job.gpu ? ' selected' : '';
    h += '<option value="' + gpuList[g].id + '"' + sel + '>' + escH(gpuList[g].name) + '</option>';
  }
  h += '</select></div>';

  /* progress */
  var pct = Math.round(job.progress * 100);
  h += '<div class="job-progress-row">';
  h += '<div class="job-progress-bar-wrap"><div class="job-progress-bar" id="bar-' + job.id + '" style="width:' + pct + '%"></div></div>';
  h += '<span class="job-pct" id="pct-' + job.id + '">' + pct + '%</span>';
  h += '</div>';

  /* stats */
  h += '<div class="job-stats" id="stats-' + job.id + '">';
  if (job.status === 'running') {
    if (job.stage && job.fps <= 0) {
      h += '<span class="job-stage">' + escH(job.stage) + '</span>';
    } else {
      h += '<span>FPS: ' + (job.fps > 0 ? job.fps.toFixed(1) : '--') + '</span>';
      h += '<span>Frame: ' + (job.total > 0 ? job.frame + '/' + job.total : '--') + '</span>';
      h += '<span>' + job.elapsed + ' / ' + job.remaining + '</span>';
      if (job.stage) h += '<span class="job-stage">' + escH(job.stage) + '</span>';
    }
  } else if (job.status === 'finished') {
    h += '<span class="badge success">\u2713 Complete</span>';
  } else if (job.status === 'failed') {
    h += '<span class="badge error">\u2717 Failed</span>';
  } else if (job.status === 'cancelled') {
    h += '<span class="badge warning">\u2718 Cancelled</span>';
  }
  h += '</div>';

  /* actions + log toggle */
  h += '<div class="job-actions" id="actions-' + job.id + '">';
  if (job.status === 'queued') {
    h += '<button class="job-btn start" onclick="startJob(\'' + job.id + '\')">&#9654; Start</button>';
  } else if (job.status === 'running') {
    h += '<button class="job-btn cancel" onclick="cancelJob(\'' + job.id + '\')">Cancel</button>';
  } else if (job.status === 'failed' || job.status === 'cancelled') {
    h += '<button class="job-btn retry" onclick="startJob(\'' + job.id + '\')">&#8635; Retry</button>';
    h += '<button class="job-btn remove" onclick="deleteJob(\'' + job.id + '\')">Remove</button>';
  } else if (job.status === 'finished') {
    h += '<button class="job-btn remove" onclick="deleteJob(\'' + job.id + '\')">Remove</button>';
  }
  if (job.log) {
    h += '<button class="job-btn" onclick="openLogWindow(\'' + job.id + '\')">View Log</button>';
  }
  if (job.status === 'finished' && job.output) {
    h += '<button class="job-btn" onclick="openOutputFolder(\'' + job.id + '\')">Open Folder</button>';
  }
  h += '</div>';

  h += '</div>';
  return h;
}

function updateJobCard(job) {
  var pct = Math.round(job.progress * 100);
  var barEl = document.getElementById('bar-' + job.id);
  if (barEl) barEl.style.width = pct + '%';
  var pctEl = document.getElementById('pct-' + job.id);
  if (pctEl) pctEl.textContent = pct + '%';

  if (job.status === 'running') {
    var statsEl = document.getElementById('stats-' + job.id);
    if (statsEl) {
      var sh = '';
      if (job.stage && job.fps <= 0) {
        sh = '<span class="job-stage">' + escH(job.stage) + '</span>';
      } else {
        sh = '<span>FPS: ' + (job.fps > 0 ? job.fps.toFixed(1) : '--') + '</span>' +
          '<span>Frame: ' + (job.total > 0 ? job.frame + '/' + job.total : '--') + '</span>' +
          '<span>' + job.elapsed + ' / ' + job.remaining + '</span>';
        if (job.stage) sh += '<span class="job-stage">' + escH(job.stage) + '</span>';
        if (job.remote_gpu_util > 0) sh += '<span>GPU: ' + job.remote_gpu_util + '%</span>';
      }
      statsEl.innerHTML = sh;
    }
  }

  // Status changes require full re-render
  var cardEl = document.getElementById('job-' + job.id);
  if (cardEl && cardEl.className.indexOf('status-' + job.status) === -1) {
    lastJobIds = '';
  }
}

/* ---------- job actions ---------- */

function setJobGpu(jobId, value) {
  for (var i = 0; i < jobsData.length; i++) {
    if (jobsData[i].id === jobId) { jobsData[i].gpu = value; break; }
  }
}

function startJob(jobId) {
  var gpu = 'local';
  var sel = document.getElementById('gpu-' + jobId);
  if (sel) gpu = sel.value;

  if (gpu !== 'local') {
    var gpuName = '';
    for (var i = 0; i < gpuList.length; i++) {
      if (gpuList[i].id === gpu) { gpuName = gpuList[i].name; break; }
    }
    var cachedKey = gpu + '_pw';
    var pw = sessionPasswords[cachedKey] || '';
    if (!pw) {
      showPasswordDialog(gpuName || 'remote GPU', function(enteredPw) {
        if (!enteredPw) return;
        sessionPasswords[cachedKey] = enteredPw;
        doStartJob(jobId, gpu, enteredPw);
      });
      return;
    }
    doStartJob(jobId, gpu, pw);
  } else {
    doStartJob(jobId, gpu, '');
  }
}

function showPasswordDialog(gpuName, callback) {
  pwCallback = callback;
  document.getElementById('pwTitle').textContent = 'SSH Password for ' + gpuName;
  document.getElementById('pwInput').value = '';
  document.getElementById('pwOverlay').style.display = 'flex';
  setTimeout(function() { document.getElementById('pwInput').focus(); }, 100);
}

function submitPwDialog() {
  var pw = document.getElementById('pwInput').value;
  document.getElementById('pwOverlay').style.display = 'none';
  if (pwCallback) pwCallback(pw);
  pwCallback = null;
}

function cancelPwDialog() {
  document.getElementById('pwOverlay').style.display = 'none';
  if (pwCallback) pwCallback(null);
  pwCallback = null;
}

function doStartJob(jobId, gpu, pw) {
  var body = {gpu: gpu};
  if (pw) body.password = pw;
  fetch('/api/jobs/' + jobId + '/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  lastJobIds = '';
}

function cancelJob(jobId) {
  fetch('/api/jobs/' + jobId + '/cancel', {method:'POST'});
  lastJobIds = '';
}

function deleteJob(jobId) {
  fetch('/api/jobs/' + jobId, {method:'DELETE'}).then(function() {
    lastJobIds = '';
    delete expandedLogs[jobId];
  });
}

function startAllQueued() {
  for (var i = 0; i < jobsData.length; i++) {
    if (jobsData[i].status === 'queued') startJob(jobsData[i].id);
  }
}

function clearCompleted() {
  for (var i = 0; i < jobsData.length; i++) {
    var s = jobsData[i].status;
    if (s === 'finished' || s === 'failed' || s === 'cancelled') {
      fetch('/api/jobs/' + jobsData[i].id, {method:'DELETE'});
    }
  }
  lastJobIds = '';
}

var currentTab = 'active';

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabActive').className = tab === 'active' ? 'tab active' : 'tab';
  document.getElementById('tabHistory').className = tab === 'history' ? 'tab active' : 'tab';
  lastJobIds = '';
}

function openLogWindow(jobId) {
  window.location.href = '/api/jobs/' + jobId + '/log';
}

function openOutputFolder(jobId) {
  for (var i = 0; i < jobsData.length; i++) {
    if (jobsData[i].id === jobId && jobsData[i].output) {
      fetch('/api/open-folder', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({path: jobsData[i].output})});
      return;
    }
  }
}

/* ---------- cloud panel ---------- */

function cpShowOnly(id) {
  var ids = ['cpList','cpStep1','cpStep2'];
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
    showList();
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
    html += '<div class="cloud-gpu-item">';
    html += '<span class="cg-delete" onclick="event.stopPropagation();deleteGpu(' + i + ')">\u00d7</span>';
    html += '<span class="cg-edit" onclick="event.stopPropagation();editGpu(' + i + ')">&#9998;</span>';
    html += '<div class="cg-name">' + escH(g.name) + '</div>';
    html += '<div class="cg-host">' + escH(g.user + '@' + g.host + ':' + g.port) + proxy + '</div>';
    html += '<span class="cg-tag">' + type + '</span>';
    if (g.auth === 'password') html += ' <span class="cg-tag">\u{1F511} password</span>';
    if (g.partition) html += ' <span class="cg-tag">p:' + escH(g.partition) + '</span>';
    if (g.qos) html += ' <span class="cg-tag">qos:' + escH(g.qos) + '</span>';
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
    lastJobIds = '';
    renderCards();
  });
}

function showAddStep1() { editingIndex = -1; cpShowOnly('cpStep1'); }

function editGpu(idx) {
  var g = gpuConfigs[idx];
  if (!g) return;
  editingIndex = idx;
  addingType = g.type || 'direct';
  showAddStep2(addingType, g);
}

function showAddStep2(type, prefill) {
  addingType = type;
  if (!prefill) editingIndex = -1;
  var pf = prefill || {};
  cpShowOnly('cpStep2');
  var isEdit = editingIndex >= 0;
  var titles = {slurm:(isEdit?'Edit':'Add') + ' SLURM Cluster', direct:(isEdit?'Edit':'Add') + ' SSH GPU', cloud:(isEdit?'Edit':'Add') + ' Cloud GPU'};
  document.getElementById('cpStep2Title').textContent = titles[type] || 'Configure';

  var html = '';
  html += '<div class="cp-field"><label>Name</label><input id="af_name" placeholder="e.g. Lab A100" value="' + escH(pf.name || '') + '"></div>';
  html += '<div class="cp-field-row">';
  html += '<div class="cp-field"><label>Host</label><input id="af_host" placeholder="cluster.uni.edu" value="' + escH(pf.host || '') + '"></div>';
  html += '<div class="cp-field" style="max-width:80px"><label>Port</label><input id="af_port" type="number" value="' + (pf.port || 22) + '"></div>';
  html += '</div>';
  html += '<div class="cp-field"><label>Username</label><input id="af_user" placeholder="jdoe" value="' + escH(pf.user || '') + '"></div>';
  html += '<div class="cp-field"><label>Authentication</label><select id="af_auth" onchange="togglePasswordField()">';
  var authOpts = [{v:'key',l:'SSH Key (~/.ssh/id_rsa)'},{v:'password',l:'Password'},{v:'agent',l:'SSH Agent'}];
  for (var a = 0; a < authOpts.length; a++) {
    var asel = (pf.auth || 'key') === authOpts[a].v ? ' selected' : '';
    html += '<option value="' + authOpts[a].v + '"' + asel + '>' + authOpts[a].l + '</option>';
  }
  html += '</select></div>';
  var pwShow = pf.auth === 'password' ? 'block' : 'none';
  html += '<div class="cp-field" id="af_password_field" style="display:' + pwShow + '"><label>Password</label><input id="af_password" type="password" placeholder="SSH password"></div>';

  if (type === 'slurm') {
    html += '<div style="margin:14px 0 8px;font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px">SLURM Settings</div>';
    html += '<div class="cp-field"><label>Partition <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_partition" placeholder="gpu" value="' + escH(pf.partition || '') + '"></div>';
    html += '<div class="cp-field-row">';
    html += '<div class="cp-field"><label>GPU Resource</label><input id="af_gres" value="' + escH(pf.gres || 'gpu:1') + '"></div>';
    html += '<div class="cp-field"><label>Memory</label><input id="af_mem" value="' + escH(pf.mem || '32G') + '"></div>';
    html += '</div>';
    html += '<div class="cp-field"><label>Time Limit <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_time" placeholder="no limit" value="' + escH(pf.time_limit || '') + '"></div>';
    html += '<div class="cp-field-row">';
    html += '<div class="cp-field"><label>QoS <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_qos" placeholder="e.g. gpu_normal" value="' + escH(pf.qos || '') + '"></div>';
    html += '<div class="cp-field"><label>Nice <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_nice" placeholder="e.g. 10000" value="' + escH(pf.nice || '') + '"></div>';
    html += '</div>';
    html += '<div class="cp-field"><label>Extra sbatch flags <span style="color:var(--text2);font-weight:400">(comma-separated)</span></label><input id="af_extra" placeholder="e.g. --exclude=node01,--constraint=a100" value="' + escH(pf.extra_sbatch || '') + '"></div>';
  }

  html += '<div style="margin:14px 0 8px;font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px">Advanced</div>';
  html += '<div class="cp-field"><label>Jump Host / Proxy <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_proxy" placeholder="jdoe@login.hpc.university.edu" value="' + escH(pf.proxy || '') + '"></div>';
  html += '<div class="cp-field"><label>SSH Key Path <span style="color:var(--text2);font-weight:400">(optional)</span></label><input id="af_keypath" placeholder="~/.ssh/id_rsa" value="' + escH(pf.key_path || '') + '"></div>';

  document.getElementById('cpFields').innerHTML = html;
  document.getElementById('cpSaveBtn').textContent = isEdit ? 'Save Changes' : 'Add GPU';
}

function togglePasswordField() {
  var auth = document.getElementById('af_auth').value;
  var field = document.getElementById('af_password_field');
  if (field) field.style.display = auth === 'password' ? 'block' : 'none';
}

function saveNewGpu() {
  var name = document.getElementById('af_name').value.trim();
  var host = document.getElementById('af_host').value.trim();
  var user = document.getElementById('af_user').value.trim();
  if (!name || !host || !user) { alert('Fill in name, host, and username.'); return; }

  var gpu = {
    name: name, type: addingType, host: host,
    port: parseInt(document.getElementById('af_port').value) || 22,
    user: user, auth: document.getElementById('af_auth').value
  };

  var passEl = document.getElementById('af_password');
  if (passEl && passEl.value) gpu.password = passEl.value;

  var proxyEl = document.getElementById('af_proxy');
  var keyEl = document.getElementById('af_keypath');
  if (proxyEl && proxyEl.value.trim()) gpu.proxy = proxyEl.value.trim();
  if (keyEl && keyEl.value.trim()) gpu.key_path = keyEl.value.trim();

  if (addingType === 'slurm') {
    var p = document.getElementById('af_partition');
    if (p && p.value.trim()) gpu.partition = p.value.trim();
    gpu.gres = document.getElementById('af_gres').value.trim() || 'gpu:1';
    gpu.mem = document.getElementById('af_mem').value.trim() || '32G';
    var timeVal = document.getElementById('af_time').value.trim();
    if (timeVal) gpu.time_limit = timeVal;
    var qosVal = document.getElementById('af_qos').value.trim();
    if (qosVal) gpu.qos = qosVal;
    var niceVal = document.getElementById('af_nice').value.trim();
    if (niceVal) gpu.nice = niceVal;
    var extraVal = document.getElementById('af_extra').value.trim();
    if (extraVal) gpu.extra_sbatch = extraVal;
  }

  var url, body;
  if (editingIndex >= 0) {
    url = '/api/remote/configs/update';
    body = JSON.stringify({index: editingIndex, gpu: gpu});
  } else {
    url = '/api/remote/configs/save';
    body = JSON.stringify(gpu);
  }

  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: body})
    .then(function(r){return r.json()}).then(function(d) {
      gpuConfigs = d.configs;
      editingIndex = -1;
      lastJobIds = '';
      showList();
    });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detect_devices()
    port = 52845

    threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1", port=port, debug=False, use_reloader=False
        ),
        daemon=True,
    ).start()
    time.sleep(0.5)

    window = webview.create_window(
        "Video2X for Mac",
        f"http://127.0.0.1:{port}",
        width=1200,
        height=780,
        min_size=(900, 600),
        confirm_close=True,
        js_api=js_api,
    )
    webview.start()
