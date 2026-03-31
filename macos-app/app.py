#!/usr/bin/env python3
"""Video2X for Mac — native macOS GUI for video upscaling and frame interpolation."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import webview
from flask import Flask, jsonify, request, send_from_directory

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
    "input_info": {},
}
process_handle = None
PROGRESS_RE = re.compile(
    r"frame=(\d+)/(\d+)\s+\(([^)]+)\);\s+fps=([^;]+);\s+elapsed=([^;]+);\s+remaining=(.+)"
)


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

    try:
        proc = subprocess.Popen(
            [str(V2X_BIN)] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=V2X_ENV, bufsize=1, universal_newlines=True
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


# --- Flask Routes ---

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/state")
def get_state():
    return jsonify(state)


@app.route("/api/devices")
def get_devices():
    detect_devices()
    return jsonify(state["devices"])


@app.route("/api/probe", methods=["POST"])
def probe():
    path = request.json.get("path", "")
    info = probe_video(path)
    state["input_info"] = info
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
    return jsonify({"ok": True})


@app.route("/api/cancel", methods=["POST"])
def cancel():
    global process_handle
    if process_handle:
        process_handle.terminate()
        state["status"] = "idle"
        state["log"] += "\nCancelled by user.\n"
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    state["status"] = "idle"
    return jsonify({"ok": True})


@app.route("/api/browse", methods=["POST"])
def browse():
    kind = request.json.get("kind", "open")
    window = webview.windows[0] if webview.windows else None
    if not window:
        return jsonify({"path": ""})

    if kind == "open":
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video Files (*.mp4;*.mkv;*.avi;*.mov;*.webm)",)
        )
        return jsonify({"path": result[0] if result else ""})
    else:
        result = window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename="output.mp4",
            file_types=("MP4 Video (*.mp4)",)
        )
        return jsonify({"path": result if result else ""})


HTML_PAGE = """<!DOCTYPE html>
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
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  user-select: none;
  overflow: hidden;
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
.main {
  display: flex; flex: 1; overflow: hidden;
}
.sidebar {
  width: 360px; min-width: 320px;
  overflow-y: auto; padding: 20px;
  background: var(--surface);
  border-right: 1px solid rgba(255,255,255,.06);
  display: flex; flex-direction: column; gap: 20px;
}
.content { flex: 1; display: flex; flex-direction: column; }

.section-title {
  font-size: 13px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .5px; color: var(--text2); margin-bottom: 8px;
}

/* Drop zone */
.dropzone {
  border: 2px dashed rgba(255,255,255,.15);
  border-radius: var(--radius);
  padding: 24px; text-align: center;
  cursor: pointer; transition: all .2s;
}
.dropzone:hover, .dropzone.over {
  border-color: var(--accent);
  background: rgba(233,69,96,.05);
}
.dropzone .icon { font-size: 28px; margin-bottom: 8px; }
.dropzone .filename { font-size: 14px; font-weight: 500; color: var(--text); }
.dropzone .info { font-size: 12px; color: var(--text2); margin-top: 4px; }

/* Processor cards */
.proc-card {
  padding: 10px 14px; border-radius: 8px;
  cursor: pointer; transition: all .15s;
  border: 2px solid transparent;
  background: rgba(255,255,255,.03);
}
.proc-card:hover { background: rgba(255,255,255,.06); }
.proc-card.active {
  border-color: var(--accent);
  background: rgba(233,69,96,.08);
}
.proc-card .name { font-size: 14px; font-weight: 600; }
.proc-card .desc { font-size: 11px; color: var(--text2); margin-top: 2px; }

/* Settings */
.setting-row {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
}
.setting-row label { font-size: 13px; min-width: 80px; color: var(--text2); }
.setting-row select, .setting-row input {
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
.seg-control button.active {
  background: var(--accent); color: #fff;
}

/* Action button */
.btn-primary {
  width: 100%; padding: 14px; border: none; border-radius: var(--radius);
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff; font-size: 15px; font-weight: 600;
  cursor: pointer; transition: opacity .2s;
}
.btn-primary:hover { opacity: .9; }
.btn-primary:disabled { opacity: .4; cursor: not-allowed; }
.btn-cancel {
  width: 100%; padding: 14px; border: 2px solid var(--accent);
  border-radius: var(--radius); background: transparent;
  color: var(--accent); font-size: 15px; font-weight: 600;
  cursor: pointer; transition: all .2s;
}
.btn-cancel:hover { background: rgba(233,69,96,.1); }

/* Progress area */
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
.stats {
  display: flex; gap: 0; justify-content: space-around;
}
.stat { text-align: center; }
.stat .val { font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stat .lbl { font-size: 11px; color: var(--text2); margin-top: 2px; }

/* Log */
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

/* Status badges */
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 600;
}
.badge.success { background: rgba(74,222,128,.1); color: var(--success); }
.badge.error { background: rgba(233,69,96,.1); color: var(--accent); }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Video2X for Mac</h1>
    <div class="sub">ML-powered video upscaling &amp; frame interpolation</div>
  </div>
  <div class="gpu" id="gpuBadge">Detecting GPU...</div>
</div>

<div class="main">
  <div class="sidebar">

    <div>
      <div class="section-title">Input Video</div>
      <div class="dropzone" id="dropzone" onclick="browseInput()">
        <div class="icon">🎬</div>
        <div class="filename" id="inputName">Drop video here or click to browse</div>
        <div class="info" id="inputInfo"></div>
      </div>
    </div>

    <div>
      <div class="section-title">Processor</div>
      <div style="display:flex;flex-direction:column;gap:6px" id="procCards"></div>
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
        <input id="outputPath" placeholder="Auto-generated" readonly style="cursor:pointer" onclick="browseOutput()">
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

let selectedProc = 'realesrgan';
let inputPath = '';
let scale = 4;
let multiplier = 2;
let polling = null;

function init() {
  renderProcessors();
  renderSettings();
  renderAction();
  fetch('/api/devices').then(r=>r.json()).then(devs => {
    if (devs.length > 0) {
      document.getElementById('gpuBadge').textContent = '🟢 ' + devs[0].name;
    }
  });
  startPolling();
}

function renderProcessors() {
  const el = document.getElementById('procCards');
  el.innerHTML = PROCESSORS.map(p =>
    '<div class="proc-card '+(p.id===selectedProc?'active':'')+'" onclick="selectProc(\''+p.id+'\')">' +
    '<div class="name">'+p.name+'</div><div class="desc">'+p.desc+'</div></div>'
  ).join('');
}

function selectProc(id) {
  selectedProc = id;
  renderProcessors();
  renderSettings();
}

function renderSettings() {
  const proc = PROCESSORS.find(p=>p.id===selectedProc);
  const isUpscaler = selectedProc !== 'rife';
  let html = '';

  if (isUpscaler && selectedProc !== 'libplacebo') {
    html += '<div style="margin-bottom:10px"><div style="font-size:12px;color:var(--text2);margin-bottom:6px">Scale Factor</div>';
    html += '<div class="seg-control">';
    [2,3,4].forEach(s => {
      html += '<button class="'+(scale===s?'active':'')+'" onclick="scale='+s+';renderSettings()">'+s+'x</button>';
    });
    html += '</div></div>';
  } else if (selectedProc === 'libplacebo') {
    html += '<div class="setting-row"><label>Width</label><input id="outW" type="number" value="3840"></div>';
    html += '<div class="setting-row"><label>Height</label><input id="outH" type="number" value="2160"></div>';
  } else {
    html += '<div style="margin-bottom:10px"><div style="font-size:12px;color:var(--text2);margin-bottom:6px">Frame Rate Multiplier</div>';
    html += '<div class="seg-control">';
    [2,3,4].forEach(m => {
      html += '<button class="'+(multiplier===m?'active':'')+'" onclick="multiplier='+m+';renderSettings()">'+m+'x</button>';
    });
    html += '</div></div>';
  }

  html += '<div class="setting-row"><label>Model</label><select id="model">';
  proc.models.forEach(m => { html += '<option value="'+m+'">'+m+'</option>'; });
  html += '</select></div>';

  document.getElementById('settingsArea').innerHTML = html;
}

function renderAction() {
  const el = document.getElementById('actionArea');
  fetch('/api/state').then(r=>r.json()).then(s => {
    if (s.status === 'running') {
      el.innerHTML = '<button class="btn-cancel" onclick="cancelProcessing()">Cancel Processing</button>';
    } else if (s.status === 'finished') {
      el.innerHTML = '<div class="badge success">✓ Processing Complete!</div>' +
        '<button class="btn-primary" style="margin-top:10px" onclick="resetState()">Process Another</button>';
    } else if (s.status === 'failed') {
      el.innerHTML = '<div class="badge error">✗ Processing Failed</div>' +
        '<button class="btn-primary" style="margin-top:10px" onclick="resetState()">Try Again</button>';
    } else {
      el.innerHTML = '<button class="btn-primary" onclick="startProcessing()" '+(inputPath?'':'disabled')+'>Start Processing</button>';
    }
  });
}

async function browseInput() {
  const r = await fetch('/api/browse', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({kind:'open'})});
  const d = await r.json();
  if (d.path) {
    inputPath = d.path;
    document.getElementById('inputName').textContent = d.path.split('/').pop();
    const r2 = await fetch('/api/probe', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:d.path})});
    const info = await r2.json();
    if (info.width) {
      document.getElementById('inputInfo').textContent =
        info.width+'x'+info.height+' | '+info.fps+' fps | '+info.duration+'s | '+info.size_mb+' MB';
    }
    renderAction();
  }
}

async function browseOutput() {
  const r = await fetch('/api/browse', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({kind:'save'})});
  const d = await r.json();
  if (d.path) document.getElementById('outputPath').value = d.path;
}

async function startProcessing() {
  if (!inputPath) return;
  let output = document.getElementById('outputPath').value;
  if (!output) {
    const parts = inputPath.split('.');
    const ext = parts.pop();
    output = parts.join('.') + '_upscaled.' + ext;
  }

  const body = {
    input: inputPath,
    output: output,
    processor: selectedProc,
    scale: scale,
    multiplier: multiplier,
    model: document.getElementById('model')?.value || '',
    codec: document.getElementById('codec').value,
    device: 0,
  };

  if (selectedProc === 'libplacebo') {
    body.width = parseInt(document.getElementById('outW')?.value || '3840');
    body.height = parseInt(document.getElementById('outH')?.value || '2160');
  }

  await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  renderAction();
}

async function cancelProcessing() {
  await fetch('/api/cancel', {method:'POST'});
  renderAction();
}

async function resetState() {
  await fetch('/api/reset', {method:'POST'});
  renderAction();
}

function startPolling() {
  setInterval(async () => {
    const r = await fetch('/api/state');
    const s = await r.json();
    const pct = Math.round(s.progress * 100);
    document.getElementById('pctText').textContent = pct + '%';
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('statFrame').textContent = s.total > 0 ? s.frame+'/'+s.total : '--';
    document.getElementById('statFps').textContent = s.fps > 0 ? s.fps.toFixed(1) : '--';
    document.getElementById('statElapsed').textContent = s.elapsed;
    document.getElementById('statRemaining').textContent = s.remaining;

    const logEl = document.getElementById('logContent');
    if (s.log) {
      logEl.textContent = s.log;
      logEl.scrollTop = logEl.scrollHeight;
    }

    renderAction();
  }, 500);
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    detect_devices()
    port = 52845
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True
    ).start()
    time.sleep(0.5)
    webview.create_window(
        "Video2X for Mac",
        f"http://127.0.0.1:{port}",
        width=1000, height=720, min_size=(800, 600),
        confirm_close=True,
    )
    webview.start()
