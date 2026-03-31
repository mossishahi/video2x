"""Remote GPU processing via SSH — handles upload, install, processing, and download."""

import os
import re
import threading
import time

import paramiko
import yaml
from scp import SCPClient

REMOTE_V2X_DIR = "~/.local/share/venhance"
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/video2x/gpus.yaml")
EXAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_config.example.yaml")


def load_gpu_configs(path=None):
    """Load GPU target definitions from YAML config file."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        entries = data.get("gpus", [])
        configs = []
        for i, e in enumerate(entries):
            sched = e.get("scheduler", "auto")
            gpu_type = e.get("type", "slurm" if sched == "slurm" else "direct")
            slurm = e.get("slurm", {})
            configs.append({
                "id": i,
                "name": e.get("name", e.get("host", f"GPU {i}")),
                "type": gpu_type,
                "host": e.get("host", ""),
                "user": e.get("user", ""),
                "port": int(e.get("port", 22)),
                "auth": e.get("auth", "key"),
                "key_path": e.get("key_path", ""),
                "proxy": e.get("proxy", ""),
                "scheduler": sched,
                "slurm": slurm,
                "partition": slurm.get("partition", ""),
                "gres": slurm.get("gres", "gpu:1"),
                "mem": slurm.get("mem", "32G"),
                "time_limit": slurm.get("time", "02:00:00"),
            })
        return configs
    except Exception as e:
        return []


def save_gpu_configs(configs, path=None):
    """Save GPU configs back to YAML file."""
    path = path or DEFAULT_CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = []
    for c in configs:
        entry = {
            "name": c.get("name", ""),
            "type": c.get("type", "direct"),
            "host": c.get("host", ""),
            "user": c.get("user", ""),
            "port": int(c.get("port", 22)),
            "auth": c.get("auth", "key"),
        }
        if c.get("key_path"):
            entry["key_path"] = c["key_path"]
        if c.get("proxy"):
            entry["proxy"] = c["proxy"]
        if c.get("type") == "slurm":
            entry["scheduler"] = "slurm"
            slurm = {}
            if c.get("partition"): slurm["partition"] = c["partition"]
            if c.get("gres"): slurm["gres"] = c["gres"]
            if c.get("mem"): slurm["mem"] = c["mem"]
            if c.get("time_limit"): slurm["time"] = c["time_limit"]
            if slurm:
                entry["slurm"] = slurm
        else:
            entry["scheduler"] = "direct"
        raw.append(entry)
    with open(path, "w") as f:
        yaml.dump({"gpus": raw}, f, default_flow_style=False, sort_keys=False)
PROGRESS_RE = re.compile(
    r"frame=(\d+)/(\d+)\s+\(([^)]+)\);\s+fps=([^;]+);\s+elapsed=([^;]+);\s+remaining=(.+)"
)

INSTALL_SCRIPT = r"""
set -e
V2X_DIR="{v2x_dir}"

if [ -f "$V2X_DIR/build/venhance-install/bin/venhance" ]; then
    echo "INSTALL_OK"
    exit 0
fi

echo "INSTALL_STARTED"
mkdir -p "$V2X_DIR"

# Check basic dependencies
for cmd in cmake ninja git ffprobe pkg-config; do
    if ! command -v $cmd &>/dev/null; then
        echo "MISSING_DEP:$cmd"
    fi
done

# Try loading common cluster modules
module load cmake 2>/dev/null || true
module load ninja 2>/dev/null || true
module load ffmpeg 2>/dev/null || true
module load gcc 2>/dev/null || true
module load vulkan 2>/dev/null || true
module load cuda 2>/dev/null || true

# Clone if needed
if [ ! -f "$V2X_DIR/CMakeLists.txt" ]; then
    git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/k4yt3x/video2x.git "$V2X_DIR" 2>/dev/null
fi

cd "$V2X_DIR"

cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=build/venhance-install \
    -DVIDEO2X_USE_EXTERNAL_NCNN=OFF \
    -DVIDEO2X_USE_EXTERNAL_SPDLOG=OFF \
    -DVIDEO2X_USE_EXTERNAL_BOOST=OFF 2>&1

cmake --build build --config Release --parallel $(nproc) --target install 2>&1

mv build/venhance-install/bin/video2x build/venhance-install/bin/venhance 2>/dev/null

echo "INSTALL_OK"
"""

SLURM_WRAPPER = r"""#!/bin/bash
#SBATCH --job-name=vproc
#SBATCH --gres={gres}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
{partition_line}
#SBATCH --output={v2x_dir}/data/job_%j.log

module load cuda 2>/dev/null || true
module load vulkan 2>/dev/null || true

V2X="{v2x_dir}/build/venhance-install/bin/venhance"
export LD_LIBRARY_PATH="{v2x_dir}/build:$LD_LIBRARY_PATH"

cd "{v2x_dir}/build/venhance-install/share/video2x"

$V2X {args}

echo "V2X_EXIT_CODE:$?"
"""


class RemoteGPU:
    def __init__(self, state_dict):
        self.state = state_dict
        self.ssh = None
        self._proxy_ssh = None
        self.connected = False
        self._cancel = False
        self.slurm_opts = {}

    def _log(self, msg):
        self.state["log"] += msg + "\n"
        if len(self.state["log"]) > 50000:
            self.state["log"] = self.state["log"][-40000:]

    def connect(self, host, username, password=None, key_path=None, port=22, proxy=None, auth="key"):
        """Establish SSH connection, optionally through a ProxyJump host."""
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            def _auth_kwargs(target_auth, target_password, target_key_path):
                """Build paramiko auth kwargs based on auth method."""
                kw = {}
                if target_auth == "password" and target_password:
                    kw["password"] = target_password
                    kw["allow_agent"] = False
                    kw["look_for_keys"] = False
                elif target_auth == "agent":
                    kw["allow_agent"] = True
                    kw["look_for_keys"] = False
                elif target_auth == "key" and target_key_path:
                    kp = os.path.expanduser(target_key_path)
                    if os.path.isfile(kp):
                        kw["key_filename"] = kp
                    else:
                        kw["allow_agent"] = True
                        kw["look_for_keys"] = True
                else:
                    kw["allow_agent"] = True
                    kw["look_for_keys"] = True
                return kw

            sock = None
            if proxy:
                self._log(f"Connecting via jump host: {proxy}")
                proxy_ssh = paramiko.SSHClient()
                proxy_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                proxy_parts = proxy.split("@")
                proxy_user = proxy_parts[0] if len(proxy_parts) > 1 else username
                proxy_host = proxy_parts[-1]
                proxy_kw = {"hostname": proxy_host, "username": proxy_user, "timeout": 15}
                proxy_kw.update(_auth_kwargs(auth, password, key_path))
                proxy_ssh.connect(**proxy_kw)
                transport = proxy_ssh.get_transport()
                sock = transport.open_channel("direct-tcpip", (host, port), ("127.0.0.1", 0))
                self._proxy_ssh = proxy_ssh
                self._log(f"Jump host connected. Tunneling to {host}:{port}...")

            kwargs = {"hostname": host, "port": port, "username": username, "timeout": 15}
            if sock:
                kwargs["sock"] = sock
            kwargs.update(_auth_kwargs(auth, password, key_path))

            try:
                self.ssh.connect(**kwargs)
            except paramiko.ssh_exception.BadAuthenticationType as e:
                allowed = e.allowed_types if hasattr(e, 'allowed_types') else []
                self._log(f"Retrying with allowed types: {allowed}")
                transport = paramiko.Transport(sock if sock else (host, port))
                transport.connect(username=username)
                if "keyboard-interactive" in allowed and password:
                    transport.auth_interactive(username, lambda *a: [password])
                elif "publickey" in allowed:
                    transport.auth_publickey(username, paramiko.Agent().get_keys()[0])
                self.ssh = paramiko.SSHClient()
                self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh._transport = transport
            self.connected = True
            self._log(f"Connected to {username}@{host}")

            _, stdout, _ = self.ssh.exec_command("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1")
            gpu_name = stdout.read().decode().strip()
            if gpu_name:
                self._log(f"Remote GPU: {gpu_name}")
                self.state["remote_gpu_name"] = gpu_name
            else:
                self._log("Warning: No NVIDIA GPU detected on remote host.")
                self.state["remote_gpu_name"] = "Unknown"

            _, stdout, _ = self.ssh.exec_command("command -v srun sbatch 2>/dev/null && echo HAS_SLURM || echo NO_SLURM")
            has_slurm = "HAS_SLURM" in stdout.read().decode()
            self.state["remote_has_slurm"] = has_slurm
            self._log(f"SLURM: {'available' if has_slurm else 'not found (will run directly)'}")

            return True
        except Exception as e:
            self._log(f"Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.ssh:
            self.ssh.close()
        self.connected = False

    def install_video2x(self):
        """Install processing engine on the remote machine if not present."""
        self._log("Checking remote installation...")
        script = INSTALL_SCRIPT.format(v2x_dir=REMOTE_V2X_DIR)
        _, stdout, stderr = self.ssh.exec_command(f"bash -l -c '{_escape(script)}'", get_pty=True)

        for line in stdout:
            line = line.strip()
            if line == "INSTALL_OK":
                self._log("Processing engine ready on remote.")
                return True
            elif line == "INSTALL_STARTED":
                self._log("Installing on remote (this takes a few minutes on first run)...")
            elif line.startswith("MISSING_DEP:"):
                self._log(f"  Warning: {line.split(':')[1]} not found, trying module load...")
            elif line:
                self._log(f"  {line}")

        err = stderr.read().decode().strip()
        if err:
            self._log(f"Install errors:\n{err}")
        return False

    def upload_video(self, local_path):
        """Upload video to remote via SCP."""
        filename = os.path.basename(local_path)
        remote_dir = f"{REMOTE_V2X_DIR}/data"
        self.ssh.exec_command(f"mkdir -p {remote_dir}")
        time.sleep(0.5)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        self._log(f"Uploading {filename} ({size_mb:.1f} MB)...")

        with SCPClient(self.ssh.get_transport(), progress=self._scp_progress) as scp_client:
            scp_client.put(local_path, f"{remote_dir}/{filename}")

        self._log(f"Upload complete.")
        return f"{remote_dir}/{filename}"

    def _scp_progress(self, filename, size, sent):
        pct = int(sent / size * 100) if size > 0 else 0
        self.state["upload_progress"] = pct

    def process_video(self, remote_input, args_str, use_slurm=False):
        """Run video2x on the remote machine."""
        filename = os.path.basename(remote_input)
        base, ext = os.path.splitext(filename)
        remote_output = f"{REMOTE_V2X_DIR}/data/{base}_upscaled{ext}"

        full_args = f'-i "{remote_input}" -o "{remote_output}" {args_str}'

        if use_slurm:
            return self._process_slurm(full_args, remote_output)
        else:
            return self._process_direct(full_args, remote_output)

    def _process_direct(self, args, remote_output):
        """Run processing directly on the remote node."""
        v2x_bin = f"{REMOTE_V2X_DIR}/build/venhance-install/bin/venhance"
        models_dir = f"{REMOTE_V2X_DIR}/build/venhance-install/share/video2x"

        cmd = f"cd {models_dir} && LD_LIBRARY_PATH={REMOTE_V2X_DIR}/build:$LD_LIBRARY_PATH {v2x_bin} {args}"
        self._log(f"Starting remote processing...")

        _, stdout, _ = self.ssh.exec_command(f"bash -l -c '{_escape(cmd)}'", get_pty=True)

        for line in stdout:
            if self._cancel:
                break
            clean = line.replace("\x1b[K", "").replace("\r", "").strip()
            if not clean:
                continue
            m = PROGRESS_RE.search(clean)
            if m:
                self.state["frame"] = int(m.group(1))
                self.state["total"] = int(m.group(2))
                self.state["fps"] = float(m.group(4))
                self.state["elapsed"] = m.group(5)
                self.state["remaining"] = m.group(6).strip()
                if self.state["total"] > 0:
                    self.state["progress"] = self.state["frame"] / self.state["total"]
            else:
                self._log(clean)

        return remote_output

    def _process_slurm(self, args, remote_output):
        """Submit as SLURM job and poll for completion."""
        gres = self.slurm_opts.get("gres", "gpu:1")
        mem = self.slurm_opts.get("mem", "32G")
        time_limit = self.slurm_opts.get("time", "02:00:00")
        partition = self.slurm_opts.get("partition", "")
        partition_line = f"#SBATCH --partition={partition}" if partition else ""
        script = SLURM_WRAPPER.format(
            v2x_dir=REMOTE_V2X_DIR, args=args,
            gres=gres, mem=mem, time_limit=time_limit, partition_line=partition_line,
        )
        script_path = f"{REMOTE_V2X_DIR}/data/run_proc.sh"

        self.ssh.exec_command(f"mkdir -p {REMOTE_V2X_DIR}/data")
        time.sleep(0.3)

        sftp = self.ssh.open_sftp()
        with sftp.open(os.path.expanduser(script_path.replace("~", ".")), "w") as f:
            f.write(script)
        sftp.close()

        self._log("Submitting SLURM job...")
        _, stdout, _ = self.ssh.exec_command(f"sbatch {script_path}")
        output = stdout.read().decode().strip()
        self._log(output)

        job_match = re.search(r"(\d+)", output)
        if not job_match:
            self._log("Failed to get SLURM job ID")
            return None
        job_id = job_match.group(1)
        self._log(f"Job {job_id} submitted. Waiting...")

        log_path = f"{REMOTE_V2X_DIR}/data/job_{job_id}.log"
        last_size = 0

        while not self._cancel:
            time.sleep(3)
            _, stdout, _ = self.ssh.exec_command(f"squeue -j {job_id} -h -o %T 2>/dev/null")
            status = stdout.read().decode().strip()

            if not status or status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                break
            elif status == "PENDING":
                self._log(f"Job {job_id}: waiting in queue...")
                continue

            _, stdout, _ = self.ssh.exec_command(f"tail -c +{last_size} {log_path} 2>/dev/null")
            new_data = stdout.read().decode()
            last_size += len(new_data)

            for line in new_data.split("\n"):
                clean = line.strip()
                if not clean:
                    continue
                m = PROGRESS_RE.search(clean)
                if m:
                    self.state["frame"] = int(m.group(1))
                    self.state["total"] = int(m.group(2))
                    self.state["fps"] = float(m.group(4))
                    self.state["elapsed"] = m.group(5)
                    self.state["remaining"] = m.group(6).strip()
                    if self.state["total"] > 0:
                        self.state["progress"] = self.state["frame"] / self.state["total"]
                elif "V2X_EXIT_CODE:0" in clean:
                    self._log("Remote processing complete!")
                else:
                    self._log(clean)

        return remote_output

    def download_result(self, remote_path, local_dir):
        """Download processed video from remote."""
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, filename)
        self._log(f"Downloading {filename}...")

        with SCPClient(self.ssh.get_transport(), progress=self._scp_progress) as scp_client:
            scp_client.get(remote_path, local_path)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        self._log(f"Downloaded to {local_path} ({size_mb:.1f} MB)")
        return local_path

    def cancel(self):
        self._cancel = True


def _escape(s):
    return s.replace("'", "'\"'\"'")
