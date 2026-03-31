"""Remote GPU processing via SSH — handles upload, install, processing, and download."""

import os
import re
import threading
import time

import paramiko
from scp import SCPClient

REMOTE_V2X_DIR = "~/video2x_remote"
PROGRESS_RE = re.compile(
    r"frame=(\d+)/(\d+)\s+\(([^)]+)\);\s+fps=([^;]+);\s+elapsed=([^;]+);\s+remaining=(.+)"
)

INSTALL_SCRIPT = r"""
set -e
V2X_DIR="{v2x_dir}"

if [ -f "$V2X_DIR/build/video2x-install/bin/video2x" ]; then
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
        https://github.com/k4yt3x/video2x.git "$V2X_DIR"
fi

cd "$V2X_DIR"

cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=build/video2x-install \
    -DVIDEO2X_USE_EXTERNAL_NCNN=OFF \
    -DVIDEO2X_USE_EXTERNAL_SPDLOG=OFF \
    -DVIDEO2X_USE_EXTERNAL_BOOST=OFF 2>&1

cmake --build build --config Release --parallel $(nproc) --target install 2>&1

echo "INSTALL_OK"
"""

SLURM_WRAPPER = r"""#!/bin/bash
#SBATCH --job-name=video2x
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output={v2x_dir}/data/job_%j.log

module load cuda 2>/dev/null || true
module load vulkan 2>/dev/null || true

V2X="{v2x_dir}/build/video2x-install/bin/video2x"
export LD_LIBRARY_PATH="{v2x_dir}/build:$LD_LIBRARY_PATH"

cd "{v2x_dir}/build/video2x-install/share/video2x"

$V2X {args}

echo "V2X_EXIT_CODE:$?"
"""


class RemoteGPU:
    def __init__(self, state_dict):
        self.state = state_dict
        self.ssh = None
        self.connected = False
        self._cancel = False

    def _log(self, msg):
        self.state["log"] += msg + "\n"
        if len(self.state["log"]) > 50000:
            self.state["log"] = self.state["log"][-40000:]

    def connect(self, host, username, password=None, key_path=None, port=22):
        """Establish SSH connection."""
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            kwargs = {"hostname": host, "port": port, "username": username, "timeout": 15}
            if key_path and os.path.isfile(key_path):
                kwargs["key_filename"] = key_path
            elif password:
                kwargs["password"] = password
            else:
                kwargs["key_filename"] = os.path.expanduser("~/.ssh/id_rsa")
            self.ssh.connect(**kwargs)
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
        """Install video2x on the remote machine if not present."""
        self._log("Checking remote video2x installation...")
        script = INSTALL_SCRIPT.format(v2x_dir=REMOTE_V2X_DIR)
        _, stdout, stderr = self.ssh.exec_command(f"bash -l -c '{_escape(script)}'", get_pty=True)

        for line in stdout:
            line = line.strip()
            if line == "INSTALL_OK":
                self._log("video2x is ready on remote.")
                return True
            elif line == "INSTALL_STARTED":
                self._log("Installing video2x on remote (this takes a few minutes on first run)...")
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
        """Run video2x directly on the remote node."""
        v2x_bin = f"{REMOTE_V2X_DIR}/build/video2x-install/bin/video2x"
        models_dir = f"{REMOTE_V2X_DIR}/build/video2x-install/share/video2x"

        cmd = f"cd {models_dir} && LD_LIBRARY_PATH={REMOTE_V2X_DIR}/build:$LD_LIBRARY_PATH {v2x_bin} {args}"
        self._log(f"Running: video2x {args.split('-o')[0].strip()} ...")

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
        script = SLURM_WRAPPER.format(v2x_dir=REMOTE_V2X_DIR, args=args)
        script_path = f"{REMOTE_V2X_DIR}/data/run_v2x.sh"

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
