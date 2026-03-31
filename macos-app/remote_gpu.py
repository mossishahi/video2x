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
                "time_limit": slurm.get("time", ""),
                "qos": slurm.get("qos", ""),
                "nice": slurm.get("nice", ""),
                "extra_sbatch": slurm.get("extra_sbatch", ""),
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
            if c.get("qos"): slurm["qos"] = c["qos"]
            if c.get("nice"): slurm["nice"] = c["nice"]
            if c.get("extra_sbatch"): slurm["extra_sbatch"] = c["extra_sbatch"]
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
BIN="$V2X_DIR/venhance"
APPIMAGE="$V2X_DIR/venhance.AppImage"

if [ -f "$BIN" ] || [ -f "$APPIMAGE" ]; then
    echo "INSTALL_OK"
    exit 0
fi

echo "INSTALL_STARTED"
mkdir -p "$V2X_DIR/data"

# Download the prebuilt AppImage (~50MB, works on any Linux x86_64)
echo "Downloading prebuilt binary..."
RELEASE_URL="https://github.com/k4yt3x/video2x/releases/download/6.4.0/Video2X-x86_64.AppImage"

if command -v curl &>/dev/null; then
    curl -fSL --retry 3 "$RELEASE_URL" -o "$APPIMAGE" 2>&1
elif command -v wget &>/dev/null; then
    wget --tries=3 "$RELEASE_URL" -O "$APPIMAGE" 2>&1
else
    echo "INSTALL_FAILED: Neither curl nor wget available"
    exit 1
fi

# Verify it actually downloaded (not an HTML error page)
FILE_SIZE=$(stat -c%s "$APPIMAGE" 2>/dev/null || stat -f%z "$APPIMAGE" 2>/dev/null || echo 0)
echo "Downloaded: ${FILE_SIZE} bytes"
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "INSTALL_FAILED: Download too small (${FILE_SIZE} bytes), likely failed"
    cat "$APPIMAGE" 2>/dev/null | head -5
    exit 1
fi

chmod +x "$APPIMAGE"

# Extract the AppImage (works even without FUSE, which many clusters lack)
echo "Extracting..."
cd "$V2X_DIR"
"$APPIMAGE" --appimage-extract 2>&1

# Create a wrapper that runs the extracted binary
if [ -d "$V2X_DIR/squashfs-root" ]; then
    cat > "$BIN" << 'WRAPPER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/squashfs-root/usr/lib:$LD_LIBRARY_PATH"
cd "$DIR/squashfs-root/usr/share/video2x"
exec "$DIR/squashfs-root/usr/bin/video2x" "$@"
WRAPPER
    chmod +x "$BIN"
    echo "INSTALL_OK"
else
    echo "INSTALL_FAILED: Could not extract AppImage"
    exit 1
fi
"""

SLURM_WRAPPER = r"""#!/bin/bash
#SBATCH --job-name=vproc
#SBATCH --gres={gres}
#SBATCH --mem={mem}
{time_line}
{partition_line}
{qos_line}
{nice_line}
{extra_sbatch}
#SBATCH --output={v2x_dir}/data/job_%j.log

set -x
module load cuda 2>/dev/null || true
module load vulkan 2>/dev/null || true

V2X_DIR="{v2x_dir}"
export LD_LIBRARY_PATH="$V2X_DIR/squashfs-root/usr/lib:$LD_LIBRARY_PATH"
cd "$V2X_DIR/squashfs-root/usr/share/video2x"

echo "Running on $(hostname) with GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU found"

"$V2X_DIR/squashfs-root/usr/bin/video2x" {args}

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

    def _resolve_path(self, path):
        """Replace ~ with actual home directory."""
        home = getattr(self, '_remote_home', None)
        if home and path.startswith("~"):
            return path.replace("~", home, 1)
        return path

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
                self._proxy_ssh = proxy_ssh
                self._log(f"Jump host connected.")

                # Verify the hop works
                # Use SSH ControlMaster to establish ONE connection and reuse it.
                # This avoids "too many auth failures" since only 1 auth exchange happens.
                ctl_path = f"/tmp/.venhance_ssh_{host}"
                base_opts = f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
                ctl_opts = f"-o ControlPath={ctl_path}"

                # Kill any stale control socket
                proxy_ssh.exec_command(f"rm -f {ctl_path}")
                time.sleep(0.3)

                # Try to establish a ControlMaster with each key individually
                keys = []
                _, stdout, _ = proxy_ssh.exec_command("ls ~/.ssh/id_ed25519 ~/.ssh/id_rsa 2>/dev/null")
                for line in stdout.read().decode().strip().split("\n"):
                    if line.strip():
                        keys.append(line.strip())

                self._log(f"Establishing connection to {host}...")
                connected_hop = False

                for key in keys:
                    self._log(f"  Trying key: {os.path.basename(key)}...")
                    master_cmd = (
                        f"SSH_AUTH_SOCK= ssh -f -N -o ControlMaster=yes {ctl_opts} {base_opts} "
                        f"-o IdentitiesOnly=yes -o IdentityAgent=none -o IdentityFile={key} "
                        f"-o PreferredAuthentications=publickey "
                        f"{username}@{host} 2>&1; echo EXIT:$?"
                    )
                    _, stdout, _ = proxy_ssh.exec_command(master_cmd, timeout=10)
                    result = stdout.read().decode()

                    # Check if control socket was created
                    _, stdout2, _ = proxy_ssh.exec_command(f"ssh -O check {ctl_opts} {username}@{host} 2>&1")
                    check = stdout2.read().decode()
                    if "running" in check.lower() or "pid" in check.lower():
                        self._log(f"  Connected with {os.path.basename(key)}")
                        connected_hop = True
                        break
                    else:
                        self._log(f"  Failed: {result.strip()[:80]}")
                    time.sleep(0.3)

                if not connected_hop:
                    raise Exception(f"Cannot reach {host} from {proxy_host}. None of the SSH keys worked.")

                ssh_prefix = f"ssh {ctl_opts} {base_opts} {username}@{host}"
                # Use a relay that runs commands on the target via the proxy
                self.ssh = _ProxyRelay(proxy_ssh, ssh_prefix)
                self._log(f"Connected to {host} via {proxy_host}")
            else:
                kwargs = {"hostname": host, "port": port, "username": username, "timeout": 15}
                kwargs.update(_auth_kwargs(auth, password, key_path))
                try:
                    self.ssh.connect(**kwargs)
                except paramiko.ssh_exception.BadAuthenticationType as e:
                    allowed = getattr(e, 'allowed_types', [])
                    if "keyboard-interactive" in allowed and password:
                        self._log("Using keyboard-interactive auth...")
                        transport = paramiko.Transport((host, port))
                        transport.start_client()
                        transport.auth_interactive(username, lambda *a: [password] * len(a[2]))
                        self.ssh._transport = transport
                    else:
                        raise
            self.connected = True
            self._log(f"Connected to {username}@{host}")
            time.sleep(1)

            # Resolve ~ to actual home directory
            _, stdout, _ = self.ssh.exec_command("echo $HOME")
            raw = stdout.read()
            home = (raw.decode().strip() if isinstance(raw, bytes) else raw.strip())
            if home:
                self._remote_home = home
                self._log(f"Home: {home}")
            else:
                self._remote_home = f"/home/{username}"

            def _read(stdout):
                raw = stdout.read()
                return raw.decode().strip() if isinstance(raw, bytes) else raw.strip()

            try:
                _, stdout, _ = self.ssh.exec_command("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1")
                gpu_name = _read(stdout)
                if gpu_name:
                    self._log(f"Remote GPU: {gpu_name}")
                    self.state["remote_gpu_name"] = gpu_name
                else:
                    self.state["remote_gpu_name"] = "Unknown"
            except Exception as e:
                self._log(f"GPU detection skipped: {e}")
                self.state["remote_gpu_name"] = "Unknown"

            try:
                _, stdout, _ = self.ssh.exec_command("command -v srun sbatch 2>/dev/null && echo HAS_SLURM || echo NO_SLURM")
                has_slurm = "HAS_SLURM" in _read(stdout)
                self.state["remote_has_slurm"] = has_slurm
                self._log(f"SLURM: {'available' if has_slurm else 'not found (will run directly)'}")
            except Exception as e:
                self._log(f"SLURM detection skipped: {e}")
                self.state["remote_has_slurm"] = False

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
        script = INSTALL_SCRIPT.replace("{v2x_dir}", self._resolve_path(REMOTE_V2X_DIR))
        _, stdout, stderr = self.ssh.exec_command(f"bash -l <<'VENHANCE_EOF'\n{script}\nVENHANCE_EOF", get_pty=True)

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
        remote_dir = f"{self._resolve_path(REMOTE_V2X_DIR)}/data"
        self.ssh.exec_command(f"mkdir -p {remote_dir}")
        time.sleep(0.5)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        self._log(f"Uploading {filename} ({size_mb:.1f} MB)...")

        if isinstance(self.ssh, _ProxyRelay):
            self._upload_via_proxy(local_path, f"{remote_dir}/{filename}")
        else:
            with SCPClient(self.ssh.get_transport(), progress=self._scp_progress) as scp_client:
                scp_client.put(local_path, f"{remote_dir}/{filename}")

        self._log(f"Upload complete.")
        return f"{remote_dir}/{filename}"

    def _upload_via_proxy(self, local_path, remote_path):
        """Upload through the jump host using scp on the proxy."""
        proxy = self._proxy_ssh
        filename = os.path.basename(local_path)
        tmp_path = f"/tmp/{filename}"
        with SCPClient(proxy.get_transport(), progress=self._scp_progress) as scp_client:
            scp_client.put(local_path, tmp_path)
        # Now move from proxy to target
        target_host = remote_path.split(":")[0] if ":" in remote_path else ""
        self.ssh.exec_command(f"mkdir -p $(dirname {remote_path})")
        time.sleep(0.3)
        # Copy is already on the same node since _ProxyShell runs on the target
        proxy.exec_command(f"scp -o StrictHostKeyChecking=no {tmp_path} $(hostname):{remote_path}")
        time.sleep(1)
        # Fallback: the file might already be accessible if proxy and target share filesystem
        self.ssh.exec_command(f"cp {tmp_path} {remote_path} 2>/dev/null || true")

    def _scp_progress(self, filename, size, sent):
        pct = int(sent / size * 100) if size > 0 else 0
        self.state["upload_progress"] = pct

    def process_video(self, remote_input, args_str, use_slurm=False):
        """Run processing on the remote machine."""
        v2x_dir = self._resolve_path(REMOTE_V2X_DIR)
        filename = os.path.basename(remote_input)
        base, ext = os.path.splitext(filename)
        remote_output = f"{v2x_dir}/data/{base}_upscaled{ext}"

        full_args = f'-i "{remote_input}" -o "{remote_output}" {args_str}'

        if use_slurm:
            return self._process_slurm(full_args, remote_output)
        else:
            return self._process_direct(full_args, remote_output)

    def _process_direct(self, args, remote_output):
        """Run processing directly on the remote node."""
        v2x_bin = f"{self._resolve_path(REMOTE_V2X_DIR)}/venhance"

        cmd = f"{v2x_bin} {args}"
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
        time_limit = self.slurm_opts.get("time", "")
        partition = self.slurm_opts.get("partition", "")
        qos = self.slurm_opts.get("qos", "")
        nice = self.slurm_opts.get("nice", "")
        extra = self.slurm_opts.get("extra_sbatch", "")

        time_line = f"#SBATCH --time={time_limit}" if time_limit else ""
        partition_line = f"#SBATCH --partition={partition}" if partition else ""
        qos_line = f"#SBATCH --qos={qos}" if qos else ""
        nice_line = f"#SBATCH --nice={nice}" if nice else ""
        extra_sbatch = "\n".join(f"#SBATCH {e.strip()}" for e in extra.split(",") if e.strip()) if extra else ""

        v2x_dir = self._resolve_path(REMOTE_V2X_DIR)
        script = SLURM_WRAPPER.format(
            v2x_dir=v2x_dir, args=args,
            gres=gres, mem=mem, time_line=time_line,
            partition_line=partition_line, qos_line=qos_line,
            nice_line=nice_line, extra_sbatch=extra_sbatch,
        )
        script_path = f"{v2x_dir}/data/run_proc.sh"

        self.ssh.exec_command(f"mkdir -p {v2x_dir}/data")
        time.sleep(0.5)

        # Write the script via echo to avoid SFTP issues
        escaped_script = script.replace("'", "'\\''")
        self.ssh.exec_command(f"echo '{escaped_script}' > {script_path}")
        time.sleep(0.3)
        self.ssh.exec_command(f"chmod +x {script_path}")
        time.sleep(0.2)

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

        log_path = f"{v2x_dir}/data/job_{job_id}.log"
        last_size = 0

        while not self._cancel:
            time.sleep(5)
            _, stdout, _ = self.ssh.exec_command(f"squeue -j {job_id} -h -o %T 2>/dev/null")
            status = stdout.read().decode().strip()

            if status == "PENDING":
                self._log(f"Job {job_id}: waiting in queue...")
                continue

            # Read the job log for progress
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
                elif "V2X_EXIT_CODE:" in clean:
                    self._log(f"Processing failed: {clean}")
                else:
                    self._log(clean)

            # Check if job is done
            if not status or status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                break

        # Dump full job log
        self._log(f"\n--- SLURM job {job_id} finished (status: {status or 'completed'}) ---")
        _, stdout, _ = self.ssh.exec_command(f"cat {log_path} 2>/dev/null | tail -30")
        tail = stdout.read().decode().strip()
        if tail:
            self._log(tail)

        # Verify output exists
        _, stdout, _ = self.ssh.exec_command(f"ls -la {remote_output} 2>&1")
        ls_out = stdout.read().decode().strip()
        if "No such file" in ls_out:
            self._log(f"\nOutput file not found: {remote_output}")
            self._log("The SLURM job may have failed. Check the log above for errors.")
            return None

        return remote_output

    def download_result(self, remote_path, local_dir):
        """Download processed video from remote."""
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, filename)
        self._log(f"Downloading {filename}...")

        if isinstance(self.ssh, _ProxyRelay):
            self._download_via_proxy(remote_path, local_path)
        else:
            with SCPClient(self.ssh.get_transport(), progress=self._scp_progress) as scp_client:
                scp_client.get(remote_path, local_path)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        self._log(f"Downloaded to {local_path} ({size_mb:.1f} MB)")
        return local_path

    def _download_via_proxy(self, remote_path, local_path):
        """Download through the jump host."""
        proxy = self._proxy_ssh
        filename = os.path.basename(remote_path)
        tmp_path = f"/tmp/{filename}"
        # Copy from target to proxy (shared filesystem or scp)
        self.ssh.exec_command(f"cp {remote_path} {tmp_path} 2>/dev/null || true")
        time.sleep(1)
        proxy.exec_command(f"cp {remote_path} {tmp_path} 2>/dev/null || true")
        time.sleep(1)
        # Download from proxy to local
        with SCPClient(proxy.get_transport(), progress=self._scp_progress) as scp_client:
            scp_client.get(tmp_path, local_path)

    def cancel(self):
        self._cancel = True


class _ProxyRelay:
    """Runs commands on a remote target by relaying through a jump host via ssh."""

    def __init__(self, proxy_ssh, ssh_prefix):
        self._proxy = proxy_ssh
        self._prefix = ssh_prefix

    def exec_command(self, cmd, get_pty=False, timeout=None):
        """Execute a command on the target node via the proxy."""
        escaped = cmd.replace("'", "'\\''")
        relay_cmd = f"{self._prefix} '{escaped}'"
        return self._proxy.exec_command(relay_cmd, timeout=timeout)

    def get_transport(self):
        return self._proxy.get_transport()

    def close(self):
        pass


def _escape(s):
    return s.replace("'", "'\"'\"'")
