"""
Run Claude Code in ephemeral Cloud Run containers with GCS data transfer.

This module provides a high-level interface for running Claude Code tasks
in isolated containers. It handles:
- Content-addressed input caching (tar + hash -> GCS)
- Container setup (nodejs, claude-code, non-root user)
- Output retrieval and transcript extraction

Usage:
    from safetytooling.infra.cloud_run import ClaudeCodeJob, ClaudeCodeJobConfig

    config = ClaudeCodeJobConfig(
        project_id="my-project",
        gcs_bucket="my-bucket",
    )

    result = ClaudeCodeJob(config).run(
        task="Review the code in input/repo for security issues",
        inputs={"repo": Path("/tmp/my_repo")},
        system_prompt="You are a security reviewer...",
    )

    print(result.response)
    print(result.transcript)
    # result.output_dir contains files from /workspace/output/

Workspace layout (inside container):
    /workspace/           <- Claude's working directory
    |-- input/
    |   |-- repo/         <- from inputs={"repo": Path(...)}
    |   +-- data.json     <- from inputs={"data.json": Path(...)}
    +-- output/           <- Claude writes results here (retrieved automatically)

GCS layout:
    gs://bucket/
    |-- claude-code-inputs/
    |   +-- {md5hash}.tar.gz    <- Content-addressed, shared across jobs
    +-- claude-code-outputs/
        +-- {job_id}.tar.gz     <- Per-job, cleaned up after retrieval

GCS Lifecycle Policy (recommended):
    Set up automatic cleanup with:

    gsutil lifecycle set lifecycle.json gs://your-bucket

    # lifecycle.json:
    {
      "rule": [
        {"action": {"type": "Delete"}, "condition": {"age": 1, "matchesPrefix": ["claude-code-outputs/"]}},
        {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["claude-code-inputs/"]}}
      ]
    }
"""

import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from google.cloud import storage
from google.cloud.run_v2 import JobsClient

from .cloud_run_job import (
    CloudRunJob,
    CloudRunJobConfig,
    CloudRunJobError,
)

# Cloud Run image with gcloud CLI pre-installed
CLOUDRUN_IMAGE = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"

# GCS prefixes
GCS_INPUTS_PREFIX = "claude-code-inputs"
GCS_OUTPUTS_PREFIX = "claude-code-outputs"


@dataclass
class ClaudeCodeJobConfig:
    """Configuration for Claude Code job execution.

    Args:
        project_id: GCP project ID (required)
        gcs_bucket: GCS bucket for file I/O (required)
        region: GCP region (default: us-central1)
        model: Claude model to use (default: claude-opus-4-5-20251101)
        max_turns: Maximum conversation turns (default: 100)
        timeout: Job timeout in seconds (default: 600)
        cpu: vCPUs - 1, 2, 4, or 8 (default: 2)
        memory: Memory limit up to 32Gi (default: 4Gi)
        skip_permissions: Use --dangerously-skip-permissions (default: True)
    """

    project_id: str
    gcs_bucket: str
    region: str = "us-central1"
    model: str = "claude-opus-4-5-20251101"
    max_turns: int = 100
    timeout: int = 600
    cpu: str = "2"
    memory: str = "4Gi"
    skip_permissions: bool = True


@dataclass
class ClaudeCodeJobResult:
    """Result from running Claude Code."""

    response: str  # Claude's stdout
    transcript: list[dict] | None  # Parsed session from .claude/
    output_dir: Path  # Local dir with contents of /workspace/output
    returncode: int
    duration_seconds: float


class ClaudeCodeJobError(Exception):
    """Raised when Claude Code job operations fail."""

    pass


class ClaudeCodeJob:
    """
    Run Claude Code in an ephemeral Cloud Run container.

    Handles input/output via GCS with content-addressed caching for inputs.
    Supports parallel execution via shared GCP clients.

    Caching behavior:
        - In-memory cache by input paths: prevents redundant tarring within a process
        - GCS cache by content hash: prevents redundant uploads across processes/machines

    Example:
        # Single job
        config = ClaudeCodeJobConfig(project_id="my-project", gcs_bucket="my-bucket")
        result = ClaudeCodeJob(config).run(
            task="Analyze the code in input/repo",
            inputs={"repo": Path("./my_repo")},
        )

        # Parallel jobs (share clients to avoid connection overhead)
        jobs_client = JobsClient()
        storage_client = storage.Client()

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    ClaudeCodeJob(config, jobs_client=jobs_client, storage_client=storage_client).run,
                    task=task,
                    inputs=inputs,
                )
                for task, inputs in work_items
            ]
    """

    # Class-level caches for parallel execution efficiency
    _inputs_cache: ClassVar[dict[str, str]] = {}
    _inputs_locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: ClaudeCodeJobConfig,
        jobs_client: JobsClient | None = None,
        storage_client: storage.Client | None = None,
    ):
        self.config = config
        self._jobs_client = jobs_client
        self._storage_client = storage_client
        self._owns_clients = False

        # Validate required config
        if not self.config.project_id:
            raise ClaudeCodeJobError("project_id is required in ClaudeCodeJobConfig")
        if not self.config.gcs_bucket:
            raise ClaudeCodeJobError("gcs_bucket is required in ClaudeCodeJobConfig")

    def _ensure_clients(self) -> None:
        """Create GCP clients if not provided."""
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self.config.project_id)
            self._owns_clients = True
        if self._jobs_client is None:
            self._jobs_client = JobsClient()
            self._owns_clients = True

    @classmethod
    def _get_inputs_lock(cls, inputs_key: str) -> threading.Lock:
        with cls._locks_lock:
            if inputs_key not in cls._inputs_locks:
                cls._inputs_locks[inputs_key] = threading.Lock()
            return cls._inputs_locks[inputs_key]

    @classmethod
    def _compute_inputs_key(cls, inputs: dict[str, Path]) -> str:
        path_strs = sorted(f"{name}:{Path(p).expanduser().resolve()}" for name, p in inputs.items())
        key_str = "|".join(path_strs)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _compute_content_hash(self, tar_path: Path) -> str:
        md5 = hashlib.md5()
        with open(tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _tar_inputs(self, inputs: dict[str, Path], tar_path: Path) -> None:
        def _add_path_to_tar(tar: tarfile.TarFile, local_path: Path, arcname: str) -> None:
            if local_path.is_file():
                ti = tarfile.TarInfo(name=arcname)
                ti.size = local_path.stat().st_size
                ti.mtime = 0
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o644
                with open(local_path, "rb") as f:
                    tar.addfile(ti, f)
            elif local_path.is_dir():
                ti = tarfile.TarInfo(name=arcname + "/")
                ti.type = tarfile.DIRTYPE
                ti.mtime = 0
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o755
                tar.addfile(ti)
                for child in sorted(local_path.iterdir()):
                    child_arcname = f"{arcname}/{child.name}"
                    _add_path_to_tar(tar, child, child_arcname)

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for name, local_path in sorted(inputs.items()):
                local_path = Path(local_path).expanduser().resolve()
                if not local_path.exists():
                    raise ClaudeCodeJobError(f"Input not found: {local_path}")
                _add_path_to_tar(tar, local_path, name)

        tar_data = tar_buffer.getvalue()
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb", mtime=0) as gz:
            gz.write(tar_data)
        tar_path.write_bytes(gz_buffer.getvalue())

    def _upload_to_gcs_if_needed(self, tar_path: Path) -> str:
        content_hash = self._compute_content_hash(tar_path)
        gcs_path = f"{GCS_INPUTS_PREFIX}/{content_hash}.tar.gz"

        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(gcs_path)

        if blob.exists():
            print(f"GCS cache hit: {content_hash[:12]}...")
        else:
            print(f"Uploading inputs: {tar_path.stat().st_size / 1024 / 1024:.1f}MB")
            blob.upload_from_filename(str(tar_path))
            print(f"Uploaded to: gs://{self.config.gcs_bucket}/{gcs_path}")

        return gcs_path

    def _get_or_upload_inputs(self, inputs: dict[str, Path]) -> str:
        inputs_key = self._compute_inputs_key(inputs)

        with self._get_inputs_lock(inputs_key):
            if inputs_key in self._inputs_cache:
                gcs_path = self._inputs_cache[inputs_key]
                print(f"Input cache hit (in-memory): {inputs_key[:12]}...")
                return gcs_path

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                tar_path = Path(f.name)

            try:
                self._tar_inputs(inputs, tar_path)
                gcs_path = self._upload_to_gcs_if_needed(tar_path)
                self._inputs_cache[inputs_key] = gcs_path
                return gcs_path
            finally:
                tar_path.unlink(missing_ok=True)

    def _download_outputs(self, gcs_path: str, local_dir: Path) -> None:
        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            print(f"Warning: No outputs at gs://{self.config.gcs_bucket}/{gcs_path}")
            return

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = Path(f.name)

        try:
            blob.download_to_filename(str(tar_path))
            local_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(local_dir)

            print(f"Downloaded outputs to: {local_dir}")
            blob.delete()
        finally:
            tar_path.unlink(missing_ok=True)

    def _extract_transcript(self, output_dir: Path) -> list[dict] | None:
        claude_home = output_dir / "claude_home"
        if not claude_home.exists():
            return None

        for jsonl_file in claude_home.rglob("*.jsonl"):
            try:
                transcript = [json.loads(line) for line in jsonl_file.read_text().strip().split("\n") if line.strip()]
                return transcript
            except Exception:
                pass

        return None

    def _build_container_script(
        self,
        task: str,
        input_gcs_path: str | None,
        output_gcs_path: str,
        system_prompt: str | None,
        claude_config_dir: Path | None,
        api_key: str,
    ) -> str:
        claude_flags = [
            "-p",
            f"--model {self.config.model}",
            f"--max-turns {self.config.max_turns}",
        ]
        if self.config.skip_permissions:
            claude_flags.append("--dangerously-skip-permissions")

        claude_flags_str = " ".join(claude_flags)

        system_prompt_arg = ""
        if system_prompt:
            system_prompt_arg = '--system-prompt "$(cat /tmp/system_prompt.txt)"'

        input_section = ""
        if input_gcs_path:
            input_section = f"""
# Download and extract inputs
echo "Downloading inputs..."
gcloud storage cp "gs://{self.config.gcs_bucket}/{input_gcs_path}" /tmp/input.tar.gz
mkdir -p /workspace/input
cd /workspace/input
tar xzf /tmp/input.tar.gz
echo "Inputs extracted:"
find /workspace/input -type f | head -20
"""

        claude_config_section = ""
        if claude_config_dir:
            claude_config_section = """
# Set up Claude config (custom commands, etc.)
if [ -d /workspace/input/.claude ]; then
    cp -r /workspace/input/.claude /home/claude/.claude
    chown -R claude:claude /home/claude/.claude
fi
"""

        script = f"""
set -e
echo "=== Claude Code Job Starting ==="

# Install dependencies
echo "Installing dependencies..."
apt-get update -qq && apt-get install -y -qq git nodejs npm > /dev/null 2>&1
npm install -g @anthropic-ai/claude-code@latest > /dev/null 2>&1
echo "Dependencies installed"

# Create non-root user (Claude Code requires this)
useradd -m -s /bin/bash claude 2>/dev/null || true

# Create workspace
mkdir -p /workspace/input /workspace/output
chown -R claude:claude /workspace

{input_section}

{claude_config_section}

# Write system prompt to file if provided
cat > /tmp/system_prompt.txt << 'SYSTEM_PROMPT_EOF'
{system_prompt or ""}
SYSTEM_PROMPT_EOF

# Write task to file
cat > /tmp/task.txt << 'TASK_EOF'
{task}
TASK_EOF

# Set permissions
chown -R claude:claude /workspace

# Run Claude Code as non-root user
echo "Starting Claude Code..."
set +e
su - claude -c '
export ANTHROPIC_API_KEY="{api_key}"
git config --global user.email "claude@example.com"
git config --global user.name "Claude"
cd /workspace
claude {claude_flags_str} {system_prompt_arg} "$(cat /tmp/task.txt)" > /workspace/output/response.txt 2>&1
echo $? > /workspace/output/exitcode.txt
'
set -e

echo "Claude Code finished"

# Copy Claude home for transcript
cp -r /home/claude/.claude /workspace/output/claude_home 2>/dev/null || true

# Tar and upload outputs
echo "Uploading outputs..."
cd /workspace/output
tar czf /tmp/output.tar.gz .
gcloud storage cp /tmp/output.tar.gz "gs://{self.config.gcs_bucket}/{output_gcs_path}"

echo "=== Claude Code Job Complete ==="
"""
        return script

    def run(
        self,
        task: str,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
    ) -> ClaudeCodeJobResult:
        """
        Run Claude Code on a task with given inputs.

        Args:
            task: The prompt/instruction for Claude Code
            inputs: Dict of {name: local_path} to make available at /workspace/input/{name}
            system_prompt: Optional system prompt / constitution
            claude_config_dir: Optional path to .claude/ directory with custom commands
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)

        Returns:
            ClaudeCodeJobResult with response, transcript, and output files
        """
        start_time = time.time()

        # Get API key from param or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeCodeJobError("API key required: pass api_key param or set ANTHROPIC_API_KEY env var")

        self._ensure_clients()

        # Generate unique job ID for outputs
        job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        output_gcs_path = f"{GCS_OUTPUTS_PREFIX}/{job_id}.tar.gz"

        # Handle inputs
        input_gcs_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if inputs:
                all_inputs = dict(inputs)
                if claude_config_dir and claude_config_dir.exists():
                    all_inputs[".claude"] = claude_config_dir
                input_gcs_path = self._get_or_upload_inputs(all_inputs)
            elif claude_config_dir and claude_config_dir.exists():
                input_gcs_path = self._get_or_upload_inputs({".claude": claude_config_dir})

            # Build container script
            script = self._build_container_script(
                task=task,
                input_gcs_path=input_gcs_path,
                output_gcs_path=output_gcs_path,
                system_prompt=system_prompt,
                claude_config_dir=claude_config_dir,
                api_key=api_key,
            )

            # Configure Cloud Run job
            cloud_run_config = CloudRunJobConfig(
                image=CLOUDRUN_IMAGE,
                project_id=self.config.project_id,
                gcs_bucket=self.config.gcs_bucket,
                region=self.config.region,
                cpu=self.config.cpu,
                memory=self.config.memory,
                timeout=self.config.timeout + 120,
                name_prefix=f"claude-{job_id[:12]}",
                env={},
            )

            # Run the job
            try:
                with CloudRunJob(
                    cloud_run_config,
                    jobs_client=self._jobs_client,
                    storage_client=self._storage_client,
                ) as job:
                    result = job.run(script, timeout=self.config.timeout)
            except CloudRunJobError as e:
                raise ClaudeCodeJobError(f"Cloud Run job failed: {e}")

            # Download outputs
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            self._download_outputs(output_gcs_path, output_dir)

            # Read response
            response_file = output_dir / "response.txt"
            response = response_file.read_text() if response_file.exists() else ""

            # Read exit code
            exitcode_file = output_dir / "exitcode.txt"
            returncode = 1
            if exitcode_file.exists():
                try:
                    returncode = int(exitcode_file.read_text().strip())
                except ValueError:
                    pass

            # Extract transcript
            transcript = self._extract_transcript(output_dir)

            # Copy outputs to persistent location
            persistent_output_dir = Path(tempfile.mkdtemp(prefix="claude_output_"))
            for item in output_dir.iterdir():
                dest = persistent_output_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, symlinks=True, ignore_dangling_symlinks=True)
                else:
                    shutil.copy2(item, dest)

            duration = time.time() - start_time

            return ClaudeCodeJobResult(
                response=response,
                transcript=transcript,
                output_dir=persistent_output_dir,
                returncode=returncode,
                duration_seconds=duration,
            )
