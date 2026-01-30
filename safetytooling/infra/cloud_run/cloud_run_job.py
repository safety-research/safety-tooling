"""
GCP Cloud Run Jobs for ephemeral container execution.

Uses Cloud Run Jobs + Cloud Storage for isolated job execution.
Inputs/outputs are transferred as tarballs with content-addressed caching.

Usage:
    from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

    config = CloudRunJobConfig(
        image="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
        project_id="my-project",
        gcs_bucket="my-bucket",
    )

    with CloudRunJob(config) as job:
        job.send_inputs({"repo": Path("./my_repo"), "data.json": Path("./data.json")})
        result = job.run("python /workspace/input/repo/script.py")
        output_dir = job.receive_outputs()

    # Job and GCS files are automatically cleaned up

Data Flow:
    1. Local tars inputs -> content hash -> upload to GCS (if not cached)
    2. Container downloads and extracts inputs from GCS
    3. Container works on local filesystem (/workspace)
    4. Container tars outputs and uploads to GCS
    5. Local downloads and extracts outputs from GCS

Workspace layout (inside container):
    /workspace/
    ├── input/
    │   ├── repo/         <- from inputs={"repo": Path(...)}
    │   └── data.json     <- from inputs={"data.json": Path(...)}
    └── output/           <- Command writes results here

GCS layout:
    gs://bucket/
    ├── cloudrun-inputs/
    │   └── {md5hash}.tar.gz    <- Content-addressed, shared across jobs
    └── cloudrun-outputs/
        └── {job_id}.tar.gz     <- Per-job, cleaned up after retrieval
"""

import gzip
import hashlib
import io
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from google.cloud import logging as cloud_logging
from google.cloud import run_v2, storage
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import (
    CreateJobRequest,
    DeleteJobRequest,
    EnvVar,
    RunJobRequest,
)

# GCS prefixes
GCS_INPUTS_PREFIX = "cloudrun-inputs"
GCS_OUTPUTS_PREFIX = "cloudrun-outputs"


@dataclass
class CloudRunJobConfig:
    """Configuration for Cloud Run Job execution.

    Args:
        image: Container image (must have gcloud CLI installed)
        project_id: GCP project ID (required)
        gcs_bucket: GCS bucket for file I/O (required)
        region: GCP region (default: us-central1)
        cpu: vCPUs - 1, 2, 4, or 8 (default: 2)
        memory: Memory limit up to 32Gi (default: 4Gi)
        timeout: Job timeout in seconds, max 86400 (default: 3600)
        env: Environment variables dict
        name_prefix: Prefix for job names (default: ephemeral)
    """

    image: str
    project_id: str
    gcs_bucket: str
    region: str = "us-central1"
    cpu: str = "2"
    memory: str = "4Gi"
    timeout: int = 3600
    env: dict[str, str] = field(default_factory=dict)
    name_prefix: str = "ephemeral"


@dataclass
class JobResult:
    """Result from running a command in Cloud Run Job."""

    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class CloudRunJobError(Exception):
    """Raised when Cloud Run Job operations fail."""

    pass


class CloudRunJob:
    """
    Context manager for ephemeral Cloud Run Job execution.

    Creates a job on entry, cleans up job and GCS files on exit.
    Uses tarballs with content-addressed caching for efficient file transfer.

    Caching behavior:
        - In-memory cache by input paths: prevents redundant tarring within a process
        - GCS cache by content hash: prevents redundant uploads across processes/machines

    For parallel execution, pass shared clients to avoid creating hundreds
    of gRPC connections simultaneously (which causes contention):

        # Create shared clients once
        jobs_client = JobsClient()
        storage_client = storage.Client()

        # Pass to each CloudRunJob
        with CloudRunJob(config, jobs_client=jobs_client, storage_client=storage_client) as job:
            ...
    """

    # Class-level caches for parallel execution efficiency
    _inputs_cache: ClassVar[dict[str, str]] = {}
    _inputs_locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        config: CloudRunJobConfig,
        jobs_client: JobsClient | None = None,
        storage_client: storage.Client | None = None,
    ):
        self.config = config
        self._job_id: str | None = None
        self._job_name: str | None = None
        self._jobs_client: JobsClient | None = jobs_client
        self._storage_client: storage.Client | None = storage_client
        self._owns_clients: bool = False
        self._input_gcs_path: str | None = None
        self._output_gcs_path: str | None = None

        # Validate required config
        if not self.config.project_id:
            raise CloudRunJobError("project_id is required in CloudRunJobConfig")
        if not self.config.gcs_bucket:
            raise CloudRunJobError("gcs_bucket is required in CloudRunJobConfig")

    def __enter__(self) -> "CloudRunJob":
        if self._jobs_client is None:
            self._jobs_client = JobsClient()
            self._owns_clients = True
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self.config.project_id)
            self._owns_clients = True

        self._job_id = f"{self.config.name_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self._output_gcs_path = f"{GCS_OUTPUTS_PREFIX}/{self._job_id}.tar.gz"

        print(f"CloudRunJob initialized: {self._job_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._cleanup()
        else:
            print(f"Skipping cleanup due to error: {exc_val}")
            if self._output_gcs_path:
                print(f"Job outputs at: gs://{self.config.gcs_bucket}/{self._output_gcs_path}")
        return False

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def gcs_bucket(self) -> str:
        return self.config.gcs_bucket

    @classmethod
    def _get_inputs_lock(cls, inputs_key: str) -> threading.Lock:
        """Get or create a lock for a specific inputs key (thread-safe)."""
        with cls._locks_lock:
            if inputs_key not in cls._inputs_locks:
                cls._inputs_locks[inputs_key] = threading.Lock()
            return cls._inputs_locks[inputs_key]

    @classmethod
    def _compute_inputs_key(cls, inputs: dict[str, Path]) -> str:
        """Compute a cache key from input paths (for in-memory cache)."""
        path_strs = sorted(f"{name}:{Path(p).expanduser().resolve()}" for name, p in inputs.items())
        key_str = "|".join(path_strs)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _compute_content_hash(self, tar_path: Path) -> str:
        """Compute MD5 hash of tarball contents (for GCS cache)."""
        md5 = hashlib.md5()
        with open(tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _tar_inputs(self, inputs: dict[str, Path], tar_path: Path) -> None:
        """Create a deterministic gzipped tarball from inputs.

        Deterministic = same inputs produce same tarball (for content-addressed caching).
        Achieved by: sorting entries, zeroing mtime/uid/gid, consistent permissions.
        """

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
                    raise CloudRunJobError(f"Input not found: {local_path}")
                _add_path_to_tar(tar, local_path, name)

        tar_data = tar_buffer.getvalue()
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb", mtime=0) as gz:
            gz.write(tar_data)
        tar_path.write_bytes(gz_buffer.getvalue())

    def _upload_to_gcs_if_needed(self, tar_path: Path) -> str:
        """Upload tarball to GCS if not already cached (content-addressed)."""
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

    def send_inputs(self, inputs: dict[str, Path]) -> None:
        """
        Send inputs to be available in the container at /workspace/input/.

        Uses two levels of caching:
        1. In-memory: same input paths within a process skip re-tarring
        2. GCS: same content hash across processes/machines skips re-uploading

        Args:
            inputs: Dict mapping names to local paths.
                    {"repo": Path("./my_repo")} -> /workspace/input/repo/
                    {"data.json": Path("./data.json")} -> /workspace/input/data.json
        """
        inputs_key = self._compute_inputs_key(inputs)

        with self._get_inputs_lock(inputs_key):
            if inputs_key in self._inputs_cache:
                self._input_gcs_path = self._inputs_cache[inputs_key]
                print(f"Input cache hit (in-memory): {inputs_key[:12]}...")
                return

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                tar_path = Path(f.name)

            try:
                self._tar_inputs(inputs, tar_path)
                self._input_gcs_path = self._upload_to_gcs_if_needed(tar_path)
                self._inputs_cache[inputs_key] = self._input_gcs_path
            finally:
                tar_path.unlink(missing_ok=True)

    def receive_outputs(self, local_dir: Path | None = None) -> Path:
        """
        Download outputs from /workspace/output/ in the container.

        Args:
            local_dir: Where to extract outputs. Default: new temp directory.

        Returns:
            Path to directory containing the outputs.
        """
        if not self._output_gcs_path:
            raise CloudRunJobError("No outputs to receive (job not run yet?)")

        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(self._output_gcs_path)

        if not blob.exists():
            print(f"Warning: No outputs at gs://{self.config.gcs_bucket}/{self._output_gcs_path}")
            if local_dir is None:
                local_dir = Path(tempfile.mkdtemp(prefix="cloudrun_output_"))
            local_dir.mkdir(parents=True, exist_ok=True)
            return local_dir

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = Path(f.name)

        try:
            blob.download_to_filename(str(tar_path))

            if local_dir is None:
                local_dir = Path(tempfile.mkdtemp(prefix="cloudrun_output_"))
            local_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(local_dir)

            print(f"Downloaded outputs to: {local_dir}")

            # Clean up the output blob
            blob.delete()

            return local_dir
        finally:
            tar_path.unlink(missing_ok=True)

    def _cleanup(self) -> None:
        """Clean up Cloud Run job (but not cached inputs - those are shared)."""
        if self._job_name and self._jobs_client:
            print(f"Deleting job: {self._job_name}")
            try:
                request = DeleteJobRequest(name=self._job_name)
                operation = self._jobs_client.delete_job(request=request)
                operation.result(timeout=60)
            except Exception as e:
                print(f"Warning: Failed to delete job: {e}")

    def run(self, command: str, timeout: int | None = None) -> JobResult:
        """
        Run a command in the Cloud Run container.

        The command runs with:
        - Working directory: /workspace
        - Inputs at: /workspace/input/ (if send_inputs() was called)
        - Outputs from: /workspace/output/ (call receive_outputs() after)

        Args:
            command: Shell command to execute
            timeout: Override timeout in seconds (default: from config)

        Returns:
            JobResult with returncode, stdout, stderr, duration
        """
        timeout = timeout or self.config.timeout
        start_time = time.time()

        env_vars = [EnvVar(name=k, value=v) for k, v in self.config.env.items()]

        # Build input download section
        input_section = ""
        if self._input_gcs_path:
            input_section = f"""
echo "Downloading inputs..."
gcloud storage cp "gs://{self.config.gcs_bucket}/{self._input_gcs_path}" /tmp/input.tar.gz
cd /workspace/input
tar xzf /tmp/input.tar.gz
echo "Inputs extracted:"
find /workspace/input -type f | head -20 || true
"""

        script = f"""
set -e
echo "Starting job: {self._job_id}"

mkdir -p /workspace/input /workspace/output
cd /workspace

{input_section}

echo "Running command..."
set +e
{command}
EXIT_CODE=$?
set -e

echo "Command exited with code: $EXIT_CODE"

# Tar and upload outputs
echo "Uploading outputs..."
cd /workspace/output
if [ "$(ls -A . 2>/dev/null)" ]; then
    tar czf /tmp/output.tar.gz .
    gcloud storage cp /tmp/output.tar.gz "gs://{self.config.gcs_bucket}/{self._output_gcs_path}"
    echo "Outputs uploaded"
else
    echo "No outputs to upload"
fi

# Also store exit code in output for retrieval
echo $EXIT_CODE > /tmp/exitcode.txt
gcloud storage cp /tmp/exitcode.txt "gs://{self.config.gcs_bucket}/{GCS_OUTPUTS_PREFIX}/{self._job_id}_exitcode.txt"

echo "Job completed"
exit 0
"""

        job = run_v2.Job()
        container = run_v2.Container()
        container.image = self.config.image
        container.command = ["/bin/bash", "-c"]
        container.args = [script]
        container.env = env_vars
        container.resources.limits = {
            "cpu": self.config.cpu,
            "memory": self.config.memory,
        }

        job.template.template.containers.append(container)
        job.template.template.max_retries = 0
        job.template.template.timeout = f"{timeout}s"

        parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
        request = CreateJobRequest(parent=parent, job=job, job_id=self._job_id)

        print(f"Creating job: {self._job_id}")
        operation = self._jobs_client.create_job(request=request)
        created_job = operation.result()
        self._job_name = created_job.name
        print(f"Job created: {self._job_name}")

        print("Executing job...")
        run_request = RunJobRequest(name=self._job_name)
        run_operation = self._jobs_client.run_job(request=run_request)

        try:
            execution = run_operation.result(timeout=timeout + 60)
            print(f"Job execution completed: {execution.name}")
        except Exception as e:
            raise CloudRunJobError(f"Job execution failed: {e}")

        duration = time.time() - start_time

        # Read exit code from GCS
        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        returncode = 1
        try:
            exitcode_blob = bucket.blob(f"{GCS_OUTPUTS_PREFIX}/{self._job_id}_exitcode.txt")
            if exitcode_blob.exists():
                returncode = int(exitcode_blob.download_as_text().strip())
                exitcode_blob.delete()
        except Exception:
            pass

        return JobResult(
            returncode=returncode,
            stdout="",  # stdout is in Cloud Run logs, not captured here
            stderr="",  # stderr is in Cloud Run logs, not captured here
            duration_seconds=duration,
        )

    def get_logs(self) -> str:
        """Fetch logs from Cloud Logging for this job."""
        if not self._job_name:
            return ""

        try:
            logging_client = cloud_logging.Client(project=self.config.project_id)

            filter_str = f'resource.type="cloud_run_job" AND resource.labels.job_name="{self._job_id}"'

            entries = []
            for entry in logging_client.list_entries(filter_=filter_str, max_results=1000):
                if hasattr(entry, "payload"):
                    entries.append(str(entry.payload))
                elif hasattr(entry, "text_payload"):
                    entries.append(entry.text_payload)

            return "\n".join(entries)
        except Exception as e:
            return f"Failed to fetch logs: {e}"
