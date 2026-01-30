"""
CloudRunClient - Run commands in ephemeral GCP Cloud Run containers.

Usage:
    from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig

    client = CloudRunClient(CloudRunClientConfig(
        project_id="my-project",
        gcs_bucket="my-bucket",
    ))

    # Single run
    result = client.run(command="echo hello", inputs={"data": Path("./data")})
    print(result.stdout)

    # Run same command 100 times in parallel
    results = client.run(command="echo hello", inputs={"data": Path("./data")}, n=100)

Data Flow:
    1. Local tars inputs -> content hash -> upload to GCS (if not cached)
    2. Container downloads and extracts inputs from GCS
    3. Container runs command
    4. Container tars outputs and uploads to GCS
    5. Local downloads and extracts outputs from GCS

Workspace layout (inside container):
    /workspace/
    ├── input/      <- inputs extracted here
    └── output/     <- command writes results here
"""

import gzip
import hashlib
import io
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from google.cloud import run_v2, storage
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import (
    CreateJobRequest,
    DeleteJobRequest,
    EnvVar,
    RunJobRequest,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# GCS prefixes
GCS_INPUTS_PREFIX = "cloudrun-inputs"
GCS_OUTPUTS_PREFIX = "cloudrun-outputs"


@dataclass
class CloudRunClientConfig:
    """Configuration for CloudRunClient.

    Args:
        project_id: GCP project ID (required)
        gcs_bucket: GCS bucket for file I/O (required)
        name_prefix: Prefix for job names (default: "cloudrun")
        region: GCP region (default: us-central1)
        image: Container image - must have gcloud CLI (default: google-cloud-cli:slim)
        cpu: vCPUs - 1, 2, 4, or 8 (default: 1)
        memory: Memory limit up to 32Gi (default: 2Gi)
        timeout: Default job timeout in seconds (default: 600)
        env: Environment variables dict
    """

    project_id: str
    gcs_bucket: str
    name_prefix: str = "cloudrun"
    region: str = "us-central1"
    image: str = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"
    cpu: str = "1"
    memory: str = "2Gi"
    timeout: int = 600
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class CloudRunResult:
    """Result from running a command in Cloud Run."""

    stdout: str
    stderr: str
    output_dir: Path  # Local dir with /workspace/output contents
    returncode: int
    duration_seconds: float
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.returncode == 0


class CloudRunClientError(Exception):
    """Raised when CloudRunClient operations fail."""

    pass


class CloudRunClient:
    """
    Client for running commands in ephemeral Cloud Run containers.

    Example:
        client = CloudRunClient(CloudRunClientConfig(
            project_id="my-project",
            gcs_bucket="my-bucket",
        ))

        # Single run
        result = client.run(command="python script.py", inputs={"repo": repo_path})

        # Run 100 times in parallel
        results = client.run(command="python script.py", inputs={"repo": repo_path}, n=100)
    """

    # Class-level caches for efficiency across parallel runs
    _inputs_cache: ClassVar[dict[str, str]] = {}
    _inputs_locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: CloudRunClientConfig):
        self.config = config

        if not self.config.project_id:
            raise CloudRunClientError("project_id is required")
        if not self.config.gcs_bucket:
            raise CloudRunClientError("gcs_bucket is required")

        # Create shared GCP clients
        self._jobs_client = JobsClient()
        self._storage_client = storage.Client(project=self.config.project_id)

    def run(
        self,
        tasks: list[dict] | str,
        inputs: dict[str, Path] | None = None,
        timeout: int | None = None,
        n: int | None = None,
        max_workers: int | None = None,
        progress: bool = True,
    ) -> dict[str, list[CloudRunResult]] | CloudRunResult | list[CloudRunResult]:
        """
        Run command(s) in Cloud Run container(s).

        Two calling conventions:

        1. Task list (new, preferred):
            tasks = [
                {"id": "task-1", "command": "...", "inputs": {...}, "n": 10},
                {"id": "task-2", "command": "...", "inputs": {...}, "n": 5},
            ]
            results = client.run(tasks)
            # Returns: {"task-1": [Result, ...], "task-2": [Result, ...]}

        2. Single command (legacy):
            result = client.run(command="echo hello", inputs={...})
            results = client.run(command="echo hello", inputs={...}, n=10)

        Task dict fields:
            id: Unique identifier for this task (required)
            command: Shell command to execute (required)
            inputs: Dict mapping names to local paths (optional)
            n: Number of times to run this task (default: 1)
            timeout: Per-task timeout override (optional)

        Args:
            tasks: List of task dicts, OR a command string (legacy)
            inputs: Inputs for legacy single-command mode
            timeout: Timeout for legacy mode (default: from config)
            n: Repetition count for legacy mode
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Returns:
            Task list mode: dict mapping task id -> list of results
            Legacy mode: CloudRunResult or list[CloudRunResult]
        """
        # Legacy single-command mode
        if isinstance(tasks, str):
            command = tasks
            timeout = timeout or self.config.timeout

            if n is None:
                return self._run_single(command, inputs, timeout)

            # Parallel runs of same command
            results: dict[int, CloudRunResult] = {}

            def run_one_legacy(idx: int) -> tuple[int, CloudRunResult]:
                try:
                    result = self._run_single(command, inputs, timeout)
                    return idx, result
                except Exception as e:
                    return idx, CloudRunResult(
                        stdout="",
                        stderr="",
                        output_dir=Path("/dev/null"),
                        returncode=1,
                        duration_seconds=0,
                        error=e,
                    )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_one_legacy, i): i for i in range(n)}

                iterator = as_completed(futures)
                if progress and tqdm is not None:
                    iterator = tqdm(iterator, total=n, desc="Running jobs")

                for future in iterator:
                    idx, result = future.result()
                    results[idx] = result

            return [results[i] for i in range(n)]

        # New task list mode
        return self._run_tasks(tasks, max_workers=max_workers, progress=progress)

    def _run_tasks(
        self,
        tasks: list[dict],
        max_workers: int | None = None,
        progress: bool = True,
    ) -> dict[str, list[CloudRunResult]]:
        """Run a list of tasks, each potentially multiple times."""
        # Expand tasks into individual jobs
        # Each job is (task_id, run_index, command, inputs, timeout)
        jobs = []
        for task in tasks:
            task_id = task["id"]
            command = task["command"]
            task_inputs = task.get("inputs")
            task_timeout = task.get("timeout", self.config.timeout)
            task_n = task.get("n", 1)

            for i in range(task_n):
                jobs.append((task_id, i, command, task_inputs, task_timeout))

        # Run all jobs in parallel
        results_by_task: dict[str, dict[int, CloudRunResult]] = {task["id"]: {} for task in tasks}

        def run_one_job(job: tuple) -> tuple[str, int, CloudRunResult]:
            task_id, run_idx, command, task_inputs, task_timeout = job
            try:
                result = self._run_single(command, task_inputs, task_timeout)
                return task_id, run_idx, result
            except Exception as e:
                return (
                    task_id,
                    run_idx,
                    CloudRunResult(
                        stdout="",
                        stderr="",
                        output_dir=Path("/dev/null"),
                        returncode=1,
                        duration_seconds=0,
                        error=e,
                    ),
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_one_job, job): job for job in jobs}

            iterator = as_completed(futures)
            if progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(jobs), desc="Running jobs")

            for future in iterator:
                task_id, run_idx, result = future.result()
                results_by_task[task_id][run_idx] = result

        # Convert to ordered lists
        final_results: dict[str, list[CloudRunResult]] = {}
        for task in tasks:
            task_id = task["id"]
            task_n = task.get("n", 1)
            final_results[task_id] = [results_by_task[task_id][i] for i in range(task_n)]

        return final_results

    def _run_single(
        self,
        command: str,
        inputs: dict[str, Path] | None,
        timeout: int,
    ) -> CloudRunResult:
        """Run a single command in a Cloud Run container."""
        start_time = time.time()
        job_id = f"{self.config.name_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        job_name = None
        output_gcs_path = f"{GCS_OUTPUTS_PREFIX}/{job_id}.tar.gz"

        try:
            # Upload inputs if provided
            input_gcs_path = None
            if inputs:
                input_gcs_path = self._upload_inputs(inputs)

            # Build and run job
            job_name = self._create_and_run_job(
                job_id=job_id,
                command=command,
                input_gcs_path=input_gcs_path,
                output_gcs_path=output_gcs_path,
                timeout=timeout,
            )

            # Download outputs
            output_dir = self._download_outputs(output_gcs_path)

            # Read captured stdout/stderr/exitcode
            stdout = (output_dir / "stdout.txt").read_text() if (output_dir / "stdout.txt").exists() else ""
            stderr = (output_dir / "stderr.txt").read_text() if (output_dir / "stderr.txt").exists() else ""
            returncode = 1
            if (output_dir / "exitcode.txt").exists():
                try:
                    returncode = int((output_dir / "exitcode.txt").read_text().strip())
                except ValueError:
                    pass

            duration = time.time() - start_time

            return CloudRunResult(
                stdout=stdout,
                stderr=stderr,
                output_dir=output_dir,
                returncode=returncode,
                duration_seconds=duration,
            )

        finally:
            # Clean up job (but not cached inputs)
            if job_name:
                self._delete_job(job_name)

    def _upload_inputs(self, inputs: dict[str, Path]) -> str:
        """Upload inputs to GCS with content-addressed caching."""
        inputs_key = self._compute_inputs_key(inputs)

        with self._get_inputs_lock(inputs_key):
            if inputs_key in self._inputs_cache:
                return self._inputs_cache[inputs_key]

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                tar_path = Path(f.name)

            try:
                self._tar_inputs(inputs, tar_path)
                gcs_path = self._upload_to_gcs_if_needed(tar_path)
                self._inputs_cache[inputs_key] = gcs_path
                return gcs_path
            finally:
                tar_path.unlink(missing_ok=True)

    def _create_and_run_job(
        self,
        job_id: str,
        command: str,
        input_gcs_path: str | None,
        output_gcs_path: str,
        timeout: int,
    ) -> str:
        """Create and execute a Cloud Run job. Returns job name."""
        # Build input download section
        input_section = ""
        if input_gcs_path:
            input_section = f"""
echo "Downloading inputs..."
gcloud storage cp "gs://{self.config.gcs_bucket}/{input_gcs_path}" /tmp/input.tar.gz
cd /workspace/input
tar xzf /tmp/input.tar.gz
"""

        script = f"""
set -e
mkdir -p /workspace/input /workspace/output
cd /workspace

{input_section}

echo "Running command..."
set +e
( {command} ) > /tmp/cmd_stdout.txt 2> /tmp/cmd_stderr.txt
EXIT_CODE=$?
set -e

cp /tmp/cmd_stdout.txt /workspace/output/stdout.txt
cp /tmp/cmd_stderr.txt /workspace/output/stderr.txt
echo $EXIT_CODE > /workspace/output/exitcode.txt

echo "Uploading outputs..."
cd /workspace/output
tar czf /tmp/output.tar.gz .
gcloud storage cp /tmp/output.tar.gz "gs://{self.config.gcs_bucket}/{output_gcs_path}"
exit 0
"""

        env_vars = [EnvVar(name=k, value=v) for k, v in self.config.env.items()]

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
        request = CreateJobRequest(parent=parent, job=job, job_id=job_id)

        operation = self._jobs_client.create_job(request=request)
        created_job = operation.result()
        job_name = created_job.name

        run_request = RunJobRequest(name=job_name)
        run_operation = self._jobs_client.run_job(request=run_request)

        try:
            run_operation.result(timeout=timeout + 120)
        except Exception as e:
            raise CloudRunClientError(f"Job execution failed: {e}")

        return job_name

    def _download_outputs(self, output_gcs_path: str) -> Path:
        """Download outputs from GCS."""
        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(output_gcs_path)

        output_dir = Path(tempfile.mkdtemp(prefix="cloudrun_output_"))

        if not blob.exists():
            return output_dir

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            tar_path = Path(f.name)

        try:
            blob.download_to_filename(str(tar_path))
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(output_dir)
            blob.delete()
            return output_dir
        finally:
            tar_path.unlink(missing_ok=True)

    def _delete_job(self, job_name: str) -> None:
        """Delete a Cloud Run job."""
        try:
            request = DeleteJobRequest(name=job_name)
            operation = self._jobs_client.delete_job(request=request)
            operation.result(timeout=60)
        except Exception:
            pass  # Best effort cleanup

    @classmethod
    def _get_inputs_lock(cls, inputs_key: str) -> threading.Lock:
        """Get or create a lock for a specific inputs key."""
        with cls._locks_lock:
            if inputs_key not in cls._inputs_locks:
                cls._inputs_locks[inputs_key] = threading.Lock()
            return cls._inputs_locks[inputs_key]

    @classmethod
    def _compute_inputs_key(cls, inputs: dict[str, Path]) -> str:
        """Compute a cache key from input paths."""
        path_strs = sorted(f"{name}:{Path(p).expanduser().resolve()}" for name, p in inputs.items())
        return hashlib.md5("|".join(path_strs).encode()).hexdigest()

    def _tar_inputs(self, inputs: dict[str, Path], tar_path: Path) -> None:
        """Create a deterministic gzipped tarball from inputs."""

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
                    _add_path_to_tar(tar, child, f"{arcname}/{child.name}")

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for name, local_path in sorted(inputs.items()):
                local_path = Path(local_path).expanduser().resolve()
                if not local_path.exists():
                    raise CloudRunClientError(f"Input not found: {local_path}")
                _add_path_to_tar(tar, local_path, name)

        tar_data = tar_buffer.getvalue()
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb", mtime=0) as gz:
            gz.write(tar_data)
        tar_path.write_bytes(gz_buffer.getvalue())

    def _upload_to_gcs_if_needed(self, tar_path: Path) -> str:
        """Upload tarball to GCS if not already cached."""
        md5 = hashlib.md5()
        with open(tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        content_hash = md5.hexdigest()

        gcs_path = f"{GCS_INPUTS_PREFIX}/{content_hash}.tar.gz"
        bucket = self._storage_client.bucket(self.config.gcs_bucket)
        blob = bucket.blob(gcs_path)

        if not blob.exists():
            blob.upload_from_filename(str(tar_path))

        return gcs_path
