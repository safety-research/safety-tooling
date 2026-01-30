"""
CloudRunClient - Run commands in ephemeral GCP Cloud Run containers.

Usage:
    from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig, CloudRunTask

    client = CloudRunClient(CloudRunClientConfig(
        project_id="my-project",
        gcs_bucket="my-bucket",
    ))

    # Create tasks
    tasks = [
        CloudRunTask(id="task-1", command="echo hello", n=10),
        CloudRunTask(id="task-2", command="echo world", n=5),
    ]

    # Run all and get results
    results = client.run(tasks)
    # Returns: {task1: [Result, ...], task2: [Result, ...]}

    # Or stream results as they complete
    for task, run_idx, result in client.run_stream(tasks):
        print(f"{task.id}[{run_idx}]: {result.success}")

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
from typing import ClassVar, Iterator

from google.cloud import run_v2, storage
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import (
    CreateJobRequest,
    DeleteJobRequest,
    EnvVar,
    EnvVarSource,
    RunJobRequest,
    SecretKeySelector,
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
        env: Environment variables dict (plain text values)
        secrets: Dict mapping env var names to Secret Manager secret names.
                 Secrets are injected securely via GCP Secret Manager, not embedded in scripts.
                 Format: {"ENV_VAR_NAME": "secret-name"} or {"ENV_VAR_NAME": "projects/proj/secrets/name"}
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
    secrets: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CloudRunTask:
    """A task to run in Cloud Run.

    Frozen (immutable) so it can be used as a dict key.

    Args:
        id: Unique identifier for this task
        command: Shell command to execute
        inputs: Tuple of (name, path) pairs for files to upload (tuple for hashability)
        n: Number of times to run this task (default: 1)
        timeout: Job timeout in seconds (default: None, uses client config)
    """

    id: str
    command: str
    inputs: tuple[tuple[str, str | Path], ...] | None = None
    n: int = 1
    timeout: int | None = None

    def inputs_as_dict(self) -> dict[str, Path] | None:
        """Convert inputs tuple to dict for internal use."""
        if self.inputs is None:
            return None
        return {name: Path(path) for name, path in self.inputs}


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

        tasks = [
            CloudRunTask(id="task-1", command="python script.py", n=10),
            CloudRunTask(id="task-2", command="python other.py", n=5),
        ]

        # Get all results at once
        results = client.run(tasks)
        # Returns: {task1: [Result, ...], task2: [Result, ...]}

        # Or stream as they complete
        for task, run_idx, result in client.run_stream(tasks):
            print(f"{task.id}[{run_idx}]: {result.success}")
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
        tasks: list[CloudRunTask],
        max_workers: int | None = None,
        progress: bool = True,
    ) -> dict[CloudRunTask, list[CloudRunResult]]:
        """
        Run tasks in Cloud Run containers.

        Args:
            tasks: List of CloudRunTask objects
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Returns:
            Dict mapping task -> list of results (one per run if n > 1)

        Example:
            tasks = [
                CloudRunTask(id="task-1", command="echo hello", n=10),
                CloudRunTask(id="task-2", command="echo world", n=5),
            ]
            results = client.run(tasks)
            for task, result_list in results.items():
                print(f"{task.id}: {len(result_list)} results")
        """
        # Collect all results from streaming
        results_by_task: dict[CloudRunTask, dict[int, CloudRunResult]] = {task: {} for task in tasks}

        for task, run_idx, result in self.run_stream(tasks, max_workers=max_workers, progress=progress):
            results_by_task[task][run_idx] = result

        # Convert to ordered lists
        final_results: dict[CloudRunTask, list[CloudRunResult]] = {}
        for task in tasks:
            final_results[task] = [results_by_task[task][i] for i in range(task.n)]

        return final_results

    def run_stream(
        self,
        tasks: list[CloudRunTask],
        max_workers: int | None = None,
        progress: bool = True,
    ) -> Iterator[tuple[CloudRunTask, int, CloudRunResult]]:
        """
        Run tasks and yield results as they complete.

        Args:
            tasks: List of CloudRunTask objects
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Yields:
            Tuples of (task, run_index, result) as each job completes

        Example:
            for task, run_idx, result in client.run_stream(tasks):
                print(f"{task.id}[{run_idx}]: {result.success}")
                save_to_db(task, run_idx, result)
        """
        # Build task lookup by id
        task_by_id = {task.id: task for task in tasks}

        # Expand tasks into individual jobs
        # Each job is (task_id, run_index)
        jobs = []
        for task in tasks:
            for i in range(task.n):
                jobs.append((task.id, i))

        def run_one_job(job: tuple[str, int]) -> tuple[str, int, CloudRunResult]:
            task_id, run_idx = job
            task = task_by_id[task_id]
            timeout = task.timeout or self.config.timeout
            try:
                result = self._run_single(task.command, task.inputs_as_dict(), timeout)
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
                task = task_by_id[task_id]
                yield task, run_idx, result

    def run_single(
        self,
        command: str,
        inputs: dict[str, Path] | None = None,
        timeout: int | None = None,
    ) -> CloudRunResult:
        """
        Convenience method to run a single command.

        Args:
            command: Shell command to execute
            inputs: Dict mapping names to local paths
            timeout: Job timeout in seconds (default: from config)

        Returns:
            CloudRunResult
        """
        # Convert inputs dict to tuple for CloudRunTask
        inputs_tuple = None
        if inputs:
            inputs_tuple = tuple((name, str(path)) for name, path in inputs.items())

        task = CloudRunTask(
            id="_single",
            command=command,
            inputs=inputs_tuple,
            timeout=timeout or self.config.timeout,
        )
        results = self.run([task])
        return results[task][0]

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

        # Build env vars - plain text values
        env_vars = [EnvVar(name=k, value=v) for k, v in self.config.env.items()]

        # Add secrets as env vars via Secret Manager (secure injection)
        for env_name, secret_name in self.config.secrets.items():
            # Handle both short names and full paths
            if not secret_name.startswith("projects/"):
                secret_path = f"projects/{self.config.project_id}/secrets/{secret_name}"
            else:
                secret_path = secret_name

            secret_env = EnvVar(
                name=env_name,
                value_source=EnvVarSource(
                    secret_key_ref=SecretKeySelector(
                        secret=secret_path,
                        version="latest",
                    )
                ),
            )
            env_vars.append(secret_env)

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
