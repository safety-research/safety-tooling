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
        service_account: Service account email for the job (default: uses project default).
                        Use a restricted service account to limit what the container can access.
                        Format: "name@project.iam.gserviceaccount.com"
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
    service_account: str | None = None


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
    _inputs_cache: ClassVar[dict[str, str]] = {}  # inputs_key -> gcs_path
    _inputs_locks: ClassVar[dict[str, threading.Lock]] = {}
    _locks_lock: ClassVar[threading.Lock] = threading.Lock()
    _job_cache: ClassVar[dict[str, str]] = {}  # config_hash -> job_name
    _job_locks: ClassVar[dict[str, threading.Lock]] = {}  # config_hash -> lock

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
        # Generate unique output path for this execution
        execution_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        output_gcs_path = f"{GCS_OUTPUTS_PREFIX}/{execution_id}.tar.gz"

        # Upload inputs if provided
        input_gcs_path = None
        if inputs:
            input_gcs_path = self._upload_inputs(inputs)

        # Get or create reusable job, then run execution
        # (job_id parameter is unused in new implementation)
        self._create_and_run_job(
            job_id="",  # Unused
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
        # Note: No cleanup needed - jobs are reused, executions auto-cleanup

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

    def _get_or_create_job(self, timeout: int) -> str:
        """Get or create a reusable Cloud Run job for this config.

        Jobs are cached by config hash. The job uses a generic script that reads
        INPUT_GCS_PATH, OUTPUT_GCS_PATH, and COMMAND from environment variables,
        which are passed via execution overrides.

        Returns:
            Full job name (projects/.../jobs/...)
        """
        config_hash = self._compute_config_hash()

        with self._get_job_lock(config_hash):
            if config_hash in self._job_cache:
                return self._job_cache[config_hash]

            job_id = f"{self.config.name_prefix}-{config_hash[:16]}"

            # Generic script that reads from env vars
            # INPUT_GCS_PATH, OUTPUT_GCS_PATH, COMMAND are set per-execution via overrides
            script = f"""
set -e
mkdir -p /workspace/input /workspace/output
cd /workspace

# Download inputs if INPUT_GCS_PATH is set
if [ -n "$INPUT_GCS_PATH" ]; then
    echo "Downloading inputs from $INPUT_GCS_PATH..."
    gcloud storage cp "gs://{self.config.gcs_bucket}/$INPUT_GCS_PATH" /tmp/input.tar.gz
    cd /workspace/input
    tar xzf /tmp/input.tar.gz
    cd /workspace
fi

echo "Running command..."
set +e
( eval "$COMMAND" ) > /tmp/cmd_stdout.txt 2> /tmp/cmd_stderr.txt
EXIT_CODE=$?
set -e

cp /tmp/cmd_stdout.txt /workspace/output/stdout.txt
cp /tmp/cmd_stderr.txt /workspace/output/stderr.txt
echo $EXIT_CODE > /workspace/output/exitcode.txt

echo "Uploading outputs to $OUTPUT_GCS_PATH..."
cd /workspace/output
tar czf /tmp/output.tar.gz .
gcloud storage cp /tmp/output.tar.gz "gs://{self.config.gcs_bucket}/$OUTPUT_GCS_PATH"
exit 0
"""

            # Build env vars - plain text values from config
            env_vars = [EnvVar(name=k, value=v) for k, v in self.config.env.items()]

            # Add secrets as env vars via Secret Manager (secure injection)
            for env_name, secret_name in self.config.secrets.items():
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

            # Add placeholder env vars (overridden at execution time)
            env_vars.append(EnvVar(name="INPUT_GCS_PATH", value=""))
            env_vars.append(EnvVar(name="OUTPUT_GCS_PATH", value=""))
            env_vars.append(EnvVar(name="COMMAND", value="true"))  # Default no-op

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

            if self.config.service_account:
                job.template.template.service_account = self.config.service_account

            parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
            request = CreateJobRequest(parent=parent, job=job, job_id=job_id)

            try:
                operation = self._jobs_client.create_job(request=request)
                created_job = operation.result()
                job_name = created_job.name
            except Exception as e:
                # Job might already exist (from previous process/session)
                if "already exists" in str(e).lower():
                    job_name = f"{parent}/jobs/{job_id}"
                else:
                    raise CloudRunClientError(f"Failed to create job: {e}")

            self._job_cache[config_hash] = job_name
            return job_name

    def _run_job_execution(
        self,
        job_name: str,
        command: str,
        input_gcs_path: str | None,
        output_gcs_path: str,
        timeout: int,
    ) -> None:
        """Run an execution of an existing job with specific inputs/outputs/command.

        Uses RunJobRequest.Overrides to pass per-execution environment variables.
        """
        # Build env var overrides for this execution
        env_overrides = [
            run_v2.EnvVar(name="OUTPUT_GCS_PATH", value=output_gcs_path),
            run_v2.EnvVar(name="COMMAND", value=command),
        ]
        if input_gcs_path:
            env_overrides.append(run_v2.EnvVar(name="INPUT_GCS_PATH", value=input_gcs_path))

        # Create container override with env vars
        container_override = run_v2.RunJobRequest.Overrides.ContainerOverride(
            env=env_overrides,
        )

        # Create task override with timeout
        task_override = run_v2.RunJobRequest.Overrides(
            container_overrides=[container_override],
            timeout=f"{timeout}s",
        )

        run_request = RunJobRequest(
            name=job_name,
            overrides=task_override,
        )

        run_operation = self._jobs_client.run_job(request=run_request)

        try:
            run_operation.result(timeout=timeout + 120)
        except Exception as e:
            raise CloudRunClientError(f"Job execution failed: {e}")

    def _create_and_run_job(
        self,
        job_id: str,  # Kept for backward compatibility but unused
        command: str,
        input_gcs_path: str | None,
        output_gcs_path: str,
        timeout: int,
    ) -> str:
        """Create (or reuse) and execute a Cloud Run job. Returns job name.

        This method now reuses job definitions across executions. Each execution
        passes its specific command, inputs, and outputs via environment variable
        overrides. This avoids hitting the 1000 job quota.
        """
        job_name = self._get_or_create_job(timeout)

        self._run_job_execution(
            job_name=job_name,
            command=command,
            input_gcs_path=input_gcs_path,
            output_gcs_path=output_gcs_path,
            timeout=timeout,
        )

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

    @classmethod
    def _get_inputs_lock(cls, inputs_key: str) -> threading.Lock:
        """Get or create a lock for a specific inputs key."""
        with cls._locks_lock:
            if inputs_key not in cls._inputs_locks:
                cls._inputs_locks[inputs_key] = threading.Lock()
            return cls._inputs_locks[inputs_key]

    @classmethod
    def _get_job_lock(cls, config_hash: str) -> threading.Lock:
        """Get or create a lock for a specific config hash."""
        with cls._locks_lock:
            if config_hash not in cls._job_locks:
                cls._job_locks[config_hash] = threading.Lock()
            return cls._job_locks[config_hash]

    def _compute_config_hash(self) -> str:
        """Compute a hash of the job configuration (everything that goes into job definition).

        This includes: image, cpu, memory, env vars, secrets, service_account, region, project.
        Does NOT include: command, inputs, outputs, timeout (those vary per execution).
        """
        config_parts = [
            self.config.project_id,
            self.config.region,
            self.config.image,
            self.config.cpu,
            self.config.memory,
            self.config.service_account or "",
            self.config.gcs_bucket,
        ]
        # Add sorted env vars
        for k, v in sorted(self.config.env.items()):
            config_parts.append(f"env:{k}={v}")
        # Add sorted secrets
        for k, v in sorted(self.config.secrets.items()):
            config_parts.append(f"secret:{k}={v}")

        return hashlib.md5("|".join(config_parts).encode()).hexdigest()

    @classmethod
    def _compute_inputs_key(cls, inputs: dict[str, Path]) -> str:
        """Compute a cache key from input paths."""
        path_strs = sorted(f"{name}:{Path(p).expanduser().resolve()}" for name, p in inputs.items())
        return hashlib.md5("|".join(path_strs).encode()).hexdigest()

    def _tar_inputs(self, inputs: dict[str, Path], tar_path: Path) -> None:
        """Create a gzipped tarball from inputs, preserving file permissions.

        Normalizes mtime for deterministic hashes (enables GCS deduplication)
        while preserving file permissions.
        """

        def normalize_mtime(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
            tarinfo.mtime = 0
            return tarinfo

        # Create uncompressed tar first, then gzip with mtime=0
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for name, local_path in sorted(inputs.items()):
                local_path = Path(local_path).expanduser().resolve()
                if not local_path.exists():
                    raise CloudRunClientError(f"Input not found: {local_path}")
                tar.add(local_path, arcname=name, filter=normalize_mtime)

        # Gzip with mtime=0 and no filename for deterministic output
        gz_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=gz_buffer, mode="wb", mtime=0) as gz:
            gz.write(tar_buffer.getvalue())
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
