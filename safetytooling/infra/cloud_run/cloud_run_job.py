"""
GCP Cloud Run Jobs for ephemeral container execution.

Uses Cloud Run Jobs + Cloud Storage for isolated job execution.
Files are transferred via explicit gcloud storage cp (no GCSFuse).

Usage:
    from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

    config = CloudRunJobConfig(
        image="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
        project_id="my-project",
        gcs_bucket="my-bucket",
    )

    with CloudRunJob(config) as job:
        job.send_files({"local/repo": "input/repo"})
        result = job.run("python /workspace/input/repo/script.py")
        job.receive_files({"output/results.json": "local/results.json"})

    # Job and GCS files are automatically cleaned up

Data Flow:
    1. Local uploads inputs to GCS via Python SDK
    2. Container downloads inputs from GCS via `gcloud storage cp`
    3. Container works on local filesystem (/workspace)
    4. Container uploads outputs to GCS via `gcloud storage cp`
    5. Local downloads outputs from GCS via Python SDK
"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from google.cloud import logging as cloud_logging
from google.cloud import run_v2, storage
from google.cloud.run_v2 import JobsClient
from google.cloud.run_v2.types import (
    CreateJobRequest,
    DeleteJobRequest,
    EnvVar,
    RunJobRequest,
)


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
    Uses explicit gcloud storage cp for file I/O (no GCSFuse).

    For parallel execution, pass shared clients to avoid creating hundreds
    of gRPC connections simultaneously (which causes contention):

        # Create shared clients once
        jobs_client = JobsClient()
        storage_client = storage.Client()

        # Pass to each CloudRunJob
        with CloudRunJob(config, jobs_client=jobs_client, storage_client=storage_client) as job:
            ...
    """

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
        self._gcs_prefix: str | None = None

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

        self._job_id = (
            f"{self.config.name_prefix}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        )
        self._gcs_prefix = f"jobs/{self._job_id}"

        print(f"CloudRunJob initialized: {self._job_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._cleanup()
        else:
            print(f"Skipping cleanup due to error: {exc_val}")
            print(
                f"Job artifacts at: gs://{self.config.gcs_bucket}/{self._gcs_prefix}/"
            )
        return False

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def gcs_bucket(self) -> str:
        return self.config.gcs_bucket

    def _cleanup(self) -> None:
        if self._job_name and self._jobs_client:
            print(f"Deleting job: {self._job_name}")
            try:
                request = DeleteJobRequest(name=self._job_name)
                operation = self._jobs_client.delete_job(request=request)
                operation.result(timeout=60)
            except Exception as e:
                print(f"Warning: Failed to delete job: {e}")

        if self._gcs_prefix and self._storage_client:
            print(f"Cleaning up GCS: gs://{self.config.gcs_bucket}/{self._gcs_prefix}/")
            try:
                bucket = self._storage_client.bucket(self.config.gcs_bucket)
                blobs = list(bucket.list_blobs(prefix=self._gcs_prefix))
                for blob in blobs:
                    blob.delete()
            except Exception as e:
                print(f"Warning: Failed to cleanup GCS: {e}")

    def send_files(self, file_map: dict[str, str]) -> None:
        bucket = self._storage_client.bucket(self.config.gcs_bucket)

        for local_path, remote_path in file_map.items():
            local_path = Path(local_path).expanduser()
            gcs_path = f"{self._gcs_prefix}/{remote_path}"

            if local_path.is_dir():
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative = file_path.relative_to(local_path)
                        blob_path = f"{gcs_path}/{relative}"
                        blob = bucket.blob(blob_path)
                        blob.upload_from_filename(str(file_path))
                print(
                    f"Uploaded directory: {local_path} -> gs://{self.config.gcs_bucket}/{gcs_path}/"
                )
            else:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                print(
                    f"Uploaded file: {local_path} -> gs://{self.config.gcs_bucket}/{gcs_path}"
                )

    def receive_files(self, file_map: dict[str, str]) -> None:
        bucket = self._storage_client.bucket(self.config.gcs_bucket)

        for remote_path, local_path in file_map.items():
            local_path = Path(local_path).expanduser()
            gcs_path = f"{self._gcs_prefix}/{remote_path}"

            blobs = list(bucket.list_blobs(prefix=gcs_path))

            if not blobs:
                print(
                    f"Warning: No files found at gs://{self.config.gcs_bucket}/{gcs_path}"
                )
                continue

            if len(blobs) == 1 and blobs[0].name == gcs_path:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blobs[0].download_to_filename(str(local_path))
                print(
                    f"Downloaded file: gs://{self.config.gcs_bucket}/{gcs_path} -> {local_path}"
                )
            else:
                local_path.mkdir(parents=True, exist_ok=True)
                for blob in blobs:
                    if blob.name.endswith("/"):
                        continue
                    relative = blob.name[len(gcs_path) :].lstrip("/")
                    if not relative:
                        continue
                    dest = local_path / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(dest))
                print(
                    f"Downloaded directory: gs://{self.config.gcs_bucket}/{gcs_path}/ -> {local_path}/"
                )

    def run(self, command: str, timeout: int | None = None) -> JobResult:
        timeout = timeout or self.config.timeout
        start_time = time.time()

        env_vars = [EnvVar(name=k, value=v) for k, v in self.config.env.items()]

        gcs_input = f"gs://{self.config.gcs_bucket}/{self._gcs_prefix}/input"
        gcs_output = f"gs://{self.config.gcs_bucket}/{self._gcs_prefix}/output"

        script = f"""
set -e
echo "Starting job: {self._job_id}"

mkdir -p /workspace/input /workspace/output
cd /workspace

echo "Downloading inputs from {gcs_input}/"
gcloud storage cp -r "{gcs_input}/*" /workspace/input/ 2>/dev/null || echo "No inputs to download"

echo "Input files:"
find /workspace/input -type f | head -20 || true

echo "Running command..."
set +e
{command}
EXIT_CODE=$?
set -e

echo "Command exited with code: $EXIT_CODE"

echo "Uploading outputs to {gcs_output}/"
if [ -d /workspace/output ] && [ "$(ls -A /workspace/output 2>/dev/null)" ]; then
    gcloud storage cp -r /workspace/output/* "{gcs_output}/"
    echo "Outputs uploaded"
else
    echo "No outputs to upload"
fi

echo $EXIT_CODE | gcloud storage cp - "{gcs_output}/exitcode.txt"

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

        bucket = self._storage_client.bucket(self.config.gcs_bucket)

        stdout = ""
        stderr = ""
        returncode = 1

        try:
            stdout_blob = bucket.blob(f"{self._gcs_prefix}/output/stdout.txt")
            if stdout_blob.exists():
                stdout = stdout_blob.download_as_text()
        except Exception:
            pass

        try:
            stderr_blob = bucket.blob(f"{self._gcs_prefix}/output/stderr.txt")
            if stderr_blob.exists():
                stderr = stderr_blob.download_as_text()
        except Exception:
            pass

        try:
            exitcode_blob = bucket.blob(f"{self._gcs_prefix}/output/exitcode.txt")
            if exitcode_blob.exists():
                returncode = int(exitcode_blob.download_as_text().strip())
        except Exception:
            pass

        return JobResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )

    def get_logs(self) -> str:
        if not self._job_name:
            return ""

        try:
            logging_client = cloud_logging.Client(project=self.config.project_id)

            filter_str = (
                f'resource.type="cloud_run_job" '
                f'AND resource.labels.job_name="{self._job_id}"'
            )

            entries = []
            for entry in logging_client.list_entries(
                filter_=filter_str, max_results=1000
            ):
                if hasattr(entry, "payload"):
                    entries.append(str(entry.payload))
                elif hasattr(entry, "text_payload"):
                    entries.append(entry.text_payload)

            return "\n".join(entries)
        except Exception as e:
            return f"Failed to fetch logs: {e}"
