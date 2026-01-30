"""Integration tests for cloud_run module.

These tests require:
- GCP credentials (gcloud auth application-default login)
- A GCP project with Cloud Run, GCS, and Secret Manager enabled
- An Anthropic API key stored in Secret Manager

Run with: pytest tests/test_cloud_run_integration.py -v --run-integration

Environment variables needed:
- GCP_PROJECT_ID: Your GCP project ID
- GCS_BUCKET: GCS bucket for file transfer
- API_KEY_SECRET: Name of the Secret Manager secret containing the Anthropic API key
                  (e.g., "anthropic-api-key-username")
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def integration_enabled(request):
    """Check if integration tests should run."""
    if not request.config.getoption("--run-integration", default=False):
        pytest.skip("Integration tests skipped. Use --run-integration to run.")


@pytest.fixture
def gcp_config():
    """Get GCP config from environment or skip."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    gcs_bucket = os.environ.get("GCS_BUCKET")

    if not project_id or not gcs_bucket:
        pytest.skip("GCP_PROJECT_ID and GCS_BUCKET environment variables required.")

    return {"project_id": project_id, "gcs_bucket": gcs_bucket}


@pytest.fixture
def api_key_secret():
    """Get API key secret name or skip."""
    secret_name = os.environ.get("API_KEY_SECRET")
    if not secret_name:
        pytest.skip("API_KEY_SECRET environment variable required (name of Secret Manager secret).")
    return secret_name


class TestCloudRunClient:
    """Integration tests for CloudRunClient."""

    def test_simple_command(self, integration_enabled, gcp_config):
        """Test running a simple command and getting output."""
        from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig

        config = CloudRunClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            cpu="1",
            memory="512Mi",
            timeout=300,
        )

        client = CloudRunClient(config)
        result = client.run_single(command="echo 'hello from cloud run' && echo 'error test' >&2")

        assert result.returncode == 0, f"Command failed: {result.error}"
        assert "hello from cloud run" in result.stdout
        assert "error test" in result.stderr

    def test_file_roundtrip(self, integration_enabled, gcp_config):
        """Test sending files in and getting files out."""
        from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig

        config = CloudRunClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            cpu="1",
            memory="512Mi",
            timeout=120,
        )

        # Create test input file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            (input_dir / "test.txt").write_text("hello world")

            client = CloudRunClient(config)

            # Read input, transform, write to output
            result = client.run_single(
                command="cat /workspace/input/data/test.txt | tr 'a-z' 'A-Z' > /workspace/output/result.txt",
                inputs={"data": input_dir},
            )

            assert result.returncode == 0, f"Command failed: {result.error}"
            assert result.output_dir is not None

            result_file = result.output_dir / "result.txt"
            assert result_file.exists()
            assert result_file.read_text().strip() == "HELLO WORLD"

    def test_multiple_tasks(self, integration_enabled, gcp_config):
        """Test running multiple tasks in parallel."""
        from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig, CloudRunTask

        config = CloudRunClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            cpu="1",
            memory="512Mi",
            timeout=120,
        )

        tasks = [
            CloudRunTask(id="task-1", command="echo 'task one'", n=2),
            CloudRunTask(id="task-2", command="echo 'task two'", n=3),
        ]

        client = CloudRunClient(config)
        results = client.run(tasks)

        # Results keyed by task object
        task1, task2 = tasks
        assert task1 in results
        assert task2 in results
        assert len(results[task1]) == 2
        assert len(results[task2]) == 3

        for r in results[task1]:
            assert r.returncode == 0, f"Task failed: {r.error}"
            assert "task one" in r.stdout
        for r in results[task2]:
            assert r.returncode == 0, f"Task failed: {r.error}"
            assert "task two" in r.stdout

    def test_streaming_results(self, integration_enabled, gcp_config):
        """Test streaming results as they complete."""
        from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig, CloudRunTask

        config = CloudRunClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            cpu="1",
            memory="512Mi",
            timeout=120,
        )

        tasks = [
            CloudRunTask(id="task-1", command="echo 'task one'", n=2),
            CloudRunTask(id="task-2", command="echo 'task two'", n=2),
        ]

        client = CloudRunClient(config)

        # Collect streamed results
        streamed = []
        for task, run_idx, result in client.run_stream(tasks, progress=False):
            streamed.append((task.id, run_idx, result.success))

        # Should have 4 total results (2 + 2)
        assert len(streamed) == 4

        # Check we got results for both tasks
        task_ids = {item[0] for item in streamed}
        assert task_ids == {"task-1", "task-2"}


class TestClaudeCodeClient:
    """Integration tests for ClaudeCodeClient - full stack including Claude.

    These tests require an Anthropic API key stored in GCP Secret Manager.
    Set API_KEY_SECRET to the name of your secret (e.g., "anthropic-api-key-username").
    """

    def test_simple_task(self, integration_enabled, gcp_config, api_key_secret):
        """Test Claude Code executing a simple task."""
        from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

        config = ClaudeCodeClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            api_key_secret=api_key_secret,
            timeout=300,
        )

        client = ClaudeCodeClient(config)
        result = client.run_single(
            task="Write 'integration test passed' to /workspace/output/result.txt and then say 'done'",
        )

        assert result.returncode == 0, f"Task failed: {result.error}"
        assert "done" in result.response.lower()
        assert result.transcript is not None
        assert len(result.transcript) > 0

        # Check file was written
        if result.output_dir:
            result_file = result.output_dir / "result.txt"
            if result_file.exists():
                assert "integration test passed" in result_file.read_text().lower()

    def test_with_input_files(self, integration_enabled, gcp_config, api_key_secret):
        """Test Claude Code reading input files and producing output."""
        from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

        config = ClaudeCodeClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            api_key_secret=api_key_secret,
            timeout=300,
        )

        # Create test input
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "repo"
            input_dir.mkdir()
            (input_dir / "data.txt").write_text("The secret number is 42.")

            client = ClaudeCodeClient(config)
            result = client.run_single(
                task=(
                    "Read the file at /workspace/input/repo/data.txt. What is the secret number? Say just the number."
                ),
                inputs={"repo": input_dir},
            )

        assert result.returncode == 0, f"Task failed: {result.error}"
        assert "42" in result.response

    def test_with_system_prompt(self, integration_enabled, gcp_config, api_key_secret):
        """Test Claude Code with a custom system prompt / constitution."""
        from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

        config = ClaudeCodeClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            api_key_secret=api_key_secret,
            timeout=300,
        )

        system_prompt = "You are a helpful assistant. Always end your responses with 'CUSTOM_MARKER_12345'."

        client = ClaudeCodeClient(config)
        result = client.run_single(
            task="Say hello.",
            system_prompt=system_prompt,
        )

        assert result.returncode == 0, f"Task failed: {result.error}"
        assert "CUSTOM_MARKER_12345" in result.response

    def test_multiple_tasks(self, integration_enabled, gcp_config, api_key_secret):
        """Test running multiple different Claude Code tasks in parallel."""
        from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig, ClaudeCodeTask

        config = ClaudeCodeClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            api_key_secret=api_key_secret,
            timeout=300,
        )

        tasks = [
            ClaudeCodeTask(id="math", task="What is 2 + 2? Reply with just the number.", n=2),
            ClaudeCodeTask(id="greeting", task="Say 'hello world' and nothing else.", n=2),
        ]

        client = ClaudeCodeClient(config)
        results = client.run(tasks)

        # Results keyed by task object
        math_task, greeting_task = tasks
        assert math_task in results
        assert greeting_task in results
        assert len(results[math_task]) == 2
        assert len(results[greeting_task]) == 2

        for r in results[math_task]:
            assert "4" in r.response
        for r in results[greeting_task]:
            assert "hello" in r.response.lower()

    def test_streaming_results(self, integration_enabled, gcp_config, api_key_secret):
        """Test streaming Claude Code results as they complete."""
        from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig, ClaudeCodeTask

        config = ClaudeCodeClientConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            api_key_secret=api_key_secret,
            timeout=300,
        )

        tasks = [
            ClaudeCodeTask(id="task-1", task="Say 'one'", n=2),
            ClaudeCodeTask(id="task-2", task="Say 'two'", n=2),
        ]

        client = ClaudeCodeClient(config)

        # Collect streamed results
        streamed = []
        for task, run_idx, result in client.run_stream(tasks, progress=False):
            streamed.append((task.id, run_idx, result.success))

        # Should have 4 total results (2 + 2)
        assert len(streamed) == 4

        # Check we got results for both tasks
        task_ids = {item[0] for item in streamed}
        assert task_ids == {"task-1", "task-2"}
