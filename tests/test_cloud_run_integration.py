"""Integration tests for cloud_run module.

These tests require:
- GCP credentials (gcloud auth application-default login)
- ANTHROPIC_API_KEY environment variable
- A GCP project with Cloud Run and GCS enabled

Run with: pytest tests/test_cloud_run_integration.py -v --run-integration

Environment variables needed:
- GCP_PROJECT_ID: Your GCP project ID
- GCS_BUCKET: GCS bucket for file transfer
- ANTHROPIC_API_KEY: Anthropic API key for Claude Code tests
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
        pytest.skip("GCP_PROJECT_ID and GCS_BUCKET environment variables required. Set them to run integration tests.")

    return {"project_id": project_id, "gcs_bucket": gcs_bucket}


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key or skip."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY environment variable required.")
    return api_key


class TestCloudRunJob:
    """Integration tests for low-level CloudRunJob."""

    def test_simple_command(self, integration_enabled, gcp_config):
        """Test running a simple command and getting output."""
        from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

        config = CloudRunJobConfig(
            image="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            cpu="1",
            memory="512Mi",
            timeout=300,
        )

        with CloudRunJob(config) as job:
            result = job.run("echo 'hello from cloud run' && echo 'error test' >&2")

        assert result.returncode == 0
        assert "hello from cloud run" in result.stdout
        assert "error test" in result.stderr

    def test_file_roundtrip(self, integration_enabled, gcp_config):
        """Test sending files in and getting files out."""
        from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

        config = CloudRunJobConfig(
            image="gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
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

            with CloudRunJob(config) as job:
                job.send_inputs({"data": input_dir})

                # Read input, transform, write to output
                result = job.run("cat /workspace/input/data/test.txt | tr 'a-z' 'A-Z' > /workspace/output/result.txt")

                # receive_outputs() is idempotent - can call multiple times
                output_dir = job.receive_outputs()

            assert result.returncode == 0
            assert output_dir is not None

            result_file = output_dir / "result.txt"
            assert result_file.exists()
            assert result_file.read_text().strip() == "HELLO WORLD"


class TestClaudeCodeJob:
    """Integration tests for ClaudeCodeJob - full stack including Claude."""

    def test_simple_task(self, integration_enabled, gcp_config, anthropic_api_key):
        """Test Claude Code executing a simple task."""
        from safetytooling.infra.cloud_run import ClaudeCodeJob, ClaudeCodeJobConfig

        config = ClaudeCodeJobConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            timeout=300,
        )

        result = ClaudeCodeJob(config).run(
            task="Write 'integration test passed' to /workspace/output/result.txt and then say 'done'",
            api_key=anthropic_api_key,
        )

        assert result.returncode == 0
        assert "done" in result.response.lower()
        assert result.transcript is not None
        assert len(result.transcript) > 0

        # Check file was written
        if result.output_dir:
            result_file = result.output_dir / "result.txt"
            if result_file.exists():
                assert "integration test passed" in result_file.read_text().lower()

    def test_with_input_files(self, integration_enabled, gcp_config, anthropic_api_key):
        """Test Claude Code reading input files and producing output."""
        from safetytooling.infra.cloud_run import ClaudeCodeJob, ClaudeCodeJobConfig

        config = ClaudeCodeJobConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            timeout=300,
        )

        # Create test input
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "repo"
            input_dir.mkdir()
            (input_dir / "data.txt").write_text("The secret number is 42.")

            result = ClaudeCodeJob(config).run(
                task=(
                    "Read the file at /workspace/input/repo/data.txt. What is the secret number? Say just the number."
                ),
                inputs={"repo": input_dir},
                api_key=anthropic_api_key,
            )

        assert result.returncode == 0
        assert "42" in result.response

    def test_with_system_prompt(self, integration_enabled, gcp_config, anthropic_api_key):
        """Test Claude Code with a custom system prompt / constitution."""
        from safetytooling.infra.cloud_run import ClaudeCodeJob, ClaudeCodeJobConfig

        config = ClaudeCodeJobConfig(
            project_id=gcp_config["project_id"],
            gcs_bucket=gcp_config["gcs_bucket"],
            timeout=300,
        )

        system_prompt = "You are a helpful assistant. Always end your responses with 'CUSTOM_MARKER_12345'."

        result = ClaudeCodeJob(config).run(
            task="Say hello.",
            system_prompt=system_prompt,
            api_key=anthropic_api_key,
        )

        assert result.returncode == 0
        assert "CUSTOM_MARKER_12345" in result.response
