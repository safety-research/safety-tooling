"""
Run Claude Code in ephemeral Cloud Run containers.

This module provides a high-level interface for running Claude Code tasks
in isolated containers. It builds on CloudRunJob, adding:
- Claude Code specific setup (nodejs, claude-code CLI, non-root user)
- System prompt / constitution support
- Transcript extraction from .claude/ directory

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
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage
from google.cloud.run_v2 import JobsClient

from .cloud_run_job import (
    CloudRunJob,
    CloudRunJobConfig,
    CloudRunJobError,
)

# Default Cloud Run image with gcloud CLI pre-installed
DEFAULT_IMAGE = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"


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
        cpu: vCPUs - 1, 2, 4, or 8 (default: 1)
        memory: Memory limit up to 32Gi (default: 2Gi)
        skip_permissions: Use --dangerously-skip-permissions (default: True)
        image: Container image (default: google-cloud-cli:slim). Must have gcloud CLI.
        pre_claude_command: Shell command to run before Claude Code (e.g., git config)
        post_claude_command: Shell command to run after Claude Code
    """

    project_id: str
    gcs_bucket: str
    region: str = "us-central1"
    model: str = "claude-opus-4-5-20251101"
    max_turns: int = 100
    timeout: int = 600
    cpu: str = "1"
    memory: str = "2Gi"
    skip_permissions: bool = True
    image: str = DEFAULT_IMAGE
    pre_claude_command: str | None = None
    post_claude_command: str | None = None


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

    Uses CloudRunJob for the underlying infrastructure, adding Claude Code
    specific setup and transcript extraction.

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

    def __init__(
        self,
        config: ClaudeCodeJobConfig,
        jobs_client: JobsClient | None = None,
        storage_client: storage.Client | None = None,
    ):
        self.config = config
        self._jobs_client = jobs_client
        self._storage_client = storage_client

        # Validate required config
        if not self.config.project_id:
            raise ClaudeCodeJobError("project_id is required in ClaudeCodeJobConfig")
        if not self.config.gcs_bucket:
            raise ClaudeCodeJobError("gcs_bucket is required in ClaudeCodeJobConfig")

    def _extract_transcript(self, output_dir: Path) -> list[dict] | None:
        """Extract conversation transcript from .claude/ directory."""
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
        system_prompt: str | None,
        api_key: str,
    ) -> str:
        """Build the shell script that runs inside the container."""
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

        pre_command_section = ""
        if self.config.pre_claude_command:
            pre_command_section = self.config.pre_claude_command

        post_command_section = ""
        if self.config.post_claude_command:
            post_command_section = self.config.post_claude_command

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

# Set up Claude config if provided
if [ -d /workspace/input/.claude ]; then
    cp -r /workspace/input/.claude /home/claude/.claude
    chown -R claude:claude /home/claude/.claude
fi

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
cd /workspace
{pre_command_section}
claude {claude_flags_str} {system_prompt_arg} "$(cat /tmp/task.txt)" > /workspace/output/response.txt 2>&1
echo $? > /workspace/output/exitcode.txt
{post_command_section}
'
set -e

echo "Claude Code finished"

# Copy Claude home for transcript extraction
cp -r /home/claude/.claude /workspace/output/claude_home 2>/dev/null || true

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

        # Build the command script
        script = self._build_container_script(
            task=task,
            system_prompt=system_prompt,
            api_key=api_key,
        )

        # Configure Cloud Run job
        cloud_run_config = CloudRunJobConfig(
            image=self.config.image,
            project_id=self.config.project_id,
            gcs_bucket=self.config.gcs_bucket,
            region=self.config.region,
            cpu=self.config.cpu,
            memory=self.config.memory,
            timeout=self.config.timeout + 120,  # Extra buffer for setup
            name_prefix="claude",
            env={},
        )

        # Run the job
        try:
            with CloudRunJob(
                cloud_run_config,
                jobs_client=self._jobs_client,
                storage_client=self._storage_client,
            ) as job:
                # Send inputs (if any)
                if inputs or claude_config_dir:
                    all_inputs = dict(inputs) if inputs else {}
                    if claude_config_dir and claude_config_dir.exists():
                        all_inputs[".claude"] = claude_config_dir
                    job.send_inputs(all_inputs)

                # Run Claude Code (outputs are downloaded automatically)
                result = job.run(script, timeout=self.config.timeout)
                output_dir = result.output_dir

        except CloudRunJobError as e:
            raise ClaudeCodeJobError(f"Cloud Run job failed: {e}")

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

        duration = time.time() - start_time

        return ClaudeCodeJobResult(
            response=response,
            transcript=transcript,
            output_dir=output_dir,
            returncode=returncode,
            duration_seconds=duration,
        )
