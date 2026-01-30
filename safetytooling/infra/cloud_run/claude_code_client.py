"""
ClaudeCodeClient - Run Claude Code in ephemeral Cloud Run containers.

This is a thin wrapper over CloudRunClient that adds:
- Claude Code CLI installation
- Non-root user setup (required by Claude Code)
- System prompt / constitution support
- Transcript extraction from .claude/ directory

Usage:
    from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

    # Reads GCLOUD_PROJECT_ID and GCLOUD_GCS_BUCKET from environment
    client = ClaudeCodeClient()

    # Or pass explicitly
    client = ClaudeCodeClient(project_id="my-project", gcs_bucket="my-bucket")

    # Single task
    result = client.run(
        task="Review the code for security issues",
        inputs={"repo": Path("./my_repo")},
    )

    # Same task N times (for measuring variance)
    results = client.run(
        task="Review the code for security issues",
        inputs={"repo": Path("./my_repo")},
        n=100,
    )
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .cloud_run_client import (
    CloudRunClient,
    CloudRunClientConfig,
    CloudRunClientError,
    CloudRunResult,
)


@dataclass
class ClaudeCodeClientConfig:
    """Configuration for Claude Code client.

    Args:
        project_id: GCP project ID (required)
        gcs_bucket: GCS bucket for file I/O (required)
        name_prefix: Prefix for Cloud Run job names (default: "claude")
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
    name_prefix: str = "claude"
    region: str = "us-central1"
    model: str = "claude-opus-4-5-20251101"
    max_turns: int = 100
    timeout: int = 600
    cpu: str = "1"
    memory: str = "2Gi"
    skip_permissions: bool = True
    image: str = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"
    pre_claude_command: str | None = None
    post_claude_command: str | None = None


@dataclass
class ClaudeCodeResult:
    """Result from running Claude Code."""

    response: str  # Claude's stdout
    transcript: list[dict] | None  # Parsed session from .claude/
    output_dir: Path  # Local dir with contents of /workspace/output
    returncode: int
    duration_seconds: float
    error: Exception | None = None  # Set if this task failed

    @property
    def success(self) -> bool:
        return self.error is None and self.returncode == 0


class ClaudeCodeClientError(Exception):
    """Raised when Claude Code client operations fail."""

    pass


class ClaudeCodeClient:
    """
    Client for running Claude Code in ephemeral Cloud Run containers.

    Thin wrapper over CloudRunClient that adds Claude Code specific setup.

    Example:
        # From environment (GCLOUD_PROJECT_ID, GCLOUD_GCS_BUCKET)
        client = ClaudeCodeClient()

        # Explicit
        client = ClaudeCodeClient(project_id="my-project", gcs_bucket="my-bucket")

        # With a config object
        config = ClaudeCodeClientConfig(
            project_id="my-project",
            gcs_bucket="my-bucket",
            name_prefix="claude-monitor",
            timeout=300,
        )
        client = ClaudeCodeClient(config)

        # Single task
        result = client.run(task="Review this code", inputs={"repo": repo})

        # Same task 100 times in parallel
        results = client.run(task="Review this code", inputs={"repo": repo}, n=100)
    """

    def __init__(
        self,
        config_or_project_id: ClaudeCodeClientConfig | str | None = None,
        gcs_bucket: str | None = None,
        **kwargs,
    ):
        # Accept either a config object or individual args
        if isinstance(config_or_project_id, ClaudeCodeClientConfig):
            self.config = config_or_project_id
        else:
            # Build config from args
            project_id = config_or_project_id or os.environ.get("GCLOUD_PROJECT_ID")
            gcs_bucket = gcs_bucket or os.environ.get("GCLOUD_GCS_BUCKET")

            if not project_id:
                raise ClaudeCodeClientError("project_id required: pass as argument or set GCLOUD_PROJECT_ID env var")
            if not gcs_bucket:
                raise ClaudeCodeClientError("gcs_bucket required: pass as argument or set GCLOUD_GCS_BUCKET env var")

            self.config = ClaudeCodeClientConfig(
                project_id=project_id,
                gcs_bucket=gcs_bucket,
                **kwargs,
            )

        # Create the underlying CloudRunClient
        cloud_run_config = CloudRunClientConfig(
            project_id=self.config.project_id,
            gcs_bucket=self.config.gcs_bucket,
            name_prefix=self.config.name_prefix,
            region=self.config.region,
            image=self.config.image,
            cpu=self.config.cpu,
            memory=self.config.memory,
            timeout=self.config.timeout + 120,  # Extra buffer for setup
            env={},
        )
        self._cloud_run = CloudRunClient(cloud_run_config)

    def _build_claude_command(
        self,
        task: str,
        system_prompt: str | None,
        api_key: str,
    ) -> str:
        """Build the shell script that runs Claude Code inside the container."""
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
claude {claude_flags_str} {system_prompt_arg} "$(cat /tmp/task.txt)"
echo $? > /tmp/claude_exitcode.txt
{post_command_section}
'
CLAUDE_EXIT=$?
set -e

# Get the real exit code from Claude
if [ -f /tmp/claude_exitcode.txt ]; then
    CLAUDE_EXIT=$(cat /tmp/claude_exitcode.txt)
fi

echo "Claude Code finished with exit code: $CLAUDE_EXIT"

# Copy Claude home for transcript extraction
mkdir -p /workspace/output
cp -r /home/claude/.claude /workspace/output/claude_home 2>/dev/null || true

echo "=== Claude Code Job Complete ==="
exit $CLAUDE_EXIT
"""
        return script

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

    def _convert_result(self, cloud_run_result: CloudRunResult) -> ClaudeCodeResult:
        """Convert CloudRunResult to ClaudeCodeResult."""
        transcript = self._extract_transcript(cloud_run_result.output_dir)

        return ClaudeCodeResult(
            response=cloud_run_result.stdout,
            transcript=transcript,
            output_dir=cloud_run_result.output_dir,
            returncode=cloud_run_result.returncode,
            duration_seconds=cloud_run_result.duration_seconds,
            error=cloud_run_result.error,
        )

    def run(
        self,
        task: str,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
        n: int | None = None,
        max_workers: int | None = None,
        progress: bool = True,
    ) -> ClaudeCodeResult | list[ClaudeCodeResult]:
        """
        Run Claude Code task(s).

        Args:
            task: The prompt/instruction for Claude Code
            inputs: Dict of {name: local_path} to make available at /workspace/input/{name}
            system_prompt: Optional system prompt / constitution
            claude_config_dir: Optional path to .claude/ directory with custom commands
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
            n: Run the same task N times in parallel (returns list of results)
            max_workers: Max concurrent jobs when running multiple (default: unlimited)
            progress: Show progress bar when running multiple (requires tqdm)

        Returns:
            ClaudeCodeResult for single task, list[ClaudeCodeResult] for n > 1

        Examples:
            # Single task
            result = client.run(task="Review this code", inputs={"repo": repo})

            # Same task 100 times in parallel
            results = client.run(task="Review this code", inputs={"repo": repo}, n=100)
        """
        # Get API key from param or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeCodeClientError("API key required: pass api_key param or set ANTHROPIC_API_KEY env var")

        # Build inputs dict
        all_inputs = dict(inputs) if inputs else {}
        if claude_config_dir and claude_config_dir.exists():
            all_inputs[".claude"] = claude_config_dir

        # Build the Claude command
        command = self._build_claude_command(
            task=task,
            system_prompt=system_prompt,
            api_key=api_key,
        )

        # Delegate to CloudRunClient
        result = self._cloud_run.run(
            command=command,
            inputs=all_inputs if all_inputs else None,
            timeout=self.config.timeout,
            n=n,
            max_workers=max_workers,
            progress=progress,
        )

        # Convert results
        if isinstance(result, list):
            return [self._convert_result(r) for r in result]
        return self._convert_result(result)
