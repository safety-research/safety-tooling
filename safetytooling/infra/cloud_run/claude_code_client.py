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

    # Single task (convenience method)
    result = client.run_single(
        task="Review the code for security issues",
        inputs={"repo": Path("./my_repo")},
    )

    # Multiple tasks in parallel
    results = client.run([
        {"id": "review-1", "task": "Review for XSS", "inputs": {"repo": repo}, "n": 10},
        {"id": "review-2", "task": "Review for SQLi", "inputs": {"repo": repo}, "n": 5},
    ])
    # Returns: {"review-1": [Result, ...], "review-2": [Result, ...]}
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .cloud_run_client import (
    CloudRunClient,
    CloudRunClientConfig,
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
        timeout: Job timeout in seconds (default: 600). Should include ~2min for setup
                 (installing nodejs, npm, claude-code CLI, creating user).
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

        # Single task (convenience method)
        result = client.run_single(task="Review this code", inputs={"repo": repo})

        # Multiple tasks in parallel
        results = client.run([
            {"id": "review-1", "task": "Review for XSS", "inputs": {"repo": repo}, "n": 10},
            {"id": "review-2", "task": "Review for SQLi", "inputs": {"repo": repo}, "n": 5},
        ])
        # Returns: {"review-1": [Result, ...], "review-2": [Result, ...]}
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
            timeout=self.config.timeout,
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
        tasks: list[dict],
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
        max_workers: int | None = None,
        progress: bool = True,
    ) -> dict[str, list[ClaudeCodeResult]]:
        """
        Run Claude Code tasks.

        Args:
            tasks: List of task dicts with fields:
                id: Unique identifier for this task (required)
                task: The prompt/instruction for Claude Code (required)
                inputs: Dict of {name: local_path} (optional)
                system_prompt: System prompt / constitution (optional, overrides global)
                n: Number of times to run this task (default: 1)
            system_prompt: Default system prompt for tasks that don't specify one
            claude_config_dir: Optional path to .claude/ directory with custom commands
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Returns:
            Dict mapping task id -> list of results

        Example:
            results = client.run([
                {"id": "review-1", "task": "Review for XSS", "inputs": {"repo": repo}, "n": 10},
                {"id": "review-2", "task": "Review for SQLi", "inputs": {"repo": repo}, "n": 5},
            ])
            # Returns: {"review-1": [Result, ...], "review-2": [Result, ...]}
        """
        # Get API key from param or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeCodeClientError("API key required: pass api_key param or set ANTHROPIC_API_KEY env var")

        # Build CloudRunClient tasks from ClaudeCode tasks
        cloud_run_tasks = []
        for task_dict in tasks:
            task_id = task_dict["id"]
            task_prompt = task_dict["task"]
            task_inputs = task_dict.get("inputs")
            task_system_prompt = task_dict.get("system_prompt", system_prompt)
            task_n = task_dict.get("n", 1)

            # Build inputs
            all_inputs = dict(task_inputs) if task_inputs else {}
            if claude_config_dir and claude_config_dir.exists():
                all_inputs[".claude"] = claude_config_dir

            # Build command for this task
            command = self._build_claude_command(
                task=task_prompt,
                system_prompt=task_system_prompt,
                api_key=api_key,
            )

            cloud_run_tasks.append(
                {
                    "id": task_id,
                    "command": command,
                    "inputs": all_inputs if all_inputs else None,
                    "timeout": self.config.timeout,
                    "n": task_n,
                }
            )

        # Run all tasks
        cloud_run_results = self._cloud_run.run(
            tasks=cloud_run_tasks,
            max_workers=max_workers,
            progress=progress,
        )

        # Convert results
        final_results: dict[str, list[ClaudeCodeResult]] = {}
        for task_id, results_list in cloud_run_results.items():
            final_results[task_id] = [self._convert_result(r) for r in results_list]

        return final_results

    def run_single(
        self,
        task: str,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
    ) -> ClaudeCodeResult:
        """
        Convenience method to run a single Claude Code task.

        Args:
            task: The prompt/instruction for Claude Code
            inputs: Dict of {name: local_path} to make available at /workspace/input/{name}
            system_prompt: Optional system prompt / constitution
            claude_config_dir: Optional path to .claude/ directory with custom commands
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)

        Returns:
            ClaudeCodeResult

        Example:
            result = client.run_single(
                task="Review this code for security issues",
                inputs={"repo": Path("./my_repo")},
            )
        """
        results = self.run(
            tasks=[{"id": "_single", "task": task, "inputs": inputs}],
            system_prompt=system_prompt,
            claude_config_dir=claude_config_dir,
            api_key=api_key,
            progress=False,
        )
        return results["_single"][0]
