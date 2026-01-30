"""
Run Claude Code in ephemeral Cloud Run containers.

This module provides a high-level interface for running Claude Code tasks
in isolated containers. It builds on CloudRunJob, adding:
- Claude Code specific setup (nodejs, claude-code CLI, non-root user)
- System prompt / constitution support
- Transcript extraction from .claude/ directory

Usage:
    from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

    client = ClaudeCodeClient(ClaudeCodeClientConfig(
        project_id="my-project",
        gcs_bucket="my-bucket",
    ))

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

    # Multiple different tasks
    results = client.run(tasks=[
        {"task": "Review for XSS", "inputs": {"repo": repo}},
        {"task": "Review for SQLi", "inputs": {"repo": repo}},
    ])
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage
from google.cloud.run_v2 import JobsClient

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .cloud_run_job import (
    CloudRunJob,
    CloudRunJobConfig,
    CloudRunJobError,
)

# Default Cloud Run image with gcloud CLI pre-installed
DEFAULT_IMAGE = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim"


@dataclass
class ClaudeCodeClientConfig:
    """Configuration for Claude Code client.

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
        return self.error is None


class ClaudeCodeClientError(Exception):
    """Raised when Claude Code client operations fail."""

    pass


class ClaudeCodeClient:
    """
    Client for running Claude Code in ephemeral Cloud Run containers.

    Example:
        client = ClaudeCodeClient(ClaudeCodeClientConfig(
            project_id="my-project",
            gcs_bucket="my-bucket",
        ))

        # Single task
        result = client.run(task="Review this code", inputs={"repo": repo})

        # Same task 100 times
        results = client.run(task="Review this code", inputs={"repo": repo}, n=100)

        # Multiple different tasks
        results = client.run(tasks=[
            {"task": "Review for XSS", "inputs": {"repo": repo}},
            {"task": "Review for SQLi", "inputs": {"repo": repo}},
        ])
    """

    def __init__(self, config: ClaudeCodeClientConfig):
        self.config = config

        # Validate required config
        if not self.config.project_id:
            raise ClaudeCodeClientError("project_id is required")
        if not self.config.gcs_bucket:
            raise ClaudeCodeClientError("gcs_bucket is required")

        # Create shared GCP clients
        self._jobs_client = JobsClient()
        self._storage_client = storage.Client(project=self.config.project_id)

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

    def _run_single(
        self,
        task: str,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
    ) -> ClaudeCodeResult:
        """Run a single Claude Code task."""
        start_time = time.time()

        # Get API key from param or environment
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeCodeClientError("API key required: pass api_key param or set ANTHROPIC_API_KEY env var")

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
            raise ClaudeCodeClientError(f"Cloud Run job failed: {e}")

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

        return ClaudeCodeResult(
            response=response,
            transcript=transcript,
            output_dir=output_dir,
            returncode=returncode,
            duration_seconds=duration,
        )

    def run(
        self,
        task: str | None = None,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        claude_config_dir: Path | None = None,
        api_key: str | None = None,
        n: int | None = None,
        tasks: list[dict] | None = None,
        max_workers: int | None = None,
        progress: bool = True,
    ) -> ClaudeCodeResult | list[ClaudeCodeResult]:
        """
        Run Claude Code task(s).

        Args:
            task: The prompt/instruction for Claude Code (for single or repeated runs)
            inputs: Dict of {name: local_path} to make available at /workspace/input/{name}
            system_prompt: Optional system prompt / constitution
            claude_config_dir: Optional path to .claude/ directory with custom commands
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
            n: Run the same task N times (returns list of results)
            tasks: List of task dicts for different tasks (returns list of results)
                   Each dict can have: task, inputs, system_prompt
            max_workers: Max concurrent jobs when running multiple (default: unlimited)
            progress: Show progress bar when running multiple (requires tqdm)

        Returns:
            ClaudeCodeResult for single task, list[ClaudeCodeResult] for multiple

        Examples:
            # Single task
            result = client.run(task="Review this code", inputs={"repo": repo})

            # Same task 100 times
            results = client.run(task="Review this code", inputs={"repo": repo}, n=100)

            # Multiple different tasks
            results = client.run(tasks=[
                {"task": "Review for XSS", "inputs": {"repo": repo}},
                {"task": "Review for SQLi", "inputs": {"repo": repo}},
            ])
        """
        # Validate args
        if tasks is not None and task is not None:
            raise ClaudeCodeClientError("Cannot specify both 'task' and 'tasks'")
        if tasks is None and task is None:
            raise ClaudeCodeClientError("Must specify either 'task' or 'tasks'")
        if n is not None and tasks is not None:
            raise ClaudeCodeClientError("Cannot specify both 'n' and 'tasks'")

        # Single task, no repetition
        if task is not None and n is None:
            return self._run_single(
                task=task,
                inputs=inputs,
                system_prompt=system_prompt,
                claude_config_dir=claude_config_dir,
                api_key=api_key,
            )

        # Build task list
        if tasks is not None:
            task_list = tasks
        else:
            # Same task N times
            task_list = [{"task": task, "inputs": inputs, "system_prompt": system_prompt} for _ in range(n)]

        # Get API key once
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeCodeClientError("API key required: pass api_key param or set ANTHROPIC_API_KEY env var")

        # Run in parallel
        results: dict[int, ClaudeCodeResult] = {}

        def run_one(idx: int, task_dict: dict) -> tuple[int, ClaudeCodeResult]:
            try:
                result = self._run_single(
                    task=task_dict["task"],
                    inputs=task_dict.get("inputs"),
                    system_prompt=task_dict.get("system_prompt"),
                    claude_config_dir=task_dict.get("claude_config_dir"),
                    api_key=api_key,
                )
                return idx, result
            except Exception as e:
                # Return a result with error set
                return idx, ClaudeCodeResult(
                    response="",
                    transcript=None,
                    output_dir=Path("/dev/null"),
                    returncode=1,
                    duration_seconds=0,
                    error=e,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_one, i, t): i for i, t in enumerate(task_list)}

            # Use tqdm if available and requested
            iterator = as_completed(futures)
            if progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Running Claude Code")

            for future in iterator:
                idx, result = future.result()
                results[idx] = result

        # Return in original order
        return [results[i] for i in range(len(task_list))]


# Backwards compatibility aliases
ClaudeCodeJobConfig = ClaudeCodeClientConfig
ClaudeCodeJobResult = ClaudeCodeResult
ClaudeCodeJobError = ClaudeCodeClientError
ClaudeCodeJob = ClaudeCodeClient
