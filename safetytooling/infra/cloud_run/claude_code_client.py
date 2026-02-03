"""
ClaudeCodeClient - Run Claude Code in ephemeral Cloud Run containers.

This is a thin wrapper over CloudRunClient that adds:
- Claude Code CLI installation
- Non-root user setup (required by Claude Code)
- System prompt / constitution support
- Transcript extraction from .claude/ directory

Usage:
    from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig, ClaudeCodeTask

    # Create client (reads GCLOUD_PROJECT_ID and GCLOUD_GCS_BUCKET from environment)
    client = ClaudeCodeClient()

    # Or pass explicitly
    client = ClaudeCodeClient(project_id="my-project", gcs_bucket="my-bucket")

    # Create tasks
    tasks = [
        ClaudeCodeTask(
            id="review-1",
            task="Review the code for XSS vulnerabilities",
            inputs=(("repo", "./my_repo"),),
            n=10,
        ),
        ClaudeCodeTask(
            id="review-2",
            task="Review for SQL injection",
            inputs=(("repo", "./my_repo"),),
            system_prompt="You are a security expert.",
            n=5,
        ),
    ]

    # Run all and get results
    results = client.run(tasks)
    # Returns: {task1: [Result, ...], task2: [Result, ...]}

    # Or stream results as they complete
    for task, run_idx, result in client.run_stream(tasks):
        print(f"{task.id}[{run_idx}]: {result.success}")
        save_to_db(task, run_idx, result)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .cloud_run_client import (
    CloudRunClient,
    CloudRunClientConfig,
    CloudRunResult,
    CloudRunTask,
)

# Pre-built image with Claude Code, git, nodejs, npm, and non-root user
# Saves ~2 minutes of setup time per job
DEFAULT_CLAUDE_CODE_IMAGE = "gcr.io/fellows-safety-research/claude-code-runner:latest"


# Default tools for code review: exploration, editing, and web search
# These allow exploring and modifying code, but no arbitrary shell execution
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Bash(cd *)",
    "Bash(ls *)",
    "Bash(find *)",
    "Bash(cat *)",
    "Bash(head *)",
    "Bash(tail *)",
    "Bash(wc *)",
    "Bash(file *)",
    "Bash(git log *)",
    "Bash(git show *)",
    "Bash(git diff *)",
    "Bash(git status *)",
    "Bash(git branch *)",
    "Bash(git commit *)",  # For committing review findings
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
        timeout: Job timeout in seconds (default: 300). Pre-built image needs no setup time.
        cpu: vCPUs - 1, 2, 4, or 8 (default: 1)
        memory: Memory limit up to 32Gi (default: 2Gi)
        skip_permissions: Use --dangerously-skip-permissions (default: False).
                         If True, ignores allowed_tools and skips all permission prompts.
        allowed_tools: Tools to allow without prompting (default: DEFAULT_ALLOWED_TOOLS).
                      Only used when skip_permissions=False.
                      Patterns like "Bash(git *)" allow specific commands.
                      Since Cloud Run is non-interactive, Claude will fail if it tries to use
                      a tool that isn't in this list (no way to prompt for permission).
                      See https://code.claude.com/docs/en/settings#permission-rule-syntax
        image: Container image (default: pre-built claude-code-runner).
               The default image has Claude Code pre-installed, saving ~2 min setup.
               Set to "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim" for stock image.
        api_key_secret: Name of Secret Manager secret containing ANTHROPIC_API_KEY (required).
                       The secret is injected securely via GCP Secret Manager at runtime.
                       Format: "secret-name" or "projects/proj/secrets/name"
        service_account: Service account email for the job (default: uses project default).
                        SECURITY: Use a restricted service account to limit container access.
                        See README for setup instructions.
                        Format: "name@project.iam.gserviceaccount.com"
    """

    project_id: str
    gcs_bucket: str
    name_prefix: str = "claude"
    region: str = "us-central1"
    model: str = "claude-opus-4-5-20251101"
    max_turns: int = 100
    timeout: int = 300  # Reduced from 600 - no setup time needed with pre-built image
    cpu: str = "1"
    memory: str = "2Gi"
    skip_permissions: bool = False
    allowed_tools: tuple[str, ...] | None = DEFAULT_ALLOWED_TOOLS
    image: str = DEFAULT_CLAUDE_CODE_IMAGE
    api_key_secret: str | None = None
    service_account: str | None = None


# Instructions prepended to task when output_instructions=True
OUTPUT_INSTRUCTIONS = """Write any output files to /workspace/output/ so they can be retrieved after the job completes.
Input files are available at /workspace/input/.

"""


@dataclass(frozen=True)
class ClaudeCodeTask:
    """A Claude Code task to run in Cloud Run.

    Frozen (immutable) so it can be used as a dict key.

    Args:
        id: Unique identifier for this task
        task: The prompt/instruction for Claude Code
        inputs: Tuple of (name, path) pairs for files to upload (tuple for hashability)
        system_prompt: Optional system prompt / constitution
        pre_claude_command: Shell command to run before Claude Code (e.g., git config, repo setup)
        post_claude_command: Shell command to run after Claude Code
        n: Number of times to run this task (default: 1)
        output_instructions: Prepend instructions telling Claude where to write output files (default: True)
    """

    id: str
    task: str
    inputs: tuple[tuple[str, str | Path], ...] | None = None
    system_prompt: str | None = None
    pre_claude_command: str | None = None
    post_claude_command: str | None = None
    n: int = 1
    output_instructions: bool = True

    def to_cloud_run_task(
        self,
        config: ClaudeCodeClientConfig,
    ) -> CloudRunTask:
        """Convert to a CloudRunTask by building the full shell script.

        Args:
            config: Client configuration with model, max_turns, etc.

        Returns:
            CloudRunTask ready to execute
        """
        # Optionally prepend output instructions
        task = self.task
        if self.output_instructions:
            task = OUTPUT_INSTRUCTIONS + task

        command = _build_claude_command(
            task=task,
            system_prompt=self.system_prompt,
            pre_claude_command=self.pre_claude_command,
            post_claude_command=self.post_claude_command,
            config=config,
        )
        return CloudRunTask(
            id=self.id,
            command=command,
            inputs=self.inputs,
            n=self.n,
            timeout=config.timeout,
        )


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


def _build_claude_command(
    task: str,
    system_prompt: str | None,
    pre_claude_command: str | None,
    post_claude_command: str | None,
    config: ClaudeCodeClientConfig,
) -> str:
    """Build the shell script that runs Claude Code inside the container.

    Note: ANTHROPIC_API_KEY is passed via Cloud Run environment variables,
    not embedded in this script. This is more secure as it:
    - Doesn't appear in the job spec visible to anyone with jobs.get permission
    - Doesn't appear in process listings
    - Doesn't risk appearing in logs

    The script auto-detects whether it's running in a pre-built image (with Claude
    already installed) or a stock image (needs installation).
    """
    claude_flags = [
        "-p",
        f"--model {config.model}",
        f"--max-turns {config.max_turns}",
    ]
    if config.skip_permissions:
        claude_flags.append("--dangerously-skip-permissions")
    elif config.allowed_tools:
        # Pass all tools as space-separated arguments after --allowedTools
        # Use double quotes for tools with special chars (works inside single-quoted shell command)
        quoted_tools = []
        for tool in config.allowed_tools:
            if "(" in tool or " " in tool or "*" in tool:
                quoted_tools.append(f'"{tool}"')
            else:
                quoted_tools.append(tool)
        tools_str = " ".join(quoted_tools)
        claude_flags.append(f"--allowedTools {tools_str}")

    claude_flags_str = " ".join(claude_flags)

    system_prompt_arg = ""
    if system_prompt:
        system_prompt_arg = '--system-prompt "$(cat /tmp/system_prompt.txt)"'

    pre_command_section = pre_claude_command or ""
    post_command_section = post_claude_command or ""

    script = f"""
set -e
echo "=== Claude Code Job Starting ==="

# Auto-detect if we need to install dependencies
# Pre-built image has claude already installed, stock image does not
if command -v claude &> /dev/null; then
    echo "Using pre-built image (claude already installed)"
else
    echo "Installing dependencies (stock image)..."
    apt-get update -qq && apt-get install -y -qq git nodejs npm > /dev/null 2>&1
    npm install -g @anthropic-ai/claude-code@latest > /dev/null 2>&1
    echo "Dependencies installed"
fi

# Create non-root user if it doesn't exist (pre-built image already has it)
id -u claude &>/dev/null || useradd -m -s /bin/bash claude

# Wrap git to force --no-verify on commits (skip hooks for security)
cat > /usr/local/bin/git << 'GITWRAPPER'
#!/bin/bash
if [[ "$1" == "commit" ]]; then
    exec /usr/bin/git commit --no-verify "${{@:2}}"
else
    exec /usr/bin/git "$@"
fi
GITWRAPPER
chmod +x /usr/local/bin/git

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

# Write pre-claude command to script file (quoted heredoc handles all special chars)
cat > /tmp/pre_claude.sh << 'PRE_CLAUDE_EOF'
{pre_command_section}
PRE_CLAUDE_EOF
chmod +x /tmp/pre_claude.sh

# Write post-claude command to script file
cat > /tmp/post_claude.sh << 'POST_CLAUDE_EOF'
{post_command_section}
POST_CLAUDE_EOF
chmod +x /tmp/post_claude.sh

# Set permissions
chown -R claude:claude /workspace

# Run Claude Code as non-root user
# Use 'su claude' (not 'su - claude') to preserve environment including ANTHROPIC_API_KEY
echo "Starting Claude Code..."
set +e
su claude -c 'cd /workspace && /tmp/pre_claude.sh && claude {claude_flags_str} {system_prompt_arg} "$(cat /tmp/task.txt)"; echo $? > /tmp/claude_exitcode.txt; /tmp/post_claude.sh'
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


def _extract_transcript(output_dir: Path) -> list[dict] | None:
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


def _convert_result(cloud_run_result: CloudRunResult) -> ClaudeCodeResult:
    """Convert CloudRunResult to ClaudeCodeResult."""
    transcript = _extract_transcript(cloud_run_result.output_dir)

    return ClaudeCodeResult(
        response=cloud_run_result.stdout,
        transcript=transcript,
        output_dir=cloud_run_result.output_dir,
        returncode=cloud_run_result.returncode,
        duration_seconds=cloud_run_result.duration_seconds,
        error=cloud_run_result.error,
    )


class ClaudeCodeClient:
    """
    Client for running Claude Code in ephemeral Cloud Run containers.

    Thin wrapper over CloudRunClient that handles Claude Code specific setup.

    Example:
        # From environment (GCLOUD_PROJECT_ID, GCLOUD_GCS_BUCKET)
        client = ClaudeCodeClient()

        # Explicit
        client = ClaudeCodeClient(project_id="my-project", gcs_bucket="my-bucket")

        # Create tasks
        tasks = [
            ClaudeCodeTask(id="review-1", task="Review for XSS", n=10),
            ClaudeCodeTask(id="review-2", task="Review for SQLi", n=5),
        ]

        # Run all and get results
        results = client.run(tasks)
        # Returns: {task1: [Result, ...], task2: [Result, ...]}

        # Or stream as they complete
        for task, run_idx, result in client.run_stream(tasks):
            print(f"{task.id}[{run_idx}]: {result.success}")
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

        # Build secrets dict for Cloud Run
        secrets = {}
        if self.config.api_key_secret:
            secrets["ANTHROPIC_API_KEY"] = self.config.api_key_secret

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
            secrets=secrets,
            service_account=self.config.service_account,
        )
        self._cloud_run = CloudRunClient(cloud_run_config)

    def run(
        self,
        tasks: list[ClaudeCodeTask],
        max_workers: int | None = None,
        progress: bool = True,
    ) -> dict[ClaudeCodeTask, list[ClaudeCodeResult]]:
        """
        Run Claude Code tasks.

        Args:
            tasks: List of ClaudeCodeTask objects
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Returns:
            Dict mapping task -> list of results

        Note:
            ANTHROPIC_API_KEY is injected via GCP Secret Manager (configured via api_key_secret).
            This is more secure than passing the key directly.

        Example:
            tasks = [
                ClaudeCodeTask(id="review-1", task="Review for XSS", n=10),
                ClaudeCodeTask(id="review-2", task="Review for SQLi", n=5),
            ]
            results = client.run(tasks)
            for task, result_list in results.items():
                print(f"{task.id}: {len(result_list)} results")
        """
        # Collect all results from streaming
        results_by_task: dict[ClaudeCodeTask, dict[int, ClaudeCodeResult]] = {task: {} for task in tasks}

        for task, run_idx, result in self.run_stream(tasks, max_workers=max_workers, progress=progress):
            results_by_task[task][run_idx] = result

        # Convert to ordered lists
        final_results: dict[ClaudeCodeTask, list[ClaudeCodeResult]] = {}
        for task in tasks:
            final_results[task] = [results_by_task[task][i] for i in range(task.n)]

        return final_results

    def run_stream(
        self,
        tasks: list[ClaudeCodeTask],
        max_workers: int | None = None,
        progress: bool = True,
    ) -> Iterator[tuple[ClaudeCodeTask, int, ClaudeCodeResult]]:
        """
        Run Claude Code tasks and yield results as they complete.

        Args:
            tasks: List of ClaudeCodeTask objects
            max_workers: Max concurrent jobs (default: unlimited)
            progress: Show progress bar (requires tqdm)

        Yields:
            Tuples of (task, run_index, result) as each job completes

        Note:
            ANTHROPIC_API_KEY is injected via GCP Secret Manager (configured via api_key_secret).
            This is more secure than passing the key directly.

        Example:
            for task, run_idx, result in client.run_stream(tasks):
                print(f"{task.id}[{run_idx}]: {result.success}")
                save_to_db(task, run_idx, result)
        """
        if not self.config.api_key_secret:
            raise ClaudeCodeClientError(
                "api_key_secret is required in config. "
                "Store your ANTHROPIC_API_KEY in GCP Secret Manager and reference it via api_key_secret."
            )

        if not self.config.service_account:
            raise ClaudeCodeClientError(
                "service_account is required in config for security. "
                "Create a restricted service account and specify it via service_account. "
                "See README 'Security Hardening' section for setup instructions."
            )

        # Build lookup from CloudRunTask id back to ClaudeCodeTask
        claude_task_by_id = {task.id: task for task in tasks}

        # Convert ClaudeCodeTasks to CloudRunTasks
        cloud_run_tasks = [task.to_cloud_run_task(self.config) for task in tasks]

        # Stream results from CloudRunClient
        for cloud_task, run_idx, cloud_result in self._cloud_run.run_stream(
            cloud_run_tasks, max_workers=max_workers, progress=progress
        ):
            claude_task = claude_task_by_id[cloud_task.id]
            yield claude_task, run_idx, _convert_result(cloud_result)

    def run_single(
        self,
        task: str,
        inputs: dict[str, Path] | None = None,
        system_prompt: str | None = None,
        pre_claude_command: str | None = None,
        post_claude_command: str | None = None,
        output_instructions: bool = True,
    ) -> ClaudeCodeResult:
        """
        Convenience method to run a single Claude Code task.

        Args:
            task: The prompt/instruction for Claude Code
            inputs: Dict of {name: local_path} to make available at /workspace/input/{name}
            system_prompt: Optional system prompt / constitution
            pre_claude_command: Shell command to run before Claude Code
            post_claude_command: Shell command to run after Claude Code
            output_instructions: Prepend instructions telling Claude where to write output files (default: True)

        Returns:
            ClaudeCodeResult

        Note:
            ANTHROPIC_API_KEY is injected via GCP Secret Manager (configured via api_key_secret).
        """
        # Convert inputs dict to tuple for ClaudeCodeTask
        inputs_tuple = None
        if inputs:
            inputs_tuple = tuple((name, str(path)) for name, path in inputs.items())

        claude_task = ClaudeCodeTask(
            id="_single",
            task=task,
            inputs=inputs_tuple,
            system_prompt=system_prompt,
            pre_claude_command=pre_claude_command,
            post_claude_command=post_claude_command,
            output_instructions=output_instructions,
        )
        results = self.run([claude_task], progress=False)
        return results[claude_task][0]
