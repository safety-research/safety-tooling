# Cloud Run Module

Run many Claude Code instances in parallel on ephemeral GCP Cloud Run containers, with GCS-based file I/O. Send files in, get files back.

**Primary use case**: Batch processing with Claude Code - review many PRs, analyze many codebases, run many investigations in parallel without managing infrastructure.

## Quick Start

```python
from pathlib import Path
from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeClientConfig

client = ClaudeCodeClient(ClaudeCodeClientConfig(
    project_id="my-gcp-project",
    gcs_bucket="my-bucket",
))

# Single task (convenience method)
result = client.run_single(
    task="Review this PR for security issues",
    inputs={"repo": Path("./my_repo")},
)
print(result.response)

# Multiple different tasks in parallel
results = client.run([
    {"id": "xss-review", "task": "Review for XSS vulnerabilities", "inputs": {"repo": repo}, "n": 10},
    {"id": "sqli-review", "task": "Review for SQL injection", "inputs": {"repo": repo}, "n": 10},
])
# Returns: {"xss-review": [Result, ...], "sqli-review": [Result, ...]}

# Aggregate results
for task_id, task_results in results.items():
    flagged = sum(1 for r in task_results if r.success and "vulnerability" in r.response.lower())
    print(f"{task_id}: flagged in {flagged}/{len(task_results)} runs")
```

## Custom Containers (low-level)

```python
from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig

client = CloudRunClient(CloudRunClientConfig(
    project_id="my-project",
    gcs_bucket="my-bucket",
    image="gcr.io/my-project/my-image:latest",
))

# Single command (convenience method)
result = client.run_single(
    command="cd /workspace/input/repo && python script.py",
    inputs={"repo": Path("./my_repo"), "data.json": Path("./config.json")},
)

print(f"Exit code: {result.returncode}")
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"Outputs at: {result.output_dir}")

# Multiple commands in parallel
results = client.run([
    {"id": "task-1", "command": "python script1.py", "inputs": {"repo": repo}, "n": 10},
    {"id": "task-2", "command": "python script2.py", "inputs": {"repo": repo}, "n": 5},
])
```

## Prerequisites

1. **GCP Project** with Cloud Run and Cloud Storage APIs enabled
2. **GCS Bucket** for file transfer
3. **Authentication**: `gcloud auth application-default login` or service account
4. **Anthropic API Key** (for ClaudeCodeClient): Set `ANTHROPIC_API_KEY` env var or pass to `run()`

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐
│ Local files │────▶│ GCS bucket  │────▶│ Cloud Run container         │
│             │     │             │     │                             │
│ inputs={    │     │ inputs/     │     │  /workspace/input/repo/     │
│   "repo": / │     │ {hash}.tgz  │     │  /workspace/output/         │
│ }           │     │             │     │                             │
└─────────────┘     │ outputs/    │◀────│  (your command runs here)   │
       ▲            │ {job}.tgz   │     └─────────────────────────────┘
       │            └─────────────┘
       └──────────────────┘
         Download outputs
```

1. **Tar and hash** local inputs (deterministic tarball for caching)
2. **Upload to GCS** (skipped if hash already exists - content-addressed)
3. **Create Cloud Run Job** that downloads inputs, runs command, uploads outputs
4. **Download outputs** and return result

## API Reference

### ClaudeCodeClientConfig

```python
ClaudeCodeClientConfig(
    project_id: str,       # GCP project ID (required)
    gcs_bucket: str,       # GCS bucket for file I/O (required)
    region: str = "us-central1",
    model: str = "claude-opus-4-5-20251101",
    max_turns: int = 100,  # Max conversation turns
    timeout: int = 600,    # Job timeout in seconds
    cpu: str = "1",        # vCPUs: 1, 2, 4, or 8 (Claude Code is I/O bound)
    memory: str = "2Gi",   # Up to 32Gi
    skip_permissions: bool = True,  # --dangerously-skip-permissions
    image: str = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
    pre_claude_command: str | None = None,   # Run before claude (e.g., git config)
    post_claude_command: str | None = None,  # Run after claude
)
```

**`pre_claude_command` / `post_claude_command`**: Shell commands to run before/after Claude Code. Useful for:
- Git config: `'git config --global user.email "bot@example.com" && git config --global user.name "Bot"'`
- Installing dependencies: `'pip install some-package'`
- Post-processing outputs

### ClaudeCodeClient.run()

```python
results = client.run(
    tasks: list[dict],                # List of task dicts (see below)
    system_prompt: str = None,        # Default system prompt for all tasks
    claude_config_dir: Path = None,   # Custom .claude/ directory
    api_key: str = None,              # Default: ANTHROPIC_API_KEY env var
    max_workers: int = None,          # Max parallel jobs (default: unlimited)
    progress: bool = True,            # Show progress bar (requires tqdm)
)
# Returns: dict mapping task id -> list of ClaudeCodeResult
```

**Task dict fields:**
- `id` (required): Unique identifier for this task
- `task` (required): The prompt/instruction for Claude Code
- `inputs` (optional): Dict of {name: local_path}
- `system_prompt` (optional): Override the default system prompt for this task
- `n` (optional): Number of times to run this task (default: 1)

### ClaudeCodeClient.run_single()

```python
result = client.run_single(
    task: str,                        # The prompt for Claude Code
    inputs: dict[str, Path] = None,   # Files at /workspace/input/
    system_prompt: str = None,        # Custom system prompt / constitution
    claude_config_dir: Path = None,   # Custom .claude/ directory
    api_key: str = None,              # Default: ANTHROPIC_API_KEY env var
)
# Returns: ClaudeCodeResult
```

### ClaudeCodeResult

```python
result.response          # str: Claude's stdout
result.transcript        # list[dict]: Parsed conversation from .claude/
result.output_dir        # Path: Local dir with /workspace/output contents
result.returncode        # int: Exit code
result.duration_seconds  # float: Total execution time
result.success           # bool: True if no error occurred
result.error             # Exception | None: Error if task failed
```

### CloudRunClientConfig

```python
CloudRunClientConfig(
    project_id: str,       # GCP project ID (required)
    gcs_bucket: str,       # GCS bucket for file I/O (required)
    name_prefix: str = "cloudrun",
    region: str = "us-central1",
    image: str = "gcr.io/google.com/cloudsdktool/google-cloud-cli:slim",
    cpu: str = "1",        # vCPUs: 1, 2, 4, or 8
    memory: str = "2Gi",   # Up to 32Gi
    timeout: int = 600,    # Job timeout in seconds
    env: dict = {},        # Environment variables
)
```

**Note on `image`**: Must be a full image path that Cloud Run can pull. The image must have `gcloud` CLI installed for GCS file transfer.

### CloudRunClient.run()

```python
results = client.run(
    tasks: list[dict],      # List of task dicts (see below)
    max_workers: int = None,  # Max parallel jobs (default: unlimited)
    progress: bool = True,    # Show progress bar (requires tqdm)
)
# Returns: dict mapping task id -> list of CloudRunResult
```

**Task dict fields:**
- `id` (required): Unique identifier for this task
- `command` (required): Shell command to execute
- `inputs` (optional): Dict mapping names to local paths
- `n` (optional): Number of times to run this task (default: 1)
- `timeout` (optional): Override the default timeout for this task

### CloudRunClient.run_single()

```python
result = client.run_single(
    command: str,                   # Shell command to execute
    inputs: dict[str, Path] = None, # Files at /workspace/input/
    timeout: int = None,            # Job timeout (default: from config)
)
# Returns: CloudRunResult
```

### CloudRunResult

```python
result.stdout            # str: Command stdout
result.stderr            # str: Command stderr
result.output_dir        # Path: Local dir with /workspace/output contents
result.returncode        # int: Exit code
result.duration_seconds  # float: Total execution time
result.success           # bool: True if no error and returncode == 0
result.error             # Exception | None: Error if job failed
```

## Workspace Layout

Inside the container:

```
/workspace/              <- Working directory
├── input/
│   ├── repo/            <- from inputs={"repo": Path(...)}
│   └── data.json        <- from inputs={"data.json": Path(...)}
└── output/              <- Write results here (retrieved after job completes)
```

## Caching

Two levels of caching minimize redundant work:

1. **In-memory cache**: Same input paths within a process skip re-tarring
2. **GCS cache**: Same content hash across processes/machines skips re-uploading

Content hashing uses MD5 of the tarball. Tarballs are deterministic (sorted entries, zeroed timestamps) so identical files = identical hash = cache hit.

## GCS Cleanup

Outputs are deleted after download. For inputs, set up a lifecycle policy:

```bash
cat > lifecycle.json << 'EOF'
{
  "rule": [
    {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["cloudrun-inputs/"]}}
  ]
}
EOF

gsutil lifecycle set lifecycle.json gs://your-bucket
```

## Cost

Cloud Run Jobs pricing (as of 2024):
- ~$0.00002400/vCPU-second
- ~$0.00000250/GiB-second

A typical 2-minute job with 1 vCPU and 2GB RAM costs ~$0.003.

## Troubleshooting

**"API key required"**: Set `ANTHROPIC_API_KEY` env var or pass `api_key=` to `run()`.

**Job timeout**: Increase `timeout` in config. Default is 600s for ClaudeCodeClient, 3600s for CloudRunClient.

**"quota exceeded"**: Run `gcloud auth application-default set-quota-project PROJECT_ID`.

**View logs**: 
```bash
gcloud logging read 'resource.type="cloud_run_job"' --project=PROJECT --limit=50
```
