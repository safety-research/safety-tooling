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

# Single task
result = client.run(
    task="Review this PR for security issues",
    inputs={"repo": Path("./my_repo")},
)
print(result.response)

# Same task 100 times (measure variance)
results = client.run(
    task="Review this code for vulnerabilities",
    inputs={"repo": Path("./my_repo")},
    n=100,
)
flagged = sum(1 for r in results if r.success and "vulnerability" in r.response.lower())
print(f"Flagged in {flagged}/{len(results)} runs")

# Multiple different tasks
results = client.run(tasks=[
    {"task": "Review for XSS", "inputs": {"repo": repo}},
    {"task": "Review for SQLi", "inputs": {"repo": repo}},
])
```

## Custom Containers (low-level)

```python
from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

config = CloudRunJobConfig(
    image="gcr.io/my-project/my-image:latest",  # Full image path
    project_id="my-project",
    gcs_bucket="my-bucket",
)

with CloudRunJob(config) as job:
    job.send_inputs({"repo": Path("./my_repo"), "data.json": Path("./config.json")})
    result = job.run("cd /workspace/input/repo && python script.py")
    output_dir = job.receive_outputs()

print(f"Exit code: {result.returncode}")
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"Outputs at: {output_dir}")
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
result = client.run(
    task: str,                        # The prompt for Claude Code
    inputs: dict[str, Path] = None,   # Files at /workspace/input/
    system_prompt: str = None,        # Custom system prompt / constitution
    claude_config_dir: Path = None,   # Custom .claude/ directory
    api_key: str = None,              # Default: ANTHROPIC_API_KEY env var
    n: int = None,                    # Run same task N times (returns list)
    tasks: list[dict] = None,         # Run different tasks (returns list)
    max_workers: int = None,          # Max parallel jobs (default: unlimited)
    progress: bool = True,            # Show progress bar (requires tqdm)
)
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

### CloudRunJobConfig

```python
CloudRunJobConfig(
    image: str,            # Full container image path (e.g., "gcr.io/project/image:tag")
    project_id: str,       # GCP project ID (required)
    gcs_bucket: str,       # GCS bucket for file I/O (required)
    region: str = "us-central1",
    cpu: str = "1",        # vCPUs: 1, 2, 4, or 8
    memory: str = "2Gi",   # Up to 32Gi
    timeout: int = 3600,   # Job timeout in seconds
    env: dict = {},        # Environment variables
    name_prefix: str = "ephemeral",
)
```

**Note on `image`**: Must be a full image path that Cloud Run can pull. The image must have `gcloud` CLI installed for GCS file transfer.

### CloudRunJob

```python
with CloudRunJob(config) as job:
    # Send inputs to /workspace/input/
    job.send_inputs({"name": Path("local/path")})
    
    # Run command
    result = job.run("your-command", timeout=300)
    
    # Get outputs from /workspace/output/ (idempotent, can call multiple times)
    output_dir = job.receive_outputs()

# result.stdout and result.stderr contain command output
```

## Workspace Layout

Inside the container:

```
/workspace/              <- Working directory
├── input/
│   ├── repo/            <- from inputs={"repo": Path(...)}
│   └── data.json        <- from inputs={"data.json": Path(...)}
└── output/              <- Write results here (retrieved by receive_outputs)
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

**Job timeout**: Increase `timeout` in config. Default is 600s for ClaudeCodeClient, 3600s for CloudRunJob.

**"quota exceeded"**: Run `gcloud auth application-default set-quota-project PROJECT_ID`.

**View logs**: 
```bash
gcloud logging read 'resource.type="cloud_run_job"' --project=PROJECT --limit=50
```
