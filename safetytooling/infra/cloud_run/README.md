# Cloud Run Module

Run Claude Code in ephemeral GCP Cloud Run containers with GCS-based file I/O.

## Quick Start

```python
from safetytooling.infra.cloud_run import ClaudeCodeJob, ClaudeCodeJobConfig

config = ClaudeCodeJobConfig(
    project_id="my-gcp-project",
    gcs_bucket="my-bucket",
)

result = ClaudeCodeJob(config).run(
    task="List the files in input/repo and describe what this project does",
    inputs={"repo": Path("./my_repo")},
)

print(result.response)
print(result.transcript)  # Full conversation history
```

## Prerequisites

1. **GCP Project** with Cloud Run and Cloud Storage APIs enabled
2. **GCS Bucket** for file transfer
3. **Authentication**: `gcloud auth application-default login` or service account
4. **Anthropic API Key**: Set `ANTHROPIC_API_KEY` env var or pass to `run()`

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐
│ Local files │────▶│ GCS bucket  │────▶│ Cloud Run container         │
│             │     │             │     │  ┌─────────────────────┐    │
│ inputs={    │     │ inputs/     │     │  │ Claude Code         │    │
│   "repo": / │     │ {hash}.tgz  │     │  │ - reads /workspace/ │    │
│ }           │     │             │     │  │ - writes /output/   │    │
└─────────────┘     │ outputs/    │◀────│  └─────────────────────┘    │
       ▲            │ {job}.tgz   │     └─────────────────────────────┘
       │            └─────────────┘
       │                  │
       └──────────────────┘
         Download outputs
```

1. **Tar and hash** local inputs
2. **Upload to GCS** (skipped if hash already exists - content-addressed caching)
3. **Create Cloud Run Job** that:
   - Downloads inputs from GCS
   - Installs Claude Code
   - Runs your task
   - Uploads outputs to GCS
4. **Download outputs** and return result

## API Reference

### ClaudeCodeJobConfig

```python
ClaudeCodeJobConfig(
    project_id: str,       # GCP project ID (required)
    gcs_bucket: str,       # GCS bucket for file I/O (required)
    region: str = "us-central1",
    model: str = "claude-opus-4-5-20251101",
    max_turns: int = 100,  # Max conversation turns
    timeout: int = 600,    # Job timeout in seconds
    cpu: str = "2",        # vCPUs: 1, 2, 4, or 8
    memory: str = "4Gi",   # Up to 32Gi
    skip_permissions: bool = True,  # --dangerously-skip-permissions
)
```

### ClaudeCodeJob.run()

```python
result = ClaudeCodeJob(config).run(
    task: str,                        # The prompt for Claude Code
    inputs: dict[str, Path] = None,   # Files to make available
    system_prompt: str = None,        # Custom system prompt / constitution
    claude_config_dir: Path = None,   # Custom .claude/ directory
    api_key: str = None,              # Default: ANTHROPIC_API_KEY env var
)
```

### ClaudeCodeJobResult

```python
result.response          # str: Claude's stdout
result.transcript        # list[dict]: Parsed conversation from .claude/
result.output_dir        # Path: Local dir with /workspace/output contents
result.returncode        # int: Exit code
result.duration_seconds  # float: Total execution time
```

## Workspace Layout

Inside the container, Claude Code sees:

```
/workspace/              <- Working directory
├── input/
│   ├── repo/            <- from inputs={"repo": Path(...)}
│   └── data.json        <- from inputs={"data.json": Path(...)}
└── output/              <- Claude writes results here
```

Reference inputs in your task as `input/repo`, `input/data.json`, etc.

## Parallel Execution

For running many jobs, share GCP clients to avoid connection overhead:

```python
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from google.cloud.run_v2 import JobsClient

# Create shared clients once
jobs_client = JobsClient()
storage_client = storage.Client()

config = ClaudeCodeJobConfig(project_id="...", gcs_bucket="...")

def run_task(task, inputs):
    return ClaudeCodeJob(
        config,
        jobs_client=jobs_client,
        storage_client=storage_client,
    ).run(task=task, inputs=inputs)

# Run in parallel
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(run_task, t, i) for t, i in work_items]
    results = [f.result() for f in futures]
```

## Caching

Two levels of caching minimize redundant work:

1. **In-memory cache**: Same input paths within a process skip re-tarring
2. **GCS cache**: Same content hash across processes/machines skips re-uploading

Content hashing uses MD5 of the tarball, so identical files = identical hash = cache hit.

## GCS Cleanup

Outputs are deleted after download. For inputs, set up a lifecycle policy:

```bash
# lifecycle.json
{
  "rule": [
    {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["claude-code-inputs/"]}}
  ]
}

gsutil lifecycle set lifecycle.json gs://your-bucket
```

## Low-Level: CloudRunJob

For custom containers (not Claude Code), use `CloudRunJob` directly:

```python
from safetytooling.infra.cloud_run import CloudRunJob, CloudRunJobConfig

config = CloudRunJobConfig(
    image="python:3.11-slim",
    project_id="my-project",
    gcs_bucket="my-bucket",
)

with CloudRunJob(config) as job:
    job.send_files({"local/data": "input/data"})
    result = job.run("python /workspace/input/data/script.py")
    job.receive_files({"output/results.json": "local/results.json"})

print(result.stdout)
```

## Cost

Cloud Run Jobs pricing (as of 2024):
- ~$0.00002400/vCPU-second
- ~$0.00000250/GiB-second

A typical 2-minute Claude Code review with 2 vCPUs and 4GB RAM costs ~$0.006.

## Troubleshooting

**"API key required"**: Set `ANTHROPIC_API_KEY` env var or pass `api_key=` to `run()`.

**Job timeout**: Increase `timeout` in config. Default is 600s (10 min).

**"quota exceeded"**: Run `gcloud auth application-default set-quota-project PROJECT_ID`.

**View logs**: 
```bash
gcloud logging read 'resource.type="cloud_run_job"' --project=PROJECT --limit=50
```
