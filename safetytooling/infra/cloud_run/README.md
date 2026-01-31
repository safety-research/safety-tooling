# Cloud Run Module

Run many Claude Code instances in parallel on ephemeral GCP Cloud Run containers. Send files in, get artifacts back.

**Primary use case**: Batch processing with Claude Code - review many PRs, analyze many codebases, run many investigations in parallel without managing infrastructure.

## Quick Start

```python
from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeTask

client = ClaudeCodeClient(
    project_id="my-project",
    gcs_bucket="my-bucket",
    api_key_secret="anthropic-api-key-USERNAME",  # Your Secret Manager secret name
)

# Single task
result = client.run_single(
    task="Review this code for security issues",
    inputs={"repo": "./my_repo"},
)
print(result.response)

# Multiple tasks in parallel
tasks = [
    ClaudeCodeTask(
        id="haiku",
        task="Write a haiku about this code",
        inputs=(("repo", "./my_repo"),),
        n=5,
    ),
    ClaudeCodeTask(
        id="limerick", 
        task="Write a limerick about this code",
        inputs=(("repo", "./my_repo"),),
        n=5,
    ),
]
results = client.run(tasks)  # Runs 10 jobs, uploads repo only once

for task in tasks:
    print(f"\n{task.id}:")
    for r in results[task]:
        print(r.response)

# Stream results as they complete
for task, run_idx, result in client.run_stream(tasks):
    print(f"{task.id}[{run_idx}]: {result.response[:50]}...")
```

## Custom Containers (low-level)

```python
from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig, CloudRunTask

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
tasks = [
    CloudRunTask(id="task-1", command="python script1.py", inputs=(("repo", "./repo"),), n=10),
    CloudRunTask(id="task-2", command="python script2.py", inputs=(("repo", "./repo"),), n=5),
]
results = client.run(tasks)
```

## Prerequisites

1. **GCP Project** with Cloud Run, Cloud Storage, and Secret Manager APIs enabled
2. **GCS Bucket** for file transfer
3. **Authentication**: `gcloud auth application-default login` or service account
4. **Anthropic API Key** (for ClaudeCodeClient): Stored in GCP Secret Manager (see below)

## API Key Setup (Required for ClaudeCodeClient)

The Anthropic API key is stored securely in GCP Secret Manager and injected at runtime. This avoids exposing the key in scripts, logs, or job specs.

**One-time setup:**

```bash
# 1. Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com --project=YOUR_PROJECT

# 2. Create your secret (use your username for per-user secrets)
echo -n "sk-ant-api03-YOUR_KEY" | gcloud secrets create anthropic-api-key-USERNAME \
    --data-file=- \
    --project=YOUR_PROJECT

# 3. Grant Cloud Run access to read it
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding anthropic-api-key-USERNAME \
    --project=YOUR_PROJECT \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**To rotate the key later:**

```bash
echo -n "NEW_KEY" | gcloud secrets versions add anthropic-api-key-USERNAME \
    --data-file=- \
    --project=YOUR_PROJECT
```

The client uses `version="latest"` so new keys are picked up automatically on the next job.

## Security Hardening (Recommended)

By default, Cloud Run jobs use the project's default compute service account, which often has broad permissions (e.g., `roles/editor`). This means a misbehaving Claude could:

- Access **all** GCS buckets in the project
- Read/write other GCP services
- Make outbound network requests anywhere

**Recommendation**: Create a dedicated service account with minimal permissions.

**One-time setup:**

```bash
PROJECT=your-project
BUCKET=your-bucket
SECRET=anthropic-api-key-USERNAME

# 1. Create restricted service account
gcloud iam service-accounts create claude-runner \
    --display-name="Claude Code Runner (restricted)" \
    --project=$PROJECT

SA_EMAIL="claude-runner@${PROJECT}.iam.gserviceaccount.com"

# 2. Grant ONLY access to your specific bucket (not all buckets)
# objectUser = create, get, delete, list objects (no admin capabilities)
gsutil iam ch "serviceAccount:${SA_EMAIL}:objectUser" "gs://${BUCKET}"

# 3. Grant access to read your API key secret
gcloud secrets add-iam-policy-binding $SECRET \
    --project=$PROJECT \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# 4. Grant permission to write Cloud Run job logs
gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/logging.logWriter"
```

**Use it in your code:**

```python
client = ClaudeCodeClient(
    project_id="my-project",
    gcs_bucket="my-bucket",
    api_key_secret="anthropic-api-key-USERNAME",
    service_account="claude-runner@my-project.iam.gserviceaccount.com",  # Restricted!
)
```

**What this limits:**
- Container can only access `my-bucket`, not other buckets in the project
- Container can only read the specific API key secret
- Container cannot access BigQuery, Pub/Sub, other Cloud Run jobs, etc.
- **Without this, Claude could take over your entire GCP project** - don't skip this step!

**What this doesn't limit:**
- Outbound network access (Claude could exfiltrate data to external URLs)
- Anthropic API usage (Claude could use your API key for other purposes)

For the "yolo Claude" use case, the main risks are data exfiltration and API key abuse.
Containers are ephemeral (destroyed after job), so there's no persistence risk.

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐
│ Local files │────▶│ GCS bucket  │────▶│ Cloud Run container         │
│             │     │             │     │                             │
│ inputs={    │     │ inputs/     │     │  /workspace/input/repo/     │
│   "repo": / │     │ {hash}.tgz  │     │  /workspace/output/         │
│ }           │  ▲  │  (cached)   │     │                             │
└─────────────┘  │  │ outputs/    │◀────│  (Claude Code runs here)    │
       ▲         │  │ {job}.tgz   │     └─────────────────────────────┘
       │         │  └─────────────┘
       │         │
       │         └── Thread lock: concurrent requests for same
       │             inputs block until first upload completes,
       │             then use cached hash
       └──────────────────┘
         Download outputs
```

1. **Tar and hash** local inputs (deterministic tarball)
2. **Check in-memory cache** - same path in this process? Use cached GCS path
3. **Check GCS cache** - same content hash? Skip upload
4. **Thread blocking** - concurrent requests for same inputs wait for first to finish, then use its cached result
5. **Create Cloud Run Job** that downloads inputs, runs command, uploads outputs
6. **Download outputs** and return result

This means running `n=100` with the same inputs only tars and uploads once.

## API Reference

### ClaudeCodeTask

```python
ClaudeCodeTask(
    id: str,                    # Unique identifier (required)
    task: str,                  # Prompt/instruction for Claude Code (required)
    inputs: tuple | None,       # Tuple of (name, path) pairs, e.g. (("repo", "./repo"),)
    system_prompt: str | None,  # Custom system prompt / constitution
    pre_claude_command: str | None,   # Shell command before Claude
    post_claude_command: str | None,  # Shell command after Claude
    n: int = 1,                 # Number of times to run this task
)
```

Note: `inputs` uses tuples (not dicts) so the task is hashable and can be used as a dict key.

### ClaudeCodeClientConfig

```python
ClaudeCodeClientConfig(
    project_id: str,       # GCP project ID (required)
    gcs_bucket: str,       # GCS bucket for file I/O (required)
    api_key_secret: str,   # Secret Manager secret name (required)
    service_account: str,  # Restricted service account (required, see Security Hardening)
    region: str = "us-central1",
    model: str = "claude-opus-4-5-20251101",
    max_turns: int = 100,  # Max conversation turns
    timeout: int = 300,    # Job timeout in seconds (reduced - pre-built image has no setup)
    cpu: str = "1",        # vCPUs: 1, 2, 4, or 8
    memory: str = "2Gi",   # Up to 32Gi
    skip_permissions: bool = True,  # --dangerously-skip-permissions
    image: str = DEFAULT_CLAUDE_CODE_IMAGE,  # Pre-built image with Claude Code
)
```

### ClaudeCodeClient.run()

```python
results = client.run(
    tasks: list[ClaudeCodeTask],  # List of tasks
    max_workers: int = None,      # Max parallel jobs (default: unlimited)
    progress: bool = True,        # Show progress bar (requires tqdm)
)
# Returns: dict[ClaudeCodeTask, list[ClaudeCodeResult]]
```

Note: API key is injected via Secret Manager (configured in `api_key_secret`).

### ClaudeCodeClient.run_stream()

```python
for task, run_idx, result in client.run_stream(tasks):
    # Process each result as it completes
    print(f"{task.id}[{run_idx}]: {result.success}")
# Yields: tuple[ClaudeCodeTask, int, ClaudeCodeResult]
```

### ClaudeCodeClient.run_single()

```python
result = client.run_single(
    task: str,                        # The prompt for Claude Code
    inputs: dict[str, Path] = None,   # Files at /workspace/input/
    system_prompt: str = None,        # Custom system prompt / constitution
    pre_claude_command: str = None,   # Shell command before Claude
    post_claude_command: str = None,  # Shell command after Claude
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

### CloudRunTask

```python
CloudRunTask(
    id: str,                # Unique identifier (required)
    command: str,           # Shell command to execute (required)
    inputs: tuple | None,   # Tuple of (name, path) pairs
    n: int = 1,             # Number of times to run this task
    timeout: int | None,    # Override default timeout
)
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
    secrets: dict = {},    # Secret Manager secrets as env vars
    service_account: str = None,  # Restricted service account (see Security Hardening)
)
```

### CloudRunClient.run()

```python
results = client.run(
    tasks: list[CloudRunTask],  # List of tasks
    max_workers: int = None,    # Max parallel jobs (default: unlimited)
    progress: bool = True,      # Show progress bar (requires tqdm)
)
# Returns: dict[CloudRunTask, list[CloudRunResult]]
```

### CloudRunClient.run_stream()

```python
for task, run_idx, result in client.run_stream(tasks):
    # Process each result as it completes
    print(f"{task.id}[{run_idx}]: {result.success}")
# Yields: tuple[CloudRunTask, int, CloudRunResult]
```

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
│   ├── repo/            <- from inputs=(("repo", Path(...)),)
│   └── data.json        <- from inputs=(("data.json", Path(...)),)
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

## Testing

The module has two test suites:

### Unit tests (no GCP required)

Test caching, thread-safety, and deterministic tarring locally:

```bash
pytest tests/test_cloud_run_caching.py -v
```

These tests verify:
- **Deterministic tarring**: Same content produces identical hashes
- **In-memory caching**: Prevents redundant tarring within a session
- **GCS caching**: Prevents redundant uploads (mocked)
- **Thread-safety**: Concurrent uploads of same inputs only tar/upload once
- **Lock correctness**: Verifies locks actually block concurrent access

### Integration tests (requires GCP)

Full end-to-end tests against real Cloud Run:

```bash
export GCP_PROJECT_ID="your-project"
export GCS_BUCKET="your-bucket"
export API_KEY_SECRET="anthropic-api-key-USERNAME"  # Secret Manager secret name
export SERVICE_ACCOUNT="claude-runner@your-project.iam.gserviceaccount.com"  # Restricted SA

pytest tests/test_cloud_run_integration.py -v --run-integration
```

**Recommended workflow**: Run unit tests during development, integration tests before merging.

## Troubleshooting

**"api_key_secret is required"**: Set `api_key_secret` in your config pointing to a Secret Manager secret.

**"Secret not found"**: Ensure the secret exists and has at least one version:
```bash
gcloud secrets versions list YOUR_SECRET_NAME --project=YOUR_PROJECT
```

**"Permission denied" on secret**: Grant the Cloud Run service account access:
```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding YOUR_SECRET_NAME \
    --project=YOUR_PROJECT \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

**"Invalid API key"**: The secret value may be incorrect. Add a new version:
```bash
echo -n "sk-ant-api03-CORRECT_KEY" | gcloud secrets versions add YOUR_SECRET_NAME \
    --data-file=- --project=YOUR_PROJECT
```

**Job timeout**: Increase `timeout` in config. Default is 600s.

**"quota exceeded"**: Run `gcloud auth application-default set-quota-project PROJECT_ID`.

**View logs**: 
```bash
gcloud logging read 'resource.type="cloud_run_job"' --project=PROJECT --limit=50
```
