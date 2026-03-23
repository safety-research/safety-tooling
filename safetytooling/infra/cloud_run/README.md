# Cloud Run Module

Run Claude Code instances in parallel on ephemeral GCP Cloud Run containers. Send files in, get transcripts and artifacts back.

## Quick Start

```python
from safetytooling.infra.cloud_run import ClaudeCodeClient, ClaudeCodeTask

client = ClaudeCodeClient(
    project_id="my-project",
    gcs_bucket="my-bucket",
    api_key_secret="anthropic-api-key-USERNAME",
    service_account="claude-runner@my-project.iam.gserviceaccount.com",
)

# Run a single task
result = client.run_single(
    task="Review this code for security issues",
    inputs={"repo": "./my_repo"},
)
print(result.response)       # Claude's stdout
print(result.transcript)     # Full conversation transcript (list of dicts)
print(result.output_dir)     # Path to downloaded /workspace/output/ contents
print(result.success)        # True if no error and exit code 0

# Run many tasks in parallel
tasks = [
    ClaudeCodeTask(id="review-xss", task="Review for XSS", inputs=(("repo", "./my_repo"),), n=10),
    ClaudeCodeTask(id="review-sqli", task="Review for SQL injection", inputs=(("repo", "./my_repo"),), n=5),
]
results = client.run(tasks)  # Uploads repo once, runs 15 jobs

for task in tasks:
    for r in results[task]:
        print(f"{task.id}: {r.response[:80]}")
```

### Streaming Results

```python
for task, run_idx, result in client.run_stream(tasks):
    print(f"{task.id}[{run_idx}]: {result.response[:80]}...")
    save_to_db(task, run_idx, result)
```

## Prerequisites

1. **GCP project** with Cloud Run, Cloud Storage, and Secret Manager APIs enabled
2. **GCS bucket** for file transfer between local machine and containers
3. **Authentication**: `gcloud auth application-default login`
4. **Anthropic API key** stored in GCP Secret Manager (see setup below)
5. **Restricted service account** (see setup below)

### One-Time GCP Setup

```bash
PROJECT=your-project
BUCKET=your-bucket

# Store your API key in Secret Manager
echo -n "sk-ant-api03-YOUR_KEY" | gcloud secrets create anthropic-api-key-USERNAME \
    --data-file=- --project=$PROJECT

# Create a restricted service account (limits what the container can access)
gcloud iam service-accounts create claude-runner \
    --display-name="Claude Code Runner (restricted)" --project=$PROJECT
SA_EMAIL="claude-runner@${PROJECT}.iam.gserviceaccount.com"

# Grant access to your bucket only
gsutil iam ch "serviceAccount:${SA_EMAIL}:objectUser" "gs://${BUCKET}"

# Grant access to your API key secret
gcloud secrets add-iam-policy-binding anthropic-api-key-USERNAME \
    --project=$PROJECT --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

# Grant log writing
gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:${SA_EMAIL}" --role="roles/logging.logWriter"
```

To rotate the API key later:
```bash
echo -n "NEW_KEY" | gcloud secrets versions add anthropic-api-key-USERNAME \
    --data-file=- --project=$PROJECT
```

## API

### ClaudeCodeTask

```python
ClaudeCodeTask(
    id="my-task",                        # Unique identifier
    task="Analyze this repo",            # Prompt for Claude Code
    inputs=(("repo", "./my_repo"),),     # Files to upload (tuple of pairs for hashability)
    system_prompt="You are a reviewer",  # Custom system prompt
    pre_claude_command="git config ...", # Shell command before Claude
    post_claude_command="cat report.md", # Shell command after Claude
    root_setup_command="echo ...",       # Shell command as root before dropping to claude user
    n=5,                                 # Run this task 5 times
    output_instructions=True,            # Tell Claude about /workspace/input and /workspace/output
)
```

### ClaudeCodeClient

Constructor accepts a `ClaudeCodeClientConfig` or individual kwargs (passed through to config):

```python
client = ClaudeCodeClient(
    project_id="my-project",              # Or set GCLOUD_PROJECT_ID env var
    gcs_bucket="my-bucket",               # Or set GCLOUD_GCS_BUCKET env var
    api_key_secret="anthropic-api-key-USERNAME",  # Secret Manager secret name (required at run time)
    service_account="claude-runner@...",   # Restricted service account (required at run time)
    model="claude-opus-4-5-20251101",     # Default model
    max_turns=100,                         # Max conversation turns
    timeout=300,                           # Job timeout in seconds
    cpu="1",                               # vCPUs: 1, 2, 4, or 8
    memory="2Gi",                          # Up to 32Gi
    skip_permissions=True,                 # --dangerously-skip-permissions (default: False)
)

# Three ways to run:
result = client.run_single(task="...", inputs={...})           # Single task, returns ClaudeCodeResult
results = client.run(tasks)                                     # Batch, returns dict[Task, list[Result]]
for task, idx, result in client.run_stream(tasks):              # Streaming, yields as completed
    ...
```

Both `run()` and `run_stream()` accept `max_workers` to control parallelism (default: unlimited).

### ClaudeCodeResult

```python
result.response          # str: Claude's stdout
result.transcript        # list[dict] | None: Full conversation transcript (see below)
result.output_dir        # Path: Local dir with /workspace/output contents
result.returncode        # int: Exit code
result.duration_seconds  # float: Wall-clock time
result.success           # bool: No error and returncode == 0
result.error             # Exception | None
```

**Transcripts**: The `transcript` field is a list of JSONL entries from Claude Code's `.claude/` session directory. Each entry has a `message` dict with `role` and `content`. When Claude Code spawns subagents (via the Task/Agent tool), the subagent's full transcript is inlined into the parent transcript at the corresponding `tool_result` block as a `subagent_transcript` field. This gives you the complete execution trace in a single structure.

`project_id` and `gcs_bucket` can also be set via `GCLOUD_PROJECT_ID` and `GCLOUD_GCS_BUCKET` env vars instead of passing them explicitly.

## How It Works

```
Local files ──tar+hash──▶ GCS bucket ──download──▶ Cloud Run container
                          (cached)                  /workspace/input/
                                                    Claude Code runs here
                          GCS bucket ◀──upload────── /workspace/output/
Download ◀────────────── (auto-deleted)
```

1. Local inputs are tarred with deterministic timestamps and content-hashed
2. Identical content is uploaded once (in-memory + GCS deduplication)
3. Concurrent requests for the same inputs block until the first upload completes
4. Cloud Run job downloads inputs, runs Claude Code, uploads outputs
5. Outputs are downloaded locally and the GCS copy is deleted

Running `n=100` with the same inputs tars and uploads only once.

### Workspace Layout (Inside Container)

```
/workspace/
├── input/
│   ├── repo/         ← from inputs=(("repo", Path(...)),)
│   └── data.json     ← from inputs=(("data.json", Path(...)),)
└── output/           ← Write results here (retrieved after job completes)
```

By default (`output_instructions=True`), Claude is told where to find inputs and write outputs.

## Low-Level: CloudRunClient

For running arbitrary commands (not Claude Code) in containers:

```python
from safetytooling.infra.cloud_run import CloudRunClient, CloudRunClientConfig, CloudRunTask

client = CloudRunClient(CloudRunClientConfig(
    project_id="my-project",
    gcs_bucket="my-bucket",
    image="gcr.io/my-project/my-image:latest",
))

result = client.run_single(
    command="cd /workspace/input/repo && python script.py",
    inputs={"repo": Path("./my_repo")},
)
print(result.stdout, result.returncode)
```

Same `run()`, `run_stream()`, and `run_single()` interface as `ClaudeCodeClient`.

## Egress Firewall (Optional)

To restrict outbound network access from containers (e.g., only allow `api.anthropic.com` and Google APIs), use VPC Direct Egress with Cloud NGFW firewall rules:

```python
client = ClaudeCodeClient(
    ...,
    vpc_network="my-egress-vpc",
    vpc_subnet="my-egress-subnet",
    vpc_egress="all-traffic",
)
```

This routes all container traffic through a VPC where FQDN-based firewall rules control which domains are reachable. See `tests/test_vpc_egress.py` for an integration test that verifies the setup.

## Testing

```bash
# Unit tests (no GCP required) - caching, thread-safety, deterministic tarring
pytest tests/test_cloud_run_caching.py -v

# Integration tests (requires GCP)
export GCP_PROJECT_ID=my-project GCS_BUCKET=my-bucket
export API_KEY_SECRET=anthropic-api-key-USERNAME
export SERVICE_ACCOUNT=claude-runner@my-project.iam.gserviceaccount.com
pytest tests/test_cloud_run_integration.py -v --run-integration
```

## Cost

Cloud Run Jobs: ~$0.003 per 2-minute job (1 vCPU, 2GB RAM).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `api_key_secret is required` | Set `api_key_secret` pointing to a Secret Manager secret |
| `Secret not found` | `gcloud secrets versions list SECRET --project=PROJECT` |
| `Permission denied` on secret | Grant the service account `secretmanager.secretAccessor` |
| Job timeout | Increase `timeout` in config |
| Quota exceeded | `gcloud auth application-default set-quota-project PROJECT` |
| View logs | `gcloud logging read 'resource.type="cloud_run_job"' --project=PROJECT --limit=50` |
