"""Cloud Run job execution for Claude Code and other tasks."""

from .claude_code_job import (
    # New names
    ClaudeCodeClient,
    ClaudeCodeClientConfig,
    ClaudeCodeClientError,
    # Backwards compatibility
    ClaudeCodeJob,
    ClaudeCodeJobConfig,
    ClaudeCodeJobError,
    ClaudeCodeJobResult,
    ClaudeCodeResult,
)
from .cloud_run_job import (
    CloudRunJob,
    CloudRunJobConfig,
    CloudRunJobError,
    JobResult,
)

__all__ = [
    # New names (preferred)
    "ClaudeCodeClient",
    "ClaudeCodeClientConfig",
    "ClaudeCodeClientError",
    "ClaudeCodeResult",
    # Backwards compatibility
    "ClaudeCodeJob",
    "ClaudeCodeJobConfig",
    "ClaudeCodeJobError",
    "ClaudeCodeJobResult",
    # Low-level
    "CloudRunJob",
    "CloudRunJobConfig",
    "CloudRunJobError",
    "JobResult",
]
