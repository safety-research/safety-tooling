# Minimal Claude Code runner image for Cloud Run Jobs
# Based on google-cloud-cli:slim which already has gcloud CLI
#
# Build:
#   docker build -t gcr.io/YOUR_PROJECT/claude-code-runner:latest .
#   docker push gcr.io/YOUR_PROJECT/claude-code-runner:latest
#
# This image pre-installs:
# - git, nodejs, npm (system packages)
# - @anthropic-ai/claude-code (global npm package)
# - Non-root 'claude' user (required by Claude Code)
#
# Saves ~2 minutes of setup time per Cloud Run Job execution.

FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

# Install system dependencies and Claude Code in one layer
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        git \
        nodejs \
        npm \
        ripgrep \
    && npm install -g @anthropic-ai/claude-code@latest \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.npm/_cacache

# Create non-root user (Claude Code requires this)
RUN useradd -m -s /bin/bash claude

# Create workspace directories
RUN mkdir -p /workspace/input /workspace/output && \
    chown -R claude:claude /workspace

# Set working directory
WORKDIR /workspace
