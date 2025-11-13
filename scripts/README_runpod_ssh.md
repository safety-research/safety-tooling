# RunPod SSH Config Generator

A Python script to automatically generate SSH config entries for your RunPod pods.

## Prerequisites

Install required packages:

```bash
pip install requests python-dotenv
```

## Setup

1. Create a `.env` file in your project root (or ensure RUNPOD_API_KEY is set in your environment):

```bash
RUNPOD_API_KEY=your_api_key_here
```

2. Get your RunPod API key from: https://www.runpod.io/console/user/settings

**Note:** The API key only needs **read access** to fetch pod information. No write permissions are required.

## Quick Start (Recommended)

**Expected default usage for Anthropic cluster:**

```bash
python safety-tooling/scripts/runpod_ssh_config.py --cluster-name "Anthropic" --user [YOUR_USERNAME] --identity-file ~/.ssh/id_ed25519
```

Replace `[YOUR_USERNAME]` with your actual username on the pods.

This will generate SSH configs for all running pods in the "Anthropic" cluster using your ED25519 SSH key.

## Usage

### Basic usage (all running pods)

```bash
cd safety-tooling/scripts
python runpod_ssh_config.py
```

This will:
- Fetch all your pods from RunPod (including Instant Cluster pods)
- Filter for running/active pods
- Generate SSH config entries
- Write to `~/.ssh/config.d/runpod`
- Backup any existing config file

### Filter by cluster name (for Instant Clusters)

```bash
python runpod_ssh_config.py --cluster-name "My Cluster"
```

### Custom output path

```bash
python runpod_ssh_config.py --output-path ~/custom-path/ssh-config
```

### Custom SSH user

```bash
python runpod_ssh_config.py --user myuser
```

### Combine cluster and identity file

```bash
python runpod_ssh_config.py --cluster-name "My Cluster" --identity-file ~/.ssh/id_rsa
```

### Specify SSH identity file (private key)

```bash
python runpod_ssh_config.py --identity-file ~/.ssh/id_rsa
```

### Don't backup existing config

```bash
python runpod_ssh_config.py --no-backup
```

## SSH Config Output

The script generates entries like:

```
Host runpod-my-pod-name
    HostName 123.45.67.89
    Port 22334
    User root
    IdentityFile ~/.ssh/id_rsa  # (if --identity-file specified)
    # Pod ID: abc123xyz
    # Pod Name: My Pod Name
```

**Note:** Hosts are automatically sorted with "controller" nodes first, then alphabetically by name.

## Using the config

After running the script, you can SSH to your pods using:

```bash
ssh runpod-my-pod-name
```

## Including in main SSH config

The script will automatically check if your `~/.ssh/config` includes the generated config file. If not, it will display a warning message.

To use the generated config, add this to the **top** of your `~/.ssh/config`:

```
Include config.d/runpod
```

Or create the main config if it doesn't exist:

```bash
echo 'Include config.d/runpod' > ~/.ssh/config
```

Alternatively, you can manually specify the config file in your SSH commands:

```bash
ssh -F ~/.ssh/config.d/runpod runpod-my-pod-name
```

## Troubleshooting

### "RUNPOD_API_KEY environment variable not set"

Make sure you have a `.env` file with your API key or set it in your environment:

```bash
export RUNPOD_API_KEY=your_api_key_here
```

**Note:** Only read access is required for the API key.

### "No pods found"

- Check that you have pods in your RunPod account
- Verify your API key is correct
- Check your internet connection

### "No running pods to configure"

The script only generates config for pods with status "RUNNING". Start your pods first.

## Features

- ✅ Fetches pods via RunPod GraphQL API (`https://api.runpod.io/graphql`)
- ✅ **Supports all pod types** including Instant Clusters
- ✅ Filters for running pods with runtime information
- ✅ Optional filtering by cluster name (for Instant Clusters)
- ✅ Generates clean SSH config entries
- ✅ Automatic directory creation
- ✅ Backs up existing config files
- ✅ Customizable output path and SSH user
- ✅ Environment variable support via .env files

## API Reference

This script uses the RunPod GraphQL API to fetch pod information with full runtime details.

