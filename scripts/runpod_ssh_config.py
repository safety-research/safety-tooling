#!/usr/bin/env python3
"""
RunPod SSH Config Generator

This script fetches all pods from your RunPod account and generates SSH config
entries for all running/active pods.

Usage:
    python runpod_ssh_config.py [--cluster-name CLUSTER_NAME] [--output-path PATH]
                                [--user USER] [--identity-file IDENTITY_FILE]
                                [--no-backup]

Environment Variables:
    RUNPOD_API_KEY: Your RunPod API key (required, read-only access sufficient)

Output:
    Writes SSH config to ~/.ssh/config.d/runpod by default
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' package is required. Install with: pip install python-dotenv")
    sys.exit(1)


class RunPodClient:
    """Client for interacting with RunPod GraphQL API."""

    GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"

    def __init__(self, api_key: str):
        """Initialize RunPod client with API key."""
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def get_all_pods(self, cluster_name: Optional[str] = None) -> List[Dict]:
        """
        Fetch all pods from RunPod using GraphQL API.

        Args:
            cluster_name: Optional cluster name to filter pods

        Returns:
            List of pod dictionaries (includes all types of pods)
        """
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    clusterId
                    desiredStatus
                    imageName
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                    }
                }
                clusters {
                    id
                    name
                }
            }
        }
        """

        try:
            # Use GraphQL endpoint
            response = requests.post(
                self.GRAPHQL_ENDPOINT, json={"query": query}, headers=self.headers, timeout=30
            )
            response.raise_for_status()

            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                print(f"GraphQL errors: {data['errors']}")
                return []

            myself_data = data.get("data", {}).get("myself", {})

            # Extract pods from GraphQL response
            pods = myself_data.get("pods", [])

            if not pods:
                print("No pods found")
                return []

            # Filter by cluster_name if specified
            if cluster_name:
                # Get clusters and build a mapping from name to ID
                clusters = myself_data.get("clusters", [])
                cluster_id = None

                for cluster in clusters:
                    if cluster.get("name") == cluster_name:
                        cluster_id = cluster.get("id")
                        break

                if cluster_id:
                    pods = [p for p in pods if p.get("clusterId") == cluster_id]
                    print(f"Filtering by cluster '{cluster_name}' (ID: {cluster_id})")
                else:
                    print(f"Warning: Cluster '{cluster_name}' not found")
                    return []

            return pods

        except requests.exceptions.RequestException as e:
            print(f"Error fetching pods from RunPod API: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                try:
                    print(f"Response body: {e.response.json()}")
                except (ValueError, AttributeError):
                    print(f"Response text: {e.response.text}")
            return []


def get_ssh_port_from_pod(pod: Dict) -> Optional[int]:
    """
    Extract SSH port from pod runtime configuration.

    Args:
        pod: Pod dictionary from GraphQL API

    Returns:
        SSH port number or None if not found
    """
    runtime = pod.get("runtime")
    if not runtime:
        return None

    ports = runtime.get("ports", [])

    # Look for SSH port (typically port 22 mapped to a public port)
    for port in ports:
        if port.get("privatePort") == 22:
            public_port = port.get("publicPort")
            if public_port:
                return int(public_port)

    return None


def get_ssh_host_from_pod(pod: Dict) -> Optional[str]:
    """
    Extract SSH hostname/IP from pod runtime configuration.

    Args:
        pod: Pod dictionary from GraphQL API

    Returns:
        IP address or None
    """
    runtime = pod.get("runtime")
    if not runtime:
        return None

    ports = runtime.get("ports", [])

    # Look for public IP
    for port in ports:
        if port.get("isIpPublic"):
            ip = port.get("ip")
            if ip:
                return ip

    # Fallback to any IP
    if ports:
        return ports[0].get("ip")

    return None


def is_pod_running(pod: Dict) -> bool:
    """
    Check if a pod is in a running state.

    Args:
        pod: Pod dictionary from GraphQL API

    Returns:
        True if pod is running and has runtime info
    """
    desired_status = pod.get("desiredStatus", "").upper()
    has_runtime = pod.get("runtime") is not None

    return desired_status == "RUNNING" and has_runtime


def generate_ssh_config_entry(
    pod: Dict, user: str = "root", identity_file: Optional[str] = None
) -> Optional[str]:
    """
    Generate SSH config entry for a pod.

    Args:
        pod: Pod dictionary from API
        user: SSH username (default: root)
        identity_file: Optional path to SSH private key

    Returns:
        SSH config entry as string, or None if pod doesn't have SSH access
    """
    pod_id = pod.get("id", "unknown")
    pod_name = pod.get("name", pod_id)

    ssh_host = get_ssh_host_from_pod(pod)
    ssh_port = get_ssh_port_from_pod(pod)

    if not ssh_host or not ssh_port:
        return None

    # Create a clean host alias
    host_alias = f"runpod-{pod_name.replace(' ', '-').lower()}"

    config_lines = [
        f"Host {host_alias}",
        f"    HostName {ssh_host}",
        f"    Port {ssh_port}",
        f"    User {user}",
    ]

    # Add IdentityFile if specified
    if identity_file:
        config_lines.append(f"    IdentityFile {identity_file}")

    config_lines.extend(
        [
            f"    # Pod ID: {pod_id}",
            f"    # Pod Name: {pod_name}",
            "",
        ]
    )

    return "\n".join(config_lines)


def write_ssh_config(config_entries: List[str], output_path: Path, backup: bool = True):
    """
    Write SSH config entries to file.

    Args:
        config_entries: List of SSH config entry strings
        output_path: Path to output file
        backup: Whether to backup existing file
    """
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file if requested
    if backup and output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_suffix(f".backup_{timestamp}")
        output_path.rename(backup_path)
        print(f"Backed up existing config to: {backup_path}")

    # Write header and entries
    header = [
        "# RunPod SSH Config",
        f"# Generated: {datetime.now().isoformat()}",
        "# This file is auto-generated by runpod_ssh_config.py",
        "",
    ]

    content = "\n".join(header) + "\n" + "\n".join(config_entries)

    output_path.write_text(content)
    print(f"SSH config written to: {output_path}")


def check_ssh_config_include(generated_config_path: Path):
    """
    Check if the main SSH config includes the generated config file.

    Args:
        generated_config_path: Path to the generated SSH config file
    """
    main_ssh_config = Path.home() / ".ssh" / "config"

    # Check if main SSH config exists
    if not main_ssh_config.exists():
        print(f"\n⚠️  Warning: {main_ssh_config} does not exist.")
        print("   To use the generated SSH config, create it with:")
        print(f"   echo 'Include {generated_config_path}' > {main_ssh_config}")
        return

    # Read main SSH config
    try:
        config_content = main_ssh_config.read_text()

        # Check if it includes the generated config
        # Support both relative and absolute paths
        generated_str = str(generated_config_path)

        # Check for exact path or expanded path
        has_include = False
        for line in config_content.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("Include"):
                parts = line_stripped.split(None, 1)
                include_path = parts[1] if len(parts) > 1 else ""
                # Expand the include path for comparison
                expanded_include = str(Path(include_path).expanduser())
                if expanded_include == generated_str or include_path == generated_str:
                    has_include = True
                    break

        if not has_include:
            print(f"\n⚠️  Warning: {main_ssh_config} does not include the generated config.")
            print(f"   Add this line to the top of your {main_ssh_config}:")
            print(f"   Include {generated_config_path}")
            print("\n   Or run:")
            print(
                f"   echo 'Include {generated_config_path}' | cat - {main_ssh_config} "
                f"> /tmp/ssh_config.tmp && mv /tmp/ssh_config.tmp {main_ssh_config}"
            )
    except Exception as e:
        print(f"\n⚠️  Warning: Could not check {main_ssh_config}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate SSH config for RunPod pods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config for all running pods
  python runpod_ssh_config.py
  
  # Filter by cluster name (for Instant Clusters)
  python runpod_ssh_config.py --cluster-name "My Cluster"
  
  # Specify SSH key
  python runpod_ssh_config.py --identity-file ~/.ssh/id_rsa
  
  # Combine options
  python runpod_ssh_config.py --cluster-name "My Cluster" --identity-file ~/.ssh/id_rsa
  
  # Custom output path
  python runpod_ssh_config.py --output-path ~/my-custom-ssh-config
        """,
    )

    parser.add_argument(
        "--cluster-name", type=str, help="Filter pods by cluster name (for Instant Clusters)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="~/.ssh/config.d/runpod",
        help="Output path for SSH config (default: ~/.ssh/config.d/runpod)",
    )

    parser.add_argument("--user", type=str, default="root", help="SSH username (default: root)")

    parser.add_argument(
        "--identity-file",
        type=str,
        help="Path to SSH private key (e.g., ~/.ssh/id_rsa)",
    )

    parser.add_argument(
        "--no-backup", action="store_true", help="Don't backup existing config file"
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)

    # Initialize client
    client = RunPodClient(api_key)

    # Fetch pods
    print("Fetching pods from RunPod...")
    if args.cluster_name:
        print(f"Filtering by cluster name: {args.cluster_name}")

    pods = client.get_all_pods(cluster_name=args.cluster_name)

    if not pods:
        print("No pods found")
        sys.exit(0)

    print(f"Found {len(pods)} total pod(s)")

    # Filter for running pods
    running_pods = [p for p in pods if is_pod_running(p)]
    print(f"Found {len(running_pods)} running pod(s)")

    if not running_pods:
        print("No running pods to configure")
        sys.exit(0)

    # Generate SSH config entries
    pod_entries = []  # List of (pod, entry) tuples
    skipped_pods = []

    for pod in running_pods:
        entry = generate_ssh_config_entry(pod, user=args.user, identity_file=args.identity_file)
        if entry:
            pod_entries.append((pod, entry))
        else:
            skipped_pods.append(pod.get("name", pod.get("id")))

    if skipped_pods:
        print(f"\nSkipped {len(skipped_pods)} pod(s) without SSH access:")
        for name in skipped_pods:
            print(f"  - {name}")

    if not pod_entries:
        print("\nNo pods with SSH access found")
        sys.exit(0)

    # Sort entries: controller pods first, then alphabetically by name
    def sort_key(pod_entry_tuple):
        pod, _ = pod_entry_tuple
        pod_name = (pod.get("name") or pod.get("id") or "").lower()
        # Return tuple: (is_not_controller, pod_name)
        # False sorts before True, so controller pods come first
        has_controller = "controller" in pod_name
        return (not has_controller, pod_name)

    pod_entries.sort(key=sort_key)

    # Extract just the config entries in sorted order
    config_entries = [entry for _, entry in pod_entries]

    print(f"\nGenerated SSH config for {len(config_entries)} pod(s)")

    # Write config file
    output_path = Path(args.output_path).expanduser()
    write_ssh_config(config_entries, output_path, backup=not args.no_backup)

    # Check if main SSH config includes this file
    check_ssh_config_include(output_path)

    print("\n✅ Done! You can now SSH to your pods using:")
    # Show first 3 examples in sorted order
    for pod, _ in pod_entries[:3]:
        pod_name = pod.get("name") or pod.get("id") or "unknown"
        host_alias = f"runpod-{pod_name.replace(' ', '-').lower()}"
        print(f"  ssh {host_alias}")

    if len(pod_entries) > 3:
        print(f"  ... and {len(pod_entries) - 3} more")


if __name__ == "__main__":
    main()
