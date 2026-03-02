"""Integration tests for VPC Direct Egress firewall on Cloud Run.

Tests that when vpc_network is configured, the Cloud NGFW firewall policy
correctly allows/blocks outbound traffic by domain.

Requires:
- GCP credentials (gcloud auth application-default login)
- VPC with Cloud NGFW firewall policy (see README.md Egress Firewall section)
- Cloud NAT with ENDPOINT_TYPE_MANAGED_PROXY_LB

Additional environment variables:
- VPC_NETWORK: VPC network name (e.g., "egress-firewall-vpc")
- VPC_SUBNET: VPC subnet name (e.g., "egress-firewall-subnet")

Run with: pytest tests/test_vpc_egress.py -v --run-integration
"""

import os
import re

import pytest

from safetytooling.infra.cloud_run import (
    ClaudeCodeClient,
    ClaudeCodeClientConfig,
    ClaudeCodeTask,
)


@pytest.fixture
def integration_enabled(request):
    if not request.config.getoption("--run-integration", default=False):
        pytest.skip("Integration tests skipped. Use --run-integration to run.")


@pytest.fixture
def vpc_config():
    """Get VPC + GCP config from environment or skip."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    gcs_bucket = os.environ.get("GCS_BUCKET")
    api_key_secret = os.environ.get("API_KEY_SECRET")
    service_account = os.environ.get("SERVICE_ACCOUNT")
    vpc_network = os.environ.get("VPC_NETWORK")
    vpc_subnet = os.environ.get("VPC_SUBNET")

    missing = []
    for name, val in [
        ("GCP_PROJECT_ID", project_id),
        ("GCS_BUCKET", gcs_bucket),
        ("API_KEY_SECRET", api_key_secret),
        ("SERVICE_ACCOUNT", service_account),
        ("VPC_NETWORK", vpc_network),
        ("VPC_SUBNET", vpc_subnet),
    ]:
        if not val:
            missing.append(name)
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")

    return ClaudeCodeClientConfig(
        project_id=project_id,
        gcs_bucket=gcs_bucket,
        api_key_secret=api_key_secret,
        service_account=service_account,
        vpc_network=vpc_network,
        vpc_subnet=vpc_subnet,
        vpc_egress="all-traffic",
        timeout=300,
    )


# Shell script that tests connectivity and prints structured results.
# Each test prints "TEST <name>: PASS" or "TEST <name>: FAIL".
TEST_SCRIPT = r"""
echo "=== VPC Egress Firewall Tests ==="

# Test allowed domain (Anthropic API)
HTTP_CODE=$(curl -4 -s -o /dev/null -w '%{http_code}' --connect-timeout 60 https://api.anthropic.com/v1/messages)
if [ "$HTTP_CODE" != "000" ]; then echo "TEST allowed_anthropic: PASS"; else echo "TEST allowed_anthropic: FAIL"; fi

# Test allowed domain (PyPI)
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 30 https://pypi.org/simple/)
if [ "$HTTP_CODE" != "000" ]; then echo "TEST allowed_pypi: PASS"; else echo "TEST allowed_pypi: FAIL"; fi

# Test allowed domain (npm)
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 30 https://registry.npmjs.org/)
if [ "$HTTP_CODE" != "000" ]; then echo "TEST allowed_npm: PASS"; else echo "TEST allowed_npm: FAIL"; fi

# Test blocked domain (example.com)
HTTP_CODE=$(curl -4 -s -o /dev/null -w '%{http_code}' --connect-timeout 15 https://example.com 2>/dev/null)
if [ "$HTTP_CODE" = "000" ]; then echo "TEST blocked_example: PASS"; else echo "TEST blocked_example: FAIL"; fi

# Test IPv6 blocked
HTTP_CODE=$(curl -6 -s -o /dev/null -w '%{http_code}' --connect-timeout 15 https://api.anthropic.com/v1/messages 2>/dev/null)
if [ "$HTTP_CODE" = "000" ]; then echo "TEST blocked_ipv6: PASS"; else echo "TEST blocked_ipv6: FAIL"; fi

echo "=== Done ==="
"""


def _parse_test_results(output: str) -> dict[str, bool]:
    """Parse 'TEST name: PASS/FAIL' lines from script output."""
    results = {}
    for match in re.finditer(r"TEST (\w+): (PASS|FAIL)", output):
        results[match.group(1)] = match.group(2) == "PASS"
    return results


class TestVPCEgress:
    """Test that VPC egress firewall blocks/allows the right domains."""

    def test_egress_firewall(self, integration_enabled, vpc_config):
        """Run all egress tests in a single Cloud Run job."""
        client = ClaudeCodeClient(vpc_config)

        tasks = [
            ClaudeCodeTask(
                id="egress-test",
                task="echo 'Tests ran in pre_claude_command'",
                pre_claude_command=TEST_SCRIPT,
                output_instructions=False,
                n=1,
            ),
        ]

        results = client.run(tasks)
        task = tasks[0]
        result = results[task][0]

        assert result.returncode == 0, f"Job failed: {result.error}"

        parsed = _parse_test_results(result.response)
        assert len(parsed) >= 4, f"Expected >= 4 test results, got {len(parsed)}: {parsed}"

        # Allowed domains should connect
        assert parsed.get("allowed_anthropic"), "api.anthropic.com should be reachable"
        assert parsed.get("allowed_pypi"), "pypi.org should be reachable"
        assert parsed.get("allowed_npm"), "registry.npmjs.org should be reachable"

        # Blocked domains should not connect
        assert parsed.get("blocked_example"), "example.com should be blocked"
        assert parsed.get("blocked_ipv6"), "IPv6 should be blocked"
