"""Unit tests for CloudRunClient caching and thread-safety.

These tests don't require GCP credentials - they test the local caching
and tarring logic without actually running Cloud Run jobs.
"""

import hashlib
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_repo():
    """Create a temporary repo directory with some files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "repo"
        repo.mkdir()
        (repo / "file1.txt").write_text("hello world")
        (repo / "file2.txt").write_text("goodbye world")
        subdir = repo / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")
        yield repo


@pytest.fixture
def mock_gcs_client():
    """Mock GCS client that tracks uploads."""
    uploads = []
    existing_blobs = set()

    class MockBlob:
        def __init__(self, name):
            self.name = name

        def exists(self):
            return self.name in existing_blobs

        def upload_from_filename(self, path):
            uploads.append(self.name)
            existing_blobs.add(self.name)

    class MockBucket:
        def blob(self, name):
            return MockBlob(name)

    class MockClient:
        def bucket(self, name):
            return MockBucket()

    return MockClient(), uploads, existing_blobs


class TestDeterministicTarring:
    """Test that tarring is deterministic (same content = same hash)."""

    def test_same_content_same_hash(self, temp_repo):
        """Same directory content should produce identical tarballs."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        # Create client with mocked GCP clients
        with patch.object(CloudRunClient, "__init__", lambda self, config: None):
            client = CloudRunClient.__new__(CloudRunClient)
            client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")

            inputs = {"repo": temp_repo}

            # Create two tarballs
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f1:
                tar1 = Path(f1.name)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f2:
                tar2 = Path(f2.name)

            try:
                client._tar_inputs(inputs, tar1)
                client._tar_inputs(inputs, tar2)

                # Compare hashes
                hash1 = hashlib.md5(tar1.read_bytes()).hexdigest()
                hash2 = hashlib.md5(tar2.read_bytes()).hexdigest()

                assert hash1 == hash2, "Same content should produce same hash"
            finally:
                tar1.unlink(missing_ok=True)
                tar2.unlink(missing_ok=True)

    def test_different_content_different_hash(self, temp_repo):
        """Different directory content should produce different hashes."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        with patch.object(CloudRunClient, "__init__", lambda self, config: None):
            client = CloudRunClient.__new__(CloudRunClient)
            client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")

            # Create first tarball
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f1:
                tar1 = Path(f1.name)
            client._tar_inputs({"repo": temp_repo}, tar1)
            hash1 = hashlib.md5(tar1.read_bytes()).hexdigest()

            # Modify content
            (temp_repo / "file1.txt").write_text("modified content")

            # Create second tarball
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f2:
                tar2 = Path(f2.name)
            client._tar_inputs({"repo": temp_repo}, tar2)
            hash2 = hashlib.md5(tar2.read_bytes()).hexdigest()

            try:
                assert hash1 != hash2, "Different content should produce different hash"
            finally:
                tar1.unlink(missing_ok=True)
                tar2.unlink(missing_ok=True)


class TestInputsCaching:
    """Test the in-memory and GCS caching of inputs."""

    def test_cache_key_computation(self, temp_repo):
        """Test that cache keys are computed correctly."""
        from safetytooling.infra.cloud_run.cloud_run_client import CloudRunClient

        inputs = {"repo": temp_repo}
        key1 = CloudRunClient._compute_inputs_key(inputs)

        # Same inputs = same key
        key2 = CloudRunClient._compute_inputs_key(inputs)
        assert key1 == key2

        # Different name = different key
        inputs2 = {"other": temp_repo}
        key3 = CloudRunClient._compute_inputs_key(inputs2)
        assert key1 != key3

    def test_inmemory_cache_prevents_retarring(self, temp_repo, mock_gcs_client):
        """In-memory cache should prevent re-tarring same inputs."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        mock_client, uploads, _ = mock_gcs_client

        # Clear class-level cache
        CloudRunClient._inputs_cache.clear()
        CloudRunClient._inputs_locks.clear()

        with patch.object(CloudRunClient, "__init__", lambda self, config: None):
            client = CloudRunClient.__new__(CloudRunClient)
            client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")
            client._storage_client = mock_client

            inputs = {"repo": temp_repo}

            # First upload
            path1 = client._upload_inputs(inputs)
            assert len(uploads) == 1

            # Second upload - should use cache, no new upload
            path2 = client._upload_inputs(inputs)
            assert len(uploads) == 1  # Still 1
            assert path1 == path2

    def test_gcs_cache_prevents_reupload(self, temp_repo):
        """GCS cache should prevent re-uploading same content."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        uploads = []
        existing_blobs = set()

        class MockBlob:
            def __init__(self, name):
                self.name = name

            def exists(self):
                return self.name in existing_blobs

            def upload_from_filename(self, path):
                uploads.append(self.name)
                existing_blobs.add(self.name)

        class MockBucket:
            def blob(self, name):
                return MockBlob(name)

        class MockClient:
            def bucket(self, name):
                return MockBucket()

        # Clear class-level cache to force GCS check
        CloudRunClient._inputs_cache.clear()
        CloudRunClient._inputs_locks.clear()

        with patch.object(CloudRunClient, "__init__", lambda self, config: None):
            client = CloudRunClient.__new__(CloudRunClient)
            client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")
            client._storage_client = MockClient()

            inputs = {"repo": temp_repo}

            # First upload
            path1 = client._upload_inputs(inputs)
            assert len(uploads) == 1

            # Clear in-memory cache but keep GCS "cache" (existing_blobs)
            CloudRunClient._inputs_cache.clear()

            # Second upload - should check GCS and skip upload
            path2 = client._upload_inputs(inputs)
            assert len(uploads) == 1  # Still 1 - GCS cache hit
            assert path1 == path2


class TestThreadSafety:
    """Test thread-safety of the caching mechanism."""

    def test_lock_blocks_concurrent_access(self, temp_repo):
        """Verify the lock actually blocks threads - not just outcome testing."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        # Track concurrent executions inside the critical section
        inside_critical_section = [0]
        max_concurrent = [0]
        section_lock = threading.Lock()
        barrier = threading.Barrier(5)  # Ensure all threads start together

        original_tar = CloudRunClient._tar_inputs

        def slow_tar_that_tracks_concurrency(self, inputs, tar_path):
            # Record entering critical section
            with section_lock:
                inside_critical_section[0] += 1
                if inside_critical_section[0] > max_concurrent[0]:
                    max_concurrent[0] = inside_critical_section[0]

            # Simulate slow work - if lock works, only 1 thread is here at a time
            time.sleep(0.05)

            # Record leaving critical section
            with section_lock:
                inside_critical_section[0] -= 1

            return original_tar(self, inputs, tar_path)

        def mock_upload(self, tar_path):
            return f"cloudrun-inputs/test.tar.gz"

        # Clear caches
        CloudRunClient._inputs_cache.clear()
        CloudRunClient._inputs_locks.clear()

        with patch.object(CloudRunClient, "_tar_inputs", slow_tar_that_tracks_concurrency):
            with patch.object(CloudRunClient, "_upload_to_gcs_if_needed", mock_upload):
                with patch.object(CloudRunClient, "__init__", lambda self, config: None):
                    client = CloudRunClient.__new__(CloudRunClient)
                    client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")

                    inputs = {"repo": temp_repo}

                    def upload_after_barrier():
                        barrier.wait()  # All threads start at same time
                        return client._upload_inputs(inputs)

                    # Run 5 concurrent uploads, all starting together
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(upload_after_barrier) for _ in range(5)]
                        [f.result() for f in futures]

                    # If lock works: max_concurrent should be 1
                    # If lock broken: max_concurrent would be > 1
                    assert max_concurrent[0] == 1, (
                        f"Lock should ensure only 1 thread in critical section at a time, "
                        f"but saw {max_concurrent[0]} concurrent"
                    )

    def test_concurrent_uploads_same_inputs(self, temp_repo):
        """Multiple threads uploading same inputs should only tar/upload once."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        tar_count = [0]
        upload_count = [0]
        tar_lock = threading.Lock()
        upload_lock = threading.Lock()

        original_tar = CloudRunClient._tar_inputs
        original_upload = CloudRunClient._upload_to_gcs_if_needed

        def counting_tar(self, inputs, tar_path):
            with tar_lock:
                tar_count[0] += 1
            # Add small delay to increase chance of race conditions
            time.sleep(0.01)
            return original_tar(self, inputs, tar_path)

        def counting_upload(self, tar_path):
            with upload_lock:
                upload_count[0] += 1
            # Return fake GCS path
            return f"cloudrun-inputs/{hashlib.md5(tar_path.read_bytes()).hexdigest()}.tar.gz"

        # Clear caches
        CloudRunClient._inputs_cache.clear()
        CloudRunClient._inputs_locks.clear()

        with patch.object(CloudRunClient, "_tar_inputs", counting_tar):
            with patch.object(CloudRunClient, "_upload_to_gcs_if_needed", counting_upload):
                with patch.object(CloudRunClient, "__init__", lambda self, config: None):
                    client = CloudRunClient.__new__(CloudRunClient)
                    client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")

                    inputs = {"repo": temp_repo}

                    # Run 10 concurrent uploads
                    results = []
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        futures = [executor.submit(client._upload_inputs, inputs) for _ in range(10)]
                        results = [f.result() for f in futures]

                    # All should return the same path
                    assert len(set(results)) == 1, "All threads should get same GCS path"

                    # Should only tar and upload once
                    assert tar_count[0] == 1, f"Should tar once, got {tar_count[0]}"
                    assert upload_count[0] == 1, f"Should upload once, got {upload_count[0]}"

    def test_concurrent_uploads_different_inputs(self, temp_repo):
        """Multiple threads with different inputs should all upload."""
        from safetytooling.infra.cloud_run.cloud_run_client import (
            CloudRunClient,
            CloudRunClientConfig,
        )

        upload_count = [0]
        upload_lock = threading.Lock()

        def counting_upload(self, tar_path):
            with upload_lock:
                upload_count[0] += 1
            return f"cloudrun-inputs/{hashlib.md5(tar_path.read_bytes()).hexdigest()}.tar.gz"

        # Clear caches
        CloudRunClient._inputs_cache.clear()
        CloudRunClient._inputs_locks.clear()

        # Create multiple different repos
        repos = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                repo = Path(tmpdir) / f"repo{i}"
                repo.mkdir()
                (repo / "file.txt").write_text(f"content {i}")
                repos.append(repo)

            with patch.object(CloudRunClient, "_upload_to_gcs_if_needed", counting_upload):
                with patch.object(CloudRunClient, "__init__", lambda self, config: None):
                    client = CloudRunClient.__new__(CloudRunClient)
                    client.config = CloudRunClientConfig(project_id="test", gcs_bucket="test-bucket")

                    # Run concurrent uploads with different inputs
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(client._upload_inputs, {"repo": repo}) for repo in repos]
                        results = [f.result() for f in futures]

                    # All should return different paths
                    assert len(set(results)) == 5, "Different inputs should get different paths"

                    # Should upload 5 times
                    assert upload_count[0] == 5, f"Should upload 5 times, got {upload_count[0]}"


class TestInputsLocking:
    """Test the per-inputs-key locking mechanism."""

    def test_lock_per_inputs_key(self):
        """Each unique inputs key should get its own lock."""
        from safetytooling.infra.cloud_run.cloud_run_client import CloudRunClient

        # Clear locks
        CloudRunClient._inputs_locks.clear()

        lock1 = CloudRunClient._get_inputs_lock("key1")
        lock2 = CloudRunClient._get_inputs_lock("key2")
        lock1_again = CloudRunClient._get_inputs_lock("key1")

        assert lock1 is lock1_again, "Same key should return same lock"
        assert lock1 is not lock2, "Different keys should have different locks"

    def test_lock_creation_thread_safe(self):
        """Lock creation itself should be thread-safe."""
        from safetytooling.infra.cloud_run.cloud_run_client import CloudRunClient

        # Clear locks
        CloudRunClient._inputs_locks.clear()

        locks = []
        lock = threading.Lock()

        def get_lock():
            result = CloudRunClient._get_inputs_lock("shared-key")
            with lock:
                locks.append(result)

        # Create many threads trying to get the same lock
        threads = [threading.Thread(target=get_lock) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have gotten the same lock object
        assert len(set(id(l) for l in locks)) == 1, "All threads should get same lock"
