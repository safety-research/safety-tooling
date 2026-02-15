"""Unit tests for FileBasedCacheManager memory eviction logic.

These tests exercise the in-memory cache eviction without making any API calls.
"""

import tempfile
from pathlib import Path

from safetytooling.apis.inference.cache_manager import FileBasedCacheManager, total_size


def _make_data(size_chars: int, num_keys: int = 1) -> dict:
    """Create a dict whose in-memory size is roughly proportional to size_chars."""
    return {f"k{i}": "x" * (size_chars // num_keys) for i in range(num_keys)}


class TestAddEntrySelfEviction:
    """Regression tests for the self-eviction bug in add_entry.

    When a bin file is re-loaded from disk with a larger size (because
    save_cache wrote new entries to disk without updating in_memory_cache),
    add_entry previously left self.sizes and self.in_memory_cache out of
    sync. This caused a KeyError on the next eviction cycle.
    """

    def test_re_add_larger_bin_does_not_crash(self):
        """Re-adding a bin that triggers eviction must not self-evict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small = _make_data(200_000)  # ~0.2 MB
            large = _make_data(400_000)  # ~0.4 MB

            limit = total_size(small) + total_size(large) + 0.01
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=limit)

            bin_a = Path(tmpdir) / "binA.json"
            bin_b = Path(tmpdir) / "binB.json"
            bin_c = Path(tmpdir) / "binC.json"

            # Fill cache: bin_a is the smallest entry
            cm.add_entry(bin_a, small)
            cm.add_entry(bin_b, large)

            # Re-add bin_a with larger contents (simulates disk reload after
            # save_cache grew the bin). This must evict bin_b, not bin_a itself.
            cm.add_entry(bin_a, large)

            # Before the fix, sizes had bin_a but in_memory_cache didn't.
            # This next add triggers eviction → KeyError on bin_a.pop().
            cm.add_entry(bin_c, large)  # must not crash

    def test_re_add_keeps_sizes_consistent(self):
        """After re-adding a bin, sizes and in_memory_cache must agree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=100)

            bin_a = Path(tmpdir) / "binA.json"
            small = _make_data(100_000)
            big = _make_data(500_000)

            cm.add_entry(bin_a, small)
            cm.add_entry(bin_a, big)

            assert set(cm.sizes.keys()) == set(cm.in_memory_cache.keys())
            assert bin_a in cm.in_memory_cache
            assert cm.in_memory_cache[bin_a] is big

    def test_re_add_does_not_double_count(self):
        """Re-adding a bin must not inflate total_usage_mb."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=100)

            bin_a = Path(tmpdir) / "binA.json"
            data = _make_data(200_000)
            expected_size = total_size(data)

            cm.add_entry(bin_a, data)
            assert abs(cm.total_usage_mb - expected_size) < 0.001

            # Re-add with identical data — total should not change
            cm.add_entry(bin_a, data)
            assert abs(cm.total_usage_mb - expected_size) < 0.001

    def test_oversized_entry_not_leaked(self):
        """If an entry is too large for the cache, it must not leak in in_memory_cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_limit = 0.001  # ~1 KB
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=tiny_limit)

            bin_a = Path(tmpdir) / "binA.json"
            huge = _make_data(1_000_000)  # way over 1 KB

            result = cm.add_entry(bin_a, huge)

            assert result is False
            assert bin_a not in cm.in_memory_cache
            assert bin_a not in cm.sizes
            assert cm.total_usage_mb == 0
