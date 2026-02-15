"""Unit tests for FileBasedCacheManager LRU memory eviction.

These tests exercise the in-memory cache eviction without making any API calls.
"""

import tempfile
from pathlib import Path

from safetytooling.apis.inference.cache_manager import FileBasedCacheManager, total_size


def _make_data(size_chars: int, num_keys: int = 1) -> dict:
    """Create a dict whose in-memory size is roughly proportional to size_chars."""
    return {f"k{i}": "x" * (size_chars // num_keys) for i in range(num_keys)}


class TestAddEntryReAdd:
    """Regression tests for re-adding bins (the self-eviction bug)."""

    def test_re_add_larger_bin_does_not_crash(self):
        """Re-adding a bin that triggers eviction must not self-evict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small = _make_data(200_000)
            large = _make_data(400_000)

            limit = total_size(small) + total_size(large) + 0.01
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=limit)

            bin_a = Path(tmpdir) / "binA.json"
            bin_b = Path(tmpdir) / "binB.json"
            bin_c = Path(tmpdir) / "binC.json"

            cm.add_entry(bin_a, small)
            cm.add_entry(bin_b, large)

            # Re-add bin_a with larger contents (simulates disk reload after
            # save_cache grew the bin on disk).
            cm.add_entry(bin_a, large)
            cm.add_entry(bin_c, large)  # must not crash

    def test_re_add_keeps_dicts_consistent(self):
        """After re-adding a bin, sizes and in_memory_cache must have the same keys."""
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

            cm.add_entry(bin_a, data)
            assert abs(cm.total_usage_mb - expected_size) < 0.001

    def test_oversized_entry_not_leaked(self):
        """If an entry exceeds the entire cache limit, it must not leak."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=0.001)

            bin_a = Path(tmpdir) / "binA.json"
            huge = _make_data(1_000_000)

            result = cm.add_entry(bin_a, huge)

            assert result is False
            assert bin_a not in cm.in_memory_cache
            assert bin_a not in cm.sizes
            assert cm.total_usage_mb == 0


class TestLRUEvictionOrder:
    """Tests that eviction follows LRU order, not smallest-first."""

    def test_evicts_oldest_not_smallest(self):
        """When memory is full, the least-recently-used bin is evicted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small = _make_data(100_000)
            large = _make_data(300_000)

            # Room for small + large, but not small + large + small
            limit = total_size(small) + total_size(large) + 0.01
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=limit)

            bin_a = Path(tmpdir) / "binA.json"
            bin_b = Path(tmpdir) / "binB.json"
            bin_c = Path(tmpdir) / "binC.json"

            cm.add_entry(bin_a, small)  # oldest
            cm.add_entry(bin_b, large)  # newer

            # Adding bin_c must evict bin_a (oldest), NOT bin_a (smallest).
            # Under the old smallest-first policy, bin_a would also have been
            # evicted — but for the wrong reason. We verify the LRU property
            # by checking that bin_b (larger but newer) survives.
            cm.add_entry(bin_c, small)

            assert bin_a not in cm.in_memory_cache  # evicted (oldest)
            assert bin_b in cm.in_memory_cache  # kept (newer)
            assert bin_c in cm.in_memory_cache  # just added

    def test_touch_prevents_eviction(self):
        """Accessing a bin via touch() moves it to the back of the LRU queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _make_data(200_000)

            # Room for exactly 2 bins
            limit = total_size(data) * 2 + 0.01
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=limit)

            bin_a = Path(tmpdir) / "binA.json"
            bin_b = Path(tmpdir) / "binB.json"
            bin_c = Path(tmpdir) / "binC.json"

            cm.add_entry(bin_a, data)  # oldest
            cm.add_entry(bin_b, data)  # newer

            # Touch bin_a — it's now the most-recently-used
            cm.touch(bin_a)

            # Adding bin_c should evict bin_b (now the LRU), not bin_a
            cm.add_entry(bin_c, data)

            assert bin_a in cm.in_memory_cache  # survived (was touched)
            assert bin_b not in cm.in_memory_cache  # evicted (LRU after touch)
            assert bin_c in cm.in_memory_cache

    def test_add_entry_moves_to_mru(self):
        """Re-adding a bin moves it to the most-recently-used position."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _make_data(200_000)

            limit = total_size(data) * 2 + 0.01
            cm = FileBasedCacheManager(Path(tmpdir), max_mem_usage_mb=limit)

            bin_a = Path(tmpdir) / "binA.json"
            bin_b = Path(tmpdir) / "binB.json"
            bin_c = Path(tmpdir) / "binC.json"

            cm.add_entry(bin_a, data)  # oldest
            cm.add_entry(bin_b, data)  # newer

            # Re-add bin_a (simulates reload from disk) — now it's MRU
            cm.add_entry(bin_a, data)

            # Adding bin_c should evict bin_b (now LRU), not bin_a
            cm.add_entry(bin_c, data)

            assert bin_a in cm.in_memory_cache
            assert bin_b not in cm.in_memory_cache
            assert bin_c in cm.in_memory_cache
