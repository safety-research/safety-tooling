"""Tests for SQLiteCacheManager.

Tests the SQLite-based cache backend against the same interface as FileBasedCacheManager,
including basic save/load, batch operations, compression, schema versioning, and stats.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

# Import directly to avoid pulling in the full api chain (needs pydub etc.)
from safetytooling.apis.inference.sqlite_cache_manager import (
    SCHEMA_VERSION,
    CacheStats,
    SQLiteCacheManager,
    _compress,
    _decompress,
)
from safetytooling.data_models.embedding import EmbeddingParams, EmbeddingResponseBase64
from safetytooling.data_models.inference import LLMParams, LLMResponse, StopReason
from safetytooling.data_models.messages import ChatMessage, MessageRole, Prompt


def _make_prompt(content: str = "Hello") -> Prompt:
    return Prompt(messages=[ChatMessage(role=MessageRole.user, content=content)])


def _make_params(model: str = "gpt-4o-mini", temperature: float = 0.0) -> LLMParams:
    return LLMParams(model=model, temperature=temperature)


def _make_response(completion: str = "Hi there!", cost: float = 0.001) -> LLMResponse:
    return LLMResponse(
        model_id="gpt-4o-mini",
        completion=completion,
        cost=cost,
        stop_reason=StopReason.STOP_SEQUENCE,
    )


class TestCompression:
    def test_roundtrip(self):
        original = '{"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}'
        assert _decompress(_compress(original)) == original

    def test_empty_string(self):
        assert _decompress(_compress("")) == ""

    def test_unicode(self):
        text = "Hello 世界 🌍 café"
        assert _decompress(_compress(text)) == text

    def test_compression_reduces_size(self):
        large = "x" * 10000
        compressed = _compress(large)
        assert len(compressed) < len(large.encode("utf-8"))


class TestSQLiteCacheManagerBasic:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt()
            params = _make_params()
            responses = [_make_response()]

            cm.save_cache(prompt, params, responses)
            loaded = cm.maybe_load_cache(prompt, params)

            assert loaded is not None
            assert loaded.responses is not None
            assert len(loaded.responses) == 1
            assert loaded.responses[0].completion == "Hi there!"

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt("not cached")
            params = _make_params()

            result = cm.maybe_load_cache(prompt, params)
            assert result is None

    def test_different_prompts_different_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()

            prompt1 = _make_prompt("Hello")
            prompt2 = _make_prompt("Goodbye")
            resp1 = [_make_response("Response 1")]
            resp2 = [_make_response("Response 2")]

            cm.save_cache(prompt1, params, resp1)
            cm.save_cache(prompt2, params, resp2)

            loaded1 = cm.maybe_load_cache(prompt1, params)
            loaded2 = cm.maybe_load_cache(prompt2, params)

            assert loaded1 is not None and loaded1.responses is not None
            assert loaded2 is not None and loaded2.responses is not None
            assert loaded1.responses[0].completion == "Response 1"
            assert loaded2.responses[0].completion == "Response 2"

    def test_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt()
            params = _make_params()

            cm.save_cache(prompt, params, [_make_response("first")])
            cm.save_cache(prompt, params, [_make_response("second")])

            loaded = cm.maybe_load_cache(prompt, params)
            assert loaded is not None and loaded.responses is not None
            assert loaded.responses[0].completion == "second"

    def test_different_models_different_dbs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt()
            params1 = _make_params(model="gpt-4o-mini")
            params2 = _make_params(model="claude-sonnet-4-20250514")

            cm.save_cache(prompt, params1, [_make_response("gpt4")])
            cm.save_cache(prompt, params2, [_make_response("sonnet")])

            # Each model gets its own .sqlite file
            sqlite_files = list(Path(tmpdir).glob("*.sqlite"))
            assert len(sqlite_files) == 2

            loaded1 = cm.maybe_load_cache(prompt, params1)
            loaded2 = cm.maybe_load_cache(prompt, params2)
            assert loaded1 is not None and loaded1.responses is not None
            assert loaded2 is not None and loaded2.responses is not None
            assert loaded1.responses[0].completion == "gpt4"
            assert loaded2.responses[0].completion == "sonnet"


class TestBatchLookup:
    def test_batch_all_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()

            prompts = [_make_prompt(f"prompt_{i}") for i in range(5)]
            for i, p in enumerate(prompts):
                cm.save_cache(p, params, [_make_response(f"resp_{i}")])

            results = cm.maybe_load_cache_batch(prompts, params)
            assert len(results) == 5
            assert all(v is not None for v in results.values())

    def test_batch_partial_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()

            prompts = [_make_prompt(f"prompt_{i}") for i in range(5)]
            # Only cache first 3
            for i in range(3):
                cm.save_cache(prompts[i], params, [_make_response(f"resp_{i}")])

            results = cm.maybe_load_cache_batch(prompts, params)
            cached = [h for h, v in results.items() if v is not None]
            missed = [h for h, v in results.items() if v is None]
            assert len(cached) == 3
            assert len(missed) == 2

    def test_batch_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()
            results = cm.maybe_load_cache_batch([], params)
            assert results == {}


class TestSchemaVersioning:
    def test_stale_schema_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt()
            params = _make_params()

            # Save normally
            cm.save_cache(prompt, params, [_make_response()])

            # Manually update schema version in DB to simulate old entry
            db_path, prompt_hash = cm.get_cache_file(prompt, params)
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE responses SET schema_version = ? WHERE prompt_hash = ?",
                (SCHEMA_VERSION - 1, prompt_hash),
            )
            conn.commit()
            conn.close()

            loaded = cm.maybe_load_cache(prompt, params)
            assert loaded is None
            assert cm.stats.stale == 1


class TestCacheStats:
    def test_hit_miss_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            prompt = _make_prompt()
            params = _make_params()

            # Miss
            cm.maybe_load_cache(prompt, params)
            assert cm.stats.misses == 1
            assert cm.stats.hits == 0

            # Save and hit
            cm.save_cache(prompt, params, [_make_response()])
            cm.maybe_load_cache(prompt, params)
            assert cm.stats.hits == 1
            assert cm.stats.writes == 1

    def test_hit_rate(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

        stats.hits = 3
        stats.misses = 1
        assert stats.hit_rate == 0.75

    def test_summary_format(self):
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.writes = 10
        stats.saved_cost = 0.05
        summary = stats.summary()
        assert "10 hits" in summary
        assert "5 misses" in summary
        assert "66.7%" in summary


class TestDiagnostics:
    def test_entry_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()

            assert cm.entry_count() == 0

            for i in range(5):
                cm.save_cache(_make_prompt(f"p{i}"), params, [_make_response()])

            assert cm.entry_count() == 5
            assert cm.entry_count(params) == 5

    def test_db_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()
            cm.save_cache(_make_prompt(), params, [_make_response()])

            sizes = cm.db_sizes()
            assert len(sizes) == 1
            assert all(v > 0 for v in sizes.values())


class TestModeration:
    def test_save_and_load_moderation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            texts = ["hello world", "test text"]

            # Should return None before saving
            assert cm.maybe_load_moderation(texts) is None

            # Save and reload
            cm.save_moderation(texts, [])
            loaded = cm.maybe_load_moderation(texts)
            assert loaded is not None
            assert loaded.texts == texts


class TestEmbeddings:
    def test_save_and_load_embeddings(self):
        pytest.importorskip("numpy")
        import base64

        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = EmbeddingParams(model_id="text-embedding-3-small", texts=["hello"])
            embedding = np.array([0.1, 0.2, 0.3], dtype="float32")
            response = EmbeddingResponseBase64(
                model_id="text-embedding-3-small",
                embeddings=[base64.b64encode(embedding.tobytes()).decode()],
                tokens=5,
                cost=0.0001,
            )

            assert cm.maybe_load_embeddings(params) is None

            cm.save_embeddings(params, response)
            loaded = cm.maybe_load_embeddings(params)
            assert loaded is not None
            assert len(loaded.embeddings) == 1


class TestWALMode:
    """Verify SQLite is configured with WAL mode for concurrent reads."""

    def test_wal_mode_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = SQLiteCacheManager(Path(tmpdir))
            params = _make_params()
            cm.save_cache(_make_prompt(), params, [_make_response()])

            db_path, _ = cm.get_cache_file(_make_prompt(), params)
            conn = sqlite3.connect(str(db_path))
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            assert mode == "wal"
