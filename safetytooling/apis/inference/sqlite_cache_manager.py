"""SQLite-based cache manager for LLM inference responses.

Replaces the JSON bin-file approach with per-model SQLite databases.
Key improvements:
- O(1) lookup by primary key (no loading entire bin files)
- WAL mode for concurrent readers + single writer without blocking
- Batch lookups via SQL IN clause
- Built-in hit/miss statistics
- Schema versioning for safe pydantic model evolution
- zstd compression for response blobs (~5-8x savings)
"""

import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import zstandard as zstd

from safetytooling.data_models import (
    BatchPrompt,
    EmbeddingParams,
    EmbeddingResponseBase64,
    LLMCache,
    LLMCacheModeration,
    LLMParams,
    LLMResponse,
    Prompt,
    TaggedModeration,
)
from safetytooling.data_models.hashable import deterministic_hash

from .cache_manager import BaseCacheManager

LOGGER = logging.getLogger(__name__)

# Bump this when LLMCache/LLMResponse pydantic schema changes.
# Old entries with mismatched version are treated as cache misses.
SCHEMA_VERSION = 1

# zstd compression level (3 is a good speed/ratio tradeoff)
ZSTD_LEVEL = 3

_compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
_decompressor = zstd.ZstdDecompressor()


def _compress(data: str) -> bytes:
    return _compressor.compress(data.encode("utf-8"))


def _decompress(data: bytes) -> str:
    return _decompressor.decompress(data).decode("utf-8")


@dataclass
class CacheStats:
    """Track cache hit/miss statistics for a session."""

    hits: int = 0
    misses: int = 0
    stale: int = 0  # schema version mismatch
    writes: int = 0
    saved_cost: float = 0.0
    _start_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses + self.stale
        return self.hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses + self.stale

    def summary(self) -> str:
        elapsed = time.time() - self._start_time
        return (
            f"Cache stats ({elapsed:.0f}s): "
            f"{self.hits} hits, {self.misses} misses, {self.stale} stale "
            f"({self.hit_rate:.1%} hit rate), "
            f"{self.writes} writes, ${self.saved_cost:.4f} saved"
        )


def _init_db_sync(db_path: Path) -> None:
    """Initialize the SQLite database schema synchronously.

    Called once per database file. Uses WAL mode for concurrent read access.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                prompt_hash TEXT PRIMARY KEY,
                params_hash TEXT NOT NULL,
                response_blob BLOB NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                cost REAL DEFAULT 0.0,
                total_tokens INTEGER DEFAULT 0
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_params ON responses(params_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON responses(last_accessed)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS moderation (
                text_hash TEXT PRIMARY KEY,
                response_blob BLOB NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                params_hash TEXT PRIMARY KEY,
                response_blob BLOB NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """
        )

        conn.commit()
    finally:
        conn.close()


def _db_path_for_model(cache_dir: Path, params: LLMParams) -> Path:
    """Get the SQLite database path for a given model."""
    return cache_dir / f"{params.model_hash()}.sqlite"


class SQLiteCacheManager(BaseCacheManager):
    """SQLite-based cache with per-model database files.

    Each model gets its own .sqlite file in the cache directory.
    Uses WAL mode for concurrent readers and zstd compression for response blobs.
    """

    def __init__(self, cache_dir: Path):
        super().__init__(cache_dir, num_bins=0)  # bins not used
        self.stats = CacheStats()
        self._initialized_dbs: set[Path] = set()
        self._conn_cache: dict[Path, sqlite3.Connection] = {}

    def _ensure_db(self, db_path: Path) -> None:
        """Ensure the database at db_path is initialized (idempotent)."""
        if db_path not in self._initialized_dbs:
            _init_db_sync(db_path)
            self._initialized_dbs.add(db_path)

    def _get_sync_conn(self, db_path: Path) -> sqlite3.Connection:
        """Get a reusable synchronous connection for the given database."""
        self._ensure_db(db_path)
        conn = self._conn_cache.get(db_path)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        self._conn_cache[db_path] = conn
        return conn

    # ──────────────────────────────────────────────
    # LLM response cache
    # ──────────────────────────────────────────────

    def get_cache_file(self, prompt: Prompt, params: LLMParams) -> tuple[Path, str]:
        """Return (db_path, prompt_hash) — kept for interface compat."""
        prompt_hash = prompt.model_hash()
        db_path = _db_path_for_model(self.cache_dir, params)
        return db_path, prompt_hash

    def maybe_load_cache(self, prompt: Prompt, params: LLMParams) -> LLMCache | None:
        db_path, prompt_hash = self.get_cache_file(prompt, params)
        conn = self._get_sync_conn(db_path)

        row = conn.execute(
            "SELECT response_blob, schema_version, cost FROM responses WHERE prompt_hash = ?",
            (prompt_hash,),
        ).fetchone()

        if row is None:
            self.stats.misses += 1
            return None

        blob, version, cost = row

        if version != SCHEMA_VERSION:
            self.stats.stale += 1
            LOGGER.info(f"Cache stale (schema v{version} != v{SCHEMA_VERSION}) for {prompt_hash}")
            return None

        # Update access metadata
        now = time.time()
        conn.execute(
            "UPDATE responses SET last_accessed = ?, access_count = access_count + 1 WHERE prompt_hash = ?",
            (now, prompt_hash),
        )
        conn.commit()

        json_str = _decompress(blob)
        cache_entry = LLMCache.model_validate_json(json_str)

        self.stats.hits += 1
        self.stats.saved_cost += cost or 0.0

        return cache_entry

    def maybe_load_cache_batch(self, prompts: Sequence[Prompt], params: LLMParams) -> dict[str, LLMCache | None]:
        """Batch lookup — chunked SQL queries for multiple prompts.

        SQLite limits bind variables to 999 per query, so large batches
        are automatically chunked.
        """
        if not prompts:
            return {}

        db_path = _db_path_for_model(self.cache_dir, params)
        conn = self._get_sync_conn(db_path)

        hashes = {p.model_hash(): p for p in prompts}
        hash_list = list(hashes.keys())

        results: dict[str, LLMCache | None] = {h: None for h in hash_list}
        now = time.time()
        found_hashes = []

        # SQLite SQLITE_MAX_VARIABLE_NUMBER is 999 by default
        chunk_size = 900
        for i in range(0, len(hash_list), chunk_size):
            chunk = hash_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT prompt_hash, response_blob, schema_version, cost FROM responses WHERE prompt_hash IN ({placeholders})",
                chunk,
            ).fetchall()

            for prompt_hash, blob, version, cost in rows:
                if version != SCHEMA_VERSION:
                    self.stats.stale += 1
                    continue

                json_str = _decompress(blob)
                cache_entry = LLMCache.model_validate_json(json_str)
                results[prompt_hash] = cache_entry
                found_hashes.append(prompt_hash)

                self.stats.hits += 1
                self.stats.saved_cost += cost or 0.0

        # Update access metadata for all found entries
        for i in range(0, len(found_hashes), chunk_size):
            chunk = found_hashes[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            conn.execute(
                f"UPDATE responses SET last_accessed = ?, access_count = access_count + 1 WHERE prompt_hash IN ({placeholders})",
                [now, *chunk],
            )
        if found_hashes:
            conn.commit()

        # Count misses
        self.stats.misses += len(hash_list) - len(found_hashes)

        return results

    def process_cached_responses(
        self,
        prompt: Union[BatchPrompt, Prompt],
        params: LLMParams,
        n: int,
        insufficient_valids_behaviour: str,
        print_prompt_and_response: bool,
        empty_completion_threshold: float = 0.5,
    ) -> Tuple[
        List[LLMResponse | List[LLMResponse] | None],
        List[LLMCache | None],
        List[LLMResponse | List[LLMResponse] | None],
    ]:
        cached_responses = []
        cached_results = []
        failed_cache_responses = []

        prompts = prompt.prompts if isinstance(prompt, BatchPrompt) else [prompt]

        # Batch lookup for all prompts at once
        batch_results = self.maybe_load_cache_batch(prompts, params)

        for individual_prompt in prompts:
            prompt_hash = individual_prompt.model_hash()
            cached_result = batch_results.get(prompt_hash)

            if cached_result is not None and cached_result.responses is not None:
                responses_list = cached_result.responses
                db_path, _ = self.get_cache_file(prompt=individual_prompt, params=params)
                LOGGER.info(f"Loaded cache for prompt from {db_path}")

                prop_empty_completions = sum(1 for response in responses_list if response.completion == "") / len(
                    responses_list
                )

                if prop_empty_completions > empty_completion_threshold:
                    if len(responses_list) == 1:
                        LOGGER.warning("Cache does not contain completion; likely due to recitation")
                    else:
                        LOGGER.warning(
                            f"Proportion of cache responses that contain empty completions ({prop_empty_completions}) is greater than threshold {empty_completion_threshold}. Likely due to recitation"
                        )
                    failed_cache_response = responses_list
                    cached_result = None
                    cached_response = None
                else:
                    cached_response = responses_list
                    if insufficient_valids_behaviour != "continue":
                        assert len(responses_list) == n, f"cache is inconsistent with n={n}\n{responses_list}"
                    if print_prompt_and_response:
                        individual_prompt.pretty_print(responses_list)

                    failed_cache_response = None
            else:
                cached_response = None
                cached_result = None
                failed_cache_response = None
            cached_responses.append(cached_response)
            cached_results.append(cached_result)
            failed_cache_responses.append(failed_cache_response)

        assert (
            len(cached_results) == len(prompts) == len(cached_responses) == len(failed_cache_responses)
        ), f"""Different number of cached_results, prompts, cached_responses and failed_cached_responses when they should all be length {len(prompts)}!
                Number of prompts: {len(prompts)} \n Number of cached_results: {len(cached_results)} \n Number of cached_responses: {len(cached_responses)} \n Number of failed_cache_responses: {len(failed_cache_responses)}"""

        if all(result is None for result in cached_results):
            assert all(
                response is None for response in cached_responses
            ), f"No cached results found so responses should be length 0. Instead responses is length {len(cached_responses)}"
        return cached_responses, cached_results, failed_cache_responses

    def update_failed_cache(
        self,
        prompt: Prompt,
        responses: list[LLMResponse],
        failed_cache_responses: List[List[LLMResponse]],
    ) -> list[LLMResponse]:
        assert len(responses) == len(
            failed_cache_responses[0]
        ), f"There should be the same number of responses and failed_cache_responses! Instead we have {len(responses)} responses and {len(failed_cache_responses)} failed_cache_responses."
        for i in range(len(responses)):
            responses[i].api_failures = (failed_cache_responses[0][i].api_failures or 0) + 1

        LOGGER.info(f"Updating previous failures for prompt with {len(responses)} responses")
        return responses

    def save_cache(self, prompt: Prompt, params: LLMParams, responses: list):
        db_path, prompt_hash = self.get_cache_file(prompt, params)
        self._ensure_db(db_path)

        cache_entry = LLMCache(prompt=prompt, params=params, responses=responses)
        json_str = cache_entry.model_dump_json()
        blob = _compress(json_str)

        total_cost = sum(r.cost for r in responses if hasattr(r, "cost") and r.cost)
        total_tokens = sum(
            (r.usage.input_tokens or 0) + (r.usage.output_tokens or 0)
            for r in responses
            if hasattr(r, "usage") and r.usage is not None
        )

        now = time.time()

        conn = self._get_sync_conn(db_path)
        conn.execute(
            """INSERT OR REPLACE INTO responses
               (prompt_hash, params_hash, response_blob, schema_version, created_at, last_accessed, access_count, cost, total_tokens)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                prompt_hash,
                params.model_hash(),
                blob,
                SCHEMA_VERSION,
                now,
                now,
                total_cost,
                total_tokens,
            ),
        )
        conn.commit()
        self.stats.writes += 1

    # ──────────────────────────────────────────────
    # Moderation cache
    # ──────────────────────────────────────────────

    def _moderation_db_path(self) -> Path:
        return self.cache_dir / "moderation.sqlite"

    def get_moderation_file(self, texts: list[str]) -> tuple[Path, str]:
        hashes = [deterministic_hash(t) for t in texts]
        text_hash = deterministic_hash(" ".join(hashes))
        return self._moderation_db_path(), text_hash

    def maybe_load_moderation(self, texts: list[str]) -> LLMCacheModeration | None:
        db_path, text_hash = self.get_moderation_file(texts)
        conn = self._get_sync_conn(db_path)

        row = conn.execute(
            "SELECT response_blob, schema_version FROM moderation WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()

        if row is None:
            return None

        blob, version = row
        if version != SCHEMA_VERSION:
            return None

        json_str = _decompress(blob)
        return LLMCacheModeration.model_validate_json(json_str)

    def save_moderation(self, texts: list[str], moderation: list[TaggedModeration]):
        db_path, text_hash = self.get_moderation_file(texts)
        conn = self._get_sync_conn(db_path)

        cache_entry = LLMCacheModeration(texts=texts, moderation=moderation)
        json_str = cache_entry.model_dump_json()
        blob = _compress(json_str)

        conn.execute(
            "INSERT OR REPLACE INTO moderation (text_hash, response_blob, schema_version, created_at) VALUES (?, ?, ?, ?)",
            (text_hash, blob, SCHEMA_VERSION, time.time()),
        )
        conn.commit()

    # ──────────────────────────────────────────────
    # Embeddings cache
    # ──────────────────────────────────────────────

    def _embeddings_db_path(self) -> Path:
        return self.cache_dir / "embeddings.sqlite"

    def get_embeddings_file(self, params: EmbeddingParams) -> tuple[Path, str]:
        params_hash = params.model_hash()
        return self._embeddings_db_path(), params_hash

    def maybe_load_embeddings(self, params: EmbeddingParams) -> EmbeddingResponseBase64 | None:
        db_path, params_hash = self.get_embeddings_file(params)
        conn = self._get_sync_conn(db_path)

        row = conn.execute(
            "SELECT response_blob, schema_version FROM embeddings WHERE params_hash = ?",
            (params_hash,),
        ).fetchone()

        if row is None:
            return None

        blob, version = row
        if version != SCHEMA_VERSION:
            return None

        json_str = _decompress(blob)
        return EmbeddingResponseBase64.model_validate_json(json_str)

    def save_embeddings(self, params: EmbeddingParams, response: EmbeddingResponseBase64):
        db_path, params_hash = self.get_embeddings_file(params)
        conn = self._get_sync_conn(db_path)

        json_str = response.model_dump_json()
        blob = _compress(json_str)

        conn.execute(
            "INSERT OR REPLACE INTO embeddings (params_hash, response_blob, schema_version, created_at) VALUES (?, ?, ?, ?)",
            (params_hash, blob, SCHEMA_VERSION, time.time()),
        )
        conn.commit()

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    def close(self):
        """Close all cached database connections."""
        for conn in self._conn_cache.values():
            conn.close()
        self._conn_cache.clear()

    def __del__(self):
        self.close()

    # ──────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────

    def print_stats(self):
        LOGGER.info(self.stats.summary())

    def db_sizes(self) -> dict[str, float]:
        """Return size in MB of each SQLite database in the cache dir."""
        sizes = {}
        for f in self.cache_dir.glob("*.sqlite"):
            sizes[f.name] = f.stat().st_size / (1024 * 1024)
        return sizes

    def entry_count(self, params: LLMParams | None = None) -> int:
        """Count cached entries, optionally filtered by model."""
        if params is not None:
            db_path = _db_path_for_model(self.cache_dir, params)
            if not db_path.exists():
                return 0
            conn = self._get_sync_conn(db_path)
            row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
            return row[0]

        total = 0
        for db_file in self.cache_dir.glob("*.sqlite"):
            if db_file.name in ("moderation.sqlite", "embeddings.sqlite"):
                continue
            conn = self._get_sync_conn(db_file)
            row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
            total += row[0]
        return total
