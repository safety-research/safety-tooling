"""One-shot migration: legacy per-params-hash .sqlite files → model-partitioned v2 files.

Background: schema v1 named each cache file by `params.model_hash()`, which
swept in every kwarg (seed, temperature, max_tokens, …). High-cardinality
kwargs (esp. cache-busting seeds) exploded the file count and triggered
"OperationalError: unable to open database file" during concurrent init.

Schema v2 partitions on `params.model` only, and uses composite PK
(prompt_hash, params_hash) so different params for the same prompt live as
separate rows in the same file.

This script walks legacy files, reads each row's embedded `LLMCache.params`
to determine the model, and re-inserts the row into the new v2 file. The
existing row contents (blob, version, timestamps, cost, tokens) are preserved
verbatim — no API re-calls.

Usage:
    python -m safetytooling.apis.inference.migrate_sqlite_cache <cache_dir>
    python -m safetytooling.apis.inference.migrate_sqlite_cache .cache/api --dry-run

Idempotent: legacy files are deleted only after successful migration; re-runs
will find no legacy files and exit cleanly. Migrated entries use
INSERT OR IGNORE so re-runs don't overwrite newer cached values.
"""

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

from safetytooling.data_models import LLMCache

from .sqlite_cache_manager import (
    SCHEMA_VERSION,
    _db_path_for_model,
    _decompress,
    _init_db_sync,
)

LOGGER = logging.getLogger(__name__)


_RESERVED_NAMES = {"moderation.sqlite", "embeddings.sqlite"}


def _is_legacy(path: Path) -> bool:
    """v2 files are named `model_*.sqlite`; everything else (besides reserved
    moderation/embeddings) is legacy v1."""
    return (
        path.suffix == ".sqlite"
        and not path.name.startswith("model_")
        and path.name not in _RESERVED_NAMES
    )


def _read_legacy_rows(legacy_path: Path) -> list[tuple]:
    """Return all rows from a legacy .sqlite file.

    Returns tuples of (prompt_hash, params_hash, blob, version, created_at,
    last_accessed, access_count, cost, total_tokens).
    """
    conn = sqlite3.connect(str(legacy_path))
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.execute(
            "SELECT prompt_hash, params_hash, response_blob, schema_version, "
            "created_at, last_accessed, access_count, cost, total_tokens "
            "FROM responses"
        )
        return cursor.fetchall()
    finally:
        conn.close()


def _model_for_row(blob: bytes) -> str | None:
    """Extract `params.model` from a row's compressed response_blob.

    Returns None if the blob can't be decoded — these rows are skipped.
    """
    try:
        json_str = _decompress(blob)
        cache_entry = LLMCache.model_validate_json(json_str)
        return cache_entry.params.model
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(f"  skipping unreadable row: {type(e).__name__}: {e}")
        return None


def _insert_row(conn: sqlite3.Connection, row: tuple) -> bool:
    """Insert one legacy row. Uses INSERT OR IGNORE so re-runs / overlapping
    rows from multiple legacy files don't clobber newer entries. Returns True
    if a row was inserted, False if it was a no-op duplicate."""
    cursor = conn.execute(
        """INSERT OR IGNORE INTO responses
           (prompt_hash, params_hash, response_blob, schema_version,
            created_at, last_accessed, access_count, cost, total_tokens)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        row,
    )
    return cursor.rowcount > 0


def _delete_legacy(path: Path) -> None:
    """Delete a legacy .sqlite file and its WAL/SHM sidecars."""
    for suffix in (".sqlite", ".sqlite-wal", ".sqlite-shm"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def migrate_cache_dir(cache_dir: Path, *, dry_run: bool = False) -> dict:
    """Migrate all legacy files in cache_dir to v2 model-partitioned files.

    Returns a stats dict: legacy_files, migrated_rows, dropped_rows, new_files.
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")

    legacy_files = sorted(p for p in cache_dir.iterdir() if _is_legacy(p))
    LOGGER.info(f"Found {len(legacy_files)} legacy .sqlite files in {cache_dir}")
    if dry_run:
        LOGGER.info("DRY RUN — no writes will be made")

    new_conns: dict[Path, sqlite3.Connection] = {}
    stats = {
        "legacy_files": len(legacy_files),
        "migrated_rows": 0,
        "dropped_rows": 0,
        "skipped_unreadable": 0,
        "new_files": set(),
        "start_time": time.time(),
    }

    log_every = max(1, len(legacy_files) // 20)

    for i, legacy in enumerate(legacy_files):
        try:
            rows = _read_legacy_rows(legacy)
        except sqlite3.DatabaseError as e:
            LOGGER.warning(f"  skipping {legacy.name}: {type(e).__name__}: {e}")
            stats["skipped_unreadable"] += 1
            continue

        if not rows:
            # Empty legacy file — just delete it (along with any sidecars).
            if not dry_run:
                _delete_legacy(legacy)
            continue

        rows_inserted_for_file = 0
        for row in rows:
            blob = row[2]
            model = _model_for_row(blob)
            if model is None:
                stats["dropped_rows"] += 1
                continue

            # Build a dummy LLMParams to compute the new path — only model matters
            # for partitioning, so we synthesize one with just the model field.
            new_path = cache_dir / f"model_{_slug(model)}.sqlite"

            if dry_run:
                stats["migrated_rows"] += 1
                stats["new_files"].add(new_path.name)
                rows_inserted_for_file += 1
                continue

            conn = new_conns.get(new_path)
            if conn is None:
                _init_db_sync(new_path)
                conn = sqlite3.connect(str(new_path))
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=5000")
                new_conns[new_path] = conn
                stats["new_files"].add(new_path.name)

            if _insert_row(conn, row):
                stats["migrated_rows"] += 1
                rows_inserted_for_file += 1
            else:
                stats["dropped_rows"] += 1  # PK collision — keep existing

        if not dry_run and rows_inserted_for_file > 0:
            # Commit per-file so partial progress survives a crash
            new_conns[new_path].commit()

        if not dry_run:
            _delete_legacy(legacy)

        if (i + 1) % log_every == 0 or i + 1 == len(legacy_files):
            elapsed = time.time() - stats["start_time"]
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            LOGGER.info(
                f"  [{i + 1}/{len(legacy_files)}] files processed "
                f"({rate:.1f}/s, {stats['migrated_rows']} rows migrated)"
            )

    # Final commit + close
    if not dry_run:
        for conn in new_conns.values():
            conn.commit()
            conn.close()

    stats["new_files"] = sorted(stats["new_files"])
    stats["elapsed"] = time.time() - stats["start_time"]
    return stats


# Minimal model-name slugging — mirrors _slug_for_partition in sqlite_cache_manager
# without taking a direct LLMParams dep (we only have a string here).
_FS_UNSAFE = str.maketrans({c: "_" for c in '/\\:*?"<>| \t'})


def _slug(model: str) -> str:
    return model.translate(_FS_UNSAFE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    LOGGER.info(f"Schema version (target): v{SCHEMA_VERSION}")
    stats = migrate_cache_dir(args.cache_dir, dry_run=args.dry_run)
    LOGGER.info("=" * 60)
    LOGGER.info(f"  Legacy files processed: {stats['legacy_files']}")
    LOGGER.info(f"  Rows migrated:          {stats['migrated_rows']}")
    LOGGER.info(f"  Rows dropped (dup/bad): {stats['dropped_rows']}")
    LOGGER.info(f"  Files skipped (corrupt): {stats['skipped_unreadable']}")
    LOGGER.info(f"  New v2 files created:    {len(stats['new_files'])}")
    LOGGER.info(f"  Elapsed:                {stats['elapsed']:.1f}s")
    for name in stats["new_files"]:
        LOGGER.info(f"    → {name}")


if __name__ == "__main__":
    main()
