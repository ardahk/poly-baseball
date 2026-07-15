#!/usr/bin/env bash
# Daily WAL-safe backup of polybot.db + artifacts/.
#
# Keeps exactly TWO rotating archives (latest + previous) — the DB is
# cumulative, so the newest archive contains all history. manifest.jsonl is
# append-only and never rotated: it is the permanent record of what each
# day's snapshot contained even after its archive is deleted.
#
# Never `cp` the live db files: the 2026-07-11 corruption incident came from
# a naive 3-file copy of an open WAL database. `.backup` uses SQLite's online
# backup API and is safe against the running polybot.service writer.
set -euo pipefail

DB=/home/ubuntu/poly-baseball/polybot.db
SRC_DIR=/home/ubuntu/poly-baseball
DEST=/home/ubuntu/backups/polybot
STAMP=$(date -u +%Y%m%d-%H%M%S)
KEEP=2

mkdir -p "$DEST"
TMP=$(mktemp -d "$DEST/tmp.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

sqlite3 "$DB" ".backup '$TMP/polybot.db'"

INTEGRITY=$(sqlite3 "$TMP/polybot.db" 'PRAGMA integrity_check;')
if [ "$INTEGRITY" != "ok" ]; then
    echo "integrity_check FAILED: $INTEGRITY" >&2
    exit 1
fi

ROWS=$(sqlite3 -json "$TMP/polybot.db" "SELECT
    (SELECT COUNT(*) FROM price_ticks)            AS price_ticks,
    (SELECT COUNT(*) FROM decisions)              AS decisions,
    (SELECT COUNT(*) FROM trades)                 AS trades,
    (SELECT COUNT(*) FROM game_states)            AS game_states,
    (SELECT COUNT(*) FROM signals)                AS signals,
    (SELECT COUNT(*) FROM signal_counterfactuals) AS signal_counterfactuals,
    (SELECT COUNT(*) FROM equity)                 AS equity,
    (SELECT COUNT(*) FROM model_observations)     AS model_observations" \
    | sed 's/^\[//; s/\]$//')
MAX_TICK_TS=$(sqlite3 "$TMP/polybot.db" 'SELECT COALESCE(MAX(ts), 0) FROM price_ticks;')
DB_BYTES=$(stat -c %s "$TMP/polybot.db")

ARCHIVE="$DEST/polybot-backup-$STAMP.tar.zst"
tar -cf - -C "$TMP" polybot.db -C "$SRC_DIR" artifacts \
    | zstd -T0 -12 -q -o "$ARCHIVE"

zstd -t -q "$ARCHIVE"

SHA=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
SIZE=$(stat -c %s "$ARCHIVE")
printf '{"date":"%s","file":"%s","sha256":"%s","size_bytes":%s,"integrity":"ok","db_bytes":%s,"max_price_tick_ts":%s,"rows":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "$ARCHIVE")" "$SHA" "$SIZE" \
    "$DB_BYTES" "$MAX_TICK_TS" "$ROWS" >> "$DEST/manifest.jsonl"

# Rotate only after the new archive verified.
ls -1t "$DEST"/polybot-backup-*.tar.zst | tail -n +$((KEEP + 1)) | xargs -r rm --

echo "backup ok: $ARCHIVE ($SIZE bytes, sha256 $SHA)"
