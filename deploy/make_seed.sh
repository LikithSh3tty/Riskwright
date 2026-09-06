#!/usr/bin/env bash
# Produce deploy/postgres/riskwright.sql.gz from a locally loaded database.
#
# Run this on a machine that has already brought the normal stack up and let
# src/data/loader.py populate Postgres. The hosted instance never runs the
# loader: it streams 740MB of CSV through pandas, and previous_application
# alone is 1.7M rows. That belongs on a workstation, not a small cloud VM, and
# fetching the source files would mean putting Kaggle credentials on a public
# box.
#
# What ships is therefore the output of the verified loader rather than a
# second implementation of it.
#
#   bash deploy/make_seed.sh
#
# Writes deploy/postgres/riskwright.sql.gz, which is gitignored -- it is the
# dataset, and the submission instructions say the dataset does not go in git.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/deploy/postgres/riskwright.sql.gz"
CONTAINER="${CONTAINER:-riskwright-postgres}"
DB="${POSTGRES_DB:-riskwright}"
USER_NAME="${POSTGRES_USER:-riskwright}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "error: container '$CONTAINER' is not running." >&2
    echo "Bring the stack up first and let the loader finish:" >&2
    echo "    docker-compose up -d postgres loader && docker wait riskwright-loader" >&2
    exit 1
fi

# Refuse to dump an empty database. A seed image that boots cleanly and holds
# no rows is the worst outcome here: everything looks healthy and every answer
# is wrong.
echo "checking the source database is populated"
for table in application_train bureau previous_application; do
    count=$(docker exec "$CONTAINER" psql -tAX -U "$USER_NAME" -d "$DB" \
            -c "SELECT count(*) FROM ${table}")
    printf '  %-22s %s rows\n' "$table" "$count"
    if [ "$count" -lt 1 ]; then
        echo "error: ${table} is empty; run the loader before seeding." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "$OUT")"

# --no-owner and --no-privileges because the restore runs as whatever user the
# seeded image is configured with, and the grants are reapplied by
# 20-readonly-role.sh afterwards. Carrying the original ownership would make
# the restore fail on a role that does not exist yet.
echo "dumping ${DB}"
docker exec "$CONTAINER" pg_dump \
    --username "$USER_NAME" \
    --dbname "$DB" \
    --no-owner \
    --no-privileges \
    --format plain \
    | gzip -9 > "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "wrote $OUT ($SIZE)"
echo
echo "next:"
echo "  docker buildx build --platform linux/amd64 \\"
echo "      -f deploy/postgres/Dockerfile \\"
echo "      -t likithsh3tty/riskwright-postgres:v2.3 deploy/postgres"
