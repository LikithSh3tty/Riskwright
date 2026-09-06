# Postgres with the Home Credit data already in it.
#
# The hosted instance does not run the loader. src/data/loader.py streams three
# CSVs totalling 740MB through pandas -- 1.7M rows for previous_application
# alone -- and that memory spike is not something to inflict on a small cloud
# VM. It would also mean putting Kaggle credentials on a public box to fetch
# the source files.
#
# So the data is loaded once locally, by the same loader the clean-clone test
# verifies, dumped, and baked in here. What ships is the *output* of the
# verified path rather than a second implementation of it.
#
# Build from the repository root:
#   docker buildx build --platform linux/amd64 \
#       -f deploy/postgres/Dockerfile -t likithsh3tty/riskwright-postgres:v2.3 \
#       deploy/postgres
#
# The build context is deploy/postgres/ and must contain riskwright.sql.gz,
# produced by deploy/make_seed.sh. That file is gitignored: it is the dataset.

FROM postgres:16-alpine

# postgres:16-alpine runs everything in /docker-entrypoint-initdb.d exactly
# once, on first boot with an empty data directory, in filename order. The
# numeric prefixes are the ordering: the dump must restore before the role can
# be granted SELECT on tables that do not exist yet.
#
# .sql.gz is decompressed and piped to psql by the entrypoint itself, so the
# dump needs no wrapper script.
COPY riskwright.sql.gz /docker-entrypoint-initdb.d/10-riskwright.sql.gz
COPY 20-readonly-role.sh /docker-entrypoint-initdb.d/20-readonly-role.sh

# The entrypoint only executes .sh files that are readable; make the bit
# explicit rather than depending on however the host filesystem happened to
# mark it. Windows checkouts do not carry a useful mode.
RUN chmod +x /docker-entrypoint-initdb.d/20-readonly-role.sh

# Documented for whoever reads `docker inspect`. The compose file sets the
# real values; nothing secret is baked in here.
ENV POSTGRES_DB=riskwright \
    POSTGRES_USER=riskwright

EXPOSE 5432
