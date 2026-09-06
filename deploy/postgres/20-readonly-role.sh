#!/bin/sh
# Recreate the chatbot's read-only role inside the seeded image.
#
# pg_dump does not carry roles. They are cluster-level objects rather than
# database objects, so a plain dump restores the three tables and silently
# leaves riskwright_ro behind. That role is not incidental: the README states
# that the chatbot connects as a principal holding SELECT on exactly three
# tables and nothing else, and that even a bypassed SQL validation gate cannot
# write. A deployment missing the role either breaks the chatbot or -- worse --
# quietly runs it as the owner, which makes the security claim false while
# everything still appears to work.
#
# So the grants are reproduced here, matching loader.ensure_readonly_role
# statement for statement. Ordered after 10-riskwright.sql.gz because the
# GRANTs reference tables the dump creates.
#
# The password arrives from the environment at container start and is never
# baked into the image. Unset is a hard failure rather than a blank password:
# a silently passwordless role is the failure mode worth preventing.

set -eu

ROLE="${POSTGRES_RO_USER:-riskwright_ro}"
PASSWORD="${POSTGRES_RO_PASSWORD:?POSTGRES_RO_PASSWORD must be set; refusing to create a passwordless read-only role}"

echo "provisioning read-only role ${ROLE}"

# psql's own quoting, not string concatenation: :'x' interpolates as a quoted
# literal and :"x" as a quoted identifier. That is the shell equivalent of the
# parameterised psycopg2.sql approach the loader uses, and it is why the
# password never appears as bare SQL text.
psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v role="$ROLE" \
     -v ro_password="$PASSWORD" \
     -v dbname="$POSTGRES_DB" <<'SQL'

SELECT NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'role') AS need_create
\gset

\if :need_create
CREATE ROLE :"role" WITH LOGIN PASSWORD :'ro_password';
\else
ALTER ROLE :"role" WITH LOGIN PASSWORD :'ro_password';
\endif

-- Exactly SELECT, on exactly the three whitelisted tables.
GRANT CONNECT ON DATABASE :"dbname" TO :"role";
GRANT USAGE ON SCHEMA public TO :"role";
GRANT SELECT ON application_train TO :"role";
GRANT SELECT ON bureau TO :"role";
GRANT SELECT ON previous_application TO :"role";

-- Anything created later stays out of reach unless granted explicitly.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM :"role";

SQL

# Prove it rather than assume it. These are the same properties the local stack
# is checked against, and a seeded image that quietly lost them would be worse
# than one that failed to build.
echo "verifying ${ROLE} grants"
GRANTED=$(psql -tAX --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "SELECT count(*) FROM information_schema.table_privileges
        WHERE grantee = '${ROLE}' AND privilege_type = 'SELECT'")
WRITES=$(psql -tAX --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "SELECT count(*) FROM information_schema.table_privileges
        WHERE grantee = '${ROLE}' AND privilege_type <> 'SELECT'")

if [ "$GRANTED" != "3" ]; then
    echo "FATAL: expected 3 SELECT grants for ${ROLE}, found ${GRANTED}" >&2
    exit 1
fi
if [ "$WRITES" != "0" ]; then
    echo "FATAL: ${ROLE} holds ${WRITES} non-SELECT privileges" >&2
    exit 1
fi

echo "read-only role ${ROLE} ready: ${GRANTED} SELECT grants, no write privileges"
