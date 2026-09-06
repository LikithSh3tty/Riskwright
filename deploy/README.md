# Deploying Riskwright

Standing the platform up on a fresh Ubuntu VM. Three published images are
pulled and started; nothing is built on the host and no dataset is downloaded
to it.

## What this needs

| Resource | Minimum | Why |
|----------|---------|-----|
| RAM | **4 GB** | Postgres holds ~3.7M rows across three tables; the API loads a LightGBM model and a SHAP explainer into memory. 2 GB will start and then be killed under a real query. |
| vCPU | **2** | SHAP on a single prediction is CPU-bound. One core works and feels slow. |
| Disk | **40 GB** | The seeded Postgres image is the bulk of it, plus the restored data directory, which is a second copy. |
| Ports | **80** inbound | The only published port. |

The first boot restores the seeded dump before Postgres accepts connections.
Expect a few minutes before the stack reports healthy. Every later start skips
the restore.

## 1. Install Docker

Ubuntu 22.04 or 24.04, from Docker's own repository rather than the distro's,
which ships an older engine without the compose plugin:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
```

Let your user run Docker without `sudo`. The group change only applies to a new
login, so log out and back in:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

## 2. Fetch the deployment files

Only two files are needed on the host. Cloning the whole repository also works
and costs nothing:

```bash
mkdir -p ~/riskwright && cd ~/riskwright
curl -fsSLO https://raw.githubusercontent.com/LikithSh3tty/Riskwright/main/deploy/docker-compose.deploy.yml
curl -fsSLO https://raw.githubusercontent.com/LikithSh3tty/Riskwright/main/deploy/.env.example
```

## 3. Configure

```bash
cp .env.example .env
nano .env
```

Set at minimum:

| Variable | Value |
|----------|-------|
| `POSTGRES_PASSWORD` | Any value |
| `POSTGRES_RO_PASSWORD` | Any value, different from the above |
| `ANTHROPIC_API_KEY` | Your key. Needed for the chatbot only; everything else runs without it. |

`POSTGRES_RO_PASSWORD` is read on the database's **first boot** to recreate the
chatbot's read-only role inside the seeded image. Setting it later has no
effect, because the init scripts run once against an empty data directory.

Keep `.env` off the internet. It is not in the repository and not in any image.

## 4. Start

```bash
docker compose -f docker-compose.deploy.yml pull
docker compose -f docker-compose.deploy.yml up -d
```

## 5. Open port 80

Two places, and both matter. The provider's security group is the one people
forget:

```bash
# On the host
sudo ufw allow 80/tcp
sudo ufw allow OpenSSH        # do this before enabling, or you lock yourself out
sudo ufw enable
```

Then allow inbound TCP 80 in the provider's console — AWS security group, GCP
firewall rule, Azure NSG, DigitalOcean cloud firewall. A host that answers
`curl localhost` while the public address times out is almost always this.

## Checking it worked

```bash
docker compose -f docker-compose.deploy.yml ps
```

All three should report `healthy`:

```
NAME                   STATUS
riskwright-postgres    Up 3 minutes (healthy)
riskwright-api         Up 2 minutes (healthy)
riskwright-frontend    Up 2 minutes (healthy)
```

Then, from the host:

```bash
curl -s localhost/api/health
```

Expect `"status": "ok"` with `"database": {"connected": true}` and
`"model_loaded": true`. `"llm_configured": false` means the key is missing —
everything except the chatbot still works.

Confirm the seeded data actually arrived, rather than assuming a healthy
container means a populated one:

```bash
docker compose -f docker-compose.deploy.yml exec postgres \
    psql -U riskwright -d riskwright -c \
    "SELECT count(*) FROM application_train"     # expect 307511
```

And that the read-only role survived the seeding:

```bash
docker compose -f docker-compose.deploy.yml exec postgres \
    psql -U riskwright -d riskwright -c \
    "SELECT grantee, privilege_type, table_name FROM information_schema.table_privileges
     WHERE grantee = 'riskwright_ro'"
```

Expect exactly three rows, all `SELECT`. Anything else means the security
property the README claims is not true of this deployment.

Finally, open `http://<your-host>/` and check all five sections load.

## Logs

```bash
docker compose -f docker-compose.deploy.yml logs -f            # everything
docker compose -f docker-compose.deploy.yml logs -f api        # one service
docker compose -f docker-compose.deploy.yml logs --tail 100 postgres
```

The database restore is only visible in the postgres log on the very first
boot.

## Restarting and updating

```bash
docker compose -f docker-compose.deploy.yml restart            # no image change
docker compose -f docker-compose.deploy.yml restart api        # one service

docker compose -f docker-compose.deploy.yml pull               # new images
docker compose -f docker-compose.deploy.yml up -d
```

`up -d` after a pull recreates only the containers whose image changed. The
`pgdata` volume survives, so the database is not re-seeded.

To start completely fresh, including re-running the seed:

```bash
docker compose -f docker-compose.deploy.yml down -v
docker compose -f docker-compose.deploy.yml up -d
```

`-v` removes the volume and therefore the database. It is the only way to
re-run the init scripts, and the only way to change the read-only role's
password without an `ALTER ROLE`.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `postgres` unhealthy for several minutes on first boot | Normal. It is restoring the dump and does not accept connections until finished. Watch `logs postgres`. |
| `api` restarting | Almost always Postgres not ready, or a wrong `POSTGRES_PASSWORD`. Check `logs api`. |
| Site loads, chatbot refuses everything | `ANTHROPIC_API_KEY` unset. `curl -s localhost/api/health` shows `llm_configured`. |
| `curl localhost` works, public address does not | Provider firewall. See step 5. |
| Container killed, or OOM in `dmesg` | Under 4 GB of RAM. |

## What differs from the repository stack

Two things, and no third:

- Images are **pulled**, not built.
- There is **no loader service**. The data ships inside the postgres image,
  seeded from a dump of a local run of that same loader.

The healthchecks, dependency conditions, restart policy, internal network
layout, and the rule that only the frontend is published are all identical to
the verified `docker-compose.yml`.
