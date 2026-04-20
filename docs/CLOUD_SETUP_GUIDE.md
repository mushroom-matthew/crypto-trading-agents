# Cloud Setup Guide — Sysadmin Reference

**Last updated:** 2026-04-20
**Covers:** Phase 0 (Temporal on VPS — recommended), Phase 0 Alt (Temporal Cloud — $100/mo), Phase 1 (Fly.io worker), Phase 2 (AWS ECS Fargate)
**Related:** `docs/AWS_DEPLOYMENT_SCOPE.md` (architecture + cost estimates), `docs/branching/07-aws-deploy.md` (runbook)

---

## Which Phase 0 option should I use?

| Option | Cost | Uptime | Best for |
|--------|------|--------|----------|
| **VPS-hosted Temporal** (recommended) | ~$4/mo | VPS SLA (~99.9%) | Paper trading |
| **Temporal Cloud** | ~$100/mo | 99.99% multi-region | Live trading, production |

For paper trading, a $4/month Hetzner VPS running the same `temporalio/auto-setup` container you use locally is the right call. You only need Temporal Cloud if you're running live money and need enterprise SLAs.

---

## Phase 0 (Recommended) — Self-hosted Temporal on a VPS (~45 min)

**Goal:** Temporal server runs on a cheap always-on VPS. WSL2 crashing no longer kills your sessions.

### Step 1: Provision a VPS

**Hetzner** is the cheapest option with good performance:

1. Create account at [hetzner.com/cloud](https://www.hetzner.com/cloud)
2. New Project → **Add Server**
   - Location: Ashburn (US East) or Hillsboro (US West)
   - Image: **Ubuntu 24.04**
   - Type: **CX22** (2 vCPU, 4GB RAM) — $4.15/mo
   - SSH key: paste your WSL2 public key (`cat ~/.ssh/id_rsa.pub`)
   - Name: `temporal-server`
3. Click **Create & Buy Now**
4. Note the server's public IP address

> Alternatives: DigitalOcean Droplet ($6/mo), Linode Nanode ($5/mo), AWS EC2 t3.micro (~$8/mo). All work the same way.

### Step 2: Install Docker on the VPS

```bash
# SSH into your new server
ssh root@<your-vps-ip>

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Verify
docker --version
```

### Step 3: Start Temporal on the VPS

```bash
# Still on the VPS — create a docker-compose for Temporal only
mkdir -p /opt/temporal && cat > /opt/temporal/docker-compose.yml << 'EOF'
services:
  temporal-db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=temporal
      - POSTGRES_PASSWORD=temporal
      - POSTGRES_DB=temporal
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U temporal"]
      interval: 10s
      retries: 10

  temporal:
    image: temporalio/auto-setup:1.24
    environment:
      - DB=postgres12
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=temporal-db
    depends_on:
      temporal-db:
        condition: service_healthy
    ports:
      - "7233:7233"
    restart: unless-stopped

  temporal-ui:
    image: temporalio/ui:latest
    environment:
      - TEMPORAL_ADDRESS=temporal:7233
    ports:
      - "8088:8080"
    depends_on:
      - temporal
    restart: unless-stopped

volumes:
  pgdata:
EOF

cd /opt/temporal && docker compose up -d
docker compose ps   # all three should be healthy within ~60 seconds
```

### Step 4: Open firewall port 7233

On Hetzner: Project → **Firewalls** → **Create Firewall**
- Add inbound rule: TCP port **7233**, source **your home IP** (not 0.0.0.0/0 — don't expose gRPC publicly)
- Apply to `temporal-server`

To find your home IP: `curl ifconfig.me`

### Step 5: Configure your `.env`

```bash
# Replace the Temporal block in .env:
TEMPORAL_ADDRESS=<your-vps-ip>:7233
TEMPORAL_NAMESPACE=default
# Leave TEMPORAL_TLS_CERT and TEMPORAL_TLS_KEY unset — plain gRPC to your own server is fine
```

### Step 6: Start your local stack (no local temporal)

```bash
# No --profile local — don't start local temporal containers
docker compose up db app ops-api worker
```

Worker logs should show:
```
Connecting to Temporal at <your-vps-ip>:7233 (ns=default)
Temporal client ready
```

### Step 7: Smoke test

```bash
# Start a session, kill everything local, restart worker only
curl -X POST http://localhost:8081/paper-trading/sessions \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC-USD"], "timeframe": "5m"}'

docker compose down   # kill local stack — Temporal on VPS keeps running

docker compose up db worker   # restart just the worker

curl http://localhost:8081/paper-trading/sessions/<session_id>
# cycle count should be incrementing — session survived
```

### Keep Temporal updated on the VPS

```bash
ssh root@<your-vps-ip>
cd /opt/temporal
docker compose pull && docker compose up -d
```

---

## Phase 0 (Alternative) — Temporal Cloud (~30 min, ~$100/mo)

**Only use this if you need enterprise SLAs for live trading.** For paper trading the VPS approach above is identical in practice at 1/25th the cost.

### Step 1: Create a Temporal Cloud account

1. Go to [cloud.temporal.io](https://cloud.temporal.io) → Sign up (Google or GitHub)
2. Complete email verification

### Step 2: Create a namespace

1. Left nav → **Namespaces** → **Create Namespace**
2. Name: `crypto-trading` (or anything you like)
3. Region: `us-east-1` or `us-west-2` (pick whichever is closest to you)
4. Retention: 30 days (default is fine)
5. Click **Create**
6. Note the full namespace ID shown — it will look like `crypto-trading.a1b2c3d4`

### Step 3: Generate mTLS certificates

Temporal Cloud requires mutual TLS. Generate a CA cert and a client cert/key pair on your WSL2 machine, then upload the CA to Temporal.

```bash
mkdir -p ~/temporal-certs && cd ~/temporal-certs

# 1. CA private key
openssl genrsa -out ca.key 4096

# 2. Self-signed CA cert (10-year validity)
openssl req -new -x509 -days 3650 -key ca.key -out ca.pem \
  -subj "/CN=temporal-ca/O=crypto-trading"

# 3. Client private key
openssl genrsa -out client.key 4096

# 4. Client CSR
openssl req -new -key client.key -out client.csr \
  -subj "/CN=worker/O=crypto-trading"

# 5. Sign the client cert with your CA
openssl x509 -req -days 3650 -in client.csr \
  -CA ca.pem -CAkey ca.key -CAcreateserial \
  -out client.pem

# Verify
ls ~/temporal-certs/
# ca.key  ca.pem  client.csr  client.key  client.pem  ca.srl
```

### Step 4: Upload CA cert to Temporal Cloud

1. Temporal Cloud → your namespace → **CA Certificates** tab
2. Click **Add Certificate**
3. Paste the full contents of `~/temporal-certs/ca.pem`
4. Click **Add**

### Step 5: Configure your `.env`

Open `.env` and replace the Temporal block:

```bash
# Replace these local-dev values...
# TEMPORAL_ADDRESS=temporal:7233
# TEMPORAL_NAMESPACE=default

# ...with your Temporal Cloud values:
TEMPORAL_ADDRESS=crypto-trading.a1b2c3d4.tmprl.cloud:7233
TEMPORAL_NAMESPACE=crypto-trading.a1b2c3d4
TEMPORAL_TLS_CERT=/home/<your-linux-user>/temporal-certs/client.pem
TEMPORAL_TLS_KEY=/home/<your-linux-user>/temporal-certs/client.key
```

### Step 6: Start the stack (no local Temporal)

```bash
# Note: --profile local is NOT included.
# The temporal and temporal-ui containers will not start.
# Your worker will connect directly to Temporal Cloud.
docker compose up db app ops-api worker
```

Watch the worker logs for:
```
Connecting to Temporal Cloud at crypto-trading.a1b2c3d4.tmprl.cloud:7233 (ns=...) with mTLS
Temporal client ready
```

### Step 7: Smoke test crash resilience

```bash
# 1. Start a paper session
curl -X POST http://localhost:8081/paper-trading/sessions \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC-USD"], "timeframe": "5m"}'

# Note the session_id from the response.

# 2. Kill everything
docker compose down

# 3. Restart only the worker (no local temporal needed)
docker compose up db worker

# 4. Confirm the session is still running at the same cycle count
curl http://localhost:8081/paper-trading/sessions/<session_id>
```

### For local dev going forward

```bash
# Start WITH the self-hosted Temporal containers (local dev only):
docker compose --profile local up
```

---

## Phase 1 — Fly.io Remote Worker (~1 hour)

**Goal:** Worker process survives local machine shutdown. Sessions keep evaluating triggers even when your laptop is off.

### Step 1: Install Fly CLI and log in

```bash
curl -L https://fly.io/install.sh | sh
# Add to PATH if prompted, then:
fly auth login
```

### Step 2: Add cert-content env var support

Fly.io (and later ECS) inject secrets as environment variables, not files. Update `_build_tls_config()` in `agents/temporal_utils.py` to support inline cert content:

```python
def _build_tls_config() -> Optional[TLSConfig]:
    # Inline content (Fly.io / ECS — no filesystem access to cert files)
    cert_content = os.environ.get("TEMPORAL_TLS_CERT_CONTENT")
    key_content  = os.environ.get("TEMPORAL_TLS_KEY_CONTENT")
    if cert_content and key_content:
        return TLSConfig(
            client_cert=cert_content.encode(),
            client_private_key=key_content.encode(),
        )

    # File paths (local dev)
    cert_path = os.environ.get("TEMPORAL_TLS_CERT")
    key_path  = os.environ.get("TEMPORAL_TLS_KEY")
    if not (cert_path and key_path):
        return None
    with open(cert_path, "rb") as fh:
        client_cert = fh.read()
    with open(key_path, "rb") as fh:
        client_private_key = fh.read()
    return TLSConfig(client_cert=client_cert, client_private_key=client_private_key)
```

### Step 3: Create `fly.toml` in repo root

```toml
app = "crypto-trading-worker"
primary_region = "iad"   # us-east-1 equivalent — match your Temporal Cloud region

[build]
  dockerfile = "Dockerfile"

[env]
  TASK_QUEUE       = "mcp-tools"
  TEMPORAL_NAMESPACE = "crypto-trading.a1b2c3d4"

[processes]
  worker = "uv run python -m worker.main"

[deploy]
  strategy = "immediate"
```

### Step 4: Store secrets in Fly

```bash
fly secrets set \
  TEMPORAL_ADDRESS="crypto-trading.a1b2c3d4.tmprl.cloud:7233" \
  TEMPORAL_NAMESPACE="crypto-trading.a1b2c3d4" \
  OPENAI_API_KEY="sk-..." \
  LANGFUSE_SECRET_KEY="..." \
  LANGFUSE_PUBLIC_KEY="..." \
  COINBASEEXCHANGE_API_KEY="..." \
  COINBASEEXCHANGE_SECRET="..." \
  POSTGRES_HOST="your-db-host" \
  POSTGRES_PORT="5432" \
  POSTGRES_DB="botdb" \
  POSTGRES_USER="botuser" \
  POSTGRES_PASSWORD="..."

# Cert content (not file paths — Fly has no access to your local filesystem)
fly secrets set \
  TEMPORAL_TLS_CERT_CONTENT="$(cat ~/temporal-certs/client.pem)" \
  TEMPORAL_TLS_KEY_CONTENT="$(cat ~/temporal-certs/client.key)"
```

### Step 5: Deploy

```bash
fly apps create crypto-trading-worker
fly deploy
fly logs   # watch for "Temporal client ready"
```

### Step 6: Verify crash resilience

```bash
# Scale worker to 0 (simulate crash)
fly scale count worker=0

# Wait 30 seconds, then restart
fly scale count worker=1

# Confirm session cycle count is still incrementing
curl http://localhost:8081/paper-trading/sessions/<session_id>
```

---

## Phase 2 — AWS ECS Fargate (~1–2 days)

**Goal:** All three services (worker, ops-api, mcp-server) running in AWS. No local dependency at all.

### Step 1: Install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
aws --version
```

### Step 2: Install Terraform

```bash
wget https://releases.hashicorp.com/terraform/1.7.5/terraform_1.7.5_linux_amd64.zip
unzip terraform_1.7.5_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform version
```

### Step 3: Create an IAM user for deployments

**In AWS Console** (don't use your root account for day-to-day ops):

1. IAM → Users → **Create user**
2. Name: `crypto-trading-deploy`
3. Attach these managed policies:
   - `AmazonECS_FullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
   - `SecretsManagerReadWrite`
   - `CloudWatchFullAccess`
   - `IAMFullAccess` (needed for Terraform to create task roles)
4. Create an **Access Key** for CLI use → save it

```bash
aws configure
# AWS Access Key ID: <from step 4 above>
# AWS Secret Access Key: <from step 4 above>
# Default region: us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

### Step 4: Create ECR repositories

```bash
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

aws ecr create-repository --repository-name crypto-trading-worker    --region $REGION
aws ecr create-repository --repository-name crypto-trading-ops-api   --region $REGION
aws ecr create-repository --repository-name crypto-trading-mcp-server --region $REGION

# Log Docker into ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS \
  --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
```

### Step 5: Store secrets in AWS Secrets Manager

```bash
# Temporal Cloud connection
aws secretsmanager create-secret \
  --name trading/temporal \
  --region us-east-1 \
  --secret-string "{
    \"TEMPORAL_ADDRESS\": \"crypto-trading.a1b2c3d4.tmprl.cloud:7233\",
    \"TEMPORAL_NAMESPACE\": \"crypto-trading.a1b2c3d4\",
    \"TEMPORAL_TLS_CERT_CONTENT\": \"$(cat ~/temporal-certs/client.pem | sed 's/$/\\n/' | tr -d '\n')\",
    \"TEMPORAL_TLS_KEY_CONTENT\": \"$(cat ~/temporal-certs/client.key | sed 's/$/\\n/' | tr -d '\n')\"
  }"

# LLM
aws secretsmanager create-secret \
  --name trading/openai \
  --region us-east-1 \
  --secret-string '{"OPENAI_API_KEY": "sk-..."}'

# Exchange
aws secretsmanager create-secret \
  --name trading/coinbase \
  --region us-east-1 \
  --secret-string '{
    "COINBASEEXCHANGE_API_KEY": "...",
    "COINBASEEXCHANGE_SECRET": "..."
  }'

# Observability
aws secretsmanager create-secret \
  --name trading/langfuse \
  --region us-east-1 \
  --secret-string '{
    "LANGFUSE_SECRET_KEY": "...",
    "LANGFUSE_PUBLIC_KEY": "..."
  }'
```

### Step 6: Build and push images

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

docker build -t crypto-trading .

# Tag and push each service image
docker tag crypto-trading ${ECR}/crypto-trading-worker:latest
docker push ${ECR}/crypto-trading-worker:latest

docker tag crypto-trading ${ECR}/crypto-trading-ops-api:latest
docker push ${ECR}/crypto-trading-ops-api:latest

docker tag crypto-trading ${ECR}/crypto-trading-mcp-server:latest
docker push ${ECR}/crypto-trading-mcp-server:latest
```

### Step 7: Run Terraform

```bash
cd infra/terraform
terraform init
terraform validate
terraform plan    # read this output carefully before applying
                  # paste it into the runbook Test Evidence section
terraform apply   # only after reviewing the plan
```

The Terraform will provision: VPC, public/private subnets, NAT gateway, ECS cluster, 3 Fargate task definitions, 2 ALBs (ops-api + mcp-server), IAM task execution roles, CloudWatch log groups, Secrets Manager access policies.

After `apply` completes, Terraform outputs will include the ALB DNS names for ops-api and mcp-server.

### Step 8: Update React UI to point at cloud ops-api

In `ui/.env` (or wherever the ops-api URL is configured):

```bash
VITE_OPS_API_URL=https://<ops-api-alb-dns>.us-east-1.elb.amazonaws.com
```

---

## Cost Reference

| Phase | Temporal option | What you get | Monthly cost |
|-------|----------------|-------------|-------------|
| Phase 0 only | VPS (~$4) | Workflows survive crashes; worker still local | ~$4 |
| Phase 0 only | Temporal Cloud | Workflows survive crashes; worker still local | ~$100 |
| Phase 0 + 1 | VPS + Fly.io | Worker survives machine shutdown | ~$6 |
| Phase 0 + 1 | Cloud + Fly.io | Worker survives machine shutdown | ~$102 |
| Phase 0 + 1 + 2 | VPS + ECS | Full cloud; no local dependency | ~$90 |
| Phase 0 + 1 + 2 | Cloud + ECS | Full cloud; no local dependency | ~$186 |
| Phase 3+4 (live trading) | Cloud recommended | + RDS Multi-AZ, CI/CD, PagerDuty | ~+$110 |

---

## Credential Storage Cheat Sheet

| Secret | Local dev | Fly.io | ECS |
|--------|-----------|--------|-----|
| TLS cert | file path in `.env` | `fly secrets set` | Secrets Manager |
| API keys | `.env` | `fly secrets set` | Secrets Manager |
| DB password | `.env` | `fly secrets set` | Secrets Manager |
| Cert content | not needed | `TEMPORAL_TLS_CERT_CONTENT` | `TEMPORAL_TLS_CERT_CONTENT` in Secrets Manager |

---

## Checklist

### Phase 0 — COMPLETE ✅ (2026-04-20, self-hosted VPS path)
- [x] Hostinger VPS provisioned (Ubuntu 24.04, 2 vCPU / 4GB RAM)
- [x] Docker 29.4.1 installed
- [x] Temporal stack running (`temporalio/auto-setup:1.24` + postgres + temporal-ui)
- [x] Firewall: TCP 7233 locked to home IP
- [x] `.env` updated: `TEMPORAL_ADDRESS=<vps-ip>:7233`
- [x] Worker connects (plain gRPC, no TLS needed for self-hosted)
- [x] Kill + restart test passed (session resumed at cycle_count=5)

### Phase 1
- [ ] Fly CLI installed + authenticated
- [ ] `fly.toml` added to repo root
- [ ] `_build_tls_config()` updated to support inline cert content
- [ ] All secrets stored in Fly (`fly secrets list` to verify)
- [ ] `fly deploy` succeeds
- [ ] `fly logs` shows worker polling Temporal Cloud
- [ ] Machine-off test passes (triggers resume within one candle)

### Phase 2
- [ ] AWS CLI configured with deploy IAM user
- [ ] ECR repositories created
- [ ] All secrets stored in Secrets Manager
- [ ] Images built and pushed to ECR
- [ ] `terraform plan` reviewed
- [ ] `terraform apply` completed
- [ ] Ops-api reachable via ALB HTTPS
- [ ] React UI updated to point at cloud ops-api endpoint
