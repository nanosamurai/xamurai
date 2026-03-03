# CI/CD scaffolding and repo split (nanosamurai)

This repo currently contains both application code (xamurai services/workers) and a temporary
umbrella Helm chart (`charts/nanosamurai-stack`) that deploys the **whole nanosamurai stack**
(BFF + persistor + xamurai services).

Long-term, we want to avoid coupling service code changes to environment wiring. This document
describes a CI/CD structure that:

- keeps **service repos** autonomous for build/test/image publication
- keeps a single **deploy repo** as the “compatibility lockfile” for coherent stack releases
- keeps **Pulumi** focused on AWS foundation and managed services (MSK/RDS/EKS)
- supports fast "merge → deploy → smoke tests" for **dev** while allowing promotions later

> Note: xamurai used to be called **drsynth** in older docs and legacy file names.

---

## A) Repos and ownership

### 1) Service repos

Examples:
- `xamurai` (this repo)
- `samuraibff`
- `samuraipersistor`

Own:
- application code
- unit/component tests
- Docker image build + publish to ECR

Do not own (initially):
- per-environment Helm values
- ingress, HPA, PDB, service accounts, cluster wiring

### 2) `nanosamurai-platform` (Pulumi) — AWS foundation

Pulumi should own AWS resources and cluster foundation:
- VPC/subnets/NAT/RTs
- EKS cluster + node groups (CPU + GPU pools)
- IAM + IRSA / Pod Identity
- **Managed RDS Postgres** (optionally RDS Proxy)
- **Managed MSK**
- S3 buckets (recordings/artifacts, enrollment manifests)
- Route53 + ACM (if doing managed TLS)

Pulumi should *not* manage day-to-day Kubernetes Deployments/Services if Helm is already used.

Recommended layout: one Pulumi project with multiple stacks:

```
platform/dev
platform/staging
platform/prod
```

For starters: only `platform/dev` is needed.

### 3) `nanosamurai-deploy` — Kubernetes apps & releases

This repo owns the umbrella chart and environment overlays.

Own:
- umbrella Helm chart (`nanosamurai-stack`)
- per-environment values overlays (dev/staging/prod)
- ingress/controller integration, network policies, HPAs, PDBs
- (optional) smoke test Jobs/manifests

---

## B) Helm chart placement: umbrella first

For our current stage, keep the umbrella chart centralized in the deploy repo.

Reason: it encodes cross-service wiring (Kafka/RDS endpoints, ingress/auth issuer, S3 settings, etc.) and we are OK with **stack releases**.

Later evolution (optional):
- each service repo publishes a small chart (OCI)
- umbrella chart depends on versioned subcharts

---

## C) CD trigger model (no env branches)

### Service repo CI (build + publish)

On merge to `master`/`main`:
- run tests
- build image
- push to ECR tagged immutably: `sha-<GIT_SHA>` (avoid `latest`)
- open a PR to `nanosamurai-deploy` updating **dev** pinned image tags

### Deploy repo CD (deploy/promote)

Deploys are driven by changes in `nanosamurai-deploy`.

**Dev** (automated):
- merging the “bump image tags” PR triggers:
  - `helm upgrade --install ... --atomic --wait`
  - smoke tests (Job or scripted)

**Staging/Prod** (later):
- promotion PR copies a coherent set of image tags dev → staging → prod
- use **GitHub Environments approvals** for staging/prod deploy jobs

This avoids branch-per-env drift while preserving “merge → deploy quickly” in dev.

---

## D) Compatibility risk: deploy repo is the lockfile

The deploy repo pins a coherent set of:
- `samuraibff` image tag
- `samuraipersistor` image tag
- all xamurai service/worker image tags

This allows you to deploy only one service change in dev, but you always *can* promote a known-good set.

---

## E) Migrations and rollback policy

Stateless rollback is handled by Helm:
- use `helm upgrade --atomic` in CD
- rollback by reverting deploy repo commit or `helm rollback`

DB migrations (RDS) should follow expand/contract:
- run migrations as a Kubernetes Job or controlled pipeline step
- do not rely on down-migrations as your primary rollback mechanism in prod

---

## F) GitHub Actions scaffolding (this repo)

This repo now contains:
- `.github/workflows/ci.yml` — fast PR gate
  - Helm lint + Helm template unit tests
  - lightweight Python unit tests (no GPU / no Kafka / no LocalStack)
- `.github/workflows/integration-tests.yml` — manual/nightly scaffolding
  - intentionally not enabled by default (heavy ML deps + Docker required)

---

## G) AWS auth from GitHub Actions (OIDC)

For deploy workflows (in `nanosamurai-deploy`), prefer:
- GitHub OIDC → assume AWS role
- no static AWS keys in GitHub

---

## H) Observability (Grafana) access and security

### Baseline rules
- Grafana should not be publicly accessible by default.
- Prefer **private access** via VPN / SSO-aware ingress.

### Recommended options (in order)
1) **Internal-only access**
   - expose Grafana via an internal load balancer (private subnets)
   - access via VPN / SSM / corporate network

2) **Ingress + SSO/OIDC auth**
   - put Grafana behind an auth proxy or an ingress that supports OIDC
   - integrate with your IdP (Keycloak / Cognito / Google Workspace)

3) **Public endpoint with IP allowlist (only as a stopgap)**
   - still keep Grafana login enabled
   - restrict via ALB/NLB security group + WAF / ingress allowlist

Notes:
- Grafana has its own auth; but for admin access you typically want SSO and to avoid managing local users.
- Treat datasource credentials as secrets (Kubernetes Secret / ExternalSecrets).

---

## I) Future `nanosamurai-deploy` repo layout (reference)

```
nanosamurai-deploy/
  charts/
    nanosamurai-stack/
  environments/
    dev/
      values.yaml
    staging/
      values.yaml
    prod/
      values.yaml
  k8s/
    smoke-tests/
      tier1-job.yaml
      tier2-job.yaml
  .github/workflows/
    deploy-dev.yaml
    promote.yaml
```

Dev namespace recommendation: `nanosamurai`.
