@'
# MLB Market Engine - Roadmap

_Last updated: 2026-08-13_

## Project purpose (current)

A baseball prediction/simulation system, fully self-hosted on AWS, serving
as a portfolio piece for data architecture / data analysis roles.

**Note on scope change:** the AWS SAA-C03 certification track has been
dropped. The AWS migration itself continues in full - this is not a
pivot away from AWS, just away from formal exam study. Once the
migration is complete, project focus shifts to validating and improving
the prediction model itself (comparing predictions against recorded
outcomes) rather than further infrastructure/cert work.

---

## Status: Weekends 1-6 complete, infrastructure backlog cleared

### Live architecture (as of this update)

| Component | Status |
|---|---|
| RDS PostgreSQL 17.9, private subnets | Live |
| Secrets Manager (RDS master credentials) | Live |
| S3 (`mlb-engine-data-4462`) - parquet data | Live |
| ECS Fargate - 3 scheduled batch jobs | Live |
| EventBridge Scheduler - **5x daily** `auto_log_predictions`, 2x daily `record_outcomes` | Live |
| ECR (`mlb-engine-jobs`) - shared image, dispatched via `entrypoint.sh` | Live |
| ECS Fargate - standing `mlb-engine-app` service (Streamlit) | Live |
| Application Load Balancer + target group | Live |
| ACM certificate (DNS-validated) | Live |
| Route 53 hosted zone - `diamondmetrics.dev` | Live |
| `https://app.diamondmetrics.dev` | **Live, serving real predictions** |
| EC2 bastion (SSM-only) | **Live, standing permanent infrastructure** - kept intentionally for local DBeaver/SQL admin access, not slated for removal |
| GitHub Actions - image build/push, auto-redeploy on push | Live |

### What's automatic vs. manual right now

**Fully automatic, no action needed:**
- Predictions: 5x daily (9am / 11am / 2pm / 5:30pm / 8pm ET)
- Outcome resolution: 2x daily (morning / afternoon)
- Image build: triggers automatically on push to `main` (path-filtered)
- `mlb-engine-app` service redeploy after each image push (`force-new-deployment`)

**Not yet automated:**
- Nothing outstanding on the deploy path as of this update.

---

## Recent session work (2026-08-13)

- Restored the bastion after briefly decommissioning it - it's the only
  local access path for DBeaver/SQL admin work, which is a standing
  need, not a migration artifact to eventually remove.
- Diagnosed and fixed an RDS Secrets Manager credential drift/resync
  that replaced the secret's ARN - updated `ecs.tf`, `iam_ecs.tf`, and
  `prediction_logger.py` to the new ARN.
- Fixed `prediction_logger.py`'s `_fetch_db_url_from_secrets_manager()`
  to use the safe `.get()` fallback pattern (the RDS-managed secret
  payload only ever contains `username`/`password`, never
  `host`/`port`/`dbname`) - now matches `auto_log_predictions.py` and
  `record_outcomes.py`.
- Root-caused a multi-hour DBeaver connection failure: a local
  PostgreSQL server was permanently bound to port 5433 on this machine,
  silently intercepting SSM tunnels meant for RDS. Fix: use port 5434
  (or any port other than 5433) for all future SSM port-forwarding to
  RDS.
- Recovered real, uncommitted work (`calibration.py`,
  `calibration_metrics.py`, `diagnostic_player_outcomes.py`) that had
  been sitting on disk with space-separated filenames and had never
  been committed to git.
- Verified `migrations/migration_2026_05_cleanup_legacy_duplicates.sql`
  against live RDS data - no duplicate `game_predictions` rows found.
  Committed as historical documentation.
- Full file-system inventory of both repos - cleaned out session
  scratch files, old crash logs, and confirmed-dead migration exports;
  archived (not deleted) historical/pre-RDS-migration files and the
  project's original single-batter-profile origin data
  (`archive/project-origin/`).
- Diagnosed and fixed a self-inflicted live app outage: RDS credential
  resyncs done for local DBeaver access invalidated the running app's
  cached DB connection. Fix: `aws ecs update-service --force-new-deployment`
  after any RDS credential reset, even ones done purely for local tooling.
- Refreshed `mlb-engine-aws/README.md` (was still describing the
  abandoned Lambda + EventBridge plan) and added `DECISIONS.md`
  documenting real architecture tradeoffs.
- Fixed stale "Supabase" references in log text and docstrings
  (`auto_log_predictions.py`, `record_outcomes.py`,
  `prediction_logger.py`) to correctly say RDS PostgreSQL.
- Bumped GitHub Actions `checkout`/`configure-aws-credentials` from
  `@v4` to `@v5` in all three workflow files.
- Investigated the `player_outcomes` stuck-pending hypothesis from
  `diagnostic_player_outcomes.py` against live data - **not
  confirmed**. All currently-pending player rows have parent games
  that are themselves still pending (normal in-flight state, written
  within the same second as the player row), not orphaned rows with a
  final parent game. No fix needed.

---

## Backlog (not yet scheduled into a specific session)

1. **`data/` and `src/` restructuring** - deferred deliberately: 17
   scripts hardcode `./data/...` paths directly, so any folder
   reorganization needs a coordinated find-and-replace pass (plus
   `Dockerfile`/`entrypoint.sh`/workflow path updates if `.py` files
   move) rather than a same-session file shuffle.

---

## After the backlog: the actual pivot

Once the remaining backlog is cleared, the project shifts to its
stated focus:

- Let predictions + outcome recording accumulate for a meaningful
  sample (the 5x/2x daily automation is already running)
- Run `calibration_metrics.py` to check model calibration against real
  outcomes
- Revisit the v3 model re-validation sequence: Week 1 calibration
  check, Week 3 `compare_versions(("v2", "v3"))` head-to-head, targeting
  a reduction in `pitcher_4k_away` bias from ~+0.10 toward +0.05 or
  lower
- General "does the model actually predict well" analysis - the real
  data-analysis portfolio work this project is ultimately for
'@ | Out-File -Encoding utf8 README.md