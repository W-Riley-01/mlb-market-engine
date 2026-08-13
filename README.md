# MLB Market Engine — Roadmap

_Last updated: 2026-08-13_

## Project purpose (current)

A baseball prediction/simulation system, fully self-hosted on AWS, serving
as a portfolio piece for data architecture / data analysis roles.

**Note on scope change:** the AWS SAA-C03 certification track has been
dropped. The AWS migration itself continues in full — this is not a
pivot away from AWS, just away from formal exam study. Once the
migration is complete, project focus shifts to validating and improving
the prediction model itself (comparing predictions against recorded
outcomes) rather than further infrastructure/cert work.

---

## Status: Weekends 1–6 complete

### Live architecture (as of this update)

| Component | Status |
|---|---|
| RDS PostgreSQL 17.9, private subnets | Live |
| Secrets Manager (RDS master credentials) | Live |
| S3 (`mlb-engine-data-4462`) — parquet data | Live |
| ECS Fargate — 3 scheduled batch jobs | Live |
| EventBridge Scheduler — **5x daily** `auto_log_predictions`, 2x daily `record_outcomes` | Live |
| ECR (`mlb-engine-jobs`) — shared image, dispatched via `entrypoint.sh` | Live |
| ECS Fargate — standing `mlb-engine-app` service (Streamlit) | Live |
| Application Load Balancer + target group | Live |
| ACM certificate (DNS-validated) | Live |
| Route 53 hosted zone — `diamondmetrics.dev` | Live |
| `https://app.diamondmetrics.dev` | **Live, serving real predictions** |
| EC2 bastion (SSM-only) | Live, pending decommission |
| GitHub Actions — image build/push only | Live |

### What's automatic vs. manual right now

**Fully automatic, no action needed:**
- Predictions: 5x daily (9am / 11am / 2pm / 5:30pm / 8pm ET)
- Outcome resolution: 2x daily (morning / afternoon)
- Image build: triggers automatically on push to `main` (path-filtered)

**Not yet automatic (in progress — see Next Up):**
- The standing `mlb-engine-app` ECS **service** does not restart itself
  when a new image is pushed. Batch jobs don't have this problem (each
  scheduled run pulls `:latest` fresh), but the persisted app service
  does — it needs an explicit `force-new-deployment` after each image
  push, currently done manually via AWS CLI.

---

## This session's work (2026-08-13)

Closed out Weekend 6 plus several real issues found along the way:

- Fixed the Streamlit → RDS connection gap (`generate_streamlit_secrets.py`,
  generates `.streamlit/secrets.toml` from Secrets Manager at container
  startup)
- Discovered `mlb-engine-aws` (the Terraform repo) had **never been
  pushed to GitHub** — all infra source was local-only. Fixed: repo is
  now properly tracked and pushed.
- Cleaned up dead duplicate files (`Dockerfile`/`entrypoint.sh`/
  `Entrypoint.bash`) that existed in the wrong repo and were never part
  of the actual build
- Added a proper `.gitignore` to `mlb-engine-aws` (was at risk of
  committing a 6.7 MB DB dump and other files that shouldn't be in git)
- Found and closed a real, **pre-existing** scheduling gap: the original
  3x-daily prediction schedule (inherited from the old GitHub Actions
  cron) structurally missed any game with a ~12–1:30 PM ET first pitch.
  Added 11 AM ET and 8 PM ET runs for full time-zone coverage.

---

## Next up

**In progress:** Automate ECS service redeploy on image push — add a
step to `build-and-push.yml` that runs `aws ecs update-service
--force-new-deployment` for `mlb-engine-app` immediately after a
successful image push, so code changes to the app reach production
without a manual CLI step.

---

## Backlog (not yet scheduled into a specific session)

Roughly in priority order:

1. **Bastion decommission** — was waiting on proof of stable scheduled
   runs; that bar is now well exceeded (multiple confirmed automatic +
   manual runs today). Ripe to revisit: remove the bastion EC2 instance
   and the `rds_from_bastion` security group rule.
2. **Verify `ssm-params.json` contents** — flagged earlier, not yet
   confirmed either way whether it holds real values or just parameter
   names/paths. Decide whether it should ever be committed (even
   gitignored) based on that.
3. **Stray file cleanup in `MLB_Analysis`**:
   - Duplicate workflow file: untracked `.github/workflows/build-and-push.yml`
     (lowercase) alongside the real, tracked `Build and push.yml`
     (capitalized) — same naming-collision trap that caused today's
     Dockerfile/entrypoint.sh confusion in the other repo
   - `calibration metrics.py` (space in name) vs. the real
     `calibration_metrics.py`
   - Leftover April log files, a `Claude/` folder, an untracked
     migration SQL file
4. **Minor Weekend 5 leftovers**:
   - Stale "Logged to Supabase" text in log messages (harmless, just
     inaccurate wording post-migration)
   - Node.js deprecation warnings in the GitHub Actions workflow
   - `DECISIONS.md` / `README.md` updates to reflect current state
5. **`prediction_logger.py` cleanup** — bring the module-level
   `_fetch_db_url_from_secrets_manager()` in line with the `.get()`
   fallback pattern already used in `auto_log_predictions.py` and
   `record_outcomes.py`. Not currently exercised by anything live, but
   a landmine for future direct/manual invocation.

---

## After the backlog: the actual pivot

Once the infrastructure backlog is cleared, the project shifts to its
stated new focus:

- Let predictions + outcome recording accumulate for a meaningful
  sample (the 5x/2x daily automation is already running)
- Run `calibration_metrics.py` to check model calibration against real
  outcomes
- Revisit the v3 model re-validation sequence: Week 1 calibration
  check, Week 3 `compare_versions(("v2", "v3"))` head-to-head, targeting
  a reduction in `pitcher_4k_away` bias from ~+0.10 toward +0.05 or
  lower
- General "does the model actually predict well" analysis — the real
  data-analysis portfolio work this project is ultimately for
