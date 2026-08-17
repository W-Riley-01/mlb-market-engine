# MLB Market Engine

A Monte Carlo simulation engine for MLB game and player-prop predictions,
fully self-hosted on AWS. Predictions are generated on a schedule, logged
to a database, and automatically checked against real outcomes to measure
model accuracy over time.

**Live app:** [app.diamondmetrics.dev](https://app.diamondmetrics.dev)
**Infrastructure repo:** [`mlb-engine-aws`](https://github.com/W-Riley-01/mlb-engine-aws)

## What it does

- Simulates thousands of at-bats per game using Monte Carlo methods,
  producing full probability distributions rather than single-point
  predictions
- Generates game-level markets (win probability, first-5-innings, NRFI,
  total runs, starting pitcher strikeout distributions) and player-level
  props (hits, total bases, home runs, RBIs, stolen bases)
- Incorporates real weather data (temperature, wind, ballpark-relative
  carry effects) as a simulation input, not an afterthought
- Runs unattended on a schedule with five prediction passes and two outcome
  resolution passes per day and with no manual intervention
- Tracks every prediction against its actual outcome, tagged by model
  version, to support ongoing calibration analysis (Brier scores,
  reliability diagrams, version-over-version comparison)

## Architecture

Streamlit frontend → RDS PostgreSQL → scheduled ECS Fargate batch jobs →
Application Load Balancer. Full infrastructure details, including the
architecture diagram, live in the companion
[`mlb-engine-aws`](https://github.com/W-Riley-01/mlb-engine-aws) repo.

## Data model

Four core tables:

| Table | Grain |
|---|---|
| `game_predictions` | One row per game per simulation run |
| `player_predictions` | One row per batter per game |
| `game_outcomes` | One row per resolved game |
| `player_outcomes` | One row per batter per game, resolved from official boxscores |

Every prediction is tagged with a `model_version`, making direct,
apples-to-apples comparison possible across model iterations.

## Tech stack

Python · PostgreSQL · Streamlit · Docker · AWS (ECS Fargate, RDS, EventBridge
Scheduler, Secrets Manager) · Terraform · GitHub Actions

## Data sources

[MLB Stats API](https://statsapi.mlb.com) for schedules, rosters, and
outcomes; [Open-Meteo](https://open-meteo.com) for weather.

## Status

Live and running in production. Model currently on v3; ongoing work is
calibration analysis against accumulated prediction/outcome data.

## License

MIT — see [LICENSE](./LICENSE).
