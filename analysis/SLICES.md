# Frozen slice family (written 2026-10-22, before any slice was computed)

Anything computed that is not listed here is labeled **post-hoc** in REPORT.md.

## Analysis set
- Games: `lib.exclusion_flags(...).keep` → 1,460 games (v2 802, v3 658). Excludes v1, the All-Star Game,
  postseason, the pending game, rained-out attempts (schedule state Postponed/Cancelled on the predicted date),
  and the 8 predictions logged after actual first pitch.
- Player rows: status `final` and plate_appearances > 0 → 26,225 rows (×5 props).
- Dates: ET schedule date (`sched_date`) everywhere; `game_date` (UTC) is never used for grouping.
- Regimes (ET): **v2** 05-01..07-19 · **v3a** 07-20..08-04 (weather ~30% missing) · **v3b** 08-10..09-27 (weather complete).
  "v3" = v3a ∪ v3b (regular season only). Versions are never pooled in a metric, except where stated in pre-specified test (b).

## Targets
| id | definition | rows/game | cluster |
|---|---|---|---|
| home_win | home_win_prob vs home won (ML counted once) | 1 | game |
| nrfi | nrfi_prob vs no run in 1st | 1 | game |
| over_X.5 | half-run rungs 6.5..11.5 = P(total ≥ X+1), scored total ≥ X+1. Headline: 7.5, 8.5, 9.5 | 1 per rung | game |
| f5_home / f5_away / f5_tie, rl_home / rl_away | descriptive headline only | 1 | game |
| k_T_side | starter K ≥ T, T=3..7, side away/home; void if starter did not pitch | 1 per side | game |
| k5 (slices) | K ≥ 5, away and home pooled | 2 | game |
| p_1h, p_2tb, p_1hr, p_1rbi, p_1sb | player props | up to 18 | game |

Whole-number rungs (7.0..11.0) duplicate an event already covered by a half-run rung; they appear only in test (c).
No metric is ever pooled across rungs or across K thresholds.

## Headline metrics (per target × v2 / v3 / v3a / v3b)
n, events, non-events, base rate, mean predicted, **bias = mean predicted − realized rate** (positive = over-predicts;
same sign as calibration_metrics.py), Brier, Brier skill vs baseline, log loss (clip ε = 1e-6, stated), AUC,
calibration intercept (logistic, logit(p) as offset) and slope (logistic on logit(p)), ECE (secondary, equal-count bins).
All intervals: 95%, cluster bootstrap by game, B = 2000 (one-row-per-game targets: also Wilson on the realized rate).

Baselines:
- home_win: constant home-win rate (in-sample, per version). Other game targets: in-sample base rate.
- player props: (a) league rate (in-sample, per version); (b) batting-slot rate (in-sample, per version);
  (c) player's own past rate from this table, strictly earlier ET dates, all finalized rows regardless of version,
  shrunk toward the past league rate with a 20-game prior: (k + 20·league_past) / (n + 20).
  In-sample baselines (a, b) are slightly favoured; (c) is fully out-of-sample.

Continuous checks: totals residual (actual − total_mean), MAE, dispersion SD of (actual − total_mean)/total_std;
K full distribution: ranked probability score (RPS, outcomes 0..15) and randomized PIT from the stored K distribution.

Coverage sensitivity: headline bias and Brier post-stratified to the completed regular-season schedule over
month × start-time cells (weights = schedule count / sample count per cell), next to unweighted.

Monthly series (descriptive, not in BH family): Brier, bias, and Brier skill vs that month's own base rate,
per headline target, per version; September flagged separately.

## Pre-specified tests (reported on their own, outside the BH family)
- **(a)** pitcher_4k_away (K ≥ 4, away starter): bias and CI in v2, in v3, and CI on the v3 − v2 change.
- **(b)** Weather shift vs totals accuracy. shift = weather_carry_delta_ft × 0.05 runs (engine_runner.RUNS_PER_FT_CARRY).
  Primary: paired ΔMSE = (actual − total_mean)² − (actual − (total_mean − shift))² over games with shift ≠ 0,
  v2+v3 pooled (paired within game; mechanism unchanged across versions), bootstrap CI; negative = shift helps.
  Secondary: slope β of (actual − unshifted mean) on shift (β = 1 shift right-sized, 0 no signal); by version;
  bias / Brier (rungs 7.5, 8.5, 9.5) / ladder consistency / dispersion by |shift| bucket ≤0.5, 0.5–1.5, 1.5–2.5, >2.5.
- **(c)** Whole vs half rung: for each of the 5 pairs (6.5|7.0 … 10.5|11.0), Brier(whole) − Brier(half) against the
  same realized event (total ≥ X+1), paired by game, bootstrap CI; per version.

## Slice family (bias test and skill test per cell; BH at q = 0.05 across ALL cells below)
Slice targets: home_win, nrfi, over_8.5, k5, p_1h, p_2tb, p_1hr, p_1rbi, p_1sb.
Each segment is computed within v2 and within v3 separately unless noted.

| segment | levels | targets |
|---|---|---|
| regime | v2, v3a, v3b | all |
| month (ET) | v2: May, Jun, Jul; v3: Jul, Aug, Sep | all |
| start time (ET, scheduled) | <4pm, 4–7pm, 7–9pm, 9pm+ | all |
| lead time (scheduled first pitch − predicted_at) | <1h, 1–2h, ≥2h | all |
| side | away, home | k5 + player props |
| batting slot | 1–3, 4–6, 7–9 | player props |
| weather label | v3b only: Neutral (Dome), Very HR Suppressed, HR Suppressed, Neutral, HR Friendly, Very HR Friendly | all |
| weather missing | missing vs present; within v2 and within v3a | all |
| \|shift\| bucket | ≤0.5, 0.5–1.5, 1.5–2.5, >2.5 (games with weather) | all |
| park (home team) | 30 | all — exploratory |
| batting team | 30 | player props — exploratory |
| probability bucket | equal-count reliability bins | all — descriptive, not tested |

Per cell: bias test H0 bias = 0 and skill test H0 Brier skill (vs in-cell base rate) = 0; p-value from
z = estimate / bootstrap SE. Both enter one BH family.

## Minimum events
A cell is reported only with ≥ 50 events and ≥ 50 non-events; otherwise it is listed as "insufficient" and
excluded from BH. Reliability bins: equal-count, the largest k ≤ 10 such that every bin has ≥ 30 events and ≥ 30 non-events.

## Team / park
Raw cell estimates enter BH like any other cell. Rankings use empirical-Bayes shrinkage of the cell bias
(DerSimonian–Laird τ², normal-normal) and a permutation test (1,000 shuffles of the entity label across games)
of the largest |bias / SE|. Labeled exploratory regardless of outcome.

## Replication rule
A BH-significant cell is a **finding** only if its bias (or skill) has the same sign in both halves of that
version's regular season split at the median ET date. For segment types whose effect should be model-independent
(start time, lead time, park, weather label, weather missing, |shift|), the same sign must also appear in the
other version when that cell qualifies there. Anything failing is a **lead**.

Confidence labels for model problems: **high** = finding + consistent across versions where applicable;
**medium** = finding in one version only; **low** = lead or descriptive.
