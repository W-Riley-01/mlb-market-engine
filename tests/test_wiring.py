"""Regression test: each team must bat against the OPPOSING starter/bullpen, and each starter's Ks
must be credited to that starter. Uses a stub resolver (no DB, no network); needs only numpy.

Run:  python tests/test_wiring.py      (or pytest, if installed)
Seed 7; 2,000 iterations for the directional cases, 4,000 for the equal-teams case.
Equal-teams tolerance: |home win prob - 0.547| <= 0.03 (about 4 SE at n=4,000). 0.547 is the
last-at-bat edge measured with this stub before the fix (wiring cannot matter when both sides are equal).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from advanced_simulator import run_advanced_monte_carlo  # noqa: E402

OFFENSE = {
    "avg":   {"Out": 0.47, "K": 0.22, "1B": 0.14, "2B": 0.045, "3B": 0.004, "HR": 0.031, "BB": 0.08},
    "elite": {"Out": 0.30, "K": 0.12, "1B": 0.25, "2B": 0.09, "3B": 0.01, "HR": 0.10, "BB": 0.13},
}
# pitcher id -> (multiplier on hits/BB, K-rate override)
PITCHER = {"P_avg": (1.0, None), "P_bad": (1.8, None), "P_highK": (1.0, 0.40), "P_lowK": (1.0, 0.08), "BP": (1.0, None)}


class StubResolver:
    def _probs(self, batter_id, pitcher_id):
        p = dict(OFFENSE[batter_id.split("_")[0]])
        mult, k = PITCHER[pitcher_id]
        for o in ("1B", "2B", "3B", "HR", "BB"):
            p[o] *= mult
        if k is not None:
            p["K"] = k
        p["Out"] = max(0.0, 1 - sum(v for o, v in p.items() if o != "Out"))
        return p

    def generate_probabilities(self, batter_id, pitcher_id, density_ratio, as_of_date=None, **_):
        return self._probs(batter_id, pitcher_id)

    def generate_bullpen_probabilities(self, batter_id, bullpen, density_ratio, as_of_date=None):
        return self._probs(batter_id, "BP")

    def get_first_inning_stats(self, pitcher_id, as_of_date=None):
        return None


def _sim(away_off, home_off, away_sp, home_sp, n=2000):
    np.random.seed(7)
    lineup = lambda team, lvl: [{"name": f"{team}{i}", "id": f"{lvl}_{team}{i}"} for i in range(9)]
    r = run_advanced_monte_carlo(StubResolver(), lineup("AWAY", away_off), lineup("HOME", home_off),
                                 away_sp, home_sp, ["BP"], ["BP"], density_ratio=1.0, iterations=n)
    r["home_win_prob"] = 1 - r["away_win_prob"]
    hit = lambda team: np.mean([r["player_props"][f"{team}{i}"]["1+ Hits"] for i in range(9)])
    r["away_hit"], r["home_hit"] = hit("AWAY"), hit("HOME")
    return r


def test_elite_home_offense_vs_bad_away_starter_raises_home():
    r = _sim("avg", "elite", "P_bad", "P_avg")
    assert r["home_win_prob"] > 0.9, r["home_win_prob"]
    assert r["home_hit"] > r["away_hit"] + 0.2, (r["home_hit"], r["away_hit"])


def test_mirror_raises_away():
    r = _sim("elite", "avg", "P_avg", "P_bad")
    assert r["away_win_prob"] > 0.9, r["away_win_prob"]
    assert r["away_hit"] > r["home_hit"] + 0.2, (r["away_hit"], r["home_hit"])


def test_high_k_starter_ks_land_on_that_starter():
    r = _sim("avg", "avg", "P_highK", "P_lowK")
    away_k, home_k = np.mean(r["away_k_dist"]), np.mean(r["home_k_dist"])
    assert away_k > home_k + 3, (away_k, home_k)


def test_props_land_on_named_batters():
    # elite HOME hitters vs an average away starter: the props stored under HOME names must be high
    r = _sim("avg", "elite", "P_avg", "P_avg")
    assert r["home_hit"] > 0.85, r["home_hit"]
    assert r["away_hit"] < 0.75, r["away_hit"]


def test_equal_teams_home_edge():
    r = _sim("avg", "avg", "P_avg", "P_avg", n=4000)
    assert abs(r["home_win_prob"] - 0.547) <= 0.03, r["home_win_prob"]


if __name__ == "__main__":
    import contextlib
    import io
    failed = 0
    for name, fn in [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"{failed} failed")
    sys.exit(1 if failed else 0)
