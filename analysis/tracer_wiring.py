"""Part A1: tracer test of advanced_simulator side wiring with a synthetic resolver.
No DB, no network, no writes. Imports only advanced_simulator (which imports only numpy)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from advanced_simulator import run_advanced_monte_carlo  # noqa: E402

OUTCOMES = ("Out", "K", "1B", "2B", "3B", "HR", "BB")
# batter offense level -> base outcome probs (sum to 1)
OFFENSE = {
    "avg":   dict(Out=0.47, K=0.22, **{"1B": 0.14, "2B": 0.045, "3B": 0.004, "HR": 0.031}, BB=0.08),
    "elite": dict(Out=0.30, K=0.12, **{"1B": 0.25, "2B": 0.09, "3B": 0.01, "HR": 0.10}, BB=0.13),
}
# pitcher id -> (hit multiplier, K rate override or None)
PITCHER = {"P_avg": (1.0, None), "P_bad": (1.8, None), "P_highK": (1.0, 0.40), "P_lowK": (1.0, 0.08),
           "BP": (1.0, None)}


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


def lineup(team, level):
    return [{"name": f"{team}{i}", "id": f"{level}_{team}{i}"} for i in range(9)]


def run(case, away_off, home_off, away_sp, home_sp, n=4000):
    np.random.seed(7)
    r = run_advanced_monte_carlo(StubResolver(), lineup("AWAY", away_off), lineup("HOME", home_off),
                                 away_sp, home_sp, ["BP"], ["BP"], density_ratio=1.0, iterations=n,
                                 as_of_date=None)
    # mean runs per side: recover from totals and win prob is not possible; use props of names instead
    pp = r["player_props"]
    away_hit = np.mean([pp[f"AWAY{i}"]["1+ Hits"] for i in range(9)])
    home_hit = np.mean([pp[f"HOME{i}"]["1+ Hits"] for i in range(9)])
    print(f"\n{case}\n  inputs: away offense={away_off}, home offense={home_off}, "
          f"away SP={away_sp}, home SP={home_sp}")
    print(f"  away_win_prob={r['away_win_prob']:.3f}  home_win_prob={1 - r['away_win_prob']:.3f}  "
          f"mean total={r['total_mean']:.2f}")
    print(f"  away_k_dist mean={np.mean(r['away_k_dist']):.2f}  home_k_dist mean={np.mean(r['home_k_dist']):.2f}")
    print(f"  mean P(1+ hit) for players NAMED away={away_hit:.3f}  named home={home_hit:.3f}")
    return r


if __name__ == "__main__":
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # silence the simulator's banner
        pass
    run("(i) elite HOME offense vs terrible AWAY starter; avg away offense vs avg home starter",
        "avg", "elite", "P_bad", "P_avg")
    run("(ii) mirror: elite AWAY offense vs terrible HOME starter; avg home offense vs avg away starter",
        "elite", "avg", "P_avg", "P_bad")
    run("(iii) high-K AWAY starter vs low-K HOME starter; both offenses avg",
        "avg", "avg", "P_highK", "P_lowK")
    run("(control) everything average", "avg", "avg", "P_avg", "P_avg")
