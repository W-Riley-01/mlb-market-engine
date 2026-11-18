"""Checks for analysis/replay_ws5.py (lab 2 addendum A8 item 4: written and tested on 2025 files or made-up data,
never on the 2026 files). Made-up data except the end-to-end test, which runs 2025 gamePk 777505 (Reds at Tigers,
2025-06-15) from the cached 2025 schedule and box score, with its 2025 line score fetched into a temporary folder.
All outputs go to a temporary folder outside the repository.

  analysis\\.venv\\Scripts\\python.exe analysis/test_replay_ws5.py          (all checks)
  analysis\\.venv\\Scripts\\python.exe analysis/test_replay_ws5.py --quick  (skip the end-to-end run)
"""
import json
import shutil
import sys
import tempfile
import time
import traceback
import types
import urllib.request
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import replay_ws5 as R  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="replay_ws5_test_"))
EXPECT = {"date": "2030-01-02", "game_type": "W", "away": "Away Club", "home": "Home Club", "venue": "Test Dome"}
DOME = {"park": "Test Dome", "is_dome": True, "temp_f": 72.0, "humidity_pct": 50, "pressure_inhg": 29.92,
        "wind_speed_mph": 0.0, "wind_from_compass": "—", "wind_out_mph": 0.0, "wind_label": "Indoors",
        "carry_delta_ft": 0.0, "score": 50, "label": "Neutral (Dome)"}


class Guard(dict):
    """A dict whose forbidden keys raise on any access, including through items() and values()."""

    def __init__(self, data, forbidden):
        super().__init__(data)
        self.forbidden = set(forbidden)

    def _check(self, k):
        if k in self.forbidden:
            raise AssertionError(f"touched forbidden field {k!r}")

    def __getitem__(self, k):
        self._check(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._check(k)
        return super().get(k, default)

    def items(self):
        raise AssertionError("items() would expose forbidden fields")

    def values(self):
        raise AssertionError("values() would expose forbidden fields")


def make_schedule(n_games=1, game_type="W", away="Away Club", home="Home Club", venue="Test Dome"):
    g = {"gamePk": 900001, "gameDate": "2030-01-02T00:08:00Z", "gameType": game_type, "venue": {"name": venue},
         "status": {"detailedState": "Final"}, "linescore": {"innings": []},
         "teams": {"away": {"team": {"id": 1, "name": away}, "probablePitcher": {"id": 101, "fullName": "Away Ace"},
                            "leagueRecord": {"wins": 90}, "score": 3, "isWinner": True},
                   "home": {"team": {"id": 2, "name": home}, "probablePitcher": {"id": 201, "fullName": "Home Ace"},
                            "leagueRecord": {"wins": 80}, "score": 2, "isWinner": False}}}
    games = [g] + [dict(g, gamePk=900001 + i) for i in range(1, n_games)]
    return {"dates": [{"date": "2030-01-02", "games": games}]}


def make_box(n_starters=9, away_name="Away Club", h=None):
    """Made-up box score: starters with battingOrder slot*100 plus one substitute per side."""
    h = h or {}
    teams = {}
    for side, tid, base in (("away", 1, 100), ("home", 2, 200)):
        players = {}
        sp = base + 1
        players[f"ID{sp}"] = {"person": {"id": sp, "fullName": f"{side} pitcher"},
                              "stats": {"pitching": {"strikeOuts": 6}, "batting": {}}, "seasonStats": {}}
        for i in range(n_starters):
            pid = base + 10 + i
            players[f"ID{pid}"] = {"person": {"id": pid, "fullName": f"{side} batter {i + 1}"},
                                   "battingOrder": str((i + 1) * 100),
                                   "stats": {"batting": h.get(pid, {"atBats": 4, "hits": 1})}, "seasonStats": {}}
        players[f"ID{base + 50}"] = {"person": {"id": base + 50, "fullName": f"{side} sub"}, "battingOrder": "901",
                                     "stats": {"batting": {"atBats": 1}}, "seasonStats": {}}
        teams[side] = {"team": {"id": tid, "name": away_name if side == "away" else "Home Club"},
                       "battingOrder": [base + 10 + i for i in range(n_starters)], "pitchers": [sp, base + 2],
                       "players": players, "teamStats": {"batting": {"runs": 3 if side == "away" else 2}}}
    return {"teams": teams}


def linescore(away, home):
    return {"innings": [{"num": i + 1, "away": {"runs": a}, "home": {"runs": b}} for i, (a, b) in
                        enumerate(zip(away, home))]}


class StubResolver:
    """No data: fixed probabilities (as tests/test_wiring.py), small made-up matrices for the stop rule."""

    def __init__(self, late=False):
        dates = ["2025-05-01", "2025-09-30"] + (["2026-01-01"] if late else [])
        self.pqm = pd.DataFrame({"game_date": pd.to_datetime(dates), "pitcher": range(len(dates))})
        self.cqm = pd.DataFrame({"game_date": dates, "batter": range(len(dates))})

    def generate_probabilities(self, batter_id, pitcher_id, density_ratio, as_of_date=None, **_):
        return {"Out": 0.47, "K": 0.22, "1B": 0.14, "2B": 0.045, "3B": 0.004, "HR": 0.031, "BB": 0.09}

    def generate_bullpen_probabilities(self, batter_id, bullpen, density_ratio, as_of_date=None):
        assert list(bullpen) == [], "bullpens must be empty"
        return self.generate_probabilities(batter_id, None, density_ratio)

    def get_first_inning_stats(self, pitcher_id, as_of_date=None):
        return None


def stub_files(root: Path, box=None, ls=None, sched=None):
    """Made-up schedule, box score and line score, pre-cached so fetch_cached never goes to the network."""
    (root / "box").mkdir(parents=True, exist_ok=True)
    (root / "ls").mkdir(parents=True, exist_ok=True)
    (root / "sched.json").write_text(json.dumps(sched or make_schedule()), encoding="utf-8")
    (root / "box" / "900001.json").write_text(json.dumps(box or make_box()), encoding="utf-8")
    (root / "ls" / "900001.json").write_text(json.dumps(ls or linescore([1, 0, 0, 0, 0, 2], [0, 0, 2, 0, 0, 0])),
                                             encoding="utf-8")


def no_fetch(url, path):
    if path.exists():
        return path.read_bytes()
    raise AssertionError(f"unexpected fetch {url}")


def run_stub(root: Path, out: Path, resolver=None, **kw):
    return R.run(resolver or StubResolver(), EXPECT, root / "sched.json", root / "box", root / "ls", out,
                 iterations=kw.pop("iterations", 300), fetch=kw.pop("fetch", no_fetch),
                 weather_fn=kw.pop("weather_fn", lambda home, dt: dict(DOME)), **kw)


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------- tests

def test_stop_rule_runs_first():
    root, out = TMP / "stop", TMP / "stop_out"
    r = StubResolver(late=True)
    pqm_before = r.pqm
    # the schedule path does not exist and every other input raises: the stop must come first
    e = raises(R.StopReplay, R.run, r, EXPECT, root / "missing.json", root / "box", root / "ls", out,
               fetch=lambda *a: (_ for _ in ()).throw(AssertionError("fetched")),
               weather_fn=lambda *a: (_ for _ in ()).throw(AssertionError("weather")))
    assert "2026-01-01 or later" in str(e), e
    assert r.pqm is pqm_before, "matrices were cut before the stop rule"
    assert not (out / "ws5_prediction.json").exists()
    log = (out / "replay_ws5_log.txt").read_text(encoding="utf-8")
    assert "STOPPED" in log and log.count("content sha256") == 2, log


def test_stop_rule_limit_edge():
    ok = pd.DataFrame({"game_date": ["2025-12-31"]})
    late = pd.DataFrame({"game_date": ["2026-01-01"]})
    R.check_matrices({"a": ok, "b": ok}, [])
    raises(R.StopReplay, R.check_matrices, {"a": ok, "b": late}, [])


def test_identity_checks_pass():
    s = R.schedule_fields(make_schedule(), EXPECT["date"])
    R.identity_checks(s, R.box_fields(make_box()), dict(DOME), EXPECT)


def test_identity_two_games():
    raises(R.StopReplay, R.schedule_fields, make_schedule(n_games=2), EXPECT["date"])


def test_identity_wrong_game_type():
    s = R.schedule_fields(make_schedule(game_type="R"), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box()), dict(DOME), EXPECT)


def test_identity_wrong_teams():
    s = R.schedule_fields(make_schedule(away="Other Club"), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box(away_name="Other Club")), dict(DOME), EXPECT)


def test_identity_box_teams_differ():
    s = R.schedule_fields(make_schedule(), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box(away_name="Other Club")), dict(DOME), EXPECT)


def test_identity_wrong_venue():
    s = R.schedule_fields(make_schedule(venue="Open Park"), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box()), dict(DOME), EXPECT)


def test_identity_eight_starters():
    s = R.schedule_fields(make_schedule(), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box(n_starters=8)), dict(DOME), EXPECT)


def test_identity_not_dome():
    s = R.schedule_fields(make_schedule(), EXPECT["date"])
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box()), dict(DOME, is_dome=False), EXPECT)
    raises(R.StopReplay, R.identity_checks, s, R.box_fields(make_box()), None, EXPECT)


def test_pre_simulation_parsing_never_touches_results():
    sched = make_schedule()
    g = sched["dates"][0]["games"][0]
    for side in ("away", "home"):
        g["teams"][side] = Guard(g["teams"][side], {"leagueRecord", "score", "isWinner"})
    sched["dates"][0]["games"][0] = Guard(g, {"status", "linescore", "isTie"})
    box = make_box()
    for side in ("away", "home"):
        t = box["teams"][side]
        t["players"] = {k: Guard(p, {"stats", "seasonStats"}) for k, p in t["players"].items()}
        box["teams"][side] = Guard(t, {"teamStats"})
    s = R.schedule_fields(sched, EXPECT["date"])
    b = R.box_fields(box)
    R.identity_checks(s, b, dict(DOME), EXPECT)
    assert b["away"]["starting_nine"] == [110 + i for i in range(9)], b["away"]["starting_nine"]


def test_dome_weather_makes_no_network_call():
    def boom(*a, **kw):
        raise AssertionError("network call")
    saved = (requests.get, requests.request, requests.Session.request, urllib.request.urlopen)
    requests.get = requests.request = requests.Session.request = urllib.request.urlopen = boom
    try:
        w = R.get_game_weather("Tampa Bay Rays", "2025-10-28T00:08:00Z")
    finally:
        requests.get, requests.request, requests.Session.request, urllib.request.urlopen = saved
    assert w["is_dome"] is True and w["label"] == "Neutral (Dome)" and w["park"] == "Tropicana Field", w


def test_rerun_rule():
    root, out = TMP / "rerun", TMP / "rerun_out"
    stub_files(root)
    first = run_stub(root, out)
    pred = (out / "ws5_prediction.json").read_bytes()
    second = run_stub(root, out)  # identical bytes: continues
    assert second["prediction_sha256"] == first["prediction_sha256"]
    assert "rerun: identical, kept" in (out / "replay_ws5_log.txt").read_text(encoding="utf-8")
    changed = bytearray(pred)
    i = changed.index(b'"away_win": ') + len(b'"away_win": ')
    changed[i] = ord("1") if changed[i] != ord("1") else ord("2")
    (out / "ws5_prediction.json").write_bytes(bytes(changed))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    raises(R.RerunMismatch, run_stub, root, out)
    after = {p.name: p.read_bytes() for p in out.iterdir()}
    assert before == after, "a rerun that differs wrote something"


def test_prediction_file_has_no_results_or_times():
    root, out = TMP / "shape", TMP / "shape_out"
    stub_files(root)
    run_stub(root, out)
    pred = json.loads((out / "ws5_prediction.json").read_text(encoding="utf-8"))
    g = pred["games"][0]
    for k in ("results", "status", "away_score", "home_score"):
        assert k not in g, k
    assert "updated_at" not in pred
    raw = (out / "ws5_prediction.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    page = json.loads((out / "ws5_page.json").read_text(encoding="utf-8"))
    pg = page["games"][0]
    assert pg["status"] == "final" and pg["results"]["away_score"] == 3 and pg["results"]["home_score"] == 2
    assert pg["results"]["starter_k"] == {"away": 6, "home": 6}
    assert abs(pg["markets"]["home_win"] - (1 - pg["markets"]["away_win"])) < 1e-15
    assert len(page["beat_the_streak"]) == 5
    hits = [b["hit"] for b in page["beat_the_streak"]]
    assert hits == sorted(hits, reverse=True)
    assert page["label"] == R.LABEL


def test_starter_did_not_pitch_is_null():
    box = make_box()
    box["teams"]["home"]["players"]["ID201"]["stats"] = {"pitching": {}, "batting": {}}
    st = {"away": {"id": 101}, "home": {"id": 201}}
    lu = {s: [] for s in ("away", "home")}
    res = R.results(box, linescore([0] * 9, [0] * 9), st, lu)
    assert res["starter_k"] == {"away": 6, "home": None}, res["starter_k"]


def test_line_score_results():
    box, st, lu = make_box(), {"away": {"id": 101}, "home": {"id": 201}}, {"away": [], "home": []}
    cases = [(([1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5]), "away", True),
             (([0, 0, 0, 0, 0], [0, 0, 1, 0, 1]), "home", False),
             (([0, 1, 0, 0, 0, 3], [0, 0, 0, 1, 0, None]), "tied", False),
             (([0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]), "home", True)]
    for (a, h), led, first in cases:
        r = R.results(box, linescore(a, h), st, lu)
        assert r["led_after_5"] == led and r["run_in_first"] is first, (a, h, r["led_after_5"], r["run_in_first"])


def test_batter_line_matches_record_outcomes():
    sys.modules.setdefault("boto3", types.ModuleType("boto3"))  # imported at module level; unused here
    sys.path.insert(0, str(R.ROOT))
    import record_outcomes as ro
    lines = [({"atBats": 4, "hits": 2, "doubles": 1, "rbi": 1}, {}, "2-4, 2B, RBI"),
             ({"atBats": 5, "hits": 3, "doubles": 1, "triples": 1, "homeRuns": 1, "rbi": 4}, {}, "3-5, 2B, 3B, HR, 4 RBI"),
             ({"atBats": 3, "hits": 0}, {}, "0-3"),
             ({"atBats": 4, "hits": 2, "homeRuns": 2, "rbi": 3, "stolenBases": 1}, {}, "2-4, 2 HR, 3 RBI, SB"),
             ({"atBats": 2, "hits": 1}, {"stolenBases": 2}, "1-2, 2 SB")]
    for bat, run, want in lines:
        stats = {"batting": bat, "baseRunning": run}
        feed = {"gameData": {"status": {"statusCode": "F"}},
                "liveData": {"boxscore": {"teams": {"away": {"players": {"ID7": {"person": {"id": 7}, "stats": stats}}},
                                                    "home": {"players": {}}}}}}
        ref = ro.parse_player_outcomes(feed, [{"id": 1, "game_id": "g", "game_date": "d", "player_id": 7,
                                               "player_name": "x", "side": "away"}])[0]
        mine = R.batter_result(stats)
        for k_ref, k in (("hits", "h"), ("total_bases", "tb"), ("home_runs", "hr"), ("rbi", "rbi"),
                         ("stolen_bases", "sb"), ("at_bats", "ab"), ("doubles", "d"), ("triples", "t")):
            assert ref[k_ref] == mine[k], (k, ref[k_ref], mine[k])
        assert R.short_line(mine) == want, (R.short_line(mine), want)


def test_end_to_end_2025():
    """gamePk 777505, Cincinnati Reds at Detroit Tigers, Comerica Park, 2025-06-15; run twice."""
    pk, day = 777505, "2025-06-15"
    root = TMP / "e2e"
    (root / "box").mkdir(parents=True)
    sched = json.loads((R.API / "schedule" / f"{day}.json").read_text(encoding="utf-8"))
    for d in sched["dates"]:  # one-game copy: the replay accepts a schedule holding exactly one game
        d["games"] = [g for g in d["games"] if g["gamePk"] == pk]
    (root / "sched.json").write_text(json.dumps(sched), encoding="utf-8")
    shutil.copyfile(R.API / "box" / f"{pk}.json", root / "box" / f"{pk}.json")
    expect = {"date": day, "game_type": "R", "away": "Cincinnati Reds", "home": "Detroit Tigers",
              "venue": "Comerica Park"}
    # Comerica Park is outdoors and a 2025 forecast cannot be fetched: stand in the dome entry (no network)
    dome = lambda home, dt: R.get_game_weather("Tampa Bay Rays", dt)  # noqa: E731
    t = time.time()
    resolver = R.MatchupResolver(pqm_path=str(R.PQM), cqm_path=str(R.CQM), load_recent_form=False,
                                 load_arsenal_profile=False)
    full = (resolver.pqm, resolver.cqm)
    print(f"    load {time.time() - t:.1f}s")
    outs = []
    for k in (1, 2):
        resolver.pqm, resolver.cqm = full
        t = time.time()
        o = R.run(resolver, expect, root / "sched.json", root / "box", root / "ls", root / f"out{k}",
                  weather_fn=dome, script_path=Path(R.__file__))
        o["wall"] = time.time() - t
        outs.append(o)
    a, b = ((root / f"out{k}" / "ws5_prediction.json").read_bytes() for k in (1, 2))
    assert a == b, "the two prediction files differ"
    p = outs[0]["prediction"]
    g = p["games"][0]
    print(f"    prediction sha256 run 1 {outs[0]['prediction_sha256']}")
    print(f"    prediction sha256 run 2 {outs[1]['prediction_sha256']}")
    print(f"    run 1: simulation {outs[0]['sim_seconds']:.1f}s, wall {outs[0]['wall']:.1f}s; "
          f"run 2: simulation {outs[1]['sim_seconds']:.1f}s, wall {outs[1]['wall']:.1f}s")
    print(f"    away win {g['markets']['away_win']}; median total {g['markets']['median_total']}; "
          f"top Beat the Streak {p['beat_the_streak'][0]['name']} {p['beat_the_streak'][0]['hit']}")
    r = outs[0]["results"]
    print(f"    results: {r['away_score']}-{r['home_score']}, led after 5 {r['led_after_5']}, "
          f"run in 1st {r['run_in_first']}, starter K {r['starter_k']}")
    nulls = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif x is None:
            nulls.append(path)
    walk(json.loads((root / "out1" / "ws5_page.json").read_text(encoding="utf-8")), "page")
    print(f"    null page fields: {nulls}")
    print((root / "out1" / "replay_ws5_log.txt").read_text(encoding="utf-8").rstrip().replace("\n", "\n    | "))


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    if "--quick" in sys.argv:
        tests = [t for t in tests if t[0] != "test_end_to_end_2025"]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"\n{len(tests) - failed} passed, {failed} failed; outputs in {TMP}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
