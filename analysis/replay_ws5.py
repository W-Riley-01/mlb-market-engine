"""Website replay of 2026 World Series Game 5, Milwaukee Brewers at Tampa Bay Rays, 2026-10-28
(lab 2 addendum A8, analysis/BEAT_THE_STREAK_LAB2_ADDENDUM_A8.md, sha256 347c1710…36f2).

Runs once (A8 item 5). Order (A8 items 3 to 6):
  1. load both matrices; log latest dates and content hashes; stop on any row dated 2026-01-01 or later; cut
     both to rows before the game date;
  2. read the cached schedule, fetch the box score once; log each 2026 file's sha256;
  3. use only the pre-simulation fields of A8 item 4 step 1; run the identity checks;
  4. simulate (v4 engine as backtest_v4 runs it, empty bullpens, 5,000 iterations, seed = gamePk);
  5. write ws5_prediction.json and log its sha256 (rerun: must match byte for byte, else stop and write nothing);
  6. only then fetch the line score, read the result fields and write ws5_page.json.
Never prints or logs a 2026 field outside A8 item 4's lists; never touches seasonStats or leagueRecord.

  analysis\\.venv\\Scripts\\python.exe analysis/replay_ws5.py
"""
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_v4 import API, ROOT, starting_nine  # noqa: E402
from advanced_simulator import run_advanced_monte_carlo  # noqa: E402
from resolver import MatchupResolver  # noqa: E402
from weather import get_game_weather  # noqa: E402

WS5 = {"date": "2026-10-28", "game_type": "W", "away": "Milwaukee Brewers", "home": "Tampa Bay Rays",
       "venue": "Tropicana Field"}
PQM = ROOT / "data" / "pitch_matrix.parquet"
CQM = ROOT / "data" / "contact_matrix_env_fixed.parquet"
MATRIX_LIMIT = pd.Timestamp("2026-01-01")
ITERATIONS = 5000
MODEL_VERSION = "v4"
OUT_DIR = ROOT / "analysis" / "out" / "site"
LABEL = ("Replay, run November 2026 with the v4 engine and data through the end of the 2025 season, "
         "using the real starting lineups. Not a live pre-game prediction.")
K_THRESHOLDS = (3, 4, 5, 6, 7)
PROPS = (("hit", "1+ Hits"), ("tb2", "2+ TB"), ("hr", "1+ HR"), ("rbi", "1+ RBI"), ("sb", "1+ SB"))
SIDES = ("away", "home")


class StopReplay(Exception):
    """A registered stop: the log records it."""


class RerunMismatch(Exception):
    """A rerun produced a different prediction file: stop and write nothing (A8 item 5)."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    """A3 item 6's method, as btslab3_features.content_hash."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def check_matrices(frames: dict, lines: list, limit: pd.Timestamp = MATRIX_LIMIT) -> None:
    """Append a log line for each full matrix as loaded; then StopReplay if any row is dated on or after limit."""
    late = []
    for name, df in frames.items():
        d = pd.to_datetime(df["game_date"])
        lines.append(f"matrix {name}: rows {len(df):,}; latest game_date {d.max().date()}; "
                     f"content sha256 {content_hash(df)}")
        if (d >= limit).any():
            late.append(f"{name} has {(d >= limit).sum():,} rows dated {limit.date()} or later")
    if late:
        raise StopReplay("; ".join(late))


def cut_before(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Rows dated before day, with game_date as datetime (backtest_v4 converts it the same way)."""
    d = pd.to_datetime(df["game_date"])
    out = df[d < day]
    return out if out["game_date"].dtype == d.dtype else out.assign(game_date=d[d < day])


def fetch_cached(url: str, path: Path) -> bytes:
    """The cached file if it exists (never re-fetched); else one fetch, saved as the exact bytes returned."""
    if path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url, timeout=30) as r:
        data = r.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def schedule_fields(sched: dict, day: str) -> dict:
    """A8 item 4 step 1, schedule: gamePk, teams, gameDate, gameType, venue, probable pitchers. Nothing else."""
    days = sched.get("dates", [])
    games = [g for d in days for g in d["games"]]
    if len(games) != 1 or any(d["date"] != day for d in days):
        raise StopReplay(f"schedule holds {len(games)} games on {day}, expected exactly 1")
    g = games[0]
    teams = {}
    for s in SIDES:
        t = g["teams"][s]
        pp = t.get("probablePitcher") or {}
        teams[s] = {"id": t["team"]["id"], "name": t["team"]["name"],
                    "probable_id": pp.get("id"), "probable_name": pp.get("fullName")}
    return {"game_pk": g["gamePk"], "game_date_utc": g["gameDate"], "game_type": g["gameType"],
            "venue": g["venue"]["name"], "teams": teams}


def box_fields(box: dict) -> dict:
    """A8 item 4 step 1, box score: team IDs and names, batting order and pitcher lists, and each listed
    player's ID, name and battingOrder code. No stats, seasonStats or teamStats block is touched."""
    out = {}
    for s in SIDES:
        t = box["teams"][s]
        players = {}
        for p in t["players"].values():
            players[int(p["person"]["id"])] = {"name": p["person"]["fullName"], "battingOrder": p.get("battingOrder")}
        out[s] = {"id": t["team"]["id"], "name": t["team"]["name"], "batting_order": list(t["battingOrder"]),
                  "pitchers": list(t["pitchers"]), "players": players, "starting_nine": starting_nine(t)}
    return out


def identity_checks(sched: dict, box: dict, weather: dict | None, expect: dict) -> None:
    """A8 item 3: stop without simulating unless every check holds. (Exactly one game: schedule_fields.)"""
    bad = []
    if sched["game_type"] != expect["game_type"]:
        bad.append(f"gameType {sched['game_type']}, expected {expect['game_type']}")
    for s in SIDES:
        if sched["teams"][s]["name"] != expect[s]:
            bad.append(f"{s} team {sched['teams'][s]['name']}, expected {expect[s]}")
        if (box[s]["id"], box[s]["name"]) != (sched["teams"][s]["id"], sched["teams"][s]["name"]):
            bad.append(f"box score {s} team {box[s]['name']} does not match the schedule's")
        if len(box[s]["starting_nine"]) != 9:
            bad.append(f"{s} has {len(box[s]['starting_nine'])} starters, expected 9")
    if sched["venue"] != expect["venue"]:
        bad.append(f"venue {sched['venue']}, expected {expect['venue']}")
    if not (weather and weather.get("is_dome") is True):
        bad.append("weather.get_game_weather did not return the dome entry")
    if bad:
        raise StopReplay("identity checks failed: " + "; ".join(bad))


def k_over(k_dist) -> dict:
    """prediction_logger._k_threshold_probs with a full K distribution: share of iterations with K >= t."""
    arr = np.asarray(k_dist)
    return {str(t): float(np.mean(arr >= t)) for t in K_THRESHOLDS}


def first_inning(fi: dict | None) -> dict | None:
    if fi is None:
        return None
    return {"era": None if fi["era_1st"] is None else float(fi["era_1st"]), "scoreless": int(fi["nrfi_starts"]),
            "not_scoreless": int(fi["yrfi_starts"]), "opp_obp": float(fi["opp_obp_1st"]),
            "starts": int(fi["games_started"])}


def prediction(day: str, sched: dict, starters: dict, lineups: dict, weather: dict, res: dict) -> dict:
    """The prediction file (A8 item 6): the page shape without results, scores and status; no time stamps."""
    away_win, f5_away, f5_tie = float(res["away_win_prob"]), float(res["f5_away_win_prob"]), float(res["f5_tie_prob"])
    props = res["player_props"]
    lu = {s: [{"slot": i + 1, "id": p["id"], "name": p["name"], **{k: float(props[p["name"]][v]) for k, v in PROPS}}
              for i, p in enumerate(lineups[s])] for s in SIDES}
    game = {
        "game_id": sched["game_pk"], "away_team": sched["teams"]["away"]["name"],
        "home_team": sched["teams"]["home"]["name"], "start_time_utc": sched["game_date_utc"],
        "park": sched["venue"], "roof": "dome" if weather["is_dome"] else "open",
        "inning": None, "inning_half": None, "outs": None,
        "markets": {  # derived as prediction_logger.log_prediction does
            "away_win": away_win, "home_win": 1.0 - away_win, "f5_away": f5_away,
            "f5_home": max(1.0 - f5_away - f5_tie, 0.0), "f5_tie": f5_tie,
            "no_run_first": float(res["nrfi_prob"]), "median_total": float(res["median_total"]),
            "total_over": {str(k): float(v) for k, v in res["total_thresholds"].items()}},
        "weather": {"temp_f": weather["temp_f"], "humidity_pct": weather["humidity_pct"],
                    "pressure_inhg": weather["pressure_inhg"], "wind_mph": weather["wind_speed_mph"],
                    "wind_from": weather["wind_from_compass"], "wind_out_mph": weather["wind_out_mph"],
                    "wind_label": weather["wind_label"], "carry_ft": weather["carry_delta_ft"],
                    "hr_score": weather["score"], "hr_label": weather["label"]},
        "starters": {s: {"id": starters[s]["id"], "name": starters[s]["name"],
                         "median_k": float(res[f"{s}_pitcher_median_k"]), "k_over": k_over(res[f"{s}_k_dist"]),
                         "first_inning": first_inning(res[f"{s}_starter_first_inn"])} for s in SIDES},
        "lineups": lu,
    }
    pool = [(b, s) for s in SIDES for b in lu[s]]
    pool.sort(key=lambda x: -x[0]["hit"])  # stable: ties keep away-then-home, slot order
    other = {"away": "home", "home": "away"}
    bts = [{"name": b["name"], "team": game[f"{s}_team"], "opponent": game[f"{other[s]}_team"],
            "home": s == "home", "hit": b["hit"]} for b, s in pool[:5]]
    return {"date": day, "model_version": MODEL_VERSION, "note": "replay", "label": LABEL,
            "games": [game], "beat_the_streak": bts}


def dumps(obj: dict) -> bytes:
    """Deterministic bytes: fixed key order, full float precision (repr), LF, UTF-8."""
    return (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def batter_result(stats: dict | None) -> dict:
    """record_outcomes.parse_player_outcomes's fields and total bases."""
    if stats is None:
        return {"ab": None, "h": None, "d": None, "t": None, "hr": None, "rbi": None, "sb": None, "tb": None}
    bat = stats.get("batting") or {}
    run = stats.get("baseRunning") or {}
    ab, h = int(bat.get("atBats") or 0), int(bat.get("hits") or 0)
    d, t, hr = int(bat.get("doubles") or 0), int(bat.get("triples") or 0), int(bat.get("homeRuns") or 0)
    rbi = int(bat.get("rbi") or 0)
    sb = int(run.get("stolenBases") or bat.get("stolenBases") or 0)
    tb = (h - d - t - hr) + 2 * d + 3 * t + 4 * hr
    return {"ab": ab, "h": h, "d": d, "t": t, "hr": hr, "rbi": rbi, "sb": sb, "tb": tb}


def short_line(r: dict) -> str | None:
    """'2-4, 2B, RBI': hits-at-bats, then 2B, 3B, HR, RBI, SB, each with a count when above 1."""
    if r["h"] is None:
        return None
    parts = [f"{r['h']}-{r['ab']}"]
    for key, tag in (("d", "2B"), ("t", "3B"), ("hr", "HR"), ("rbi", "RBI"), ("sb", "SB")):
        if r[key]:
            parts.append(tag if r[key] == 1 else f"{r[key]} {tag}")
    return ", ".join(parts)


def results(box: dict, linescore: dict, starters: dict, lineups: dict) -> dict:
    """A8 item 4 step 3's result fields only. First 5 and run in the 1st from the line-score innings."""
    bt = box["teams"]
    score = {s: int(bt[s]["teamStats"]["batting"]["runs"]) for s in SIDES}
    inn = linescore["innings"]
    f5 = {s: sum((i.get(s, {}).get("runs") or 0) for i in inn[:5]) for s in SIDES}
    first = sum((inn[0].get(s, {}).get("runs") or 0) for s in SIDES) if inn else None
    ks = {}
    for s in SIDES:  # record_outcomes._starter_strikeouts: null if the starter did not pitch
        p = bt[s]["players"].get(f"ID{starters[s]['id']}")
        pitching = ((p or {}).get("stats") or {}).get("pitching") or {}
        ks[s] = int(pitching.get("strikeOuts") or 0) if pitching else None
    batters = {}
    for s in SIDES:
        rows = []
        for p in lineups[s]:
            r = batter_result((bt[s]["players"].get(f"ID{p['id']}") or {}).get("stats"))
            rows.append({"id": p["id"], "h": r["h"], "tb": r["tb"], "hr": r["hr"], "rbi": r["rbi"], "sb": r["sb"],
                         "line": short_line(r)})
        batters[s] = rows
    led = "away" if f5["away"] > f5["home"] else "home" if f5["home"] > f5["away"] else "tied"
    return {"away_score": score["away"], "home_score": score["home"], "led_after_5": led,
            "run_in_first": None if first is None else first > 0, "total_runs": score["away"] + score["home"],
            "starter_k": ks, "batters": batters}


def page(pred: dict, res: dict, updated_at: str) -> dict:
    g = pred["games"][0]
    game = {k: g[k] for k in ("game_id", "away_team", "home_team", "start_time_utc", "park", "roof")}
    game.update({"status": "final", "away_score": res["away_score"], "home_score": res["home_score"]})
    game.update({k: g[k] for k in ("inning", "inning_half", "outs", "markets", "weather", "starters", "lineups")})
    game["results"] = res
    return {"date": pred["date"], "updated_at": updated_at, "model_version": pred["model_version"],
            "note": pred["note"], "label": pred["label"], "games": [game], "beat_the_streak": pred["beat_the_streak"]}


def run(resolver, expect: dict, sched_path: Path, box_dir: Path, ls_dir: Path, out_dir: Path,
        iterations: int = ITERATIONS, fetch=fetch_cached, weather_fn=get_game_weather,
        script_path: Path = Path(__file__)) -> dict:
    """The replay. resolver.pqm / resolver.cqm hold the full matrices as loaded; they are replaced by the cut."""
    t0 = time.time()
    log_path = out_dir / "replay_ws5_log.txt"
    pred_path, page_path = out_dir / "ws5_prediction.json", out_dir / "ws5_page.json"
    lines = [f"== run {datetime.now().astimezone().isoformat(timespec='seconds')} ==",
             f"script {script_path.name} sha256 {sha256_bytes(Path(script_path).read_bytes())}"]
    write_log = True
    try:
        # 1. matrices: stop rule on the full files, then the cut
        check_matrices({"pitch": resolver.pqm, "contact": resolver.cqm}, lines)
        day = pd.Timestamp(expect["date"])
        resolver.pqm, resolver.cqm = cut_before(resolver.pqm, day), cut_before(resolver.cqm, day)
        lines.append(f"matrices cut to game_date < {day.date()}: pitch rows {len(resolver.pqm):,}, "
                     f"contact rows {len(resolver.cqm):,} ({time.time() - t0:.1f}s)")
        # 2-3. schedule, box score, pre-simulation fields, identity checks
        sched_bytes = sched_path.read_bytes()
        lines.append(f"schedule {sched_path.name} sha256 {sha256_bytes(sched_bytes)}")
        sched = schedule_fields(json.loads(sched_bytes), expect["date"])
        pk = sched["game_pk"]
        box_bytes = fetch(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", box_dir / f"{pk}.json")
        lines.append(f"box score {pk}.json sha256 {sha256_bytes(box_bytes)}")
        box_raw = json.loads(box_bytes)
        box = box_fields(box_raw)
        weather = weather_fn(sched["teams"]["home"]["name"], sched["game_date_utc"])
        identity_checks(sched, box, weather, expect)
        lines.append(f"identity checks passed: gamePk {pk}, gameType {sched['game_type']}, "
                     f"{sched['teams']['away']['name']} at {sched['teams']['home']['name']}, {sched['venue']}")
        lineups, starters = {}, {}
        for s in SIDES:
            pl = box[s]["players"]
            lineups[s] = [{"id": pid, "name": pl[pid]["name"]} for pid in box[s]["starting_nine"]]
            prob = sched["teams"][s]["probable_id"]
            sid = prob or (box[s]["pitchers"][0] if box[s]["pitchers"] else None)
            name = sched["teams"][s]["probable_name"] if prob else pl.get(sid, {}).get("name")
            starters[s] = {"id": sid, "name": name, "source": "probable" if prob else "box_first_pitcher"}
            lines.append(f"{s} starter source: {starters[s]['source']}")
        names = [p["name"] for s in SIDES for p in lineups[s]]
        if len(set(names)) < len(names) or any(st["id"] is None for st in starters.values()):
            raise StopReplay("duplicate batter name (props are keyed by name) or no starter")
        # 4. simulate as backtest_v4 does, with empty bullpens
        seed = pk % (2 ** 32)
        np.random.seed(seed)
        t_sim = time.time()
        res = run_advanced_monte_carlo(resolver, lineups["away"], lineups["home"], starters["away"]["id"],
                                       starters["home"]["id"], [], [], density_ratio=1.0, iterations=iterations)
        sim_s = time.time() - t_sim
        lines.append(f"gamePk {pk}; iterations {iterations}; seed {seed}; simulation {sim_s:.1f}s")
        # 5. prediction file; rerun rule
        pred = prediction(expect["date"], sched, starters, lineups, weather, res)
        pred_bytes = dumps(pred)
        if pred_path.exists():
            if pred_path.read_bytes() != pred_bytes:
                write_log = False
                raise RerunMismatch(f"{pred_path.name} exists and the new prediction differs; nothing written")
            lines.append(f"prediction {pred_path.name} sha256 {sha256_bytes(pred_bytes)} (rerun: identical, kept)")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path.write_bytes(pred_bytes)
            lines.append(f"prediction {pred_path.name} sha256 {sha256_bytes(pred_bytes)} (written)")
        # 6. only now: line score and result fields
        ls_bytes = fetch(f"https://statsapi.mlb.com/api/v1/game/{pk}/linescore", ls_dir / f"{pk}.json")
        lines.append(f"line score {pk}.json sha256 {sha256_bytes(ls_bytes)}")
        resb = results(box_raw, json.loads(ls_bytes), starters, lineups)
        updated = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        page_bytes = dumps(page(pred, resb, updated))
        page_path.write_bytes(page_bytes)
        lines.append(f"page {page_path.name} sha256 {sha256_bytes(page_bytes)}")
        lines.append(f"total {time.time() - t0:.1f}s; finished")
        return {"prediction_sha256": sha256_bytes(pred_bytes), "page_sha256": sha256_bytes(page_bytes),
                "sim_seconds": sim_s, "prediction": pred, "results": resb}
    except StopReplay as e:
        lines.append(f"STOPPED: {e}")
        raise
    except RerunMismatch:
        raise
    except Exception as e:
        lines.append(f"CRASHED: {type(e).__name__}")
        raise
    finally:
        if write_log:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n\n")


def main() -> int:
    t = time.time()
    resolver = MatchupResolver(pqm_path=str(PQM), cqm_path=str(CQM), load_recent_form=False,
                               load_arsenal_profile=False)
    print(f"[load] {time.time() - t:.0f}s", flush=True)
    try:
        out = run(resolver, WS5, API / "schedule" / f"{WS5['date']}.json", API / "box", API / "linescore", OUT_DIR)
    except (StopReplay, RerunMismatch) as e:
        print(f"STOPPED: {e}")
        return 1
    print(f"prediction sha256 {out['prediction_sha256']}\npage sha256 {out['page_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
