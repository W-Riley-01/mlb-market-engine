"""Fetch the public MLB Stats API schedule (read-only, no key) and cache it.

Outputs (git-ignored by the repo's *.csv rule):
  analysis/cache/schedule_2026.csv     one row per (gamePk, listed date)
  analysis/cache/first_pitch.csv       actual first-pitch time for requested gamePks

Usage:
  python analysis/fetch_schedule.py                 # schedule only
  python analysis/fetch_schedule.py 823042 824490   # also first-pitch for these gamePks
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).parent / "cache"
BASE = "https://statsapi.mlb.com/api/v1"
START, END = "2026-04-28", "2026-10-16"


def _get(url: str) -> dict:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def fetch_schedule() -> pd.DataFrame:
    url = (f"{BASE}/schedule?sportId=1&startDate={START}&endDate={END}"
           "&gameType=R,F,D,L,W,A")
    rows = []
    for d in _get(url)["dates"]:
        for g in d["games"]:
            st = g["status"]
            rows.append({
                "game_pk": g["gamePk"],
                "listed_date": d["date"],
                "official_date": g.get("officialDate"),
                "scheduled_utc": g.get("gameDate"),
                "game_type": g.get("gameType"),
                "abstract_state": st.get("abstractGameState"),
                "detailed_state": st.get("detailedState"),
                "coded_state": st.get("codedGameState"),
                "reason": st.get("reason"),
                "reschedule_date": g.get("rescheduleDate"),
                "rescheduled_from": g.get("rescheduledFrom"),
                "resumed_from": g.get("resumedFrom"),
                "double_header": g.get("doubleHeader"),
                "game_number": g.get("gameNumber"),
                "day_night": g.get("dayNight"),
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_team": g["teams"]["home"]["team"]["name"],
                "venue": g.get("venue", {}).get("name"),
            })
    return pd.DataFrame(rows)


def fetch_first_pitch(game_pks: list[int]) -> pd.DataFrame:
    rows = []
    for pk in game_pks:
        feed = _get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
                    "?fields=gameData,datetime,dateTime,officialDate,gameInfo,firstPitch,"
                    "delayDurationMinutes,status,detailedState")
        gd = feed.get("gameData", {})
        rows.append({
            "game_pk": pk,
            "scheduled_utc": gd.get("datetime", {}).get("dateTime"),
            "official_date": gd.get("datetime", {}).get("officialDate"),
            "first_pitch_utc": gd.get("gameInfo", {}).get("firstPitch"),
            "delay_minutes": gd.get("gameInfo", {}).get("delayDurationMinutes"),
            "detailed_state": gd.get("status", {}).get("detailedState"),
        })
        time.sleep(0.3)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    sched = fetch_schedule()
    sched.to_csv(CACHE / "schedule_2026.csv", index=False)
    print(f"schedule rows: {len(sched)}  distinct gamePk: {sched.game_pk.nunique()}")
    pks = [int(a) for a in sys.argv[1:]]
    if pks:
        fp = fetch_first_pitch(pks)
        fp.to_csv(CACHE / "first_pitch.csv", index=False)
        print(f"first-pitch rows: {len(fp)}  with firstPitch: {fp.first_pitch_utc.notna().sum()}")
