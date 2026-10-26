"""Integrity gate for the full v4 backtest (BACKTEST_V4_ANALYSIS_PLAN.md, section 0). Counts rows and keys only."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
OUT = ROOT / "analysis" / "out"
API = ROOT / "analysis" / "cache" / "api"
CHUNKS = "ABCDEF"
R1_PREFIX = "ledger_v4_full"
PREFIX = R1_PREFIX  # v4r2_analysis.py sets "ledger_v4r2"


def load_ledgers() -> pd.DataFrame:
    parts = []
    for c in CHUNKS:
        (f,) = OUT.glob(f"{PREFIX}_{c}_*.csv")
        parts.append(pd.read_csv(f).assign(chunk=c))
    L = pd.concat(parts, ignore_index=True)
    L["season"] = pd.to_datetime(L.date).dt.year
    return L


def box(pk: int) -> dict:
    return json.loads((API / "box" / f"{pk}.json").read_text())


def schedule_games() -> pd.DataFrame:
    """Final regular-season games by official date, from the per-date schedule files the backtest itself cached."""
    rows = []
    for f in sorted((API / "schedule").glob("*.json")):
        ds = f.stem
        if not ("2022-03-01" <= ds <= "2023-10-02" or "2024-03-01" <= ds <= "2025-10-01"):
            continue
        for day in json.loads(f.read_text()).get("dates", []):
            for g in day["games"]:
                st = g["status"]
                if g.get("gameType") != "R" or g.get("officialDate", ds) != ds:
                    continue
                rows.append({"date": ds, "game_pk": g["gamePk"], "state": st.get("detailedState"),
                             "final": st.get("abstractGameState") == "Final"
                             and st.get("detailedState") not in ("Postponed", "Cancelled"),
                             "suspended": bool(g.get("resumedFrom") or g.get("resumeDate"))
                             or "Suspend" in st.get("detailedState", "")})
    S = pd.DataFrame(rows)
    S["season"] = pd.to_datetime(S.date).dt.year
    return S


def main():
    pd.set_option("display.width", 200)
    for c in CHUNKS:
        log = (OUT / f"{PREFIX}_{c}.log").read_text()
        assert "Traceback" not in log and "\nskipped:" in log, f"chunk {c} not finished cleanly"
    print("1. all six logs end with 'skipped:' and contain no Traceback: OK")

    L = load_ledgers()
    dup = L[L.game_pk.duplicated(keep=False)]
    print(f"2. ledger rows {len(L)}; duplicate gamePks: {dup.game_pk.nunique()}")
    print(L.groupby("chunk").agg(rows=("game_pk", "size"), first=("date", "min"), last=("date", "max")).to_string())

    S = schedule_games()
    F = S[S.final & ~S.suspended]
    print("\n3. games per season: schedule (final, official date, not suspended/resumed) vs ledger")
    t = pd.DataFrame({"schedule_final": S[S.final].groupby("season").size(),
                      "of_which_suspended_or_resumed": S[S.final & S.suspended].groupby("season").size(),
                      "schedule_eligible": F.groupby("season").size(),
                      "ledger": L.groupby("season").size()}).fillna(0).astype(int)
    print(t.to_string())
    missing = F[~F.game_pk.isin(L.game_pk)]
    extra = L[~L.game_pk.isin(F.game_pk)]
    print(f"eligible schedule games not in ledger: {len(missing)}; ledger games not in eligible schedule: {len(extra)}")
    reasons = []
    for pk in missing.game_pk:
        bt = box(pk)["teams"] if (API / "box" / f"{pk}.json").exists() else None
        if bt is None:
            reasons.append("no cached box")
        elif bt["away"]["teamStats"]["batting"]["runs"] == bt["home"]["teamStats"]["batting"]["runs"]:
            reasons.append("tied_score")
        elif any(len(bt[s].get("battingOrder", [])) != 9 for s in ("away", "home")):
            reasons.append("no_full_lineup")
        else:
            reasons.append("UNEXPLAINED")
    if len(missing):
        print(missing.assign(reason=reasons).to_string(index=False))
    if len(extra):
        print(extra[["date", "game_pk", "chunk"]].to_string(index=False))

    has_box = L.game_pk.map(lambda pk: (API / "box" / f"{pk}.json").exists())
    print(f"\n4. ledger gamePks with a cached box score: {int(has_box.sum())} of {len(L)}")

    fb = L.away_starter_source.ne("probable") | L.home_starter_source.ne("probable")
    print(f"\n5. exclusions\nactual_fallback games (either side): {int(fb.sum())} -> analysis set {int((~fb).sum())}")
    print("odds_match:", L.odds_match.value_counts().to_dict())
    A = L[~fb]
    print(f"analysis set: closing ML missing {int((A.odds_away_ml_close.isna() | A.odds_home_ml_close.isna()).sum())}, "
          f"closing total missing {int(A.odds_over_close.isna().sum())}; name_collision games {int(L.name_collision.sum())}")
    nb = nj = ns = nsj = nsub = 0
    for _, r in L.iterrows():
        bt = box(r.game_pk)["teams"]
        for b in json.loads(r.batters):
            nb += 1
            pl = bt[b["side"]]["players"].get(f"ID{b['id']}")
            nj += pl is not None
            nsub += pl is None or str(pl.get("battingOrder", "")) != str(b["slot"] * 100)
        for s in ("away", "home"):
            ns += 1
            pid = r[f"{s}_starter_actual"]
            nsj += pd.notna(pid) and f"ID{int(pid)}" in bt[s]["players"]
    print(f"batter ids joined to box: {nj} of {nb} (unjoined {nb - nj}); starter ids joined: {nsj} of {ns} (unjoined {ns - nsj})")
    print(f"lineup entries whose box battingOrder is not slot x 100 (not a true starter): {nsub} of {nb}")

    ok = dup.empty and has_box.all() and "UNEXPLAINED" not in reasons and extra.empty and nj == nb and nsj == ns
    ok &= nsub == 0 or PREFIX == R1_PREFIX  # R1 lineups knowingly include substitutes (addendum 2026-10-23)
    print("\nINTEGRITY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
