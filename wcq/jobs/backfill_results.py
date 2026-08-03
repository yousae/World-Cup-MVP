"""Backfill match_results + calibration from already-logged predictions.

post_match.py is meant to fill match_results as each fixture finishes, but
two real bugs meant it never actually did during the 2026 tournament:
  1. results.py's historical-CSV fallback called a function with a keyword
     argument that doesn't exist (load_results(force_download=True) --
     load_results() takes no arguments at all), so it always raised and was
     silently swallowed by a bare except.
  2. Even with that fixed, ~6 teams have different official names between
     fixturedownload.com (the bot's schedule source) and martj42 (the
     historical CSV), so naive word-overlap matching missed those matches
     entirely. Both are now fixed in results.py.

This job re-runs result lookup for every prediction already logged in the
DB that doesn't have a result yet, then recomputes calibration once at the
end. Safe to re-run any time -- fetch_and_store_result() is a no-op for
matches that already have a stored result.

Usage: python jobs/backfill_results.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bot.storage import (
    init_db, get_all_predictions, get_result,
    compute_and_save_calibration,
)
from src.bot.results import fetch_and_store_result
from src.bot.fixtures import _UNRESOLVED_TEAM_RE


def run() -> None:
    init_db()
    preds = get_all_predictions()
    print(f"[backfill_results] {len(preds)} logged predictions to check")

    filled = skipped_placeholder = already_had_result = not_found = 0

    for p in preds:
        home, away, kickoff, mid = p["home_team"], p["away_team"], p["kickoff_utc"], p["match_id"]

        if _UNRESOLVED_TEAM_RE.match(home) or _UNRESOLVED_TEAM_RE.match(away):
            skipped_placeholder += 1
            continue

        if get_result(mid):
            already_had_result += 1
            continue

        result = fetch_and_store_result(mid, home, away, kickoff)
        if result:
            filled += 1
            print(f"[backfill_results] {home} {result['home_score']}-{result['away_score']} {away} "
                  f"(source: {result['source']})")
        else:
            not_found += 1
            print(f"[backfill_results] No result found for {home} vs {away} ({kickoff})")

    print(f"[backfill_results] done: {filled} filled, {already_had_result} already had a result, "
          f"{skipped_placeholder} skipped (unresolved bracket slots), {not_found} not found")

    calib = compute_and_save_calibration()
    if calib:
        market_note = (
            f"brier_market={calib['brier_market']:.4f} (n={calib['n_market_matches']})"
            if calib["brier_market"] is not None
            else "brier_market=n/a (no logged predictions have market data -- see WRITEUP.md)"
        )
        print(f"[backfill_results] calibration recomputed over {calib['n_matches']} matches: "
              f"brier_model={calib['brier_model']:.4f}  {market_note}")
    else:
        print("[backfill_results] calibration not computed (no matches with both a result and a "
              "model prediction yet)")


if __name__ == "__main__":
    run()
