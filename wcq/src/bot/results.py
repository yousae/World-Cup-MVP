"""Multi-source results feed for post-match scoring.

Priority order (as suggested by the project spec):
  1. Market resolution — when a Polymarket per-match binary market settles to
     ~$1 / ~$0, the winning side is the result. Zero new dependencies.
  2. football-data.org free API — covers the WC and allows ~10 req/min on the
     free tier; needs FOOTBALL_DATA_API_KEY env var.
  3. martj42 historical CSV nightly re-fetch — catches anything missed.

All three sources return the same shape: {"home_score", "away_score", "winner"}.
A 20-30 minute delay after full-time is fine for Brier scoring, so we don't
optimise for low latency.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

API_FOOTBALL_KEY: str = os.environ.get("API_FOOTBALL_KEY", "")

# API-Football (api-football.com / apisports) — direct endpoint, no RapidAPI middleware.
# If you're on RapidAPI instead, swap the base URL and headers in _result_from_api_football.
_APIFOOTBALL_BASE = "https://v3.football.api-sports.io"

# FIFA World Cup 2026 league ID in API-Football.
# Historically the men's WC is league_id=1; confirm at https://www.api-football.com/documentation-v3
_WC_LEAGUE_ID = 1
_WC_SEASON = 2026


# ---------------------------------------------------------------------------
# Source 1: Polymarket market resolution
# ---------------------------------------------------------------------------

def _outcome_from_polymarket_resolution(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Infer W/D/L from a settled Polymarket per-match market.

    A settled binary market has the winning outcome at price ≈ 1.0 (> 0.95).
    We look for outcome titles containing "home wins", "draw", "away wins", or
    team names with "win" — Polymarket title formats vary.
    """
    try:
        from src.bot.market_discovery import find_polymarket_match, get_polymarket_current_prices
        mkt = find_polymarket_match(home, away, kickoff_utc)
        if not mkt:
            return None
        prices = get_polymarket_current_prices(mkt["market_id"])
        if not prices:
            return None

        home_l = home.lower()
        away_l = away.lower()

        for outcome_label, price in prices.items():
            if price < 0.95:
                continue
            ol = outcome_label.lower()
            if "draw" in ol or "tie" in ol:
                return {"home_score": 0, "away_score": 0, "winner": "draw", "source": "polymarket"}
            if home_l in ol or "home" in ol:
                return {"home_score": 1, "away_score": 0, "winner": "home", "source": "polymarket"}
            if away_l in ol or "away" in ol:
                return {"home_score": 0, "away_score": 1, "winner": "away", "source": "polymarket"}
    except Exception as e:
        print(f"[results] Polymarket resolution: {e}")
    return None


# ---------------------------------------------------------------------------
# Source 2: API-Football (api-football.com / apisports direct)
# ---------------------------------------------------------------------------

def _result_from_api_football(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Fetch result from API-Football (api-football.com).

    Requires API_FOOTBALL_KEY env var. Free tier: 100 req/day, covers WC.
    Uses the direct apisports endpoint — if you're on RapidAPI instead,
    change the base URL to https://api-football-v1.p.rapidapi.com/v3 and
    headers to {"X-RapidAPI-Key": key, "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"}.
    """
    if not API_FOOTBALL_KEY:
        return None

    date_str = kickoff_utc[:10]  # YYYY-MM-DD
    try:
        r = requests.get(
            f"{_APIFOOTBALL_BASE}/fixtures",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params={"league": _WC_LEAGUE_ID, "season": _WC_SEASON, "date": date_str},
            timeout=15,
        )
        r.raise_for_status()
        fixtures = r.json().get("response", [])
    except Exception as e:
        print(f"[results] API-Football: {e}")
        return None

    home_words = [w.lower() for w in home.split() if len(w) > 2]
    away_words = [w.lower() for w in away.split() if len(w) > 2]

    for fix in fixtures:
        teams = fix.get("teams", {})
        ht = str(teams.get("home", {}).get("name", "")).lower()
        at = str(teams.get("away", {}).get("name", "")).lower()

        # Loose name match handles API-Football spelling differences
        if not (any(w in ht for w in home_words) and any(w in at for w in away_words)):
            continue

        status_short = fix.get("fixture", {}).get("status", {}).get("short", "")
        if status_short not in ("FT", "AET", "PEN"):
            return None  # not finished yet (FT=full time, AET=after extra time, PEN=penalties)

        goals = fix.get("goals", {})
        hs = goals.get("home")
        as_ = goals.get("away")
        if hs is None or as_ is None:
            # Fall back to score.fulltime
            ft = fix.get("score", {}).get("fulltime", {})
            hs, as_ = ft.get("home"), ft.get("away")
        if hs is None or as_ is None:
            return None

        # For knockout rounds resolved by penalties, the score stays at the
        # AET result — the team that won on pens has teams[home/away].winner=true
        if status_short == "PEN":
            winner = "home" if teams.get("home", {}).get("winner") else "away"
        else:
            winner = "home" if hs > as_ else ("away" if as_ > hs else "draw")

        return {
            "home_score": int(hs),
            "away_score": int(as_),
            "winner": winner,
            "source": "api-football",
        }
    return None


# ---------------------------------------------------------------------------
# Source 3: martj42 nightly CSV re-fetch
# ---------------------------------------------------------------------------

def _result_from_historical_csv(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Check the martj42 historical results CSV for a completed match.

    The CSV is typically updated within a few hours of full time. Re-fetches
    via historical.download_results(force=True) so a stale, previously-cached
    copy of the CSV doesn't hide a result that's actually already posted.

    Team names are normalised through tournament._FIFA_TO_HIST before
    matching, since fixturedownload.com (the bot's schedule source) and
    martj42 (this CSV) disagree on 6 official names (Czechia/Czech Republic,
    Bosnia-Herzegovina/Bosnia and Herzegovina, Türkiye/Turkey,
    Côte d'Ivoire/Ivory Coast, Cabo Verde/Cape Verde, Congo DR/DR Congo) --
    without this, matches involving those teams never matched on word
    overlap alone (e.g. "czechia" never appears as a substring of
    "czech republic"). Home/away order is also checked both ways, since WC
    matches are neutral-venue and the two sources don't always agree on
    which team is listed "home".

    martj42 files each match under its LOCAL kickoff date, while
    kickoff_utc here is a UTC timestamp -- for matches kicking off in the
    evening at a US host city, that's already past midnight UTC, one
    calendar day later than the CSV's date. Checking both kickoff_utc's
    date and the day before catches this without needing real timezone
    conversion per host city.
    """
    from datetime import datetime, timedelta
    ko_date = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00")).date()
    candidate_dates = {str(ko_date), str(ko_date - timedelta(days=1))}
    try:
        from src.data.historical import download_results
        from src.models.tournament import _FIFA_TO_HIST
        df = download_results(force=True)  # always re-fetch for post-match use
        if df is None or df.empty:
            return None

        home_hist = _FIFA_TO_HIST.get(home, home)
        away_hist = _FIFA_TO_HIST.get(away, away)
        home_words = [w.lower() for w in home_hist.split() if len(w) > 2]
        away_words = [w.lower() for w in away_hist.split() if len(w) > 2]

        date_col = df["date"].astype(str).str.slice(0, 10)
        day_matches = df[date_col.isin(candidate_dates)]

        for _, row in day_matches.iterrows():
            ht = str(row.get("home_team", "")).lower()
            at = str(row.get("away_team", "")).lower()

            same_order = any(w in ht for w in home_words) and any(w in at for w in away_words)
            swapped_order = any(w in at for w in home_words) and any(w in ht for w in away_words)
            if not (same_order or swapped_order):
                continue

            hs = int(row.get("home_score", 0))
            as_ = int(row.get("away_score", 0))
            if swapped_order and not same_order:
                # The CSV's home/away is the reverse of our fixture's home/away.
                hs, as_ = as_, hs

            winner = "home" if hs > as_ else ("away" if as_ > hs else "draw")
            return {
                "home_score": hs,
                "away_score": as_,
                "winner": winner,
                "source": "martj42",
            }
    except Exception as e:
        print(f"[results] historical CSV: {e}")
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_result(
    home: str,
    away: str,
    kickoff_utc: str,
    use_polymarket: bool = True,
    use_football_data: bool = True,
    use_historical: bool = True,
) -> dict | None:
    """Try all sources in priority order. Returns first non-None result.

    Shape: {"home_score": int, "away_score": int, "winner": str, "source": str}
    where winner ∈ {"home", "draw", "away"}.
    Returns None if the match hasn't finished or no source has the result yet.
    """
    if use_polymarket:
        r = _outcome_from_polymarket_resolution(home, away, kickoff_utc)
        if r:
            return r

    if use_football_data and API_FOOTBALL_KEY:
        r = _result_from_api_football(home, away, kickoff_utc)
        if r:
            return r

    if use_historical:
        r = _result_from_historical_csv(home, away, kickoff_utc)
        if r:
            return r

    return None


def fetch_and_store_result(
    match_id: str,
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Fetch the result and persist it to the DB if found."""
    from src.bot.storage import save_result, get_result

    existing = get_result(match_id)
    if existing:
        return existing

    result = fetch_result(home, away, kickoff_utc)
    if result:
        save_result(match_id, home, away, result["home_score"], result["away_score"])
        print(f"[results] Stored {home} {result['home_score']}-{result['away_score']} {away} (source: {result['source']})")
    return result


if __name__ == "__main__":
    # Demo: check if a known historical match can be found
    print("Testing result lookup (Brazil vs Germany, 2014-07-08)...")
    r = fetch_result("Brazil", "Germany", "2014-07-08T17:00:00Z",
                     use_polymarket=False, use_football_data=False)
    if r:
        print(f"  {r['home_score']}-{r['away_score']} winner={r['winner']} source={r['source']}")
    else:
        print("  Not found (historical CSV not yet downloaded)")
