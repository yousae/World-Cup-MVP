"""Runtime discovery of per-match Polymarket and Kalshi markets.

Per-match markets roll out only a few days before kickoff, so nothing is
hardcoded. Every pre-match job re-runs discovery to pick up newly listed
markets.

Polymarket (primary, in-play repricing):
  GET /events?slug=world-cup-matches  → all WC per-match markets
  Match by: both team names appear in the market question/title + kickoff date
  close time within ±36hr of fixture kickoff.

Kalshi (cross-platform spread; may close at kickoff, not in-play):
  Discover series dynamically via /series (search soccer/WC 2026 entries).
  For each candidate series, list events and match by team names.
  Always check close_time and status before treating as a live signal.

Results are cached in a JSON sidecar at WCQ_MARKET_CACHE_PATH (default:
data/market_cache.json) with a TTL of MARKET_CACHE_TTL_MIN minutes.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_PATH = Path(os.environ.get(
    "WCQ_MARKET_CACHE_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "market_cache.json"),
))
CACHE_TTL_MIN: int = int(os.environ.get("MARKET_CACHE_TTL_MIN", "30"))

_POLY_EVENTS_URL = f"{config.POLYMARKET_GAMMA}/events"
_POLY_MARKETS_URL = f"{config.POLYMARKET_GAMMA}/markets"
_KALSHI_SERIES_URL = f"{config.KALSHI_BASE}/series"
_KALSHI_EVENTS_URL = f"{config.KALSHI_BASE}/events"
_KALSHI_MARKETS_URL = f"{config.KALSHI_BASE}/markets"

# Team name aliases: market spelling → FIFA/model spelling
# (extends MARKET_TO_FIFA from tournament.py)
try:
    from src.models.tournament import MARKET_TO_FIFA as _M2F
    _ALIASES: dict[str, str] = dict(_M2F)
except ImportError:
    _ALIASES = {}

_ALIASES.update({
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte D'Ivoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "South Korea": "South Korea",
    "Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Curacao": "Curaçao",
    "DR Congo": "Congo DR",
    "Congo, DR": "Congo DR",
    "Iran": "Iran",
    "IR Iran": "Iran",
})


def _norm(name: str) -> str:
    """Normalise a team name: alias map, lower-case, strip whitespace."""
    name = name.strip()
    name = _ALIASES.get(name, name)
    return name.lower()


def _teams_match(market_title: str, home: str, away: str) -> bool:
    """Return True if both team names appear in the market title (case-insensitive)."""
    title_l = market_title.lower()
    home_l = _norm(home)
    away_l = _norm(away)
    # Accept partial match (e.g. "Brazil" in "Brazil vs Morocco")
    home_words = [w for w in home_l.split() if len(w) > 2]
    away_words = [w for w in away_l.split() if len(w) > 2]
    home_ok = any(w in title_l for w in home_words)
    away_ok = any(w in title_l for w in away_words)
    return home_ok and away_ok


def _kickoff_close_enough(market_end: str | None, kickoff_utc: str, window_hr: float = 36.0) -> bool:
    """Return True if the market close time is within window_hr of the kickoff."""
    if not market_end:
        return True  # no close time — assume valid
    try:
        from dateutil import parser as du
        close = du.parse(market_end)
        ko = du.parse(kickoff_utc)
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return abs((close - ko).total_seconds()) < window_hr * 3600
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _cache_get(key: str) -> Any | None:
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    age_min = (time.time() - entry.get("ts", 0)) / 60
    if age_min > CACHE_TTL_MIN:
        return None
    return entry.get("data")


def _cache_set(key: str, data: Any) -> None:
    cache = _load_cache()
    cache[key] = {"ts": time.time(), "data": data}
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Polymarket discovery
# ---------------------------------------------------------------------------

def _fetch_polymarket_wc_events() -> list[dict]:
    """Fetch all active WC 2026 W/D/W match events from the Gamma series endpoint.

    Uses series_slug=soccer-fifwc — the confirmed working discovery path.
    Filters out halftime, spreads, totals, and any non-3-way events.
    Results are cached for CACHE_TTL_MIN minutes.
    """
    cached = _cache_get("poly_wc_events")
    if cached is not None:
        return cached

    events: list[dict] = []
    n_raw = n_wrong_slug = n_filtered_title = n_no_draw_market = 0
    offset = 0
    while True:
        try:
            r = requests.get(
                _POLY_EVENTS_URL,
                params={
                    "series_slug": "soccer-fifwc",
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "offset": offset,
                },
                timeout=20,
            )
            r.raise_for_status()
            page = r.json()
        except Exception as e:
            print(f"[market_discovery] Polymarket events fetch: {e}")
            break

        if not isinstance(page, list):
            print(f"[market_discovery] Polymarket events response wasn't a list: {type(page).__name__}")
            break
        for ev in page:
            n_raw += 1
            slug  = ev.get("slug", "") or ""
            title = ev.get("title", "") or ""
            if not slug.startswith("fifwc"):
                n_wrong_slug += 1
                continue
            # Keep only full-time W/D/W events
            if any(kw in title for kw in ("Halftime", "More Markets", "Spread", "Total")):
                n_filtered_title += 1
                continue
            if not any(m.get("slug", "").endswith("-draw") for m in ev.get("markets", [])):
                n_no_draw_market += 1
                continue
            events.append(ev)

        if len(page) < 100:
            break
        offset += 100

    # This is the single most useful line for diagnosing "market columns are
    # always null" after the fact -- it says exactly which filter ate the
    # events, instead of a downstream caller silently getting None for
    # every single fixture with no way to tell why.
    print(f"[market_discovery] Polymarket WC events: {n_raw} fetched, {len(events)} usable "
          f"(dropped: {n_wrong_slug} wrong slug, {n_filtered_title} halftime/spread/total, "
          f"{n_no_draw_market} no 3-way draw market)")

    _cache_set("poly_wc_events", events)
    return events


def find_polymarket_match(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Find the Polymarket per-match 3-way market for a specific fixture.

    Searches the soccer-fifwc event series for an event whose title contains
    both team names, then combines the three binary markets (home-win / draw /
    away-win) into a single result dict with:
        outcomes = [home, "Draw", away]
        prices   = [home_win_price, draw_price, away_win_price]

    This format is directly consumable by _parse_match_market_probs() in the
    job scripts without any further changes to that function.

    Returns None if no live event is found for the fixture.
    """
    key = f"poly_match_{_norm(home)}_{_norm(away)}_{kickoff_utc[:10]}"
    cached = _cache_get(key)
    if cached is not None:
        return cached if cached else None

    home_words = {w for w in home.lower().split() if len(w) > 2}
    away_words = {w for w in away.lower().split() if len(w) > 2}

    all_events = _fetch_polymarket_wc_events()
    n_title_matched = 0
    for ev in all_events:
        ev_title = ev.get("title", "") or ""
        # Match on event title e.g. "Mexico vs. South Africa" — contains both teams
        if not _teams_match(ev_title, home, away):
            continue
        n_title_matched += 1
        end_date = ev.get("endDate") or ev.get("startDate")
        if not _kickoff_close_enough(end_date, kickoff_utc):
            print(f"[market_discovery] {home} vs {away}: matched event '{ev_title}' but its "
                  f"close time ({end_date}) is outside the {kickoff_utc} window")
            continue

        # Extract the Yes price from each of the 3 binary markets
        home_price = draw_price = away_price = None
        for m in ev.get("markets", []):
            mslug    = m.get("slug", "") or ""
            question = (m.get("question", "") or "").lower()
            try:
                import json as _json
                raw_prices = [float(p) for p in _json.loads(m.get("outcomePrices", "[]"))]
                yes_price  = raw_prices[0] if raw_prices else None
            except Exception as e:
                print(f"[market_discovery] {home} vs {away}: couldn't parse outcomePrices "
                      f"for market '{mslug}': {e}")
                yes_price = None
            if yes_price is None:
                continue

            if mslug.endswith("-draw") or "draw" in question:
                draw_price = yes_price
            elif any(w in question for w in home_words):
                home_price = yes_price
            elif any(w in question for w in away_words):
                away_price = yes_price

        # Fall back: infer any single missing leg from the ~1 sum constraint
        if home_price is None and draw_price is not None and away_price is not None:
            home_price = max(0.0, 1.0 - draw_price - away_price)
        if away_price is None and draw_price is not None and home_price is not None:
            away_price = max(0.0, 1.0 - draw_price - home_price)
        if draw_price is None and home_price is not None and away_price is not None:
            draw_price = max(0.0, 1.0 - home_price - away_price)

        if home_price is None or away_price is None:
            print(f"[market_discovery] {home} vs {away}: matched event '{ev_title}' but "
                  f"couldn't parse enough leg prices (home={home_price}, draw={draw_price}, "
                  f"away={away_price})")
            continue  # couldn't parse enough prices

        result = {
            "market_id": ev.get("slug", ""),
            "question":  ev_title,
            "outcomes":  [home, "Draw", away],
            "prices":    [home_price, draw_price or 0.0, away_price],
            "end_date":  end_date,
            "active":    ev.get("active", True),
            "closed":    ev.get("closed", False),
            "platform":  "polymarket",
        }
        _cache_set(key, result)
        return result

    if n_title_matched == 0:
        print(f"[market_discovery] {home} vs {away}: no event title matched out of "
              f"{len(all_events)} usable WC events")
    _cache_set(key, {})
    return None


def get_polymarket_current_prices(market_id: str) -> dict[str, float] | None:
    """Poll the current YES prices for a Polymarket market by ID."""
    try:
        r = requests.get(
            f"{_POLY_MARKETS_URL}/{market_id}",
            timeout=10,
        )
        r.raise_for_status()
        m = r.json()
        if isinstance(m, list):
            m = m[0] if m else {}
        import json as _json
        outcomes = _json.loads(m.get("outcomes", "[]"))
        prices = [float(p) for p in _json.loads(m.get("outcomePrices", "[]"))]
        return dict(zip(outcomes, prices)) if outcomes and prices else None
    except Exception as e:
        print(f"[market_discovery] price fetch for {market_id}: {e}")
        return None


def get_polymarket_champion_prices() -> dict[str, float]:
    """Current Polymarket champion (WC winner) prices. Always available."""
    try:
        from src.data.markets import fetch_polymarket, winner_probs
        df = fetch_polymarket()
        return winner_probs(df)
    except Exception as e:
        print(f"[market_discovery] champion prices: {e}")
        return {}


def get_kalshi_survival_prices() -> dict[str, dict[str, float]]:
    """Current Kalshi round-survival prices (KXWCROUND). Always available."""
    try:
        from src.data.markets import fetch_kalshi, kalshi_survival_probs
        df = fetch_kalshi()
        return kalshi_survival_probs(df)
    except Exception as e:
        print(f"[market_discovery] Kalshi survival: {e}")
        return {}


# ---------------------------------------------------------------------------
# Kalshi per-match discovery
# ---------------------------------------------------------------------------

def _discover_kalshi_wc_series() -> list[str]:
    """Return Kalshi series tickers likely to contain WC 2026 per-match markets."""
    cached = _cache_get("kalshi_wc_series")
    if cached is not None:
        return cached

    tickers: list[str] = []
    try:
        r = requests.get(
            _KALSHI_SERIES_URL,
            params={"limit": 200},
            timeout=20,
        )
        r.raise_for_status()
        series_list = r.json().get("series", [])
        for s in series_list:
            title = str(s.get("title", "") or s.get("name", "")).lower()
            ticker = str(s.get("ticker", ""))
            # Match any WC/World Cup soccer series that isn't the survival series we already know
            if ("world cup" in title or "wc" in ticker.lower()) and "soccer" in title or "wc2026" in ticker.lower():
                if ticker not in ("KXWCROUND", "KXMENWORLDCUP"):
                    tickers.append(ticker)
    except Exception as e:
        print(f"[market_discovery] Kalshi series discovery: {e}")

    _cache_set("kalshi_wc_series", tickers)
    return tickers


def find_kalshi_match(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict | None:
    """Find the Kalshi per-match market for a fixture, if listed.

    Checks whether the market is open at discovery time and whether close_time
    suggests it trades in-play — callers must recheck status before each poll.

    Returns None if not listed or if already closed.
    """
    key = f"kalshi_match_{_norm(home)}_{_norm(away)}_{kickoff_utc[:10]}"
    cached = _cache_get(key)
    if cached is not None:
        return cached if cached else None

    # Try known WC series first, then dynamically discovered ones
    candidate_tickers = ["KXWCMATCH", "KXWCSOCCER"] + _discover_kalshi_wc_series()

    for ticker in candidate_tickers:
        try:
            r = requests.get(
                _KALSHI_MARKETS_URL,
                params={"series_ticker": ticker, "limit": 200},
                timeout=20,
            )
            if r.status_code == 404:
                continue
            r.raise_for_status()
            mkts = r.json().get("markets", [])
        except Exception as e:
            print(f"[market_discovery] Kalshi series {ticker}: {e}")
            continue

        for m in mkts:
            title = str(m.get("title", ""))
            if not _teams_match(title, home, away):
                continue
            close_time = m.get("close_time") or m.get("expiration_time")
            if not _kickoff_close_enough(close_time, kickoff_utc):
                continue

            status = m.get("status", "open").lower()
            if status in ("settled", "closed"):
                _cache_set(key, {})
                return None

            result = {
                "ticker": m.get("ticker", ""),
                "title": title,
                "close_time": close_time,
                "status": status,
                "yes_bid": m.get("yes_bid_dollars"),
                "yes_ask": m.get("yes_ask_dollars"),
                "platform": "kalshi",
                # True only if close_time is AFTER kickoff (i.e., trades in-play)
                "trades_inplay": _close_is_after_kickoff(close_time, kickoff_utc),
            }
            _cache_set(key, result)
            return result

    _cache_set(key, {})
    return None


def get_kalshi_match_price(ticker: str) -> float | None:
    """Current mid-price for a Kalshi binary YES market."""
    try:
        r = requests.get(
            f"{_KALSHI_MARKETS_URL}/{ticker}",
            timeout=10,
        )
        r.raise_for_status()
        m = r.json().get("market", r.json())
        bid = m.get("yes_bid_dollars")
        ask = m.get("yes_ask_dollars")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2
    except Exception as e:
        print(f"[market_discovery] Kalshi price {ticker}: {e}")
    return None


def _close_is_after_kickoff(close_time: str | None, kickoff_utc: str) -> bool:
    """True if the market closes after kickoff (i.e., it trades in-play)."""
    if not close_time:
        return False
    try:
        from dateutil import parser as du
        close = du.parse(close_time)
        ko = du.parse(kickoff_utc)
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        return close > ko
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience: resolve all markets for a fixture
# ---------------------------------------------------------------------------

def resolve_fixture_markets(
    home: str,
    away: str,
    kickoff_utc: str,
) -> dict[str, Any]:
    """Return all available markets for a fixture, tiered by priority.

    Returns:
        {
          "polymarket_match":   dict | None,   # per-match, primary live signal
          "kalshi_match":       dict | None,   # per-match, cross-platform spread
          "polymarket_champion": dict[team, price],  # always available
          "kalshi_survival":    dict[team, dict[round, price]],  # always available
        }
    """
    return {
        "polymarket_match":    find_polymarket_match(home, away, kickoff_utc),
        "kalshi_match":        find_kalshi_match(home, away, kickoff_utc),
        "polymarket_champion": get_polymarket_champion_prices(),
        "kalshi_survival":     get_kalshi_survival_prices(),
    }


if __name__ == "__main__":
    print("Resolving markets for Brazil vs Morocco 2026-06-15...")
    mkts = resolve_fixture_markets("Brazil", "Morocco", "2026-06-15T18:00:00Z")
    for k, v in mkts.items():
        print(f"  {k}: {type(v).__name__} len={len(v) if v else 0}")
