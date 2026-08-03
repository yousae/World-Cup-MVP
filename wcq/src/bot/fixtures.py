"""WC 2026 match schedule management.

Downloads the full 104-match schedule (group stage known; knockout discovered
dynamically from market data as the tournament progresses) and caches it in a
local JSON file at WCQ_SCHEDULE_PATH.

Primary source: fixturedownload.com CSV feed (UTC timezone variant).
Fallback: kickoffclock.com JSON feed.

Run `python src/bot/fixtures.py --download` once to populate the cache.
After that, all functions read from the cache — no network calls.

Match window for live polling: kickoff_utc → kickoff_utc + 2h30m.
"""
from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

SCHEDULE_PATH = Path(os.environ.get(
    "WCQ_SCHEDULE_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "schedule_2026.json"),
))

_MATCH_DURATION = timedelta(hours=2, minutes=30)

# Unresolved knockout slots, before the group stage decides who's actually
# playing. The feed doesn't use "TBD" for these -- it uses group-position
# codes like "1A" (winner of Group A), "2B" (runner-up of Group B), or
# "3ABCDF" (best-third-place placeholder spanning several groups). A plain
# "TBD" check misses all of these, which is how 56 predictions ended up
# logged against fixtures where neither team was known yet.
_UNRESOLVED_TEAM_RE = re.compile(r"^\d[A-Z]+$|^TBD|^To be announced$", re.I)

# Fixturedownload.com UTC CSV feed (FIFA World Cup 2026)
_FDL_URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"
_KICKOFFCLOCK_URL = "https://www.kickoffclock.com/downloads/world-cup-2026-schedule.json"

# ---------------------------------------------------------------------------
# Internal normalisation helpers
# ---------------------------------------------------------------------------

_TEAM_ALIASES: dict[str, str] = {
    "USA":                    "United States",
    "Ivory Coast":            "Côte d'Ivoire",
    "Cape Verde":             "Cabo Verde",
    "Turkiye":                "Türkiye",
    "Korea Republic":         "South Korea",
    "IR Iran":                "Iran",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Curacao":                "Curaçao",
    "Congo, DR":              "Congo DR",
    "DR Congo":               "Congo DR",
}


def _normalise_team(name: str) -> str:
    name = name.strip()
    return _TEAM_ALIASES.get(name, name)


def _parse_kickoff(raw: str) -> str:
    """Return ISO-8601 UTC string from whatever the feed provides."""
    raw = raw.strip()
    # Try common formats
    for fmt in (
        "%d/%m/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d %b %Y %H:%M",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    # Fallback: try dateutil
    try:
        from dateutil import parser as du
        dt = du.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return raw


def _fixture_from_fdl_row(row: dict) -> dict | None:
    """Parse one row from the fixturedownload.com JSON feed."""
    home = _normalise_team(str(row.get("HomeTeam", "") or row.get("home_team", "")))
    away = _normalise_team(str(row.get("AwayTeam", "") or row.get("away_team", "")))
    if not home or not away or _UNRESOLVED_TEAM_RE.match(home) or _UNRESOLVED_TEAM_RE.match(away):
        return None

    raw_date = str(
        row.get("DateUtc", "")
        or row.get("Date", "")
        or row.get("date", "")
        or row.get("kickoff", "")
    )
    if not raw_date:
        return None

    kickoff = _parse_kickoff(raw_date)
    group = str(row.get("Group", "") or row.get("group", "")).strip() or None
    round_name = str(row.get("RoundNumber", "") or row.get("round", "")).strip()
    venue = str(row.get("Location", "") or row.get("venue", "")).strip() or None

    from src.bot.storage import make_match_id
    return {
        "match_id": make_match_id(home, away, kickoff),
        "home_team": home,
        "away_team": away,
        "kickoff_utc": kickoff,
        "venue": venue,
        "group": group,
        "stage": _infer_stage(round_name, group),
    }


def _infer_stage(round_name: str, group: str | None) -> str:
    rn = round_name.lower()
    if group or "group" in rn:
        return "group"
    if "16" in rn or "32" in rn:
        return "R32"
    if "quarter" in rn or "qf" in rn:
        return "QF"
    if "semi" in rn or "sf" in rn:
        return "SF"
    if "final" in rn:
        return "final"
    return "unknown"


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------

def download_schedule(force: bool = False) -> list[dict]:
    """Fetch the full schedule and write it to SCHEDULE_PATH.

    Returns the list of fixture dicts. Safe to re-run; pass force=True to
    overwrite an existing cache.
    """
    if SCHEDULE_PATH.exists() and not force:
        print(f"[fixtures] Schedule already cached at {SCHEDULE_PATH}. Use --force to refresh.")
        return load_schedule()

    fixtures: list[dict] = []
    errors: list[str] = []

    for url, label in [(_FDL_URL, "fixturedownload"), (_KICKOFFCLOCK_URL, "kickoffclock")]:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("fixtures", data.get("matches", data.get("data", [])))
            else:
                rows = []
            parsed = [_fixture_from_fdl_row(row) for row in rows]
            fixtures = [f for f in parsed if f is not None]
            if fixtures:
                print(f"[fixtures] Loaded {len(fixtures)} fixtures from {label}")
                break
        except Exception as e:
            errors.append(f"{label}: {e}")

    if not fixtures:
        print(f"[fixtures] All feeds failed: {errors}")
        print("[fixtures] Writing empty schedule. Run again when network is available.")

    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEDULE_PATH, "w") as f:
        json.dump({"updated_at": datetime.now(timezone.utc).isoformat(), "fixtures": fixtures}, f, indent=2)
    print(f"[fixtures] Saved {len(fixtures)} fixtures to {SCHEDULE_PATH}")
    return fixtures


def load_schedule() -> list[dict]:
    """Load cached schedule. Returns empty list if not yet downloaded."""
    if not SCHEDULE_PATH.exists():
        return []
    with open(SCHEDULE_PATH) as f:
        data = json.load(f)
    return data.get("fixtures", [])


def add_or_update_fixture(fixture: dict) -> None:
    """Upsert a single fixture into the cache (used for knockout matches discovered at runtime)."""
    fixtures = load_schedule()
    idx = next((i for i, f in enumerate(fixtures) if f["match_id"] == fixture["match_id"]), None)
    if idx is not None:
        fixtures[idx] = fixture
    else:
        fixtures.append(fixture)
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if SCHEDULE_PATH.exists():
        with open(SCHEDULE_PATH) as fh:
            existing = json.load(fh)
    existing["fixtures"] = fixtures
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(SCHEDULE_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(iso: str) -> datetime:
    """Parse an ISO-8601 string with or without timezone info."""
    try:
        from dateutil import parser as du
        dt = du.parse(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # Strip trailing Z and parse manually as UTC
        clean = iso.rstrip("Z").replace("+00:00", "")
        dt = datetime.fromisoformat(clean)
        return dt.replace(tzinfo=timezone.utc)


def get_todays_fixtures() -> list[dict]:
    """All fixtures with kickoff on today's UTC date."""
    today = _now_utc().date()
    return [
        f for f in load_schedule()
        if _parse_dt(f["kickoff_utc"]).date() == today
    ]


def get_upcoming_within(hours: float = 2.0) -> list[dict]:
    """Fixtures whose kickoff is between now and now+hours."""
    now = _now_utc()
    cutoff = now + timedelta(hours=hours)
    return [
        f for f in load_schedule()
        if now <= _parse_dt(f["kickoff_utc"]) <= cutoff
    ]


def get_live_now() -> list[dict]:
    """Fixtures currently in their match window (kickoff → kickoff + 2h30m)."""
    now = _now_utc()
    return [
        f for f in load_schedule()
        if _parse_dt(f["kickoff_utc"]) <= now <= _parse_dt(f["kickoff_utc"]) + _MATCH_DURATION
    ]


def get_recently_finished(minutes_ago: int = 90) -> list[dict]:
    """Fixtures whose match window ended within the last N minutes."""
    now = _now_utc()
    cutoff = now - timedelta(minutes=minutes_ago)
    return [
        f for f in load_schedule()
        if cutoff <= _parse_dt(f["kickoff_utc"]) + _MATCH_DURATION <= now
    ]


def find_fixture(home: str, away: str, date_str: str | None = None) -> dict | None:
    """Find a fixture by team names (approximate, case-insensitive)."""
    home_n = _normalise_team(home).lower()
    away_n = _normalise_team(away).lower()
    for f in load_schedule():
        fh = f["home_team"].lower()
        fa = f["away_team"].lower()
        if (home_n in fh or fh in home_n) and (away_n in fa or fa in away_n):
            if date_str is None or date_str in f["kickoff_utc"]:
                return f
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--today", action="store_true")
    args = ap.parse_args()
    if args.download or args.force:
        download_schedule(force=args.force)
    if args.today:
        for f in get_todays_fixtures():
            print(f"{f['home_team']} vs {f['away_team']}  {f['kickoff_utc']}")
    if not (args.download or args.force or args.today):
        sched = load_schedule()
        print(f"Cached fixtures: {len(sched)}")
        for f in sched[:5]:
            print(f" {f['home_team']} vs {f['away_team']}  {f['kickoff_utc']}")
