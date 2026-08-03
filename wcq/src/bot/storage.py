"""SQLite persistence layer for predictions, calibration, paper ledger, and dedup keys.

Configured via WCQ_DB_PATH env var (default: wcq_bot.db).
All public functions are safe to call before init_db() — they'll raise clearly.
Call init_db() once at startup in each process.
"""
from __future__ import annotations
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

_DB_PATH: str = os.environ.get("WCQ_DB_PATH", "wcq_bot.db")
_MEM_CONN: sqlite3.Connection | None = None  # persistent handle for :memory: DBs


def set_db_path(path: str | Path) -> None:
    """Override the database path (call before init_db in tests)."""
    global _DB_PATH, _MEM_CONN
    _DB_PATH = str(path)
    _MEM_CONN = None  # clear cached in-memory connection when path changes


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    global _MEM_CONN
    if _DB_PATH == ":memory:":
        # Keep a single persistent connection for :memory: — closing it would
        # destroy the database. Thread safety: single-process test use only.
        if _MEM_CONN is None:
            _MEM_CONN = sqlite3.connect(":memory:", check_same_thread=False)
            _MEM_CONN.row_factory = sqlite3.Row
        con = _MEM_CONN
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
    else:
        con = sqlite3.connect(_DB_PATH)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()


def init_db(path: str | Path | None = None) -> None:
    """Create all tables if they don't already exist. Safe to call multiple times."""
    if path is not None:
        set_db_path(path)
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id        TEXT NOT NULL UNIQUE,
                home_team       TEXT NOT NULL,
                away_team       TEXT NOT NULL,
                kickoff_utc     TEXT NOT NULL,
                p_home_model    REAL,
                p_draw_model    REAL,
                p_away_model    REAL,
                p_home_market   REAL,
                p_draw_market   REAL,
                p_away_market   REAL,
                platform        TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS match_results (
                match_id        TEXT PRIMARY KEY,
                home_team       TEXT NOT NULL,
                away_team       TEXT NOT NULL,
                home_score      INTEGER,
                away_score      INTEGER,
                winner          TEXT,
                recorded_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS calibration (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of           TEXT NOT NULL,
                n_matches       INTEGER,
                brier_model     REAL,
                brier_market    REAL,
                logloss_model   REAL,
                logloss_market  REAL,
                computed_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS paper_ledger (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id        TEXT NOT NULL,
                home_team       TEXT NOT NULL,
                away_team       TEXT NOT NULL,
                team_backed     TEXT NOT NULL,
                outcome         TEXT NOT NULL,
                platform        TEXT NOT NULL,
                market_prob     REAL NOT NULL,
                model_prob      REAL NOT NULL,
                kelly_fraction  REAL NOT NULL,
                stake           REAL NOT NULL,
                bankroll_before REAL NOT NULL,
                status          TEXT DEFAULT 'open',
                pnl             REAL,
                bankroll_after  REAL,
                placed_at       TEXT DEFAULT (datetime('now')),
                settled_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS sent_alerts (
                alert_key       TEXT PRIMARY KEY,
                sent_at         TEXT DEFAULT (datetime('now'))
            );
        """)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def save_prediction(
    match_id: str,
    home_team: str,
    away_team: str,
    kickoff_utc: str,
    model_probs: dict[str, float],
    market_probs: dict[str, float] | None = None,
    platform: str | None = None,
) -> None:
    mk = market_probs or {}
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO predictions
               (match_id, home_team, away_team, kickoff_utc,
                p_home_model, p_draw_model, p_away_model,
                p_home_market, p_draw_market, p_away_market, platform)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                match_id, home_team, away_team, kickoff_utc,
                model_probs.get("home"), model_probs.get("draw"), model_probs.get("away"),
                mk.get("home"), mk.get("draw"), mk.get("away"),
                platform,
            ),
        )


def get_prediction(match_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM predictions WHERE match_id = ?", (match_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_predictions() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM predictions ORDER BY kickoff_utc").fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_result(
    match_id: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> None:
    if home_score > away_score:
        winner = "home"
    elif home_score == away_score:
        winner = "draw"
    else:
        winner = "away"
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO match_results
               (match_id, home_team, away_team, home_score, away_score, winner)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (match_id, home_team, away_team, home_score, away_score, winner),
        )


def get_result(match_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM match_results WHERE match_id = ?", (match_id,)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_and_save_calibration() -> dict:
    """Recompute Brier score and log-loss for model and market over all settled matches.

    Three-outcome Brier score: sum over outcomes of (p - actual)^2, normalised by
    n_matches (not n_outcomes) — comparable to single-outcome Brier on binary markets.
    Log-loss is per-match, targeting only the actual outcome.

    Market-side scores are computed ONLY over matches that actually have all
    three market probabilities logged. A NULL market prob used to silently
    coerce to 0.0, which scored as a maximally wrong prediction (Brier
    contribution of exactly 1.0 every time) rather than "no data" — with
    live market capture never populating those columns during the 2026
    tournament, every match was getting graded that way, making the market
    look uniformly terrible instead of simply absent. brier_market /
    logloss_market are None (not 0.0 or 1.0) when no settled match has
    market data at all.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT p.p_home_model, p.p_draw_model, p.p_away_model,
                   p.p_home_market, p.p_draw_market, p.p_away_market,
                   r.winner
            FROM predictions p
            JOIN match_results r ON p.match_id = r.match_id
            WHERE r.winner IS NOT NULL
              AND p.p_home_model IS NOT NULL
        """).fetchall()

    if not rows:
        return {}

    brier_model = logloss_model = 0.0
    brier_market = logloss_market = 0.0
    n = len(rows)
    n_market = 0
    _OUTCOME_IDX = {"home": 0, "draw": 1, "away": 2}

    for row in rows:
        actual_idx = _OUTCOME_IDX.get(row["winner"], -1)
        if actual_idx < 0:
            continue
        model_vec = (row["p_home_model"] or 0, row["p_draw_model"] or 0, row["p_away_model"] or 0)
        has_market = all(
            row[k] is not None for k in ("p_home_market", "p_draw_market", "p_away_market")
        )

        for i, mp in enumerate(model_vec):
            actual = 1.0 if i == actual_idx else 0.0
            brier_model += (mp - actual) ** 2
        logloss_model += -math.log(max(model_vec[actual_idx], 1e-9))

        if has_market:
            n_market += 1
            market_vec = (row["p_home_market"], row["p_draw_market"], row["p_away_market"])
            for i, mk in enumerate(market_vec):
                actual = 1.0 if i == actual_idx else 0.0
                brier_market += (mk - actual) ** 2
            logloss_market += -math.log(max(market_vec[actual_idx], 1e-9))

    result = {
        "n_matches": n,
        "n_market_matches": n_market,
        "brier_model": brier_model / n,
        "brier_market": (brier_market / n_market) if n_market else None,
        "logloss_model": logloss_model / n,
        "logloss_market": (logloss_market / n_market) if n_market else None,
    }
    with _conn() as con:
        con.execute(
            """INSERT INTO calibration
               (as_of, n_matches, brier_model, brier_market, logloss_model, logloss_market)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                result["n_matches"],
                result["brier_model"],
                result["brier_market"],
                result["logloss_model"],
                result["logloss_market"],
            ),
        )
    return result


def get_latest_calibration() -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM calibration ORDER BY computed_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Paper ledger
# ---------------------------------------------------------------------------

def place_paper_bet(
    match_id: str,
    home_team: str,
    away_team: str,
    team_backed: str,
    outcome: str,
    platform: str,
    market_prob: float,
    model_prob: float,
    kelly_fraction: float,
    stake: float,
    bankroll_before: float,
) -> int:
    """Insert an open bet and return the new row id."""
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO paper_ledger
               (match_id, home_team, away_team, team_backed, outcome, platform,
                market_prob, model_prob, kelly_fraction, stake, bankroll_before)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                match_id, home_team, away_team, team_backed, outcome, platform,
                market_prob, model_prob, kelly_fraction, stake, bankroll_before,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]


def settle_paper_bet(bet_id: int, winner: str) -> dict | None:
    """Settle a bet given the actual match winner. Returns updated row or None if not found."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM paper_ledger WHERE id = ?", (bet_id,)
        ).fetchone()
        if not row or row["status"] != "open":
            return None
        row = dict(row)
        won = row["outcome"] == winner
        # Decimal odds implied by market_prob: b = (1/p) - 1. Profit on win = stake * b.
        pnl = row["stake"] * (1.0 / row["market_prob"] - 1) if won else -row["stake"]
        status = "won" if won else "lost"
        bankroll_after = row["bankroll_before"] + pnl
        con.execute(
            """UPDATE paper_ledger
               SET status = ?, pnl = ?, bankroll_after = ?, settled_at = datetime('now')
               WHERE id = ?""",
            (status, pnl, bankroll_after, bet_id),
        )
    return {**row, "status": status, "pnl": pnl, "bankroll_after": bankroll_after}


def get_open_bets_for_team(team: str) -> list[dict]:
    """Return all open bets that involve a specific team (by name in either slot)."""
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM paper_ledger
               WHERE status = 'open'
                 AND (home_team = ? OR away_team = ? OR team_backed = ?)""",
            (team, team, team),
        ).fetchall()
    return [dict(r) for r in rows]


def get_open_bets() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM paper_ledger WHERE status = 'open'"
        ).fetchall()
    return [dict(r) for r in rows]


def get_current_bankroll(initial: float = 1000.0) -> float:
    """Bankroll = initial + settled P&L − open stakes (open stakes are at risk)."""
    with _conn() as con:
        settled_pnl = con.execute(
            "SELECT COALESCE(SUM(pnl), 0.0) FROM paper_ledger WHERE status IN ('won','lost')"
        ).fetchone()[0]
        open_stakes = con.execute(
            "SELECT COALESCE(SUM(stake), 0.0) FROM paper_ledger WHERE status = 'open'"
        ).fetchone()[0]
    return initial + settled_pnl - open_stakes


def get_pnl_summary() -> dict:
    with _conn() as con:
        rows = con.execute(
            "SELECT status, COUNT(*) n, COALESCE(SUM(pnl),0) total FROM paper_ledger GROUP BY status"
        ).fetchall()
        total_wagered = con.execute(
            "SELECT COALESCE(SUM(stake),0) FROM paper_ledger"
        ).fetchone()[0]
    breakdown = {r["status"]: {"n": r["n"], "pnl": r["total"]} for r in rows}
    return {
        "breakdown": breakdown,
        "total_wagered": total_wagered,
        "current_bankroll": get_current_bankroll(),
        "roi": (get_current_bankroll() - 1000.0) / 1000.0,
    }


# ---------------------------------------------------------------------------
# Alert dedup
# ---------------------------------------------------------------------------

def is_alert_sent(alert_key: str) -> bool:
    with _conn() as con:
        return con.execute(
            "SELECT 1 FROM sent_alerts WHERE alert_key = ?", (alert_key,)
        ).fetchone() is not None


def mark_alert_sent(alert_key: str) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO sent_alerts (alert_key) VALUES (?)", (alert_key,)
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_match_id(home: str, away: str, kickoff_utc: str) -> str:
    """Stable, human-readable match ID derived from teams + date."""
    date = kickoff_utc[:10]
    norm = lambda s: s.lower().replace(" ", "_").replace("-", "_").replace("'", "")
    return f"{norm(home)}_vs_{norm(away)}_{date}"


if __name__ == "__main__":
    import tempfile, os

    # Self-test in a temp file so the real DB isn't polluted with dummy data
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp = f.name
    init_db(tmp)
    mid = make_match_id("Brazil", "Morocco", "2026-06-15T18:00:00Z")
    save_prediction(mid, "Brazil", "Morocco", "2026-06-15T18:00:00Z",
                    {"home": 0.55, "draw": 0.24, "away": 0.21},
                    {"home": 0.52, "draw": 0.26, "away": 0.22}, "polymarket")
    save_result(mid, "Brazil", "Morocco", 2, 0)
    calib = compute_and_save_calibration()
    print("Self-test calibration:", calib)
    print("Self-test P&L:", get_pnl_summary())
    os.unlink(tmp)
    print("Self-test OK")

    # Now create the real DB at the configured path (WCQ_DB_PATH env var or wcq_bot.db)
    set_db_path(os.environ.get("WCQ_DB_PATH", "wcq_bot.db"))
    init_db()
    print(f"Initialised real DB at: {_DB_PATH}")
