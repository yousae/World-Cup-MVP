"""Compare model vs. market predictions using the bot's OWN live-logged
predictions (genuine pre-match odds, captured in real time during the
tournament), joined against real outcomes from historical.py.

Usage: python run_market_comparison.py
"""
import sqlite3
import pandas as pd
from src.data.historical import load_results


def brier(p_home, p_draw, p_away, winner: str) -> float:
    actual = {"home": 1.0 if winner == "home" else 0.0,
              "draw": 1.0 if winner == "draw" else 0.0,
              "away": 1.0 if winner == "away" else 0.0}
    return (p_home - actual["home"])**2 + (p_draw - actual["draw"])**2 + (p_away - actual["away"])**2


def main():
    conn = sqlite3.connect("wcq_bot.db")
    preds = pd.read_sql_query(
        "SELECT * FROM predictions WHERE p_home_market IS NOT NULL", conn)
    conn.close()

    matches = load_results()
    matches["date_only"] = matches["date"].dt.strftime("%Y-%m-%d")
    preds["date_only"] = pd.to_datetime(preds["kickoff_utc"]).dt.strftime("%Y-%m-%d")

    print(f"{'Match':<32} {'Model H/D/A':<20} {'Market H/D/A':<20} Winner")
    print("-" * 95)

    brier_model_total, brier_market_total, n = 0.0, 0.0, 0
    for _, p in preds.iterrows():
        real = matches[
            (matches["home_team"] == p["home_team"]) &
            (matches["away_team"] == p["away_team"]) &
            (matches["date_only"] == p["date_only"])
        ]
        if real.empty:
            continue
        r = real.iloc[0]

        if r["home_score"] > r["away_score"]:
            winner = "home"
        elif r["home_score"] == r["away_score"]:
            winner = "draw"
        else:
            winner = "away"

        bm = brier(p["p_home_model"], p["p_draw_model"], p["p_away_model"], winner)
        bk = brier(p["p_home_market"], p["p_draw_market"], p["p_away_market"], winner)
        brier_model_total += bm
        brier_market_total += bk
        n += 1

        print(f"{p['home_team']} vs {p['away_team']:<18} "
              f"{p['p_home_model']:.0%}/{p['p_draw_model']:.0%}/{p['p_away_model']:.0%}   "
              f"{p['p_home_market']:.0%}/{p['p_draw_market']:.0%}/{p['p_away_market']:.0%}   "
              f"{winner}")

    print("-" * 95)
    print(f"Matches compared: {n}")
    if n == 0:
        print("No matches joined — check date/team-name formatting.")
        return
    print(f"Mean Brier — MODEL:  {brier_model_total/n:.4f}")
    print(f"Mean Brier — MARKET: {brier_market_total/n:.4f}")


if __name__ == "__main__":
    main()