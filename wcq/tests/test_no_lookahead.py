"""No-lookahead guarantees for the backtest engine.

The project's core methodological claim is that every backtested prediction
only ever uses data available before it. Nothing enforced or tested that
directly before this file -- test_smoke.py checks isolated pure functions,
and the actual walk-forward date-cutoff logic in src/backtest/engine.py had
zero coverage. These tests construct small synthetic match histories (fast,
deterministic, no network) specifically designed so a lookahead bug would
change the result, then assert it doesn't.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.elo import compute_elo
from src.backtest.engine import build_wc_backtest, build_wc_backtest_full, WC_CUTOFFS


def _match(date, home, away, hs, aw, tournament="Friendly", neutral=True):
    return {
        "date": pd.Timestamp(date), "home_team": home, "away_team": away,
        "home_score": hs, "away_score": aw, "neutral": neutral, "tournament": tournament,
    }


def _df(rows):
    return pd.DataFrame(rows)


def test_compute_elo_ignores_row_order():
    """compute_elo() must sort by date internally -- feeding it the same
    matches in a different order (e.g. as returned by an unordered SQL
    query or API page) must not change the final ratings."""
    rows = [
        _match("2001-01-01", "Alpha", "Beta", 2, 0, tournament="FIFA World Cup qualification"),
        _match("2001-03-01", "Beta", "Gamma", 1, 1, tournament="FIFA World Cup qualification"),
        _match("2001-02-01", "Gamma", "Alpha", 0, 3, tournament="FIFA World Cup qualification"),
        _match("2001-06-01", "Alpha", "Beta", 1, 1, tournament="FIFA World Cup qualification"),
    ]
    in_order = compute_elo(_df(rows))
    shuffled = compute_elo(_df([rows[3], rows[0], rows[2], rows[1]]))

    for team in ("Alpha", "Beta", "Gamma"):
        assert abs(in_order[team] - shuffled[team]) < 1e-9, team


def test_build_wc_backtest_ignores_matches_after_cutoff():
    """Elo ratings feeding a WC backtest are computed once from data strictly
    before the tournament's start date. Appending extreme, outlandish
    "future" matches (dated during or after the tournament) must not change
    a single model_prob in the resulting backtest -- if it does, the cutoff
    filter has a hole in it."""
    year = 2002
    start, end = WC_CUTOFFS[year]

    base_rows = [
        _match("1999-01-01", "Alpha", "Beta", 1, 0, tournament="FIFA World Cup qualification"),
        _match("1999-06-01", "Beta", "Gamma", 2, 2, tournament="FIFA World Cup qualification"),
        _match("2000-01-01", "Gamma", "Alpha", 0, 1, tournament="FIFA World Cup qualification"),
        _match("2001-01-01", "Alpha", "Gamma", 3, 0, tournament="FIFA World Cup qualification"),
        _match(start, "Alpha", "Beta", 1, 0, tournament="FIFA World Cup"),
        _match("2002-06-15", "Beta", "Gamma", 2, 1, tournament="FIFA World Cup"),
        _match(end, "Gamma", "Alpha", 0, 0, tournament="FIFA World Cup"),
    ]

    clean = build_wc_backtest(year, _df(base_rows))

    # Same data, plus deliberately absurd NEW matches (different team pairs,
    # so nothing is duplicated) dated during and after the tournament
    # window -- a 10-0 blowout is exactly what would visibly move Elo if it
    # leaked into training.
    noisy_rows = base_rows + [
        _match("2002-06-25", "Beta", "Alpha", 10, 0, tournament="FIFA World Cup"),
        _match("2002-08-01", "Alpha", "Beta", 0, 10, tournament="Friendly"),  # after the WC entirely
    ]
    noisy = build_wc_backtest(year, _df(noisy_rows))

    # The 3 original matches must have EXACTLY the same model_prob and Elo
    # inputs as before -- the noise rows may add new rows of their own
    # (that's expected), but must not change any pre-existing prediction.
    merged = clean.merge(
        noisy, on=["home", "away", "outcome_label"], suffixes=("_clean", "_noisy"),
    )
    assert len(merged) == len(clean)
    pd.testing.assert_series_equal(
        merged["model_prob_clean"], merged["model_prob_noisy"], check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["elo_home_clean"], merged["elo_home_noisy"], check_names=False,
    )


def test_build_wc_backtest_full_goal_diff_weight_has_no_intra_tournament_lookahead():
    """Regression test for the goal_diff_weight lookahead bug: the weight
    used to be fit on the target tournament's own full match set, so a
    later match's score could change the prediction for an earlier match
    in the same tournament. It's now fit from prior WC editions only, so
    changing a LATER match's score must leave an EARLIER match's
    model_prob completely unchanged."""
    year = 2002
    start, end = WC_CUTOFFS[year]

    # Real, CONFEDERATION-recognised teams so fit_confederation_offsets has
    # at least one cross-confederation match to fit against.
    pre_cutoff = [
        _match("1999-01-01", "Brazil", "Germany", 2, 1, tournament="Friendly"),
        _match("1999-06-01", "Brazil", "Germany", 1, 1, tournament="Friendly"),
        _match("2000-01-01", "Alpha", "Beta", 1, 0, tournament="FIFA World Cup qualification"),
        _match("2000-06-01", "Beta", "Gamma", 0, 2, tournament="FIFA World Cup qualification"),
        _match("2001-01-01", "Gamma", "Alpha", 1, 1, tournament="FIFA World Cup qualification"),
    ]
    # A prior "World Cup" edition (1998) is what goal_diff_weight now fits
    # from -- without one, weight defaults to 0.0 and the test proves nothing.
    prior_wc = [
        _match("1998-06-01", "Alpha", "Beta", 3, 0, tournament="FIFA World Cup"),
        _match("1998-06-10", "Alpha", "Gamma", 2, 0, tournament="FIFA World Cup"),
        _match("1998-06-20", "Beta", "Gamma", 0, 1, tournament="FIFA World Cup"),
    ]
    target_wc = [
        _match(start, "Alpha", "Beta", 1, 0, tournament="FIFA World Cup"),        # earliest match
        _match("2002-06-10", "Alpha", "Gamma", 4, 0, tournament="FIFA World Cup"),
        _match("2002-06-20", "Beta", "Gamma", 0, 5, tournament="FIFA World Cup"),  # will be perturbed
    ]

    matches_a = _df(pre_cutoff + prior_wc + target_wc)
    result_a = build_wc_backtest_full(year, matches_a)

    perturbed_wc = target_wc[:2] + [
        _match("2002-06-20", "Beta", "Gamma", 3, 0, tournament="FIFA World Cup"),  # Beta wins instead
    ]
    matches_b = _df(pre_cutoff + prior_wc + perturbed_wc)
    result_b = build_wc_backtest_full(year, matches_b)

    earliest_a = result_a[(result_a["home"] == "Alpha") & (result_a["away"] == "Beta")]
    earliest_b = result_b[(result_b["home"] == "Alpha") & (result_b["away"] == "Beta")]
    pd.testing.assert_series_equal(
        earliest_a["model_prob"].reset_index(drop=True),
        earliest_b["model_prob"].reset_index(drop=True),
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
    print("all no-lookahead tests passed")
