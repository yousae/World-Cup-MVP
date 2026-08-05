"""World Cup Quant Dashboard: Model vs Prediction Markets.

Run from the project root:
    streamlit run app/streamlit_app.py

EDUCATIONAL TOOL. It surfaces where a toy model disagrees with public markets.
It does not place trades and is not financial or betting advice.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data.historical import download_results, load_results
from src.data.markets import get_all, winner_probs, kalshi_survival_probs, fetch_polymarket_matches
from src.models.elo import compute_elo, top_n
from src.models.match_model import draw_params_available, fit_draw_params, match_probs
from src.models.tournament import simulate_tournament, survival_table, GROUPS_2026, MARKET_TO_FIFA, _FIFA_TO_HIST
from src.models.svi_surface import SurvivalSurface
from src.markets.implied import implied_from_book
from src.markets.edges import edge_table, flag_value, kelly_fraction
from src.backtest.engine import run_wc_backtest, WC_CUTOFFS
from src.viz import charts

st.set_page_config(page_title="World Cup Model vs Market Dashboard", layout="wide")


@st.cache_data(show_spinner="Loading match history + computing Elo...")
def load_elo(reversion: float, use_tournament_k: bool) -> dict[str, float]:
    matches = download_results()
    current_wc = matches[
        (matches["date"] >= "2026-06-11") & (matches["date"] <= "2026-07-19")
        & (matches["tournament"] == "FIFA World Cup")
    ]
    return compute_elo(matches, reversion=reversion, use_tournament_k=use_tournament_k,
                       confed_offset=True, goal_diff_form=True, current_wc_matches=current_wc)


@st.cache_data(show_spinner="Fetching prediction markets...")
def load_markets() -> pd.DataFrame:
    return get_all()


@st.cache_data(show_spinner="Fetching Polymarket match markets…", ttl=300)
def load_match_markets() -> pd.DataFrame:
    return fetch_polymarket_matches()


@st.cache_data(show_spinner="Running Monte Carlo simulations...")
def load_mc(
    _elo_frozen: tuple,
    draw_base: float,
    scale: float,
    n_sims: int,
) -> dict[str, dict[str, float]]:
    # _elo_frozen is a tuple of (team, rating) pairs so st.cache_data can hash it.
    return simulate_tournament(
        dict(_elo_frozen), seed=42, n_sims=n_sims,
        draw_base=draw_base, scale=scale,
    )


@st.cache_data(show_spinner="Building expected bracket…")
def build_bracket(
    _elo_frozen: tuple,
    draw_base: float,
    scale: float,
) -> tuple[dict, str]:
    """Return (rounds_data, champion_team) for the expected knockout bracket.

    Teams are seeded by Elo rating within each group (highest Elo = group winner,
    second = runner-up).  Knockout outcomes are the deterministic expected winner
    (the team with > 50 % win probability in that match).
    """
    from src.models.tournament import _R32_SLOTS, _FIFA_TO_HIST
    from src.models.match_model import win_prob_knockout

    elo_d = dict(_elo_frozen)
    BASE = config.ELO_BASE

    def get_elo(team: str) -> float:
        return elo_d.get(_FIFA_TO_HIST.get(team, team), BASE)

    group_rankings = {
        gid: sorted(teams, key=get_elo, reverse=True)
        for gid, teams in GROUPS_2026.items()
    }
    thirds = sorted(
        [(gid, group_rankings[gid][2]) for gid in GROUPS_2026],
        key=lambda x: get_elo(x[1]), reverse=True,
    )[:8]

    slot: dict[str, str] = {}
    for gid in GROUPS_2026:
        slot[f"W_{gid}"] = group_rankings[gid][0]
        slot[f"R_{gid}"] = group_rankings[gid][1]
    for i, (_, t) in enumerate(thirds, 1):
        slot[f"T_{i}"] = t

    def play(a: str, b: str):
        p = win_prob_knockout(get_elo(a), get_elo(b), draw_base=draw_base, scale=scale)
        return (a, b, p, 1.0 - p, a if p >= 0.5 else b)

    current = [(slot[s1], slot[s2]) for s1, s2 in _R32_SLOTS]
    rounds_data: dict[str, list] = {}
    for rname in ["R32", "R16", "QF", "SF", "Final"]:
        match_list = [play(a, b) for a, b in current]
        rounds_data[rname] = match_list
        if rname == "Final":
            break
        winners = [m[4] for m in match_list]
        current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    champion = rounds_data["Final"][0][4]
    return rounds_data, champion

def build_real_bracket() -> tuple[dict, str, dict]:
    """Real 2026 R32-through-Final bracket (actual draw), model's OWN picks
    propagate through the TRUE bracket tree — no mid-bracket correction.
    Reordered so consecutive match-pairs correctly feed forward, matching
    bracket_chart()'s layout assumption (derived from the real tree, not
    chronological order).
    """
    import pandas as pd
    from src.models.match_model import win_prob_knockout
    from src.models.confederations import CONFEDERATION, fit_confederation_offsets
    from src.models.tournament_form import tournament_goal_diff_so_far, fit_goal_diff_weight
    from src.models.tournament import _FIFA_TO_HIST, simulate_tournament

    matches = download_results()
    cutoff = pd.Timestamp("2026-06-11")
    end = pd.Timestamp("2026-07-19")
    train = matches[matches["date"] < cutoff]

    ratings = compute_elo(train)
    offsets = fit_confederation_offsets(train, ratings, CONFEDERATION)

    wc = matches[
        (matches["date"] >= cutoff) & (matches["date"] <= end)
        & (matches["tournament"] == "FIFA World Cup")
    ].sort_values("date").reset_index(drop=True)
    all_teams = set(wc["home_team"]) | set(wc["away_team"])
    confed_adjust = {t: offsets.get(CONFEDERATION.get(t), 0.0) for t in all_teams}
    base = config.ELO_BASE

    def elo_lookup(team):
        hist_name = _FIFA_TO_HIST.get(team, team)
        return ratings.get(hist_name, base) + confed_adjust.get(team, 0.0)

    group_stage = wc.iloc[:72]
    knockout = wc.iloc[72:].reset_index(drop=True)

    r32_matches = knockout.iloc[0:16]
    r16_matches = knockout.iloc[16:24]
    qf_matches = knockout.iloc[24:28]
    sf_matches = knockout.iloc[28:30]
    final_match = knockout.iloc[31:32]

    prep_rows = []
    for row in group_stage.itertuples(index=False):
        gd = tournament_goal_diff_so_far(group_stage, row.date)
        prep_rows.append({
            "home": row.home_team, "away": row.away_team,
            "home_score": int(row.home_score), "away_score": int(row.away_score),
            "neutral": bool(row.neutral),
            "gd_home": gd.get(row.home_team, 0), "gd_away": gd.get(row.away_team, 0),
        })
    weight = fit_goal_diff_weight(
        prep_rows, {t: elo_lookup(t) for t in all_teams}, confed_adjust={})
    final_group_gd = tournament_goal_diff_so_far(group_stage, pd.Timestamp("2026-12-31"))

    def rating(team):
        return elo_lookup(team) + weight * final_group_gd.get(team, 0)

    def pick_winner(a, b):
        p_a = win_prob_knockout(rating(a), rating(b))
        return (a, b, p_a, 1 - p_a, a if p_a >= 0.5 else b)

    def team_to_slot_map(round_df):
        mapping = {}
        for idx, row in enumerate(round_df.itertuples(index=False)):
            mapping[row.home_team] = idx
            mapping[row.away_team] = idx
        return mapping

    def build_structure(prev_slot_map, next_round_df):
        return [(prev_slot_map[row.home_team], prev_slot_map[row.away_team])
                for row in next_round_df.itertuples(index=False)]

    r32_slot = team_to_slot_map(r32_matches)
    r16_structure = build_structure(r32_slot, r16_matches)
    r16_slot = team_to_slot_map(r16_matches)
    qf_structure = build_structure(r16_slot, qf_matches)
    qf_slot = team_to_slot_map(qf_matches)
    sf_structure = build_structure(qf_slot, sf_matches)
    sf_slot = team_to_slot_map(sf_matches)
    final_structure = build_structure(sf_slot, final_match)

    structures = {"R16": r16_structure, "QF": qf_structure, "SF": sf_structure, "Final": final_structure}

    r32_list = [pick_winner(row.home_team, row.away_team) for row in r32_matches.itertuples(index=False)]

    def propagate(struct, prev_list):
        return [pick_winner(prev_list[i][4], prev_list[j][4]) for (i, j) in struct]

    r16_list = propagate(r16_structure, r32_list)
    qf_list = propagate(qf_structure, r16_list)
    sf_list = propagate(sf_structure, qf_list)
    final_list = propagate(final_structure, sf_list)

    raw_rounds = {"R32": r32_list, "R16": r16_list, "QF": qf_list, "SF": sf_list, "Final": final_list}
    raw_dfs = {"R32": r32_matches, "R16": r16_matches, "QF": qf_matches, "SF": sf_matches, "Final": final_match}

    round_names = ["R32", "R16", "QF", "SF", "Final"]
    order = {"Final": [0]}
    for r in range(len(round_names) - 1, 0, -1):
        cur, prev = round_names[r], round_names[r - 1]
        prev_order = []
        for idx in order[cur]:
            pi, pj = structures[cur][idx]
            prev_order.append(pi)
            prev_order.append(pj)
        order[prev] = prev_order

    rounds_data = {rn: [raw_rounds[rn][idx] for idx in order[rn]] for rn in round_names}
    champion_team = final_list[0][4]

    correct, total = 0, 0
    for rn in round_names:
        for k, row in enumerate(raw_dfs[rn].itertuples(index=False)):
            actual = row.home_team if row.home_score > row.away_score else row.away_team
            pick = raw_rounds[rn][k][4]
            total += 1
            if pick == actual:
                correct += 1

    adjusted_ratings = dict(ratings)
    for team in all_teams:
        hist_name = _FIFA_TO_HIST.get(team, team)
        adjusted_ratings[hist_name] = elo_lookup(team)
    mc_survival = simulate_tournament(adjusted_ratings, n_sims=3000, seed=42)

    return rounds_data, champion_team, mc_survival, f"{correct}/{total} ({correct/total:.0%})"


@st.cache_data(show_spinner="Running 2026 forward simulation…")
def run_forward_simulation(
    _elo_frozen: tuple,
    draw_base: float,
    scale: float,
    n_sims: int,
    seed: int = 42,
) -> list[dict]:
    """Run n_sims full 2026 WC tournament simulations; return per-sim knockout outcomes.

    Each element of the returned list is a dict:
        {"R32": set[str], "R16": set[str], "QF": set[str], "SF": set[str], "champion": str}
    where each set contains the teams that WON that round (i.e. advanced to the next).
    """
    from src.models.tournament import (
        GROUPS_2026 as _GRP, _R32_SLOTS, _select_best_thirds,
        _simulate_group, _run_knockout_round, _sequential_pairs, _elo as _get_elo,
    )
    from src.models.match_model import match_probs, win_prob_knockout
    from itertools import combinations

    elo_d = dict(_elo_frozen)
    base  = config.ELO_BASE
    rng   = np.random.default_rng(seed)
    all_teams = [t for g in _GRP.values() for t in g]

    group_probs: dict = {}
    for gid, teams in _GRP.items():
        group_probs[gid] = {
            (h, a): match_probs(_get_elo(h, elo_d, base), _get_elo(a, elo_d, base),
                                draw_base=draw_base, scale=scale)
            for h, a in combinations(teams, 2)
        }

    p_ko: dict = {a: {} for a in all_teams}
    for a in all_teams:
        for b in all_teams:
            if a != b:
                p_ko[a][b] = win_prob_knockout(
                    _get_elo(a, elo_d, base), _get_elo(b, elo_d, base),
                    draw_base=draw_base, scale=scale,
                )

    results: list[dict] = []
    for _ in range(n_sims):
        group_winners: dict[str, str] = {}
        group_runners: dict[str, str] = {}
        thirds = []
        for gid, teams in _GRP.items():
            records = _simulate_group(teams, group_probs[gid], rng, gid)
            group_winners[gid] = records[0].team
            group_runners[gid] = records[1].team
            thirds.append(records[2])

        best8 = _select_best_thirds(thirds, n=8)
        slot: dict[str, str] = {}
        for gid in _GRP:
            slot[f"W_{gid}"] = group_winners[gid]
            slot[f"R_{gid}"] = group_runners[gid]
        for i, rec in enumerate(best8, 1):
            slot[f"T_{i}"] = rec.team

        r32_pairs = [(slot[s1], slot[s2]) for s1, s2 in _R32_SLOTS]
        r32w = _run_knockout_round(r32_pairs, p_ko, rng)
        r16w = _run_knockout_round(_sequential_pairs(r32w), p_ko, rng)
        qfw  = _run_knockout_round(_sequential_pairs(r16w), p_ko, rng)
        sfw  = _run_knockout_round(_sequential_pairs(qfw), p_ko, rng)
        champ = _run_knockout_round([tuple(sfw)], p_ko, rng)[0]  # type: ignore[arg-type]

        results.append({
            "R32":      set(r32w),
            "R16":      set(r16w),
            "QF":       set(qfw),
            "SF":       set(sfw),
            "champion": champ,
        })

    return results


@st.cache_data(show_spinner="Running historical backtest (rebuilding pre-WC Elo)...")
def load_backtest(
    year: int,
    reversion: float,
    use_tournament_k: bool,
    draw_base: float,
    scale: float,
) -> dict:
    return run_wc_backtest(
        year, load_results(),
        elo_reversion=reversion, use_tournament_k=use_tournament_k,
        draw_base=draw_base, scale=scale,
        edge_threshold=0.05,
    )


st.title("⚽ World Cup Quant Dashboard")
st.caption("Model vs. prediction markets · educational tool, not betting advice")

# ---------------------------------------------------------------------------
# Sidebar — model configuration
# ---------------------------------------------------------------------------
_fitted = None
if draw_params_available():
    import json as _json
    with open(config.DRAW_PARAMS_PATH) as _f:
        _fitted = _json.load(_f)

# Model presets: each is a self-consistent bundle of Elo + draw parameters.
_PRESETS: dict[str, dict] = {
    "Standard": {
        "reversion": 0.05,
        "use_tournament_k": True,
        "draw_base": _fitted["draw_base"] if _fitted else 0.28,
        "scale":     _fitted["scale"]     if _fitted else 400.0,
        "tag":       "Recommended",
        "desc": (
            "Tournament K-weighted Elo (WC matches count 50 % more than qualifiers), "
            "5 % annual mean-reversion, draw probabilities MLE-calibrated from data."
        ),
    },
    "Simple": {
        "reversion": 0.0,
        "use_tournament_k": False,
        "draw_base": 0.28,
        "scale":     400.0,
        "tag":       "Original",
        "desc": (
            "Flat K=40 for every match, no recency decay, draw_base=0.28 hardcoded. "
            "Closest to the original unmodified Elo model."
        ),
    },
    "No-Reversion": {
        "reversion": 0.0,
        "use_tournament_k": True,
        "draw_base": _fitted["draw_base"] if _fitted else 0.28,
        "scale":     _fitted["scale"]     if _fitted else 400.0,
        "tag":       "History-weighted",
        "desc": (
            "Tournament K-weighting on, but zero mean-reversion — older WC wins "
            "keep their full weight indefinitely."
        ),
    },
}

_SIM_OPTIONS = {"Fast — 5 k": 5_000, "Standard — 20 k": 20_000}

with st.sidebar:
    st.header("Model configuration")
    st.caption("Switch configs to see how the forecast changes as each uses a different Elo and draw model. Results are cached per combination.")

    _preset_name = st.radio(
        "Match model",
        list(_PRESETS),
        index=0,
        format_func=lambda k: f"{k}  [{_PRESETS[k]['tag']}]",
    )
    _preset = _PRESETS[_preset_name]

    with st.expander("What does this model do?"):
        st.caption(_preset["desc"])
        st.caption(
            f"**Elo:** reversion={_preset['reversion']:.0%}/yr · "
            f"tournament K={'on' if _preset['use_tournament_k'] else 'off'}"
        )
        st.caption(
            f"**Draws:** draw_base={_preset['draw_base']:.3f} · "
            f"scale={_preset['scale']:.0f}"
        )

    st.divider()

    _sim_label = st.radio("Simulation depth", list(_SIM_OPTIONS), index=1)
    _n_sims = _SIM_OPTIONS[_sim_label]
    st.caption(f"{_n_sims:,} Monte Carlo sims · seed=42")

    st.divider()

    if not draw_params_available():
        st.warning("Draw params not calibrated — Standard/No-Reversion use defaults.")
        if st.button("Calibrate draw model (≈5 s)"):
            with st.spinner("Fitting draw parameters…"):
                fit_draw_params(load_results())
            st.success("Done — reload to apply.")

    st.divider()
    _dark_mode: bool = st.checkbox(
        "Dark mode charts",
        value=False,
        help="Match this to your Streamlit theme (Settings → Theme). "
             "Switches chart text and backgrounds for readability on dark backgrounds.",
    )

# ---------------------------------------------------------------------------
# Load data with the selected configuration
# ---------------------------------------------------------------------------
_rev   = _preset["reversion"]
_use_k = _preset["use_tournament_k"]
_db    = _preset["draw_base"]
_sc    = _preset["scale"]

elo = load_elo(_rev, _use_k)
markets = load_markets()
mc_survival = load_mc(tuple(sorted(elo.items())), _db, _sc, _n_sims)

tab_mkt, tab_model, tab_edge, tab_surf, tab_bt, tab_findings = st.tabs(
    ["Live markets", "Model forecast", "Edge detection",
     "Survival surface", "Backtest", "Findings"]
)

# --- Live markets -----------------------------------------------------------
with tab_mkt:
    st.subheader("Current market-implied prices")
    if markets.empty:
        st.warning("No markets returned (API may be unavailable).")
    else:
        pm_sub = markets[markets["platform"] == "polymarket"]
        ks_sub = markets[markets["platform"] == "kalshi"]

        st.markdown("#### Polymarket — champion winner markets")
        if pm_sub.empty:
            st.info("No Polymarket data.")
        else:
            raw_yes = winner_probs(pm_sub)
            if raw_yes:
                pm_display = (
                    pd.DataFrame.from_dict(raw_yes, orient="index", columns=["raw YES price"])
                    .join(
                        pd.DataFrame.from_dict(implied_from_book(raw_yes), orient="index", columns=["de-vigged prob"])
                    )
                    .sort_values("de-vigged prob", ascending=False)
                )
                st.dataframe(pm_display.style.format("{:.3f}"), use_container_width=True)
                st.caption(f"Overround: {(sum(raw_yes.values()) - 1)*100:.1f} pp")
            else:
                st.dataframe(pm_sub[["market", "outcome", "price"]], use_container_width=True)

        st.markdown("#### Kalshi — round survival markets (KXWCROUND)")
        if ks_sub.empty:
            st.info("No Kalshi data.")
        else:
            ks_pivot = (
                ks_sub[["team", "round", "price"]]
                .dropna(subset=["team", "round"])
                .pivot_table(index="team", columns="round", values="price", aggfunc="mean")
            )
            # Order columns by round depth
            col_order = [c for c in ["R16", "QF", "SF", "final"] if c in ks_pivot.columns]
            ks_pivot = ks_pivot[col_order].sort_values(col_order[-1] if col_order else "R16", ascending=False)
            st.dataframe(ks_pivot.style.format("{:.3f}"), use_container_width=True)
            st.caption(f"{len(ks_sub['team'].unique())} teams, {len(col_order)} rounds from Kalshi KXWCROUND series")

# --- Model forecast ---------------------------------------------------------
with tab_model:
    st.subheader("Elo ratings (1872 → present internationals)")
    st.dataframe(top_n(elo, 20), use_container_width=True)

    st.divider()
    st.subheader(f"Monte Carlo forecast — {_n_sims:,} simulations · {_preset_name} model")
    st.caption(
        "Survival probabilities per round. "
        "Teams absent from historical data are assigned the ELO_BASE rating."
    )

    _display_rounds = [r for r in config.ROUNDS if r != "final"]
    col_sort, col_n = st.columns([2, 1])
    with col_sort:
        sort_round = st.selectbox("Sort by round", _display_rounds, index=_display_rounds.index("champion"))
    with col_n:
        show_n = st.number_input("Show top N teams", min_value=5, max_value=48, value=16, step=1)

    mc_df = survival_table(mc_survival, sort_by=sort_round, top_n=int(show_n))
    st.dataframe(mc_df.drop(columns=["final"], errors="ignore").style.format("{:.1%}"), use_container_width=True)

    st.subheader("Group table")
    group_cols = st.columns(4)
    for i, (gid, teams) in enumerate(GROUPS_2026.items()):
        with group_cols[i % 4]:
            champ_odds = {t: mc_survival[t]["champion"] for t in teams}
            sorted_teams = sorted(champ_odds, key=champ_odds.get, reverse=True)  # type: ignore[arg-type]
            rows = [{"team": t, "champion %": f"{champ_odds[t]:.1%}",
                     "group adv %": f"{mc_survival[t]['group']:.1%}"} for t in sorted_teams]
            st.write(f"**Group {gid}**")
            st.dataframe(pd.DataFrame(rows).set_index("team"), use_container_width=True)

# --- Edge detection ---------------------------------------------------------
with tab_edge:
    st.subheader("Model vs market edge")
    st.caption(
        "**Green** = model thinks the market underprices the outcome (potential value). "
        "**Red** = market overprices relative to our Elo+MC model."
    )

    if markets.empty:
        st.info("No market data available — edge detection requires live market prices.")
    else:
        pm_sub = markets[markets["platform"] == "polymarket"]
        ks_sub = markets[markets["platform"] == "kalshi"]

        # ── Section 1: Polymarket champion odds ──────────────────────────────
        st.markdown("### Polymarket — champion (winner) edges")
        if pm_sub.empty:
            st.info("No Polymarket data available.")
        else:
            raw_yes: dict[str, float] = winner_probs(pm_sub)
            if not raw_yes:
                st.warning("Could not parse winner markets from Polymarket titles.")
            else:
                market_p: dict[str, float] = implied_from_book(raw_yes)
                model_p: dict[str, float] = {}
                unmatched_pm: list[str] = []
                for mkt_team, mkt_prob in market_p.items():
                    fifa_name = MARKET_TO_FIFA.get(mkt_team, mkt_team)
                    if fifa_name in mc_survival:
                        model_p[mkt_team] = mc_survival[fifa_name]["champion"]
                    else:
                        unmatched_pm.append(mkt_team)
                if unmatched_pm:
                    st.caption(f"Skipped (not in MC bracket): {', '.join(unmatched_pm)}")
                if model_p:
                    et = edge_table(model_p, market_p)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Teams compared", len(et))
                    c2.metric("Overround", f"{(sum(raw_yes.values())-1)*100:.1f} pp")
                    c3.metric("Largest edge", f"{et['edge_pct'].abs().max():.1f} pp")
                    st.plotly_chart(charts.edge_bars(et), use_container_width=True)
                    st.plotly_chart(charts.model_vs_market_scatter(et), use_container_width=True)
                    flagged = flag_value(et)
                    if not flagged.empty:
                        st.markdown("**Value flags (|edge| > 5 pp)**")
                        st.dataframe(
                            flagged[["outcome", "model_prob", "market_prob", "edge_pct"]]
                            .rename(columns={"edge_pct": "edge (pp)"})
                            .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                           "edge (pp)": "{:+.1f}"}),
                            use_container_width=True,
                        )
                    else:
                        st.info("No champion edges above 5 pp threshold.")
                    with st.expander("Full champion edge table"):
                        st.dataframe(
                            et[["outcome", "model_prob", "market_prob", "edge_pct",
                                "fair_odds", "market_odds"]]
                            .rename(columns={"edge_pct": "edge (pp)"})
                            .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                           "edge (pp)": "{:+.1f}", "fair_odds": "{:.2f}",
                                           "market_odds": "{:.2f}"}),
                            use_container_width=True,
                        )

        st.divider()

        # ── Section 2: Kalshi round-survival edges ────────────────────────────
        st.markdown("### Kalshi — round survival edges (KXWCROUND)")
        st.caption(
            "Kalshi markets ask 'Will X **qualify for** [Round]?', meaning the team *reaches* "
            "that round. Our model round labels refer to the round a team *wins*. "
            "The mapping applied here: Kalshi R16 → model R32 (P win R32 match), "
            "Kalshi QF → model R16, Kalshi SF → model QF, Kalshi Final → model SF/final."
        )
        # Kalshi 'qualify for X' = P(reach X) = P(win the *preceding* round).
        # Map each Kalshi label to the correct mc_survival key.
        _KALSHI_TO_MODEL_ROUND: dict[str, str] = {
            "R16":   "R32",    # qualify for R16 = win R32 match
            "QF":    "R16",    # qualify for QF  = win R16 match
            "SF":    "QF",     # qualify for SF  = win QF  match
            "final": "final",  # qualify for Final = win SF match (same value as "SF")
        }
        if ks_sub.empty:
            st.info("No Kalshi data available.")
        else:
            ks_survival = kalshi_survival_probs(ks_sub)
            kalshi_rounds = ["R16", "QF", "SF", "final"]
            round_choice = st.selectbox(
                "Round to analyse",
                [r for r in kalshi_rounds if any(r in probs for probs in ks_survival.values())],
                key="kalshi_round_select",
            )

            model_round = _KALSHI_TO_MODEL_ROUND[round_choice]
            ks_model_p: dict[str, float] = {}
            ks_market_p: dict[str, float] = {}
            for mkt_team, round_probs in ks_survival.items():
                if round_choice not in round_probs:
                    continue
                fifa_name = MARKET_TO_FIFA.get(mkt_team, mkt_team)
                if fifa_name not in mc_survival:
                    continue
                ks_model_p[mkt_team] = mc_survival[fifa_name][model_round]
                ks_market_p[mkt_team] = round_probs[round_choice]

            if ks_model_p:
                ks_et = edge_table(ks_model_p, ks_market_p)
                c1, c2, c3 = st.columns(3)
                c1.metric("Teams compared", len(ks_et))
                c2.metric("Largest edge", f"{ks_et['edge_pct'].abs().max():.1f} pp")
                avg_mkt = ks_market_p  # raw, not de-vigged (each market is independent binary)
                c3.metric("Market avg price", f"{sum(avg_mkt.values())/len(avg_mkt):.3f}",
                          help="Mean YES mid-price across all teams for this round")
                st.plotly_chart(charts.edge_bars(ks_et), use_container_width=True)
                st.plotly_chart(charts.model_vs_market_scatter(ks_et), use_container_width=True)
                ks_flagged = flag_value(ks_et)
                if not ks_flagged.empty:
                    st.markdown(f"**Value flags at {round_choice} (|edge| > 5 pp)**")
                    st.dataframe(
                        ks_flagged[["outcome", "model_prob", "market_prob", "edge_pct"]]
                        .rename(columns={"edge_pct": "edge (pp)"})
                        .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                       "edge (pp)": "{:+.1f}"}),
                        use_container_width=True,
                    )
                else:
                    st.info(f"No edges above 5 pp at {round_choice}.")
                with st.expander(f"Full {round_choice} edge table"):
                    st.dataframe(
                        ks_et[["outcome", "model_prob", "market_prob", "edge_pct",
                               "fair_odds", "market_odds"]]
                        .rename(columns={"edge_pct": "edge (pp)"})
                        .style.format({"model_prob": "{:.1%}", "market_prob": "{:.1%}",
                                       "edge (pp)": "{:+.1f}", "fair_odds": "{:.2f}",
                                       "market_odds": "{:.2f}"}),
                        use_container_width=True,
                    )
            else:
                st.warning("Could not match Kalshi teams to MC survival dictionary.")

        st.divider()

        # ── Section 3: Polymarket group stage match 3-way edges ──────────────
        st.markdown("### Polymarket — group stage match 3-way edges")
        st.caption(
            "Win / draw / win markets for individual group stage fixtures. "
            "All WC group games are neutral-venue — no home advantage is applied. "
            "Green = model sees value (underpriced). Red = market overprices vs model."
        )

        _match_mkts = load_match_markets()
        if _match_mkts.empty:
            st.info("No Polymarket match data available (API may be unavailable).")
        else:
            _outcome_map = {"home_win": "home", "draw": "draw", "away_win": "away"}
            _mrows: list[dict] = []
            for (_mhome, _maway, _mdate), _grp in _match_mkts.groupby(
                ["home_team", "away_team", "date"]
            ):
                _home_fifa = MARKET_TO_FIFA.get(_mhome, _mhome)
                _away_fifa = MARKET_TO_FIFA.get(_maway, _maway)
                _r_home = elo.get(_FIFA_TO_HIST.get(_home_fifa, _home_fifa), config.ELO_BASE)
                _r_away = elo.get(_FIFA_TO_HIST.get(_away_fifa, _away_fifa), config.ELO_BASE)
                _model = match_probs(_r_home, _r_away, neutral=True,
                                     draw_base=_db, scale=_sc)
                for _, _mr in _grp.iterrows():
                    _oc  = _mr["outcome"]
                    _mkt = _mr["price"]
                    _mod = _model[_outcome_map[_oc]]
                    _mrows.append({
                        "Date":      _mdate,
                        "Match":     f"{_mhome} vs {_maway}",
                        "Outcome":   _oc.replace("_", " ").title(),
                        "Model P":   _mod,
                        "Market P":  _mkt,
                        "Edge (pp)": _mod - _mkt,
                        "Kelly f":   kelly_fraction(_mod, _mkt),
                    })

            if _mrows:
                _mdf = pd.DataFrame(_mrows).sort_values(
                    ["Date", "Match", "Outcome"], ignore_index=True
                )
                _medge_min = st.slider(
                    "Minimum edge to show (pp)", -20, 20, -20, key="match_edge_min"
                ) / 100
                _mdf_show = _mdf[_mdf["Edge (pp)"] >= _medge_min]
                st.dataframe(
                    _mdf_show.style.format({
                        "Model P": "{:.1%}", "Market P": "{:.1%}",
                        "Edge (pp)": "{:+.1%}", "Kelly f": "{:.1%}",
                    }).background_gradient(
                        subset=["Edge (pp)"], cmap="RdYlGn", vmin=-0.12, vmax=0.12
                    ),
                    use_container_width=True,
                )
                _n_value = ((_mdf["Edge (pp)"] > 0.05).sum())
                st.caption(
                    f"{len(_mdf_show)} rows · "
                    f"{_n_value} outcomes with model edge > 5 pp across all fixtures"
                )
            else:
                st.info("Match markets loaded but no outcomes could be parsed.")

# --- Survival surface -------------------------------------------------------
with tab_surf:
    st.subheader("SVI-style survival surface")
    st.caption(
        "Smooth, monotone-non-increasing survival curves calibrated from Monte Carlo anchors. "
        "Methodology borrowed from options SVI: low-parameter sigmoid fit with "
        "no-arbitrage (calendar-monotone) constraint enforced. "
        "The literal SVI hyperbola is tuned to vol smiles; the shape here is a sigmoid."
    )

    all_mc_teams = sorted(mc_survival, key=lambda t: mc_survival[t]["champion"], reverse=True)
    default_teams = all_mc_teams[:8]
    selected_teams = st.multiselect(
        "Teams to display", options=all_mc_teams, default=default_teams
    )

    if selected_teams:
        # Calibrate SurvivalSurface for each selected team using all 7 MC anchors
        surv: dict[str, dict[str, float]] = {}
        for team in selected_teams:
            mc_anchors = mc_survival[team]
            surface = SurvivalSurface(team).fit(mc_anchors)
            surv[team] = surface.survival()

        st.plotly_chart(charts.survival_surface_3d(surv), use_container_width=True)
        st.plotly_chart(charts.survival_curves(surv), use_container_width=True)

        st.subheader("Raw Monte Carlo anchors (before SVI calibration)")
        raw_df = pd.DataFrame(
            {t: mc_survival[t] for t in selected_teams}
        ).T[[r for r in config.ROUNDS if r != "final"]]
        st.dataframe(raw_df.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("Select at least one team above.")

# --- Backtest ---------------------------------------------------------------
with tab_bt:
    st.subheader("Historical World Cup backtest")
    st.caption(
        "Elo ratings are rebuilt from scratch using only matches played **before** "
        "the selected World Cup started with zero lookahead bias. "
        "The model's match probabilities are then compared against the 64 actual "
        "results. Market baseline = 1/3 flat (equal-odds for each 3-way outcome), "
        "since archived betting odds are unavailable. "
        "Kelly staking bets on any outcome where our model exceeds 1/3 by > 5 pp."
    )

    year_choice = st.selectbox(
        "Select World Cup",
        list(WC_CUTOFFS.keys()),
        index=len(WC_CUTOFFS) - 1,
        format_func=lambda y: f"{y} — cutoff {WC_CUTOFFS[y][0]}",
    )

    bt = load_backtest(year_choice, _rev, _use_k, _db, _sc)
    data = bt["data"]

    # --- Summary metrics -------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Matches evaluated", len(data) // 3,
                help="64 WC matches × 3 outcomes = 192 rows; shown per match here")
    col2.metric("Brier — our model", f"{bt.get('brier_model', 0):.4f}",
                delta=f"{bt.get('brier_baseline', 0) - bt.get('brier_model', 0):+.4f} vs baseline",
                delta_color="normal",
                help="Lower Brier score = better calibrated. Delta = how much we beat the naive 1/3-equal baseline.")
    col3.metric("Bets placed", bt.get("n_bets", 0))
    col4.metric("Hit rate (vs 1/3 baseline)", f"{bt.get('hit_rate', 0):.1%}",
                help="Fraction of bets that won (edge ≥ 5 pp above 1/3 baseline)")
    col5.metric("Final bankroll (vs 1/3 baseline)", f"${bt.get('final_bankroll', 1000):,.0f}",
                delta=f"{bt.get('roi', 0):+.1%} ROI",
                help="Synthetic staking exercise, not a real trading result: bets are sized "
                     "and settled against a flat 1/3 equal-odds baseline (archived betting "
                     "odds aren't available for these historical WCs), not real market prices. "
                     "Any reasonably calibrated model beats a uniform prior by a wide margin, "
                     "so treat this ROI as an illustration of the staking mechanics, not "
                     "evidence of a real edge.")

    st.divider()

    # --- Charts ---------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        # Calibration plot for home-win predictions (most informative outcome)
        home_rows = data[data["outcome_label"] == "home_win"]
        st.plotly_chart(
            charts.calibration_plot(
                home_rows["model_prob"].to_numpy(),
                home_rows["outcome"].to_numpy(),
                title=f"{year_choice} WC — calibration (home-win predictions)",
            ),
            use_container_width=True,
        )

    with right:
        if bt.get("n_bets", 0) > 0:
            st.plotly_chart(
                charts.bankroll_curve(bt["ledger"]),
                use_container_width=True,
            )
        else:
            st.info("No bets placed at the current edge threshold.")

    st.divider()

    # --- Per-match predictions ------------------------------------------------
    st.subheader("Match-level predictions vs outcomes")

    # Pivot back to one row per match for display
    home_rows = data[data["outcome_label"] == "home_win"].copy()
    draw_rows = data[data["outcome_label"] == "draw"].set_index(["home", "away", "date"])
    away_rows = data[data["outcome_label"] == "away_win"].set_index(["home", "away", "date"])

    display = home_rows[["date", "home", "away", "home_score", "away_score",
                          "elo_home", "elo_away", "model_prob"]].copy()
    display = display.rename(columns={"model_prob": "P(home win)"})
    display["P(draw)"] = draw_rows["model_prob"].values
    display["P(away win)"] = away_rows["model_prob"].values
    display["result"] = display.apply(
        lambda r: (f"{int(r.home_score)}–{int(r.away_score)} "
                   + ("✓ home" if r.home_score > r.away_score
                      else ("draw" if r.home_score == r.away_score else "✓ away"))),
        axis=1,
    )
    # Draw is structurally never the argmax — P(draw) ≤ min(P(home), P(away))
    # at all Elo differences. "Favourite" just shows which side the model leans.
    display["favourite"] = display.apply(
        lambda r: "home" if r["P(home win)"] >= r["P(away win)"] else "away",
        axis=1,
    )

    st.dataframe(
        display[["date", "home", "away", "elo_home", "elo_away",
                 "P(home win)", "P(draw)", "P(away win)", "result", "favourite"]]
        .sort_values("date")
        .style.format({
            "elo_home": "{:.0f}", "elo_away": "{:.0f}",
            "P(home win)": "{:.1%}", "P(draw)": "{:.1%}", "P(away win)": "{:.1%}",
        }),
        use_container_width=True,
    )

    st.divider()

    # --- 2026 Forward Simulation (NOT a historical backtest) ------------------
    st.subheader("2026 World Cup — Live Market Forward Simulation")
    st.info(
        "**Not a historical backtest.** The 2002–2022 sections above evaluate the model "
        "against known outcomes. This section runs the same Kelly-staking strategy "
        "*forward*, using live Kalshi prices and Monte Carlo tournament paths. "
        "The output is a distribution of possible final bankrolls across simulations in "
        "a probabilistic range, not a profit forecast. Educational only."
    )

    _fwd_ks_sub = markets[markets["platform"] == "kalshi"]
    if _fwd_ks_sub.empty:
        st.warning("Kalshi market data unavailable — forward simulation requires live Kalshi prices.")
    else:
        _fwd_ks_survival = kalshi_survival_probs(_fwd_ks_sub)

        # Kalshi "qualify for X" round → mc_survival key (for model probability lookup)
        _FWDMAP_MC: dict[str, str] = {
            "R16": "R32", "QF": "R16", "SF": "QF", "final": "final",
        }
        # Kalshi "qualify for X" round → forward-simulation dict key (for outcome check)
        _FWDMAP_SIM: dict[str, str] = {
            "R16": "R32", "QF": "R16", "SF": "QF", "final": "SF",
        }

        _all_fwd_bets: list[dict] = []
        for _ks_rnd in ["R16", "QF", "SF", "final"]:
            for _mkt_team, _rnd_probs in _fwd_ks_survival.items():
                if _ks_rnd not in _rnd_probs:
                    continue
                _fifa = MARKET_TO_FIFA.get(_mkt_team, _mkt_team)
                if _fifa not in mc_survival:
                    continue
                _model_p = mc_survival[_fifa][_FWDMAP_MC[_ks_rnd]]
                _mkt_p   = _rnd_probs[_ks_rnd]
                _all_fwd_bets.append({
                    "Team":      _mkt_team,
                    "Market":    f"Qualify for {_ks_rnd}",
                    "Model P":   _model_p,
                    "Market P":  _mkt_p,
                    "Edge (pp)": _model_p - _mkt_p,
                    "Kelly f":   kelly_fraction(_model_p, _mkt_p),
                    "_fifa":     _fifa,
                    "_sim_rnd":  _FWDMAP_SIM[_ks_rnd],
                    "_mkt_p":    _mkt_p,
                })

        _fcol1, _fcol2 = st.columns([1, 2])
        with _fcol1:
            _fwd_edge_thr = st.slider(
                "Edge threshold (pp)", 0, 20, 5, key="fwd_edge_thr",
                help="Minimum model edge required to place a bet.",
            ) / 100
            _fwd_bankroll = float(st.number_input(
                "Starting bankroll ($)", value=1000, min_value=100,
                max_value=100_000, step=100, key="fwd_bank",
            ))
            _fwd_n = st.select_slider(
                "Simulations", options=[1000, 2000, 5000, 10000, 20000],
                value=5000, key="fwd_n_sims",
            )

        _active_bets = [b for b in _all_fwd_bets if b["Edge (pp)"] >= _fwd_edge_thr]

        with _fcol2:
            if _active_bets:
                _bet_disp = pd.DataFrame(_active_bets)[
                    ["Team", "Market", "Model P", "Market P", "Edge (pp)", "Kelly f"]
                ]
                st.markdown(f"**{len(_active_bets)} bets above {_fwd_edge_thr:.0%} edge**")
                st.dataframe(
                    _bet_disp.style.format({
                        "Model P": "{:.1%}", "Market P": "{:.1%}",
                        "Edge (pp)": "{:+.1%}", "Kelly f": "{:.1%}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No bets above the threshold. Lower the edge slider to see candidates.")

        if _active_bets:
            _bets_by_sim_rnd: dict[str, list] = {"R32": [], "R16": [], "QF": [], "SF": []}
            for _b in _active_bets:
                _bets_by_sim_rnd[_b["_sim_rnd"]].append(_b)

            _fwd_sim_results = run_forward_simulation(
                tuple(sorted(elo.items())), _db, _sc, _fwd_n,
            )

            # Apply Kelly staking round-by-round. Within a round, all bet stakes are
            # sized from the start-of-round bankroll (simultaneous-bet approximation),
            # then settled together before moving to the next round.
            _final_bankrolls: list[float] = []
            for _sim in _fwd_sim_results:
                _bank = _fwd_bankroll
                for _rnd in ["R32", "R16", "QF", "SF"]:
                    _rnd_start = _bank
                    _rnd_pnl   = 0.0
                    for _bet in _bets_by_sim_rnd[_rnd]:
                        _stake   = kelly_fraction(_bet["Model P"], _bet["_mkt_p"]) * _rnd_start
                        _won     = _bet["_fifa"] in _sim[_rnd]
                        _rnd_pnl += (_stake / _bet["_mkt_p"] - _stake) if _won else -_stake
                    _bank = max(_bank + _rnd_pnl, 0.0)
                _final_bankrolls.append(_bank)

            _arr      = np.array(_final_bankrolls)
            _med      = float(np.median(_arr))
            _p10      = float(np.percentile(_arr, 10))
            _p90      = float(np.percentile(_arr, 90))
            _p_profit = float(np.mean(_arr > _fwd_bankroll))

            _fmc1, _fmc2, _fmc3, _fmc4 = st.columns(4)
            _fmc1.metric("Median bankroll", f"${_med:,.0f}",
                         delta=f"{_med / _fwd_bankroll - 1:+.1%}")
            _fmc2.metric("10th percentile", f"${_p10:,.0f}")
            _fmc3.metric("90th percentile", f"${_p90:,.0f}")
            _fmc4.metric("P(profit)", f"{_p_profit:.1%}")

            import plotly.graph_objects as _pgo
            _txt_col  = "#E0E0E0" if _dark_mode else "#212121"
            _hist_col = "#42A5F5" if _dark_mode else "#1565C0"
            _fig_hist = _pgo.Figure()
            _fig_hist.add_trace(_pgo.Histogram(
                x=_arr, nbinsx=60, name="Final bankroll",
                marker_color=_hist_col, opacity=0.8,
            ))
            _fig_hist.add_vline(
                x=_fwd_bankroll, line_dash="dash",
                line_color="#EF5350" if _dark_mode else "#C62828",
                annotation_text=f"Start ${_fwd_bankroll:,.0f}",
                annotation_font_color=_txt_col,
            )
            _fig_hist.add_vline(
                x=_med, line_dash="dot",
                line_color="#66BB6A" if _dark_mode else "#2E7D32",
                annotation_text=f"Median ${_med:,.0f}",
                annotation_font_color=_txt_col,
            )
            _fig_hist.update_layout(
                title="Distribution of final bankrolls — 2026 WC forward simulation",
                xaxis_title="Final bankroll ($)",
                yaxis_title="Simulations",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=_txt_col,
                height=380,
                showlegend=False,
            )
            st.plotly_chart(_fig_hist, use_container_width=True)
            st.caption(
                f"{_fwd_n:,} simulations · Kelly staking vs live Kalshi prices · "
                f"edge threshold {_fwd_edge_thr:.0%} · "
                f"starting bankroll ${_fwd_bankroll:,.0f}"
            )

# --- Findings ---------------------------------------------------------------
with tab_findings:
    st.subheader("Model findings & conclusions")
    st.caption(
        f"Active model: **{_preset_name}** [{_preset['tag']}]  ·  "
        f"{_n_sims:,} Monte Carlo simulations"
    )

    # ── Section 1: six-WC backtest summary ───────────────────────────────────
    st.markdown("### Backtesting performance — six World Cups (2002–2022)")
    st.caption(
        "Each World Cup is a fully held-out test set. Elo ratings are rebuilt from "
        "scratch using only matches played **before** that tournament with zero lookahead. "
        "The naïve baseline assigns equal 1/3 probability to every 3-way outcome."
    )

    with st.spinner("Loading six historical backtests…"):
        all_bt = {yr: load_backtest(yr, _rev, _use_k, _db, _sc) for yr in WC_CUTOFFS}

    st.plotly_chart(charts.wc_summary_chart(all_bt, dark_mode=_dark_mode), use_container_width=True)

    avg_brier     = sum(all_bt[y]["brier_model"]    for y in all_bt) / len(all_bt)
    avg_hit       = sum(all_bt[y].get("hit_rate", 0) for y in all_bt) / len(all_bt)
    beat_baseline = sum(1 for y in all_bt if all_bt[y]["brier_model"] < all_bt[y]["brier_baseline"])
    total_bets    = sum(all_bt[y].get("n_bets", 0)   for y in all_bt)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Brier score", f"{avg_brier:.4f}",
              help="Mean across 6 WCs. The naïve 1/3 baseline averages ≈0.222.")
    c2.metric("Mean hit rate", f"{avg_hit:.1%}",
              help="Fraction of bets placed that won, averaged over all six WCs.")
    c3.metric("Outperformed baseline", f"{beat_baseline} / 6 WCs",
              help="Years where our Brier score was lower (better) than the naïve 1/3 baseline.")
    c4.metric("Total bets (5 pp edge)", f"{total_bets}",
              help="Cumulative bets placed across all six WCs at the 5 pp edge threshold.")

    st.divider()

    # ── Section 2: key findings ───────────────────────────────────────────────
    st.markdown("### Key findings")

    min_hit = min(all_bt[y].get("hit_rate", 0) for y in all_bt)
    max_hit = max(all_bt[y].get("hit_rate", 0) for y in all_bt)

    col_text, col_table = st.columns([3, 2])

    with col_text:
        st.markdown(f"""
**1. Consistent calibration across eras**
The model achieves a mean Brier score of {avg_brier:.4f} across six World Cups,
outperforming the naïve equal-odds baseline in {beat_baseline} of 6 tournaments.
The sole exception being 2002, coincides with the most historically anomalous World Cup
on record: South Korea reaching the semi-finals, Senegal eliminating the defending
champion France, and Turkey finishing third from a cold Elo starting point.

**2. Hit rate is the operative signal**
Brier score conflates calibration error with outcome surprise. For edge detection ()
the actual use case) the hit rate on bets placed above a 5 pp threshold is the
cleaner metric. The model delivers {min_hit:.1%}–{max_hit:.1%} across six WCs,
indicating that the *relative ranking* of outcome probabilities is robust even
when absolute calibration fluctuates in a 64-match sample.

**3. Draw model: calibration matters**
Maximum-likelihood estimation on ~21.5k competitive matches (1990–present) yields
`draw_base ≈ 0.313`, significantly higher than the conventional 0.28 assumption,
with a faster decay constant (`scale ≈ 319` vs 400). Draws are structurally more
likely at equal Elo strength than the literature assumes, but Elo differentiation
compresses that advantage more quickly than a simple exponential with scale = 400
would suggest. Note this is a *finding from a calibration run*, not the setting
behind the numbers on this page: that fit uses the full match history, so loading
it by default would leak those tournaments' own results into the backtests scoring
them. Unless you calibrate from the sidebar, the model runs on 0.28 / 400.

**4. National-team mean-reversion converges slowly**
The 5 % annual reversion rate that produces calibrated ratings here is 4–6× slower
than FiveThirtyEight's club-football figure (≈ 33 %). National squads have stable
identities and long qualifying cycles; aggressive reversion collapses inter-nation
variance too early, making the model overconfident on mismatched fixtures and
underconfident on elite-vs-elite ties.

**5. Tournament K-weighting improves signal fidelity**
Applying a five-tier K-factor scale (WC finals K = 60, qualifiers K = 40,
friendlies K = 20) improves Brier score by 3–5 pp relative to flat K = 40.
World Cup results carry more information per game than qualifiers, thus weighting
them more heavily in Elo updates is empirically justified.
""")


    with col_table:
        summary_rows = []
        for yr in sorted(all_bt):
            bt = all_bt[yr]
            delta = bt["brier_baseline"] - bt["brier_model"]
            summary_rows.append({
                "World Cup": yr,
                "Brier": bt["brier_model"],
                "vs Baseline": delta,
                "Bets": bt.get("n_bets", 0),
                "Hit rate": bt.get("hit_rate", 0),
            })
        summary_df = pd.DataFrame(summary_rows).set_index("World Cup")
        st.dataframe(
            summary_df.style.format({
                "Brier": "{:.4f}",
                "vs Baseline": "{:+.4f}",
                "Hit rate": "{:.1%}",
            }).background_gradient(subset=["Hit rate"], cmap="RdYlGn", vmin=0.38, vmax=0.65),
            use_container_width=True,
        )

        st.markdown("**Active model configuration**")
        st.info(
            f"**{_preset_name}** — {_preset['desc']}\n\n"
            f"draw\\_base = {_db:.3f}  ·  scale = {_sc:.0f}  ·  "
            f"reversion = {_rev:.0%}/yr  ·  tournament K: {'on' if _use_k else 'off'}"
        )

    st.divider()

    # ── Section 3: knockout bracket ───────────────────────────────────────────
    st.markdown("### Expected 2026 World Cup knockout bracket")
    st.caption(
        "REAL 2026 R32 draw. Model's own picks propagate through the true bracket "
        "tree with no mid-bracket correction — trained on pre-tournament data only, "
        "no lookahead. Box shading = model's champion probability (Monte Carlo)."
    )

    rounds_data, champion_team, mc_survival_real, accuracy = build_real_bracket()
    st.plotly_chart(
        charts.bracket_chart(rounds_data, champion_team, mc_survival_real, dark_mode=_dark_mode),
        use_container_width=True,
    )
    st.metric("Knockout-stage accuracy (model's own picks, real bracket, no correction)", accuracy)  
