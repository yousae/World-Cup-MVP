"""Plotly chart builders used by the Streamlit app."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config


def survival_surface_3d(team_survivals: dict[str, dict[str, float]]) -> go.Figure:
    """The SVI-inspired 'surface': teams x rounds x survival probability."""
    teams = list(team_survivals)
    rounds = config.ROUNDS
    z = np.array([[team_survivals[t][r] for r in rounds] for t in teams])
    fig = go.Figure(go.Surface(z=z, x=rounds, y=teams, colorscale="Viridis"))
    fig.update_layout(
        title="Tournament Survival Surface (SVI-style)",
        scene=dict(xaxis_title="round depth", yaxis_title="team",
                   zaxis_title="survival probability"),
        height=600, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def survival_curves(team_survivals: dict[str, dict[str, float]]) -> go.Figure:
    """2D version: one monotone line per team across rounds."""
    fig = go.Figure()
    for t, surv in team_survivals.items():
        fig.add_trace(go.Scatter(x=config.ROUNDS, y=[surv[r] for r in config.ROUNDS],
                                 mode="lines+markers", name=t))
    fig.update_layout(title="Survival curves (no-arb monotone)",
                      yaxis_title="P(survive)", height=450)
    return fig


def edge_bars(edge_df: pd.DataFrame) -> go.Figure:
    colors = ["#2ca02c" if e >= 0 else "#d62728" for e in edge_df["edge"]]
    fig = go.Figure(go.Bar(x=edge_df["outcome"], y=edge_df["edge_pct"],
                           marker_color=colors))
    fig.update_layout(title="Model edge vs market (percentage points)",
                      yaxis_title="edge (pp)", height=400)
    return fig


def bankroll_curve(ledger: pd.DataFrame) -> go.Figure:
    """Bankroll over time for one backtest run."""
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(ledger) + 1)),
        y=ledger["bankroll"],
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        hovertemplate="Bet %{x}<br>Bankroll: $%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=ledger["bankroll"].iloc[0] if len(ledger) > 0 else 1000,
                  line_dash="dash", line_color="gray",
                  annotation_text="starting bankroll")
    fig.update_layout(title="Bankroll over time (Kelly staking)",
                      xaxis_title="bet number", yaxis_title="bankroll ($)",
                      height=380)
    return fig


def calibration_plot(
    model_probs: "np.ndarray",
    outcomes: "np.ndarray",
    n_bins: int = 10,
    title: str = "Calibration (reliability diagram)",
) -> go.Figure:
    """Reliability diagram: predicted probability bucket vs actual win rate.

    Perfect calibration sits on the diagonal. Points above the line mean the
    model is underconfident (actual rate > predicted); below = overconfident.
    """
    import numpy as np
    bins = np.linspace(0, 1, n_bins + 1)
    centers, actuals, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (model_probs >= lo) & (model_probs < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        actuals.append(float(outcomes[mask].mean()))
        counts.append(int(mask.sum()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="perfect calibration"))
    fig.add_trace(go.Scatter(
        x=centers, y=actuals, mode="lines+markers",
        marker=dict(size=[max(6, c // 2) for c in counts], color="#2ca02c"),
        text=[f"n={c}" for c in counts],
        hovertemplate="predicted: %{x:.2f}<br>actual: %{y:.2f}<br>%{text}<extra></extra>",
        name="model",
    ))
    fig.update_layout(title=title, xaxis_title="predicted probability",
                      yaxis_title="actual win rate",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                      height=420)
    return fig


def model_vs_market_scatter(edge_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(edge_df, x="market_prob", y="model_prob", text="outcome",
                     title="Model vs market-implied probability")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="gray"))
    fig.update_traces(textposition="top center")
    fig.update_layout(height=450, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


# ---------------------------------------------------------------------------
# Findings tab charts
# ---------------------------------------------------------------------------

def wc_summary_chart(results: dict, dark_mode: bool = False) -> go.Figure:
    """Multi-WC backtest performance: Brier score and hit rate side-by-side.

    results: {year -> run_backtest() dict}
    dark_mode: True when the Streamlit app is using a dark theme.
    """
    # ── theme palette ─────────────────────────────────────────────────────────
    if dark_mode:
        bg        = "rgba(0,0,0,0)"   # transparent — inherits Streamlit dark bg
        fg        = "#E0E0E0"          # primary text / tick labels
        fg_sub    = "#90A4AE"          # secondary text (annotations, subtitles)
        gridcolor = "#2D3748"          # subtle grid lines
        bar_blue  = "#42A5F5"          # our model bar (bright, visible on dark)
        bar_red   = "#EF5350"          # baseline bar (distinct from blue)
        ref_line  = "#607D8B"          # 50 % reference line
    else:
        bg        = "rgba(0,0,0,0)"
        fg        = "#212121"
        fg_sub    = "#546E7A"
        gridcolor = "#EEEEEE"
        bar_blue  = "#1E88E5"
        bar_red   = "#EF9A9A"
        ref_line  = "#78909C"

    years  = sorted(results)
    brier_m = [results[y]["brier_model"]    for y in years]
    brier_b = [results[y]["brier_baseline"] for y in years]
    hits    = [results[y].get("hit_rate", 0) for y in years]
    n_bets  = [results[y].get("n_bets", 0)   for y in years]
    yr_str  = [str(y) for y in years]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Brier Score per World Cup  (lower = better calibration)",
            "Hit Rate per World Cup  (fraction of bets won)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=yr_str, y=brier_m, name="Our model",
        marker_color=bar_blue,
        text=[f"{b:.4f}" for b in brier_m], textposition="outside",
        textfont=dict(color=fg, size=10),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=yr_str, y=brier_b, name="Naïve 1/3 baseline",
        marker_color=bar_red,
        text=[f"{b:.4f}" for b in brier_b], textposition="outside",
        textfont=dict(color=fg, size=10),
    ), row=1, col=1)

    hit_colors = ["#43A047" if h > 0.50 else "#E53935" for h in hits]
    fig.add_trace(go.Bar(
        x=yr_str, y=hits, name="Hit rate",
        marker_color=hit_colors,
        text=[f"{h:.1%} ({n} bets)" for h, n in zip(hits, n_bets)],
        textposition="outside",
        textfont=dict(color=fg, size=10),
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0.50, line_dash="dash", line_color=ref_line,
                  annotation_text="50%",
                  annotation_font_color=fg_sub,
                  annotation_position="right",
                  row=1, col=2)

    fig.update_yaxes(range=[0, 0.30], gridcolor=gridcolor, row=1, col=1)
    fig.update_yaxes(range=[0, 0.85], tickformat=".0%", gridcolor=gridcolor, row=1, col=2)
    fig.update_xaxes(tickfont=dict(color=fg))
    fig.update_yaxes(tickfont=dict(color=fg))


    # Subplot title colour (accessed via layout.annotations added by make_subplots)
    for ann in fig.layout.annotations:
        ann.font.color = fg

    fig.update_layout(
        height=420, barmode="group",
        font=dict(color=fg),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, x=0,
            font=dict(color=fg),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=70, b=20),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
    )
    return fig


def bracket_chart(
    rounds_data: dict,
    champion_team: str,
    mc_survival: dict,
    dark_mode: bool = False,
) -> go.Figure:
    """Draw the expected 2026 WC knockout bracket as a left-to-right tree.

    rounds_data: {'R32': [(ta, tb, p_a, p_b, winner), ...], 'R16': ..., ..., 'Final': ...}
        Each tuple: team_a, team_b, P(A wins), P(B wins), expected winner.
    champion_team: the expected Final winner.
    mc_survival: {team -> {round -> float}} — used for champion probability colouring.
    dark_mode: True when the Streamlit app is using a dark theme.
    """
    # ── theme palette ─────────────────────────────────────────────────────────
    if dark_mode:
        bg              = "rgba(0,0,0,0)"
        loser_fill      = "rgba(255,255,255,0.05)"
        loser_border    = "rgba(255,255,255,0.15)"
        winner_border   = "#42A5F5"
        winner_txt      = "#90CAF9"   # light blue — readable on dark winner boxes
        loser_txt       = "#607D8B"   # muted blue-gray — de-emphasised
        champ_badge_col = "#78909C"
        header_col      = "#90A4AE"
        line_col        = "#455A64"
        title_sub_col   = "#607D8B"
        # Winner gradient: dark navy (low prob) → bright blue (high prob)
        grad_lo = (13,  71, 161)   # #0D47A1
        grad_hi = (66, 165, 245)   # #42A5F5
    else:
        bg              = "rgba(0,0,0,0)"
        loser_fill      = "#F0F4F8"
        loser_border    = "#CFD8DC"
        winner_border   = "#1565C0"
        winner_txt      = "#0D47A1"   # dark blue on light fill
        loser_txt       = "#BDBDBD"
        champ_badge_col = "#78909C"
        header_col      = "#424242"
        line_col        = "#B0BEC5"
        title_sub_col   = "#78909C"
        # Winner gradient: light blue (low prob) → deep blue (high prob)
        grad_lo = (187, 222, 251)  # #BBDEFB
        grad_hi = (21,  101, 192)  # #1565C0

    ROUND_ORDER = ["R32", "R16", "QF", "SF", "Final"]
    ROUND_LABEL = {
        "R32": "Round of 32", "R16": "Round of 16",
        "QF": "Quarter-finals", "SF": "Semi-finals", "Final": "Final",
    }
    N_MATCHES = {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "Final": 1}

    # ── coordinate constants ─────────────────────────────────────────────────
    BOX_W   = 2.5
    BOX_H   = 0.38
    TGAP    = 0.05
    RND_GAP = 0.5
    CHAMP_GAP = 0.6

    x0 = {}
    x = 0.0
    for rn in ROUND_ORDER:
        x0[rn] = x
        x += BOX_W + RND_GAP
    champ_x0 = x + CHAMP_GAP - RND_GAP

    def match_cy(rn: str, mi: int) -> float:
        step = 32 / N_MATCHES[rn]
        return step * mi + step / 2

    # ── colour helpers ───────────────────────────────────────────────────────
    all_champ_probs = [
        mc_survival.get(t, {}).get("champion", 0.0)
        for matches in rounds_data.values()
        for ta, tb, *_ in matches
        for t in (ta, tb)
    ]
    max_prob = max(all_champ_probs) if all_champ_probs else 0.20

    def fill_rgb(team: str, is_winner: bool) -> tuple[int, int, int] | None:
        """Winner fill as an (r, g, b) tuple, or None for losers (flat fill)."""
        if not is_winner:
            return None
        p = mc_survival.get(team, {}).get("champion", 0.0)
        frac = min(1.0, p / max(max_prob, 0.001))
        return (
            int(grad_lo[0] + frac * (grad_hi[0] - grad_lo[0])),
            int(grad_lo[1] + frac * (grad_hi[1] - grad_lo[1])),
            int(grad_lo[2] + frac * (grad_hi[2] - grad_lo[2])),
        )

    def fill_color(team: str, is_winner: bool) -> str:
        rgb = fill_rgb(team, is_winner)
        return loser_fill if rgb is None else f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    def winner_text_color(team: str) -> str:
        """Pick black or white text based on the actual fill luminance.

        The winner fill is a gradient keyed to champion probability, so a
        single fixed text colour cannot stay readable across it: the
        strongest team (deepest fill) was rendering dark-blue-on-dark-blue
        and its name became invisible in every box it appeared in. Uses the
        WCAG relative-luminance formula and flips to white past the standard
        0.5 threshold, which keeps contrast acceptable at both ends of the
        ramp in light and dark mode.
        """
        rgb = fill_rgb(team, True)
        if rgb is None:
            return winner_txt
        r, g, b = (c / 255.0 for c in rgb)
        lin = [(c / 12.92) if c <= 0.03928 else (((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
        luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        return "#FFFFFF" if luminance < 0.5 else "#0D47A1"

    # ── build figure ─────────────────────────────────────────────────────────
    shapes: list[dict] = []
    annotations: list[dict] = []
    line_x: list = []
    line_y: list = []

    for ri, rn in enumerate(ROUND_ORDER):
        rx = x0[rn]
        for mi, (ta, tb, pa, pb, winner) in enumerate(rounds_data[rn]):
            cy = match_cy(rn, mi)
            win_a = (winner == ta)
            win_b = (winner == tb)

            for team, y_bot, y_top, p_win, is_win in [
                (ta, cy + TGAP,           cy + TGAP + 2 * BOX_H, pa, win_a),
                (tb, cy - TGAP - 2*BOX_H, cy - TGAP,             pb, win_b),
            ]:
                champ_p = mc_survival.get(team, {}).get("champion", 0.0)
                tc = winner_text_color(team) if is_win else loser_txt
                bc = winner_border if is_win else loser_border
                bw = 1.5 if is_win else 0.8

                shapes.append(dict(
                    type="rect",
                    x0=rx, x1=rx + BOX_W,
                    y0=y_bot, y1=y_top,
                    fillcolor=fill_color(team, is_win),
                    line=dict(color=bc, width=bw),
                    layer="below",
                ))

                mid_y = (y_bot + y_top) / 2
                annotations.append(dict(
                    x=rx + 0.12, y=mid_y,
                    text=f"<b>{team[:15]}</b>",
                    showarrow=False,
                    font=dict(size=8, color=tc),
                    xanchor="left", align="left",
                ))
                annotations.append(dict(
                    x=rx + BOX_W - 0.08, y=mid_y,
                    text=f"{p_win:.0%}",
                    showarrow=False,
                    font=dict(size=8, color=tc),
                    xanchor="right", align="right",
                ))
                # Champion-probability badge sits centred between the team
                # name (left-anchored) and the win % (right-anchored) rather
                # than below the name -- the old fixed -0.18 y offset is in
                # data coordinates while the font size is in points, so at
                # smaller canvas sizes it drifted on top of the name.
                if is_win and champ_p > 0:
                    annotations.append(dict(
                        x=rx + BOX_W * 0.62, y=mid_y,
                        text=f"🏆 {champ_p:.1%}",
                        showarrow=False,
                        font=dict(size=7, color=tc),
                        xanchor="center",
                    ))

            if ri < len(ROUND_ORDER) - 1:
                next_rn = ROUND_ORDER[ri + 1]
                next_cy = match_cy(next_rn, mi // 2)
                next_rx = x0[next_rn]
                mid_x   = rx + BOX_W + RND_GAP / 2
                winner_cy_strip = (cy + TGAP + BOX_H) if win_a else (cy - TGAP - BOX_H)
                line_x.extend([rx + BOX_W, mid_x, mid_x, next_rx, None])
                line_y.extend([winner_cy_strip, winner_cy_strip, next_cy, next_cy, None])

    # ── champion box ─────────────────────────────────────────────────────────
    final_cy = match_cy("Final", 0)
    final_ta, *_ = rounds_data["Final"][0]
    win_a_final    = (champion_team == final_ta)
    champ_win_strip = (final_cy + TGAP + BOX_H) if win_a_final else (final_cy - TGAP - BOX_H)
    champ_p = mc_survival.get(champion_team, {}).get("champion", 0.0)
    champ_cx = (champ_x0 + champ_x0 + BOX_W) / 2

    shapes.append(dict(
        type="rect",
        x0=champ_x0, x1=champ_x0 + BOX_W,
        y0=final_cy - 0.8, y1=final_cy + 0.8,
        fillcolor="#F9A825",      # amber — visible on both dark and light
        line=dict(color="#E65100", width=2),
        layer="below",
    ))
    for text, y_off, sz, col in [
        ("<b>🏆 CHAMPION</b>",              0.35,  10, "#4E342E"),
        (f"<b>{champion_team}</b>",          0.00,  10, "#4E342E"),
        (f"MC champion prob: <b>{champ_p:.1%}</b>", -0.42, 8, "#5D4037"),
    ]:
        annotations.append(dict(
            x=champ_cx, y=final_cy + y_off,
            text=text, showarrow=False,
            font=dict(size=sz, color=col),
        ))

    mid_x_champ = x0["Final"] + BOX_W + (champ_x0 - x0["Final"] - BOX_W) / 2
    line_x.extend([x0["Final"] + BOX_W, mid_x_champ, mid_x_champ, champ_x0, None])
    line_y.extend([champ_win_strip, champ_win_strip, final_cy, final_cy, None])

    # ── round header labels ───────────────────────────────────────────────────
    for rn in ROUND_ORDER:
        annotations.append(dict(
            x=x0[rn] + BOX_W / 2, y=33.6,
            text=f"<b>{ROUND_LABEL[rn]}</b>",
            showarrow=False, font=dict(size=9, color=header_col),
        ))
    annotations.append(dict(
        x=champ_cx, y=33.6,
        text="<b>Champion</b>",
        showarrow=False, font=dict(size=9, color="#FB8C00"),
    ))

    # ── assemble figure ───────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        line=dict(color=line_col, width=1.3),
        showlegend=False, hoverinfo="skip",
    ))

    total_x = champ_x0 + BOX_W + 0.3
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        height=1500,
        xaxis=dict(visible=False, range=[-0.3, total_x]),
        yaxis=dict(visible=False, range=[-1.5, 35]),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        margin=dict(l=5, r=5, t=45, b=5),
        title=dict(
            text=(
                "Expected 2026 World Cup Knockout Bracket  "
                f"<span style='font-size:12px;color:{title_sub_col}'>"
                "Elo-seeded group outcomes · win % = match probability · 🏆 = MC champion prob"
                "</span>"
            ),
            font=dict(size=14, color=header_col), x=0.5,
        ),
    )
    return fig
