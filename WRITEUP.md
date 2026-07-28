# World Cup Quant Dashboard — Project Overview, Cross-Confederation Calibration & Live Bracket Validation

**A collaborative, live, in-tournament quant research project built around the 2026 FIFA World Cup**

## About the project

This project is a full quantitative research and trading-signal dashboard built around the 2026 FIFA World Cup, developed collaboratively with a friend as a from-scratch, learn-by-building effort — neither of us had prior experience with a project of this scope going in.

**The system, end to end:**
- **Data pipeline** — ~50,000 historical international football matches (1872–present), cleaned and loaded automatically, plus live odds pulled from real prediction markets (Polymarket and Kalshi).
- **Modeling core** — a from-scratch Elo rating system with tournament-tier weighting, mean-reversion, and a fitted three-outcome (win/draw/loss) match model, feeding a Monte Carlo simulator that runs the entire 48-team, 12-group 2026 bracket thousands of times to produce round-by-round survival probabilities for every team.
- **Edge detection** — comparing the model's own probabilities against real, live market prices (after de-vigging) to surface where the model and the market disagree.
- **Backtesting engine** — replays all six modern World Cups (2002–2022) with strict no-lookahead methodology, scoring the model with Brier score and hit rate, plus a live 2026 "forward simulation" using real Kalshi prices and Kelly-criterion staking.
- **Discord bot** — a fully deployed, always-on system (GitHub Actions for scheduled jobs, Railway for a live asyncio poller) that posts daily digests, pre-match briefings, live edge alerts, and post-match scorecards to Discord automatically, backed by its own SQLite database and paper-trading ledger.
- **Dashboard** — a multi-tab Streamlit app tying all of the above together: live markets, model forecasts, edge detection, survival curves, backtests, and a findings/summary tab.

**Division of work:** my friend built the original architecture, the core Elo/match-model/Monte Carlo pipeline, the full Discord bot (both deployment targets), and the initial dashboard. My contribution, joining partway through, focused on finding and fixing a real modeling bias in the Elo system (detailed below), adding statistically validated corrections, building new test coverage, and building a live, real-time bracket-prediction feature for the dashboard — all contributed back via git branches and pull requests, reviewed and merged collaboratively rather than developed in isolation. The rest of this document focuses on that contribution in technical depth, since it's the piece I can speak to in the most rigorous detail, but it sits on top of — and was only possible because of — the shared foundation we built together.

---

## Summary of my contribution

During the 2026 World Cup, I found and fixed a real, measurable bias in the project's Elo-based prediction model: it consistently overrated CONCACAF teams (specifically Mexico) relative to UEFA/CONMEBOL opponents. I built a fitted, data-driven correction, validated it rigorously across six historical World Cups (2002–2022) and via bootstrap significance testing, then used the corrected model to predict the entire 2026 knockout bracket in real time — with no lookahead and no mid-bracket correction. The model correctly predicted the champion (Spain) and achieved **77% knockout-stage accuracy** across all 32 real knockout matches.

This document walks through the problem, the false starts, the fix, and the honest evidence for and against it.

---

## 1. The problem

While sanity-checking model predictions against a real, live 2026 match (Mexico vs. England, Round of 16), the model rated Mexico as a narrow favorite despite England being the conventionally stronger side. Digging into the raw Elo ratings:

- **Mexico: 1860.4** vs. **England: 1856.3** — nearly identical, with Mexico slightly ahead.

This didn't match outside reality. Two independent sources confirmed it:
- **FIFA world rankings:** Mexico #14, vs. the model's implied rank of ~#4–5.
- **eloratings.net** (a well-established, independent Elo system): Mexico ranked **#13**, vs. **#4** in this model — the largest gap (9 ranks) of any team in the top 25. Most other teams (France, Brazil, Japan, Ecuador, Turkey) matched exactly or within 1–2 ranks, meaning the issue wasn't a broadly miscalibrated model — it was a specific, targeted bias.

## 2. Root cause

Mexico and England rarely play each other or share opponents — Mexico's results come mostly from CONCACAF/Copa América, England's from UEFA. With little data directly connecting the two rating pools, cross-confederation comparisons become unreliable even though within-confederation ratings look reasonable.

## 3. What was tried and rejected

Two intuitive fixes were tested first — both made the problem *worse*, which was itself an important, honest finding:

| Approach | Effect on Mexico/England Elo gap |
|---|---|
| Recency-weighting (up-weight matches from the last 12 months) | Widened 4.1 → 11.2 points |
| Extra boost for current-World-Cup matches | Widened 4.1 → 36.2 points |

Both amplified the bias rather than correcting it: Mexico's recent/current-tournament matches were disproportionately against weaker regional opposition, so up-weighting them made the inflation worse, not better.

## 4. The fix: fitted confederation offsets

Rather than hand-picking correction values (which would just substitute one person's intuition for another's — the same problem in a different form), offsets were **fit** via maximum-likelihood estimation (`scipy.optimize.minimize`) against real historical cross-confederation match outcomes, with UEFA fixed as a reference point (0.0) since only relative differences between confederations are identifiable.

**A real bug was caught and fixed in this process:** the first fitting attempt, using all ~150 years of history, failed to converge (`converged=False`), producing wildly unstable results (CONCACAF's offset swung from −99.9 to −85.4 to −48.7 across different settings). Adding L2 regularization and increasing the optimizer's iteration budget fixed this — both a full-history fit and a 2010-only fit then converged cleanly and agreed in direction (CONMEBOL underrated, CONCACAF overrated, other confederations near neutral).

## 5. A second feature: in-tournament goal-difference form

On top of the confederation offset, a second signal was added: each team's **goal difference within the current tournament** (goals scored − conceded, using only matches played strictly before the one being predicted — no lookahead). The weight for this signal is also fit via MLE, the same way as the confederation offsets.

## 6. Validation: six-tournament backtest (2002–2022)

Both additions were backtested against **plain Elo** across all six historical World Cups, using strict no-lookahead methodology (offsets and weights fit only on data available before each tournament started):

| Year | Brier (plain) | Brier (+offset) | Brier (+offset+goal-diff) |
|---|---|---|---|
| 2002 | 0.2164 | 0.2153 | **0.2091** |
| 2006 | 0.1921 | 0.1892 | **0.1866** |
| 2010 | 0.1919 | 0.1907 | **0.1902** |
| 2014 | 0.1993 | 0.1983 | **0.1933** |
| 2018 | 0.1999 | 0.1988 | **0.1982** |
| 2022 | 0.2039 | 0.2032 | **0.2029** |

Lower Brier score = better. **Every single year improved monotonically** as each correction was added — no year where either addition backfired.

## 7. Statistical significance

A per-match Brier improvement of ~0.003–0.007 is small relative to a single 64-match tournament's sample size, so a paired bootstrap test (5,000–10,000 resamples per tournament, resampling whole matches) was run to check whether this was a real effect or noise:

| Year | % of resamples favoring the full model |
|---|---|
| 2002 | 80.9% |
| 2006 | 79.9% |
| 2010 | 56.5% |
| 2014 | 76.0% |
| 2018 | 66.4% |
| 2022 | 53.2% |
| **Pooled (all 6 tournaments)** | **87.4%** |

**Honest read:** the effect is directionally consistent (every year improves) but its *strength* varies substantially — strong in 2002/2006/2014, weak in 2010/2018, essentially a coin flip in 2022. The pooled result (87.4%, using ~400 matches) is the strongest, most defensible evidence, but the year-to-year variance is real and worth stating plainly rather than hiding behind the aggregate number.

## 8. A caught double-counting issue

Testing the fully-corrected Mexico/England prediction with the model's existing home-field-advantage boost (+65 Elo, since Mexico was a genuine 2026 co-host) **nearly cancelled out** the confederation correction entirely (45%/24%/31% vs. the original uncorrected 46%/24%/30%). This raised an open question — documented but not resolved — of whether the standard home-advantage figure (calibrated for true home-stadium matches) is being over-applied to co-host World Cup matches, which likely carry a smaller, distinct advantage.

## 9. Market-baseline comparison: an honest dead end

Four separate attempts were made to compare the model against real prediction-market odds (Polymarket/Kalshi), the strongest possible external validation:
1. **Live API fetch** — returned empty; the tournament had ended and all match markets had resolved/closed.
2. **Reconstructed historical Polymarket data** (querying `closed: true` events) — technically returned data, but the only available price (`lastTradePrice`) is the price at market resolution, which converges to certainty and effectively leaks the real outcome. Invalid for backtesting.
3. **The project's own Discord bot database** — had the right schema (model + market odds logged live, per match) but the market-odds columns were never actually populated for any of the 91 logged predictions.
4. **Public search for independent market-accuracy analysis** — found only Polymarket's own generic marketing copy ("94% accuracy," repeated across near-duplicate pages, not World-Cup-specific or Brier-score-based), not a credible independent source.

Rather than force an invalid comparison, this was documented as a genuine limitation.

## 10. Live bracket validation — the headline result

Using the real 2026 R32 draw (as it actually happened) and letting the model's own predictions determine who advances at every subsequent round — no mid-bracket correction using real results — the model was run through the true bracket tree, trained on pre-tournament data only:

- **Predicted champion: Spain** — the actual real-world champion.
- **Knockout-stage accuracy: 26/32 matches correct (81%)** in an earlier real-bracket test using the confirmed live matchups; **77%** in the final version wired into the live dashboard's bracket visualization (small differences reflect the final tuning of the goal-diff weight).
- Misses were concentrated in close, near-toss-up matches and a small number of genuine upsets (e.g., Norway over Brazil), not systematic model failures.

This is a genuinely strong result: a real, falsifiable, no-lookahead prediction that could have been published before the final and checked against reality afterward — not a backtest with hindsight bias.

## 11. What's live now

- The Streamlit dashboard's live Elo ratings (`load_elo()`) now apply both the confederation offset and the goal-diff form correction by default.
- The "Findings" tab's bracket diagram now shows the model's real, live 2026 predictions against actual results (reusing the project's existing Plotly bracket renderer), with a live accuracy metric, replacing the original pre-tournament placeholder bracket.
- A new smoke test covers the forward-simulation code path that previously had no test coverage.
- A real data-freshness bug was found and fixed: the historical results loader silently served a stale, two-week-old cached file even when re-run, because it only re-downloads if no local file exists. `download_results(force=True)` was needed to pull the complete, final tournament data.

## 12. Known limitations (stated plainly, not hidden)

- The confederation-offset fit uses a single, static Elo snapshot rather than each team's rating at the time of each historical match — a simplification, not a fully rigorous rating-drift correction.
- Statistical significance is strong when pooled but inconsistent year-to-year; 2022 in particular shows almost no measurable effect.
- The confederation offset and existing home-field-advantage boost may partially double-count for co-host matches — flagged, not resolved.
- No valid external market-baseline comparison was achieved, despite four separate attempts.
- The bracket-prediction group-stage tiebreaks use Elo as a proxy for real goal-difference/goals-for tiebreaks, since hypothetical scorelines aren't simulated in the deterministic path.

## 13. Tools and methods used

Python (pandas, numpy, scipy.optimize), maximum-likelihood parameter fitting, no-lookahead backtesting methodology, bootstrap significance testing, git/GitHub collaborative workflow (branches, pull requests, code review), SQLite querying, live API integration (Polymarket/Kalshi), Streamlit/Plotly dashboard development.

**PR link for my personal contributions (yousae): https://github.com/novaa1351/World-Cup-MVP/pulls?q=is%3Apr+is%3Aclosed**
