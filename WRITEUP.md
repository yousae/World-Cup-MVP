# World Cup Quant Dashboard: Cross-Confederation Calibration and Live Bracket Validation

**A collaborative, in-tournament quant research project built around the 2026 FIFA World Cup**

## About the project

This is a full quantitative research and trading-signal dashboard built around the 2026 FIFA World Cup. My friend and I built it together, from scratch, as a learn-by-doing project. Neither of us had worked on anything this size before.

**The system, end to end:**
- **Data pipeline:** about 50,000 historical international matches (1872 to present), cleaned and loaded automatically, plus live odds pulled from real prediction markets (Polymarket and Kalshi).
- **Modeling core:** a from-scratch Elo rating system with tournament-tier weighting and mean-reversion, feeding a fitted three-outcome (win/draw/loss) match model and a Monte Carlo simulator that runs the full 48-team, 12-group 2026 bracket thousands of times to get round-by-round survival probabilities for every team.
- **Edge detection:** comparing the model's own probabilities against real, live market prices (after de-vigging) to see where the model and the market disagree.
- **Backtesting engine:** replays all six modern World Cups (2002 to 2022) with strict no-lookahead methodology, scored on Brier score and hit rate, plus a live 2026 "forward simulation" using real Kalshi prices and Kelly-criterion staking.
- **Discord bot:** a fully deployed, always-on system (GitHub Actions for scheduled jobs, Railway for a live asyncio poller) that posts daily digests, pre-match briefings, live edge alerts, and post-match scorecards automatically, backed by its own SQLite database and paper-trading ledger.
- **Dashboard:** a multi-tab Streamlit app that ties all of it together: live markets, model forecasts, edge detection, survival curves, backtests, and a findings tab.

**Division of work:** my friend built the original architecture, the core Elo/match-model/Monte Carlo pipeline, the whole Discord bot, and the first version of the dashboard. I joined partway through. My part was finding and fixing a real bias in the Elo system (more on that below), adding statistically validated corrections, writing new tests, and building a live, real-time bracket-prediction feature for the dashboard, all contributed back through normal git branches and pull requests, reviewed and merged together rather than built off in isolation. The rest of this document is about that contribution specifically, since it's the part I can speak to in the most technical depth, but none of it would exist without the shared foundation we built together first.

---

## Summary of my contribution

During the 2026 World Cup I found and fixed a real, measurable bias in the project's Elo model: it was consistently overrating CONCACAF teams, Mexico in particular, against UEFA and CONMEBOL opponents. I built a fitted, data-driven correction for it, validated it across six historical World Cups (2002 to 2022) with bootstrap significance testing, then used the corrected model to predict the entire 2026 knockout bracket in real time, with no lookahead and no mid-tournament adjustments. The model picked the actual champion (Spain) and got 77% of the 32 real knockout matches right.

This document walks through the problem, the dead ends, the fix, and the evidence for and against it, including the parts that didn't work out as cleanly as I'd have liked.

---

## 1. The problem

I was sanity-checking predictions against a real, live 2026 match (Mexico vs. England, Round of 16), and the model had Mexico as a narrow favorite despite England being the conventionally stronger side. Looking at the raw Elo numbers:

- Mexico: 1860.4, England: 1856.3. Basically tied, with Mexico a hair ahead.

That didn't match reality by any outside measure. Two independent sources confirmed it:
- FIFA world rankings had Mexico at #14, versus the model's implied rank of around #4 to #5.
- eloratings.net, a well-established independent Elo system, had Mexico at #13 against #4 in our model. That's the biggest gap (9 ranks) of any team in the top 25. Most other teams (France, Brazil, Japan, Ecuador, Turkey) matched within a rank or two, so this wasn't a broadly miscalibrated model. It was one specific, targeted bias.

## 2. Root cause

Mexico and England almost never play each other or share opponents. Mexico's results come mostly from CONCACAF and Copa América, England's from UEFA. With so little data directly connecting the two rating pools, cross-confederation comparisons get unreliable even when the within-confederation ratings look fine on their own.

## 3. What I tried and rejected

I tested two intuitive fixes first, and both made the problem worse, which was itself a useful, honest finding:

| Approach | Effect on Mexico/England Elo gap |
|---|---|
| Recency-weighting (up-weight matches from the last 12 months) | Widened 4.1 to 11.2 points |
| Extra boost for current-World-Cup matches | Widened 4.1 to 36.2 points |

Both made things worse because Mexico's recent and current-tournament matches were disproportionately against weaker regional opponents, so weighting them more heavily just inflated the rating further instead of correcting it.

## 4. The fix: fitted confederation offsets

Instead of hand-picking correction values, which would just swap one person's intuition for another's, I fit the offsets with maximum-likelihood estimation (`scipy.optimize.minimize`) against real historical cross-confederation match outcomes, with UEFA fixed as the reference point (0.0) since only the relative differences between confederations are identifiable.

I also caught a real bug in the process. The first fitting attempt, using the full ~150 years of history, failed to converge (`converged=False`) and produced wildly unstable results (CONCACAF's offset swung from −99.9 to −85.4 to −48.7 depending on settings). Adding L2 regularization and giving the optimizer more iterations fixed it. Both a full-history fit and a 2010-onward fit then converged cleanly and agreed on direction: CONMEBOL underrated, CONCACAF overrated, everyone else close to neutral.

## 5. A second feature: in-tournament goal-difference form

On top of the confederation offset, I added a second signal: each team's goal difference (goals scored minus goals allowed) within the current tournament, using only matches played strictly before the one being predicted, so there's no lookahead. The weight on this signal is fit with MLE the same way as the offsets.

## 6. Validation: six-tournament backtest (2002 to 2022)

I backtested both additions against plain Elo across all six historical World Cups, using strict no-lookahead methodology (offsets and weights fit only on data available before each tournament started):

| Year | Brier (plain) | Brier (+offset) | Brier (+offset+goal-diff) |
|---|---|---|---|
| 2002 | 0.2164 | 0.2153 | **0.2091** |
| 2006 | 0.1921 | 0.1892 | **0.1866** |
| 2010 | 0.1919 | 0.1907 | **0.1902** |
| 2014 | 0.1993 | 0.1983 | **0.1933** |
| 2018 | 0.1999 | 0.1988 | **0.1982** |
| 2022 | 0.2039 | 0.2032 | **0.2029** |

Lower Brier score is better. Every single year improved as each correction got added, with no year where either one backfired.

## 7. Statistical significance

A per-match Brier improvement of about 0.003 to 0.007 is small next to a single 64-match tournament's sample size, so I ran a paired bootstrap test (5,000 to 10,000 resamples per tournament, resampling whole matches) to check whether this was a real effect or just noise:

| Year | % of resamples favoring the full model |
|---|---|
| 2002 | 80.9% |
| 2006 | 79.9% |
| 2010 | 56.5% |
| 2014 | 76.0% |
| 2018 | 66.4% |
| 2022 | 53.2% |
| **Pooled (all 6 tournaments)** | **87.4%** |

The effect is directionally consistent (every year improves) but its strength moves around a lot: strong in 2002, 2006, and 2014, weak in 2010 and 2018, basically a coin flip in 2022. The pooled number, 87.4% across roughly 400 matches, is the strongest and most defensible piece of evidence here, but the year-to-year swings are real, and I'd rather say that plainly than hide behind the one aggregate number.

## 8. A double-counting issue I caught

When I tested the fully corrected Mexico/England prediction against the model's existing home-field-advantage boost (+65 Elo, since Mexico was a genuine 2026 co-host), it nearly cancelled out the confederation correction entirely (45/24/31 versus the original, uncorrected 46/24/30). That raised a question I haven't resolved: whether the standard home-advantage number, calibrated for actual home-stadium matches, is getting over-applied to co-host World Cup games, which probably carry a smaller and different kind of advantage. Flagged, not fixed.

## 9. Market-baseline comparison: an honest dead end

I made four separate attempts to compare the model against real prediction-market odds (Polymarket/Kalshi), which would have been the strongest possible outside check:

1. Live API fetch: came back empty, since the tournament had already ended and every match market had resolved and closed.
2. Reconstructed historical Polymarket data (querying `closed: true` events): this did return data, but the only price available (`lastTradePrice`) is the price at market resolution, which converges toward certainty and basically leaks the real outcome. Not usable for backtesting.
3. The project's own Discord bot database: had the right schema for this (model and market odds logged live, per match), but the market-odds columns never actually got populated for any of the 91 logged predictions.
4. A public search for independent market-accuracy writeups: found only Polymarket's own marketing copy ("94% accuracy," repeated across a bunch of near-identical pages), nothing World-Cup-specific or Brier-score-based, and nothing from a credible independent source.

Rather than force a comparison that wasn't valid, I documented it as a real limitation instead.

## 10. Live bracket validation, the headline result

Using the real 2026 Round of 32 draw as it actually happened, and letting the model's own predictions decide who advances at every round after that (no mid-bracket corrections using real results), I ran the model through the actual bracket tree, trained only on pre-tournament data:

- Predicted champion: Spain. That's who actually won.
- Knockout-stage accuracy: 26 of 32 matches correct (81%) in an earlier test run using the confirmed live matchups, 77% in the final version wired into the dashboard's bracket visualization (the small gap comes from final tuning of the goal-diff weight).
- The misses were mostly close, near-toss-up matches and a few genuine upsets (Norway over Brazil, for one), not systematic failures.

This is the result I'm proudest of: a real, falsifiable, no-lookahead prediction that could have been published before the final and checked against reality afterward, not a backtest built with the benefit of hindsight.

## 11. What's live now

- The dashboard's live Elo ratings (`load_elo()`) apply both the confederation offset and the goal-diff correction by default now.
- The Findings tab's bracket diagram shows the model's real, live 2026 predictions against actual results (using the project's existing Plotly bracket renderer), with a live accuracy metric, replacing the old pre-tournament placeholder bracket.
- A new smoke test covers the forward-simulation code path, which had no test coverage before.
- I also found and fixed a data-freshness bug: the historical results loader was silently serving a stale, two-week-old cached file even on re-run, because it only re-downloads when no local file exists. Had to call `download_results(force=True)` to actually pull the complete, final tournament data.

## 12. Known limitations

- The confederation-offset fit uses one static Elo snapshot instead of each team's rating at the time of each historical match. That's a simplification, not a fully rigorous fix for rating drift over time.
- The statistical significance is strong when pooled but inconsistent year to year. 2022 in particular shows almost no measurable effect.
- The confederation offset and the existing home-field-advantage boost may partly double-count for co-host matches. Flagged above, not resolved.
- I never got a valid external market-baseline comparison, despite four separate attempts.
- The bracket-prediction group-stage tiebreaks use Elo as a stand-in for real goal-difference and goals-for tiebreaks, since hypothetical scorelines aren't actually simulated in the deterministic path.

## 13. Independent validation pass

After the tournament wrapped, I ran the project through a quant-style validation pass (leakage audit, baseline checks, a live-versus-backtest comparison) to stress-test all of this the way a real quant shop would.

The historical backtest (2002 to 2022, walk-forward, no lookahead in the Elo training) reports a Brier score around 0.20, and scoring the actual 2026 live predictions against real outcomes (72 of the 100 logged predictions matched up cleanly to real results, since the project never scored them itself) comes in a bit better at 0.181, with a 64% hit rate, which lines up with the historical range. That's a good sign the model isn't overfit to the past. But the "beats the market" claim still hasn't actually been tested: the live market-odds logging silently failed for the whole tournament (see section 9), so every edge and ROI number in this project is measured against a flat, equal-odds strawman instead of real prices. I also found and quantified a small lookahead bug in how the "full model" comparison in section 7 fits its goal-difference weight. The honest summary is that the forecasting side of this holds up well out of sample, but the market-comparison side, the part that would actually make this a trading thesis instead of just a forecasting exercise, still doesn't exist.

## 14. Tools and methods used

Python (pandas, numpy, scipy.optimize), maximum-likelihood parameter fitting, no-lookahead backtesting methodology, bootstrap significance testing, git/GitHub collaborative workflow (branches, pull requests, code review), SQLite querying, live API integration (Polymarket/Kalshi), Streamlit/Plotly dashboard development.
