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

During the 2026 World Cup I found and fixed a real, measurable bias in the project's Elo model: it was consistently overrating CONCACAF teams, Mexico in particular, against UEFA and CONMEBOL opponents. I built a fitted, data-driven correction for it, validated it across six historical World Cups (2002 to 2022) with bootstrap significance testing, then used the corrected model to predict the entire 2026 knockout bracket in real time, with no lookahead and no mid-tournament adjustments. The model picked the actual champion (Spain) and got 81% of the 31 real knockout matches right.

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
| 2002 | 0.2164 | 0.2153 | **0.2048** |
| 2006 | 0.1921 | 0.1892 | **0.1830** |
| 2010 | 0.1919 | 0.1907 | **0.1893** |
| 2014 | 0.1993 | 0.1983 | **0.1900** |
| 2018 | 0.1999 | 0.1986 | **0.1964** |
| 2022 | 0.2039 | 0.2031 | 0.2072 |

Lower Brier score is better. (Numbers here are after fixing a lookahead bug in the goal-diff weight, see section 14, so they're a bit different from what I first reported.) The confederation offset alone improves every single year. Adding goal-diff form on top improves five of six years further, but makes 2022 slightly worse, not the clean "every year improves" story I originally had. That's a more honest picture than what I first wrote here, and I'd rather leave the miss in than paper over it.

## 7. Statistical significance

A per-match Brier improvement of about 0.003 to 0.007 is small next to a single 64-match tournament's sample size, so I ran a paired bootstrap test (5,000 to 10,000 resamples per tournament, resampling whole matches) to check whether this was a real effect or just noise:

| Year | % of resamples favoring the full model |
|---|---|
| 2002 | 88.0% |
| 2006 | 89.5% |
| 2010 | 58.7% |
| 2014 | 86.2% |
| 2018 | 72.8% |
| 2022 | 36.5% |
| **Pooled (all 6 tournaments)** | **92.8%** |

(Also updated after the section 14 lookahead fix. The pooled number actually got stronger, 92.8% versus 87.4% before, since removing the leak meant a much more stable goal-diff weight, but 2022 flipped from "roughly a coin flip" to "the full model is probably slightly worse than plain+offset that year," which is a real result and matches the Brier table above.) The effect is directionally strong in most years and genuinely weak or negative in 2010 and 2022. The pooled number, 92.8% across roughly 400 matches, is the strongest and most defensible piece of evidence here, but the year-to-year swings, and the one real miss, are worth stating plainly rather than hiding behind the one aggregate number.

## 8. A double-counting issue I caught

When I tested the fully corrected Mexico/England prediction against the model's existing home-field-advantage boost (+65 Elo, since Mexico was a genuine 2026 co-host), it nearly cancelled out the confederation correction entirely (45/24/31 versus the original, uncorrected 46/24/30). That raised a question I haven't resolved: whether the standard home-advantage number, calibrated for actual home-stadium matches, is getting over-applied to co-host World Cup games, which probably carry a smaller and different kind of advantage. Flagged, not fixed.

## 9. Market-baseline comparison: an honest dead end

I made four separate attempts to compare the model against real prediction-market odds (Polymarket/Kalshi), which would have been the strongest possible outside check:

1. Live API fetch: came back empty, since the tournament had already ended and every match market had resolved and closed.
2. Reconstructed historical Polymarket data (querying `closed: true` events): this did return data, but the only price available (`lastTradePrice`) is the price at market resolution, which converges toward certainty and basically leaks the real outcome. Not usable for backtesting.
3. The project's own Discord bot database: had the right schema for this (model and market odds logged live, per match), but the market-odds columns never actually got populated for any of the 100 logged predictions.
4. A public search for independent market-accuracy writeups: found only Polymarket's own marketing copy ("94% accuracy," repeated across a bunch of near-identical pages), nothing World-Cup-specific or Brier-score-based, and nothing from a credible independent source.

Rather than force a comparison that wasn't valid, I documented it as a real limitation instead.

## 10. Live bracket validation, the headline result

Using the real 2026 Round of 32 draw as it actually happened, and letting the model's own predictions decide who advances at every round after that (no mid-bracket corrections using real results), I ran the model through the actual bracket tree, trained only on pre-tournament data:

- Predicted champion: Spain. That's who actually won.
- Knockout-stage accuracy: 25 of 31 matches correct (81%). A 32-team knockout bracket is 16+8+4+2+1 = 31 games, not 32, so "32 real knockout matches" in an earlier version of this section was off by one.
- The misses were mostly close, near-toss-up matches and a few genuine upsets (Norway over Brazil, for one), not systematic failures.
- This number moves slightly (it was 77% at one point, then 81% again after the section 14 fixes) because the function that computes it refits its goal-diff weight fresh from `download_results()`'s current data every time it runs, rather than from a value pinned once and frozen. That's a minor reproducibility gap worth knowing about: this metric isn't a fixed historical fact the way the section 6/7 backtest tables are, it can shift by a match or two if the underlying match data gets re-fetched. 25/31 is what it produces now, consistently, as of the section 14 cleanup.

This is the result I'm proudest of: a real, falsifiable, no-lookahead prediction that could have been published before the final and checked against reality afterward, not a backtest built with the benefit of hindsight.

## 11. What's live now

- The dashboard's live Elo ratings (`load_elo()`) apply both the confederation offset and the goal-diff correction by default now.
- The Findings tab's bracket diagram shows the model's real, live 2026 predictions against actual results (using the project's existing Plotly bracket renderer), with a live accuracy metric, replacing the old pre-tournament placeholder bracket.
- A new smoke test covers the forward-simulation code path, which had no test coverage before, and `tests/test_no_lookahead.py` (added later, see section 14) covers the walk-forward guarantees themselves.
- I also found and fixed a data-freshness bug: the historical results loader was silently serving a stale, two-week-old cached file even on re-run, because it only re-downloads when no local file exists. Had to call `download_results(force=True)` to actually pull the complete, final tournament data.
- The bracket chart's text colour is now derived from each box's fill rather than hardcoded, which fixed the tournament favourite's name being invisible (section 15).

## 12. Known limitations

- The confederation-offset fit uses one static Elo snapshot instead of each team's rating at the time of each historical match. That's a simplification, not a fully rigorous fix for rating drift over time.
- The statistical significance is strong when pooled but inconsistent year to year. 2022 in particular shows almost no measurable effect.
- The confederation offset and the existing home-field-advantage boost may partly double-count for co-host matches. Flagged above, not resolved.
- I never got a valid external market-baseline comparison, despite four separate attempts.
- The bracket-prediction group-stage tiebreaks use Elo as a stand-in for real goal-difference and goals-for tiebreaks, since hypothetical scorelines aren't actually simulated in the deterministic path.

## 13. Independent validation pass

After the tournament wrapped, I ran the project through a quant-style validation pass (leakage audit, baseline checks, a live-versus-backtest comparison) to stress-test all of this the way a real quant shop would.

The historical backtest (2002 to 2022, walk-forward, no lookahead in the Elo training) reports a Brier score around 0.20, and scoring the actual 2026 live predictions against real outcomes (72 of the 100 logged predictions matched up cleanly to real results, since the project never scored them itself) comes in a bit better at 0.181, with a 64% hit rate, which lines up with the historical range. That's a good sign the model isn't overfit to the past. But the "beats the market" claim still hasn't actually been tested: the live market-odds logging silently failed for the whole tournament (see section 9), so every edge and ROI number in this project is measured against a flat, equal-odds strawman instead of real prices. I also found a small lookahead bug in how the "full model" comparison in section 7 fits its goal-difference weight. The honest summary at the time was that the forecasting side held up well out of sample, but the market-comparison side, the part that would actually make this a trading thesis instead of just a forecasting exercise, still didn't exist, and the "72 of 100 matched" number was itself hiding bugs I hadn't found yet. Section 14 below is the follow-up where I went back and actually fixed what this section found.

## 14. Cleanup and hardening pass

Section 13 found a bunch of real problems and stopped at describing them. This section is where I went back in and actually fixed the ones that were fixable, and made the ones that weren't easier to diagnose next time.

**Why match_results and calibration were actually empty.** The historical-CSV fallback in `results.py`, the one source of match results that should always work since the data's public, was calling `load_results(force_download=True)`. `load_results()` doesn't take that argument at all, so every single call raised a `TypeError` that got swallowed by a bare `except Exception`, silently, for the entire tournament. Fixed it to call the actual force-refresh function (`download_results(force=True)`). On top of that, matching a fixture to its CSV row used plain word-overlap on team names, which fails outright for the same 6 renamed teams from section 1's aliasing problem (`"czechia"` is never a substring of `"czech republic"`), and used the UTC calendar date from the fixture's kickoff timestamp, which is off by a day from the CSV's local-match-date convention for any evening kickoff at a US host city. Fixed both: results.py now runs team names through the project's existing `_FIFA_TO_HIST` map before matching, checks both team orderings (the two sources don't always agree on which side is "home" for a neutral match), and checks a 2-day window instead of an exact date string. Wrote `jobs/backfill_results.py` to catch up everything that was missed live, then ran it: **72 of 72 resolvable matches now have a stored result**, and calibration is computed for real for the first time (`brier_model ≈ 0.543` under this DB's own 3-outcome-summed convention, roughly 0.18 in the more familiar mean-per-outcome convention from section 13).

**Why 28 predictions were logged against fake teams.** `fixtures.py`'s schedule parser tries to drop unresolved knockout slots before they get saved, but it only checks for a literal `"TBD"` prefix. The actual feed marks unresolved slots with group-position codes like `"1A"` or `"3ABCDF"`, a completely different pattern that the check never caught. 28 of the 100 logged predictions had at least one side still unresolved this way, most of them (56 team-slots total across those 28 matches) had it on *both* sides, e.g. `"2A vs 2B"`, logged before either group had finished. Fixed the filter to catch both, and re-downloaded the schedule now that the tournament's over and fixturedownload.com has real team names for every match, so the cached schedule file is clean too.

**The market-odds columns.** I couldn't prove what actually happened live back in June, since Polymarket's per-match markets for this tournament are closed now and re-querying them just returns nothing (the same dead end as section 9). What I could do is add real diagnostic logging to `market_discovery.py`, so instead of a silent `None` for every fixture with no way to tell why, it now logs exactly how many candidate events were fetched, how many got filtered out and at which step, and why any given fixture's match wasn't found. If this happens again next tournament, there'll be an actual trail to follow.

**The goal-diff weight lookahead.** Confirmed and fixed the bug flagged in section 13: `build_wc_backtest_full()` used to fit its goal-difference weight on the target tournament's entire match set in one pass, meaning an early group-stage prediction was partly informed by outcomes of matches that happened after it. It's now fit purely from prior World Cup editions (pooling every WC before the one being tested), with zero reference to the target tournament's own results. This wasn't just a correctness fix, it's a straight upgrade: the leaky version only had ~48 to 64 matches to fit one parameter from and swung wildly year to year (weight estimates from -1.97 to +7.96), while pooling everything before it gives a stable weight in the 14 to 17 range every time. The pooled bootstrap significance actually went up after the fix, 92.8% versus 87.4% before, though 2022 individually got honestly worse, not better, once the leak was gone (see the corrected tables in sections 6 and 7).

**Test coverage.** Added `tests/test_no_lookahead.py`, three tests on synthetic data covering the parts of the project that had zero coverage before: that `compute_elo` doesn't care what order matches arrive in, that appending absurd future matches to the training data never changes a backtest's existing predictions, and a direct regression test for the goal-diff fix above, changing a later match's score and asserting an earlier match's prediction doesn't move.

**The flat 1/3 baseline.** The dashboard's Backtest tab already had a caption explaining the 1/3 baseline, but the "Final bankroll" / ROI metric itself didn't carry that caveat if you scrolled past the caption or screenshotted just the metric. Relabeled it and expanded the tooltip to say plainly that this is a staking-mechanics illustration against a synthetic baseline, not a measurement against real prices, and added the same caveat directly in `run_backtest()`'s docstring in the code.

**Repo housekeeping.** Deleted three branches (`add-smoke-test-coverage`, `elo-current-wc-boost`, `elo-recency-weighting`) from the shared repo that were fully merged into `main` with zero unique commits left.

## 15. Presentation pass, and a visualization bug I only found by looking at it

The last round was about making the repo readable to someone landing on it cold, and it turned up one more real bug.

**The bracket chart was hiding the champion's name.** While exporting a static version of the bracket for the README, I noticed Spain's box was blank apart from the trophy badge. The cause: winner boxes are filled with a gradient keyed to each team's Monte Carlo champion probability, running from pale blue up to deep blue, while the winner *text* colour was a single hardcoded dark blue (`#0D47A1`). Whichever team had the highest championship probability therefore got the deepest fill, and its name rendered dark-blue-on-dark-blue. In other words the bug reliably hid the name of the single most important team on the chart, which is why it had survived unnoticed: it only ever affected the tournament favourite, and only in the rounds it actually appeared in. Fixed by computing the text colour from the fill's WCAG relative luminance and flipping to white past the standard 0.5 threshold, so contrast holds at both ends of the gradient in light and dark mode. The champion-probability badge had a related problem: it was positioned at a fixed offset in *data* coordinates below the team name while its font size was in *points*, so how far below the name it landed depended on canvas size, and at smaller sizes it drifted on top of the name. Moved it inline. Both of these were live in the dashboard, not artifacts of the export.

The general lesson I took from it: I had been reading this chart in the app for weeks without noticing, because I already knew which team was supposed to be in that box and my eye filled it in. Rendering it in a different context, at a different size, is what made it visible.

**README rewritten around the result.** The old README described a tournament that hadn't happened yet, never mentioned the bracket accuracy or the live Brier score, and didn't link the write-up at all, so the strongest evidence in the project was invisible to anyone who didn't go digging. It now opens with the numbers (25/31 knockout matches, champion called, live Brier 0.181 against 0.222 for a uniform baseline, with the six-tournament historical range for context), shows the bracket image, and states the market-comparison null explicitly rather than leaving a reader to discover it in section 9.

**Repo cleanup.** Deleted a stale copy of `svi_surface.py` at the repo root that had diverged from the real one in `src/models/` and was imported by nothing, removed superseded scratch scripts (`predict_bracket_from_r32.py` and its `_v2`, `run_market_comparison.py` and its `_v2`) and promoted the surviving `run_market_comparison_v3.py` to the plain name, added a LICENSE and a real `.gitignore`. Added `render_readme_assets.py` so the README image is regenerated from the code rather than being a screenshot that silently goes stale the next time the model changes. Added a CI workflow that runs both test suites on every push and pull request, since having written the no-lookahead tests it seemed worth actually enforcing them.

**A documentation claim that wasn't true, and the fix I nearly got wrong.** The README and the dashboard's Findings tab both described the draw model as MLE-calibrated (`draw_base ≈ 0.313, scale ≈ 319`). But `data/processed/` is gitignored, so the fitted file never shipped, and a clean clone silently runs on the hardcoded `0.28 / 400` fallback. My first instinct was to fix this by committing the fitted file so the documentation became true. That would have been a mistake: the fit runs over the entire match history, which now includes the 2026 World Cup, so shipping it would have meant the six historical backtests and the 2026 bracket prediction all ran on parameters estimated partly from the results they are scored against. That is precisely the class of leak the goal-difference bug in section 14 was, reintroduced through the back door and harder to spot because it arrives as a data file rather than as code.

The right fix was the opposite: leave the defaults in place and correct the prose. It also turned out the reproducibility worry was inverted. Every number in this document was produced with `0.28 / 400`, so a clean clone reproduces them exactly; it was only the surrounding prose that was wrong. Both the README and the Findings tab now state which values are actually in effect and explain why the calibration is reported as a finding rather than loaded by default. The measured difference between the two parameter sets is small either way (roughly 0.0002 to 0.0014 of Brier score), which is exactly why it could sit there misdescribed for so long without anything looking obviously broken.

## 16. Tools and methods used

Python (pandas, numpy, scipy.optimize), maximum-likelihood parameter fitting, no-lookahead backtesting methodology, bootstrap significance testing, git/GitHub collaborative workflow (branches, pull requests, code review), SQLite querying, live API integration (Polymarket/Kalshi), Streamlit/Plotly dashboard development.
