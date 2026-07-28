# ⚽ World Cup Quant Dashboard

See **[WRITEUP.md](./WRITEUP.md)** for the full write-up: finding and fixing a real cross-confederation calibration bias in a live Elo model, validating it across six historical World Cups, and using it to correctly predict the 2026 champion in real time.

A portfolio project that pulls live prediction-market odds (Polymarket + Kalshi) for the 2026 World Cup, generates independent model probabilities from historical match data, and surfaces where the two disagree (value edges and theoretical cross-platform arbitrage) all with backtesting, an interactive Streamlit dashboard, and a Discord webhooks for live alerts.

> **Educational tool. Does not place trades and is not financial or betting advice.** Reads public market data only.

---

## Quickstart

```bash
cd wcq
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data/historical.py        
python tests/test_smoke.py           
streamlit run app/streamlit_app.py   
```

---

## Dashboard

Six tabs — all live:

| Tab | What's in it |
|-----|-------------|
| Live markets | Raw Polymarket + Kalshi prices; Kalshi round-survival pivot table |
| Model forecast | Top-20 Elo ratings; Monte Carlo survival probabilities; group tables |
| Edge detection | Model vs market edges (bar + scatter); Kalshi round-label remapping applied |
| Survival surface | SVI-style survival curves per team across tournament rounds |
| Backtest | Six historical WC backtests (2002–2022); 2026 forward simulation with Kelly staking histogram |
| Findings | Brier/hit-rate summary; 5 key findings; expected 2026 knockout bracket |


## Discord Bot

A companion bot posts automated alerts to a private Discord server:

| Notification | When | Channel |
|---|---|---|
| Daily digest | 07:00 UTC | `#daily-post` |
| Pre-match briefing | ~1hr before kickoff | `#pre-match` |
| Post-match scorecard | 10–90 min after full time | `#post-match` |
| Paper bet placed / P&L | Alongside pre/post-match | `#paper-trading` |
| Live edge alert | During match, edge > 6% | `#live-alerts` |
| Cross-platform spread | During match, Poly vs Kalshi > 4% | `#live-alerts` |

**Deployment:** GitHub Actions (3 cron jobs) + Railway (always-on live poller). See [DEPLOY.md](wcq/DEPLOY.md).

---

## Architecture

```
results.csv (50k matches)          Polymarket / Kalshi (live odds)
     ↓                                      ↓
historical.py (load/clean)           markets.py (fetch prices)
     ↓                                      ↓
elo.py (team ratings)              implied.py (de-vig → clean probs)
     ↓                                      ↓
match_model.py (win/draw/loss)             |
     ↓                                      |
tournament.py (MC bracket sim)             |
     ↓                                      |
svi_surface.py (survival curves)           |
     ↓_____________________________________|
                   edges.py (model vs market)
                         ↓
               streamlit_app.py (dashboard)
                         ↓
              src/bot/ (Discord notification layer)
```

| File | Job |
|------|-----|
| `config.py` | Shared paths, API endpoints, model knobs |
| `src/data/historical.py` | ~50k international results (1872→present), cleaned |
| `src/data/markets.py` | Polymarket + Kalshi price fetchers (public, graceful fallback) |
| `src/models/elo.py` | Elo ratings replayed over full history |
| `src/models/match_model.py` | Elo → win/draw/loss probabilities |
| `src/models/tournament.py` | 20k-simulation 48-team bracket Monte Carlo |
| `src/models/svi_surface.py` | SVI-style no-arbitrage survival surface |
| `src/markets/implied.py` | Price → implied prob, de-vig, overround |
| `src/markets/edges.py` | Model vs market edges, Kelly sizing, arb flags |
| `src/backtest/engine.py` | Staking sim, ROI, hit rate, Brier calibration |
| `src/viz/charts.py` | Plotly charts (3D surface, edge bars, calibration scatter) |
| `app/streamlit_app.py` | Dashboard (6 tabs) |
| `src/bot/` | Discord bot modules (notify, storage, poller, paper trader, …) |
| `jobs/` | GitHub Actions job scripts (digest, pre-match, post-match) |

---

## Key model details

- **Elo**: trained on all matches from 1872 → present; 5% annual mean-reversion; 5-tier K-weighting (WC finals K=60 → friendlies K=20)
- **Draw model**: `P(draw|Δelo) = draw_base × exp(-|Δelo|/scale)`, MLE-fitted; params in `data/draw_params.json`
- **Monte Carlo**: 20,000 simulations of the 48-team 2026 bracket

## How the SVI framing transfers

Options SVI fits a smooth, low-parameter, no-arbitrage curve to sparse market quotes. We reuse the *methodology*, not the literal equation:

| Options world | This project |
|---|---|
| Maturity axis | Tournament round depth (group → champion) |
| Implied-vol surface | P(team survives to that round) |
| Butterfly no-arb | Per-round survival probs form a valid distribution |
| Calendar no-arb | Survival monotone non-increasing in round depth |

The SVI hyperbola is tuned to vol smiles, so we borrow the approach (smooth + no-arb-constrained + market-calibrated), not the formula. See `src/models/svi_surface.py`.

---

## Data & legal notes

- Historical data: [martj42 international-results dataset](https://github.com/martj42/international_results) (public, no auth)
- Markets: Polymarket Gamma API + Kalshi public endpoints, read-only, no API key required
- US users: Polymarket restricts US access and Kalshi is CFTC-regulated; this project only reads public prices and simulates
