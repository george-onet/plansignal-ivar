# PlanSignal IVaR

**Inventory financial exposure, quantified per material. One number Procurement and Finance can both stand behind.**

🔗 **Live app:** [plansignal-ivar.streamlit.app](https://plansignal-ivar.streamlit.app)

---

## What it is

PlanSignal IVaR turns raw material data into **Inventory Value at Risk** — a single EUR figure for every SKU, decomposed into six independent risk dimensions plus two stress lenses. Every euro is traceable to a formula and assumptions you can audit in-app.

Built for mid-market chemical and pharmaceutical manufacturers (€50M–€500M revenue) where Procurement and Finance often calculate inventory exposure separately, on different spreadsheets, with different methods. IVaR gives both functions one source of truth.

**Not** a reorder tool, not a forecasting model, not an MRP replacement. A decision-support layer that sits on top of whatever planning system you already run.

---

## The model — six dimensions, two lenses

**Six additive dimensions** (sum to Total IVaR):

| Dimension | What it measures |
|---|---|
| Understock | Lost production / missed sales from running out |
| Overstock | Cash trapped above 1.5× safety stock + holding cost |
| LT Volatility | Extra safety stock needed to absorb lead-time variance |
| Aging / Expiry | Write-off risk on materials nearing or past shelf life |
| Tariff / Country | Revaluation exposure on trade policy change |
| Commodity Price | Replacement cost gap on input price moves |

**Two stress lenses** (surfaced separately, not summed into Total IVaR):

| Lens | What it measures |
|---|---|
| Concentration | Worst-case single-source disruption scenario |
| Margin Sensitivity | P&L amplification on Understock for high-margin SKUs |

The separation matters. Additive dimensions are expected-loss exposures. Lenses are scenario stress tests. Mixing them double-counts.

---

## How it works

1. Upload your material master (xlsx or csv). One row per SKU. Mixed numeric formats, country code variants, dirty material codes — the loader handles it.
2. Optional: upload lead-time history to unlock the LT Volatility dimension.
3. Adjust six sidebar parameters to match your business: holding cost rate, forward horizon, margin-sensitivity threshold, concentration outage multiplier, tariff exposure assumption, aging threshold.
4. Read the portfolio summary, drill into individual SKUs, export to Excel.

Every figure in every drilldown comes with a "Material inputs & model assumptions" panel that names the formula and the inputs used. No black box.

---

## Data inputs

**Required columns** (any reasonable spelling — the loader normalises):

- `material` — SKU code
- `unit_cost_eur`
- `inventory_on_hand`
- `avg_daily_demand`
- `lead_time_days`

**Optional columns** (unlock additional dimensions when present):

- `safety_stock` → Overstock (treated as 0 if absent — conservative)
- `margin_pct` → Margin Sensitivity
- `sole_source` → Concentration
- `country_of_origin` → Tariff / Country
- `shelf_life_days` + `days_on_hand` → Aging / Expiry (precise path)
- `expected_price_change_pct` → Commodity Price

Missing optional columns return €0 for that dimension — no crashes, no inferred values.

---

## Stress tested

500-SKU portfolio with planted edge cases including: mixed currency formats (`$45.20`, `12,5 €`, `£12.50`, `USD 22`), country code variants (`CN` / `chn` / `CHN` / ` BE `), 15 sole-source flag spellings, 8 negative-margin SKUs, 15 zero-margin SKUs, 8 out-of-stock SKUs, 2 past-expiry SKUs, 20 long-lead-time imports, plus an 8,866-row lead-time history file with 3 ghost materials and 20 NaN observations.

Every dimension audited. Every silent failure mode probed. Five independent SKUs spot-checked against hand-calculated LT Volatility predictions — all matched to the euro.

---

## Built with

- Python 3.11
- Streamlit
- pandas, numpy
- openpyxl

Single-file Streamlit app. No database. State is the uploaded file.

---

## Origin

PlanSignal IVaR is the upstream-materials counterpart to [PlanSignal v1](https://plansignal.streamlit.app), which handles finished-goods forecast risk. Both are portfolio projects built by [George Onet](https://www.linkedin.com/in/georgeonet/) (Rascal) — a supply chain practitioner with 7+ years in chemical and pharmaceutical planning, documenting the build journey publicly on LinkedIn.

The model reflects choices a planner makes daily. The code reflects choices a self-taught Python builder makes weekly. Both are open to scrutiny.

---

## License

No license. Code is public for review and learning. Reuse beyond personal study requires written permission.
