# PlanSignal IVaR

**Inventory Value at Risk — quantifying inventory exposure in EUR for mid-market chemical and pharmaceutical manufacturers.**

🔗 **[Live demo](https://plansignal-ivar.streamlit.app)** · 📊 [LinkedIn series](https://www.linkedin.com/in/george-onet) · 🛠 Built with Python, Pandas, Streamlit

---

## The problem

Mid-market manufacturers run two inventory conversations in parallel — and they rarely meet on the same screen.

**Plants** track local, volume-based KPIs: stock on hand, days of cover, service level. Monthly cadence. Physical units.

**Finance** tracks global, value-based KPIs: working capital, EBITDA exposure, write-off risk. Quarterly cadence. Euros.

When the gap shows up late in the year, the pressure to cut inventory fast forces decisions made on the wrong metric. Procurement and Finance reach different conclusions from different spreadsheets. The cuts that *get* made aren't always the cuts that *should* be made.

PlanSignal IVaR closes that gap by translating inventory position into one financial risk figure — per material, fully traceable, in EUR.

## What it does

Upload two standard ERP exports. In seconds, the model produces:

- **Total Financial Exposure** split into **EBITDA at Risk** (operating earnings) and **Capital at Risk** (working capital, below the EBITDA line)
- Six additive risk dimensions: **Understock**, **Aging / Expiry**, **Tariff / Country**, **Commodity Price** *(EBITDA-recoverable)* and **Overstock**, **LT Volatility** *(capital-recoverable)*
- Two stress lenses: **Supply Continuity** (sole-source outage scenario) and **Margin Sensitivity** (P&L impact on high-margin SKUs)
- A prioritised **Action List** surfacing SKUs by decision category — Trapped Working Capital, Shelf-life Risk, Single-source Liability
- A one-click **S&OP Agenda PDF** so Procurement and Finance walk into the meeting with the same sheet

Every EUR figure is independently auditable. Hover any column header for the formula. Open the per-material drilldown for the full decomposition.

## How to use it

1. Open [plansignal-ivar.streamlit.app](https://plansignal-ivar.streamlit.app) — demo data is preloaded
2. Adjust the sidebar parameters (holding cost rate, forward horizon, tariff exposure, margin threshold) to match your business
3. Upload your own data when ready — standard ERP column names auto-map

To run locally:

```bash
git clone https://github.com/george-onet/plansignal-ivar.git
cd plansignal-ivar
pip install -r requirements.txt
streamlit run ivar_app.py
```

## Methodology

PlanSignal IVaR models inventory exposure across six additive financial risk dimensions, separated into **EBITDA at Risk** (Understock, Aging, Tariff, Commodity — all flow through the P&L) and **Capital at Risk** (Overstock, LT Volatility — both rest on holding-cost-based carrying charges).

Holding cost is modeled at a fully-loaded annual rate per industry convention. Strictly, the cost-of-capital portion sits below the EBITDA line; the operating portion is the EBITDA-recoverable component. All holding-cost-based exposures are conservatively classified as Capital at Risk — making EBITDA at Risk a defensive floor estimate rather than a ceiling.

The model has been stress-tested on a 500-material messy portfolio across sixteen edge cases including format inconsistencies, sole-source flag variants, zero and negative margin SKUs, past-expiry inventory, perfect-reliability and high-volatility lead time data, and sparse-observation filters. Math reconciles to the euro at both portfolio and material level.

For dimension-by-dimension formulas and assumptions, see the in-app methodology expanders and the LinkedIn series Parts I–IIIb.

## Tech stack

Python 3.11 · Pandas · NumPy · Streamlit · SQLite (action status persistence) · ReportLab (PDF export)

## Why I built this

During my research, the same issue kept resurfacing: mid-market manufacturers face a persistent disconnect between volume-driven local KPIs and value-based global inventory targets. Plants hit their local numbers while the business still misses the global EBITDA target — and by the time the gap shows up, the cuts that get made aren't always the cuts that should be made.

PlanSignal IVaR is my answer: a tool that translates inventory position into a single financial language, aligning Procurement and Finance on a unified assessment so decisions are driven by the same clean metrics.

It's the tool I wished I'd had during my years in production and supply planning.

## About

Built by **George Onet** — supply chain planning professional, production and supply planning in complex manufacturing environments. 

🔗 [LinkedIn](https://www.linkedin.com/in/george-onet) · 📂 [PlanSignal v1](https://planning-risk-app.streamlit.app) (finished goods risk prioritisation)

## License

MIT — see [LICENSE](./LICENSE).

