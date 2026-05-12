# =============================================================================
# PlanSignal IVaR — Inventory Value at Risk
# =============================================================================
# Target   : Mid-market chemical and pharmaceutical manufacturers (€50M–€500M)
# Purpose  : Financial risk quantification per material — NOT a reorder tool.
#
# Eight risk dimensions, each expressed in EUR:
#   1. Understock          — lost production / missed sales risk
#   2. Overstock           — cash trapped / holding cost
#   3. Concentration       — worst-case single-source disruption scenario
#   4. LT Volatility       — safety stock inflation from lead-time variance
#   5. Margin Sensitivity  — P&L portion of Understock on high-margin materials (lens)
#   6. Aging / Expiry      — obsolescence / write-off risk
#   7. Tariff / Country    — revaluation on trade policy change
#   8. Commodity Price     — replacement cost gap on input price moves
#
# Holding cost default: 20 %/yr (pharma/chemical; range 20–30 %).
# Forward horizon default: 90 days (one quarter).
# =============================================================================

import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONSTANTS
# =============================================================================

APP_VERSION = "1.0"
LOG_FILE    = "ivar_usage_log.csv"

# Standard aging obsolescence rates by days-on-hand bucket
AGING_BUCKETS = [
    (30,  0.00),
    (60,  0.10),
    (90,  0.25),
    (180, 0.50),
    (999, 0.80),
]

# Countries flagged for tariff risk (2026 trade environment)
HIGH_TARIFF_COUNTRIES = {
    "CN", "US", "IN", "TW", "VN", "MX", "TR", "RU",
}

# Additive dimensions — sum into Total IVaR (pure expected loss)
RISK_COLS_ADDITIVE = [
    "understock_ivar",
    "overstock_ivar",
    "lt_volatility_ivar",
    "aging_ivar",
    "tariff_ivar",
    "commodity_ivar",
]

# Lens dimensions — surfaced separately, not summed into Total IVaR
# (Concentration is a stress scenario; Margin Sensitivity is a P&L lens
# on Understock for high-margin SKUs)
RISK_COLS_LENS = [
    "concentration_ivar",
    "margin_critical_ivar",
]

# Union — used wherever we need the full set (table display, drilldowns, charts)
RISK_COLS = RISK_COLS_ADDITIVE + RISK_COLS_LENS

RISK_LABELS = {
    "understock_ivar":      "Understock (€)",
    "overstock_ivar":       "Overstock (€)",
    "concentration_ivar":   "Concentration (€)",
    "lt_volatility_ivar":   "LT Volatility (€)",
    "margin_critical_ivar": "Margin Sensitivity (€)",
    "aging_ivar":           "Aging / Expiry (€)",
    "tariff_ivar":          "Tariff / Country (€)",
    "commodity_ivar":       "Commodity Price (€)",
    "total_ivar":           "Total IVaR (€)",
}

REQUIRED_CORE = [
    "material", "unit_cost_eur", "inventory_on_hand",
    "avg_daily_demand", "lead_time_days",
]

RENAME_CORE = {
    # material identifiers
    "sku": "material", "material_code": "material", "product_code": "material",
    "item": "material", "material_no": "material", "material_number": "material",
    # description
    "material_description": "description", "product_name": "description", "name": "description",
    # cost / margin
    "unit_cost": "unit_cost_eur", "cost": "unit_cost_eur", "cost_per_unit": "unit_cost_eur",
    "margin": "margin_pct", "gross_margin": "margin_pct", "margin_%": "margin_pct",
    # inventory
    "stock_on_hand": "inventory_on_hand", "on_hand": "inventory_on_hand",
    "inventory": "inventory_on_hand",
    "ss": "safety_stock",
    # demand / lead time
    "daily_demand": "avg_daily_demand", "demand": "avg_daily_demand",
    "lead_time": "lead_time_days", "leadtime": "lead_time_days", "lt": "lead_time_days",
    # supplier
    "vendor": "supplier",
    "single_source": "sole_source",
    # country
    "country": "country_of_origin", "origin": "country_of_origin", "coo": "country_of_origin",
    # aging
    "shelf_life": "shelf_life_days",
    # price risk
    "price_change_pct": "expected_price_change_pct",
    "price_change": "expected_price_change_pct",
}


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class IVaRParams:
    holding_cost_rate:     float = 0.20   # annual; pharma/chemical default
    horizon_days:          int   = 90
    margin_critical_pct:   float = 30.0
    sole_source_lt_factor: float = 2.0    # worst-case outage = LT × this
    tariff_change_pct:     float = 10.0   # global default; overridable per material
    obsolescence_days:     int   = 90     # aging kicks in above this threshold


# =============================================================================
# HELPERS
# =============================================================================

def normalize_material(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.upper()
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace(" ", "", regex=False)
    )


def clean_number(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
    )


def _safe(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _aging_rate(days_on_hand: float) -> float:
    for threshold, rate in AGING_BUCKETS:
        if days_on_hand <= threshold:
            return rate
    return 0.80


def log_event(name: str) -> None:
    pd.DataFrame([{"timestamp": datetime.now(), "event": name}]).to_csv(
        LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False,
    )


def _read_file(f) -> pd.DataFrame:
    return pd.read_excel(f) if f.name.lower().endswith(".xlsx") else pd.read_csv(f, sep=None, engine="python")


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("﻿", "", regex=False)
    )
    return df


# =============================================================================
# RISK DIMENSION FUNCTIONS  (each returns EUR float)
# =============================================================================

def _understock(row: pd.Series, p: IVaRParams) -> float:
    """Lost production / missed sales when stock cannot cover the replenishment cycle."""
    demand = _safe(row.get("avg_daily_demand"))
    if demand <= 0:
        return 0.0
    inventory = _safe(row.get("inventory_on_hand"))
    lt        = _safe(row.get("lead_time_days"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)
    margin    = _safe(row.get("margin_pct")) / 100

    coverage    = inventory / demand
    days_short  = max(0.0, lt - coverage)
    units_short = days_short * demand
    return units_short * cost * (1 + margin)


def _overstock(row: pd.Series, p: IVaRParams) -> float:
    """Holding cost on capital trapped in inventory above 1.5 × safety stock."""
    inventory = _safe(row.get("inventory_on_hand"))
    ss        = _safe(row.get("safety_stock"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)

    excess  = max(0.0, inventory - ss * 1.5)
    capital = excess * cost
    return capital * (p.holding_cost_rate / 365) * p.horizon_days


def _concentration(row: pd.Series, p: IVaRParams) -> float:
    """Worst-case disruption cost for sole-sourced materials (stress scenario, not EV)."""
    if str(row.get("sole_source", "N")).strip().upper() not in ("Y", "YES", "TRUE", "1"):
        return 0.0
    demand = _safe(row.get("avg_daily_demand"))
    lt     = _safe(row.get("lead_time_days"))
    cost   = _safe(row.get("unit_cost_eur"), 1.0)
    margin = _safe(row.get("margin_pct")) / 100

    outage_days  = lt * p.sole_source_lt_factor
    units_at_risk = demand * outage_days
    return units_at_risk * cost * (1 + margin)


def _lt_volatility(row: pd.Series, lt_cv_map: dict, p: IVaRParams) -> float:
    """Additional safety-stock holding cost driven by observed lead-time variability."""
    material = str(row.get("material", ""))
    cv       = lt_cv_map.get(material, np.nan)
    if np.isnan(cv) or cv <= 0:
        return 0.0

    demand = _safe(row.get("avg_daily_demand"))
    lt     = _safe(row.get("lead_time_days"))
    cost   = _safe(row.get("unit_cost_eur"), 1.0)

    z              = 1.65   # 95 % service level
    extra_ss_units = z * cv * lt * demand
    return extra_ss_units * cost * (p.holding_cost_rate / 365) * p.horizon_days


def _margin_critical(row: pd.Series, p: IVaRParams) -> float:
    """P&L amplification on high-margin materials with understock exposure."""
    margin = _safe(row.get("margin_pct"))
    if margin < p.margin_critical_pct:
        return 0.0

    demand = _safe(row.get("avg_daily_demand"))
    if demand <= 0:
        return 0.0

    inventory   = _safe(row.get("inventory_on_hand"))
    lt          = _safe(row.get("lead_time_days"))
    cost        = _safe(row.get("unit_cost_eur"), 1.0)
    coverage    = inventory / demand
    days_short  = max(0.0, lt - coverage)
    units_short = days_short * demand
    return units_short * cost * (margin / 100)


def _aging(row: pd.Series, p: IVaRParams) -> float:
    """Write-off risk from inventory age vs shelf life or standard aging buckets."""
    days_oh    = _safe(row.get("days_on_hand"))
    shelf_life = _safe(row.get("shelf_life_days"))
    inventory  = _safe(row.get("inventory_on_hand"))
    cost       = _safe(row.get("unit_cost_eur"), 1.0)
    inv_value  = inventory * cost

    if shelf_life > 0:
        remaining = max(0.0, shelf_life - days_oh)
        if remaining < p.horizon_days:
            fraction = 1.0 - (remaining / p.horizon_days)
            return inv_value * min(fraction, 1.0)
        return 0.0

    # Fallback: standard aging bucket
    if days_oh > p.obsolescence_days:
        return inv_value * _aging_rate(days_oh)
    return 0.0


def _tariff(row: pd.Series, p: IVaRParams) -> float:
    """Inventory revaluation exposure from expected tariff change."""
    coo = str(row.get("country_of_origin", "")).strip().upper()

    # Material-level override takes precedence over global param
    mat_rate = _safe(row.get("tariff_change_pct", np.nan), np.nan)
    if np.isnan(mat_rate):
        if coo not in HIGH_TARIFF_COUNTRIES:
            return 0.0
        rate = p.tariff_change_pct
    else:
        rate = mat_rate

    inventory = _safe(row.get("inventory_on_hand"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)
    return inventory * cost * (rate / 100)


def _commodity(row: pd.Series, p: IVaRParams) -> float:
    """Replacement cost gap: horizon-period demand repriced at expected input price change."""
    price_change = _safe(row.get("expected_price_change_pct"))
    if price_change <= 0:
        return 0.0
    demand = _safe(row.get("avg_daily_demand"))
    cost   = _safe(row.get("unit_cost_eur"), 1.0)
    return demand * p.horizon_days * cost * (price_change / 100)


# =============================================================================
# LEAD-TIME CV MAP
# =============================================================================

def compute_lt_cv_map(lt_history: Optional[pd.DataFrame]) -> dict:
    """Returns {material: CV(lead_time)} from historical data (min 3 observations)."""
    if lt_history is None or lt_history.empty:
        return {}
    result = {}
    for mat, grp in lt_history.groupby("material"):
        vals = grp["lead_time_days"].dropna()
        if len(vals) >= 3:
            mean = vals.mean()
            std  = vals.std(ddof=1)
            result[str(mat)] = (std / mean) if mean > 0 else 0.0
    return result


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_ivar(df: pd.DataFrame, params: IVaRParams, lt_cv_map: dict) -> pd.DataFrame:
    work = df.copy()
    work["understock_ivar"]      = work.apply(lambda r: _understock(r, params),                  axis=1)
    work["overstock_ivar"]       = work.apply(lambda r: _overstock(r, params),                   axis=1)
    work["concentration_ivar"]   = work.apply(lambda r: _concentration(r, params),               axis=1)
    work["lt_volatility_ivar"]   = work.apply(lambda r: _lt_volatility(r, lt_cv_map, params),    axis=1)
    work["margin_critical_ivar"] = work.apply(lambda r: _margin_critical(r, params),             axis=1)
    work["aging_ivar"]           = work.apply(lambda r: _aging(r, params),                       axis=1)
    work["tariff_ivar"]          = work.apply(lambda r: _tariff(r, params),                      axis=1)
    work["commodity_ivar"]       = work.apply(lambda r: _commodity(r, params),                   axis=1)
    work["total_ivar"]           = work[RISK_COLS_ADDITIVE].sum(axis=1)
    return work.sort_values("total_ivar", ascending=False)


# =============================================================================
# TEMPLATE
# =============================================================================

def build_template() -> pd.DataFrame:
    return pd.DataFrame({
        "material":                  ["CHEM-001",         "CHEM-002",        "API-003",              "EXCIP-004",                  "SOLV-005"  ],
        "description":               ["Acetic Anhydride", "Sodium Hydroxide", "Active Ingredient X",  "Microcrystalline Cellulose",  "Ethanol 96%"],
        "unit_cost_eur":             [4.20,               1.80,               85.00,                  2.50,                          3.10        ],
        "margin_pct":                [18.0,               12.0,               62.0,                   8.0,                           15.0        ],
        "inventory_on_hand":         [12000,              45000,              800,                    22000,                         8500        ],
        "safety_stock":              [5000,               15000,              400,                    8000,                          3000        ],
        "avg_daily_demand":          [400,                1200,               25,                     650,                           280         ],
        "lead_time_days":            [35,                 21,                 90,                     14,                            28          ],
        "supplier":                  ["BASF SE",          "Solvay",           "Lonza AG",             "JRS Pharma",                  "Caldic"    ],
        "sole_source":               ["Y",                "N",                "Y",                    "N",                           "N"         ],
        "country_of_origin":         ["DE",               "BE",               "CN",                   "DE",                          "NL"        ],
        "shelf_life_days":           [730,                1095,               365,                    1825,                          548         ],
        "days_on_hand":              [680,                 38,                 32,                     34,                            30          ],
        "expected_price_change_pct": [5.0,                3.0,                0.0,                    2.0,                           8.0         ],
    })


# =============================================================================
# EXCEL EXPORT
# =============================================================================

def build_excel_export(ivar_df: pd.DataFrame, params: IVaRParams) -> bytes:
    col_map = {
        "material":              "Material",
        "description":           "Description",
        "supplier":              "Supplier",
        "sole_source":           "Sole Source",
        "country_of_origin":     "Country of Origin",
        "inventory_on_hand":     "Inventory (units)",
        "unit_cost_eur":         "Unit Cost (€)",
        "understock_ivar":       "Understock IVaR (€)",
        "overstock_ivar":        "Overstock IVaR (€)",
        "concentration_ivar":    "Concentration IVaR (€)",
        "lt_volatility_ivar":    "LT Volatility IVaR (€)",
        "margin_critical_ivar":  "Margin Sensitivity IVaR (€)",
        "aging_ivar":            "Aging / Expiry IVaR (€)",
        "tariff_ivar":           "Tariff / Country IVaR (€)",
        "commodity_ivar":        "Commodity Price IVaR (€)",
        "total_ivar":            "Total IVaR (€)",
    }
    available = {k: v for k, v in col_map.items() if k in ivar_df.columns}
    export = ivar_df[list(available.keys())].copy()
    export.columns = list(available.values())
    for c in [v for v in available.values() if "(€)" in v]:
        export[c] = export[c].round(0)

    params_df = pd.DataFrame({
        "Parameter": [
            "Holding Cost Rate (annual)",
            "Forward Horizon (days)",
            "Margin Sensitivity Threshold (%)",
            "Sole-Source LT Factor (×)",
            "Global Tariff Change (%)",
            "Aging Threshold (days on hand)",
        ],
        "Value": [
            f"{params.holding_cost_rate * 100:.0f}%",
            str(params.horizon_days),
            f"{params.margin_critical_pct:.0f}%",
            f"{params.sole_source_lt_factor:.1f}×",
            f"{params.tariff_change_pct:.0f}%",
            str(params.obsolescence_days),
        ],
        "Note": [
            "Pharma/chemical benchmark: 20–30 % (APQC / nVentic)",
            "Aligns with quarterly reporting cycle",
            "Materials above this gross margin % are flagged for margin sensitivity",
            "Worst-case outage duration relative to normal lead time",
            "Applied to CN, US, IN, TW, VN, MX, TR, RU by default",
            "Days on hand above which aging IVaR activates (no shelf-life data)",
        ],
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="IVaR by Material")
        params_df.to_excel(writer, index=False, sheet_name="Model Parameters")
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# PAGE CONFIG & STYLE
# =============================================================================

st.set_page_config(page_title="PlanSignal IVaR", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 4rem; padding-bottom: 6rem; }
.ivar-title      { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0; }
.ivar-sub        { color: #5f6368; font-size: 1.0rem; margin-bottom: 0.5rem; }
.dim-note        { color: #5f6368; font-size: 0.88rem; margin-bottom: 0.4rem; }
.muted           { color: #9e9e9e; font-size: 0.80rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if "visited" not in st.session_state:
    log_event("ivar_opened")
    st.session_state["visited"] = True


# =============================================================================
# SIDEBAR — MODEL PARAMETERS
# =============================================================================

with st.sidebar:
    st.header("Model Parameters")

    st.subheader("Holding Cost")
    holding_pct = st.slider(
        "Annual holding cost rate (%)",
        min_value=5, max_value=40, value=20, step=1,
        help=(
            "Pharma / chemical benchmark: 20–30 % (APQC, nVentic). "
            "Includes cost of capital, storage, insurance, temperature handling. "
            "General manufacturing floor is ~15 %."
        ),
    )

    st.subheader("Forward Horizon")
    horizon_days = st.slider(
        "Exposure window (days)",
        min_value=30, max_value=365, value=90, step=30,
        help="90 days = one quarter, aligns with reporting cycle and typical replenishment window.",
    )

    st.subheader("Margin Sensitivity Threshold")
    margin_critical_pct = st.slider(
        "Gross margin threshold (%)",
        min_value=10, max_value=80, value=30, step=5,
        help="Materials above this gross margin % carry amplified P&L risk on stockouts.",
    )

    st.subheader("Concentration Risk")
    sole_source_lt_factor = st.slider(
        "Worst-case outage (× lead time)",
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help=(
            "Assumed disruption length as a multiple of normal lead time. "
            "2× means 'supplier down for twice the normal replenishment cycle'. "
            "This is a stress scenario, not a probability-weighted expected value."
        ),
    )

    st.subheader("Tariff Exposure")
    tariff_change_pct = st.slider(
        "Expected tariff change (%)",
        min_value=0, max_value=50, value=10, step=5,
        help=(
            "Applied to materials from high-tariff countries (CN, US, IN, TW, VN, MX, TR, RU) "
            "unless a material-level rate is provided in the data."
        ),
    )

    st.subheader("Aging Risk")
    obsolescence_days = st.slider(
        "Aging threshold (days on hand)",
        min_value=30, max_value=180, value=90, step=15,
        help="Used only when no shelf-life data is available. Aging IVaR activates above this threshold.",
    )

    params = IVaRParams(
        holding_cost_rate     = holding_pct / 100,
        horizon_days          = horizon_days,
        margin_critical_pct   = margin_critical_pct,
        sole_source_lt_factor = sole_source_lt_factor,
        tariff_change_pct     = tariff_change_pct,
        obsolescence_days     = obsolescence_days,
    )

    st.markdown("---")
    st.caption(
        f"Holding: {holding_pct}%/yr · "
        f"Horizon: {horizon_days}d · "
        f"Margin threshold: {margin_critical_pct}% · "
        f"Tariff: {tariff_change_pct}%"
    )


# =============================================================================
# HEADER
# =============================================================================

LOGO_IVAR  = "PlanSignal_IVaR_light.png"
LOGO_LIGHT = "PlanSignal_light_.PNG"

if os.path.exists(LOGO_IVAR):
    st.image(LOGO_IVAR, width=240)
elif os.path.exists(LOGO_LIGHT):
    col_logo, col_hdr = st.columns([1, 6])
    with col_logo:
        st.image(LOGO_LIGHT, width=160)
    with col_hdr:
        st.markdown('<div class="ivar-title">PlanSignal IVaR</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ivar-sub">Inventory Value at Risk — '
            'financial exposure per material, right now.</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown('<div class="ivar-title">PlanSignal IVaR</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="font-size: 1.05rem; color: #555; margin-top: -8px;">'
    'Inventory financial exposure, quantified per material.'
    '</div>'
    '<div style="font-size: 0.95rem; color: #777; font-style: italic; margin-top: 2px; margin-bottom: 8px;">'
    'One number Procurement and Finance can both stand behind.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# =============================================================================
# FILE UPLOAD
# =============================================================================

with st.expander("📂 Upload data files", expanded=True):
    main_uploaded = st.file_uploader(
        "Material data (required)", type=["csv", "xlsx"], key="ivar_main",
    )
    lt_history_uploaded = st.file_uploader(
        "Lead time history (optional — unlocks LT Volatility IVaR)",
        type=["csv", "xlsx"], key="ivar_lt",
    )

with st.expander("ℹ️ How IVaR works", expanded=False):
    st.markdown("""
**Eight risk dimensions, EUR-denominated, computed over a configurable forward horizon.**
Each dimension surfaces a different way inventory can cost the business — and quantifies it in money, not units.

**Total IVaR sums the six additive dimensions** (Understock, Overstock, LT Volatility, Aging, Tariff, Commodity) — these represent expected loss exposures.

Two further dimensions are surfaced as **lenses, not summands**:
- **Concentration** is a stress test (worst-case sole-source outage), not a probability-weighted loss
- **Margin Sensitivity** highlights the P&L portion of Understock on high-margin materials (already inside Understock, called out for visibility)

This separation keeps Total IVaR comparable to financial VaR concepts: expected loss over a horizon, with stress and amplification surfaced alongside.

---

**The more material data you provide, the more dimensions IVaR quantifies.**

4 dimensions compute from any portfolio: **Understock**, **Overstock**, **Concentration**, **LT Volatility**.

4 more unlock as you add data:
- **Margin Sensitivity exposure** — needs gross margin per material
- **Tariff exposure** — needs country of origin
- **Aging / expiry exposure** — needs shelf life and current age
- **Commodity price exposure** — needs expected price change

---

Each dimension is independently auditable. **Hover any column header in the table for a one-line methodology note.**

| Dimension | What it measures | Formula |
|---|---|---|
| **Understock** | Lost production / missed sales before next replenishment arrives | Days short × daily demand × (unit cost + margin) |
| **Overstock** | Capital cost of inventory held above 1.5 × safety stock | Excess units × unit cost × holding rate × horizon / 365 |
| **Concentration** *(lens)* | *Stress scenario* — full outage of sole-source supplier lasting LT × factor days | Outage days × daily demand × (unit cost + margin) |
| **LT Volatility** | Extra safety stock required to absorb observed lead-time variance | z (1.65) × CV(LT) × LT × daily demand × unit cost × holding rate × horizon / 365 |
| **Margin Sensitivity** *(lens)* | P&L portion of Understock exposure on high-margin materials | Shortfall units × unit cost × margin % (high-margin materials only) |
| **Aging / Expiry** | Write-off risk from inventory nearing shelf-life expiry or aged beyond threshold | Inventory value × obsolescence rate (shelf-life or aging-bucket method) |
| **Tariff / Country** | Inventory revaluation from expected tariff change on country of origin | Inventory × unit cost × tariff change % |
| **Commodity Price** | Replacement cost gap when replenishing horizon demand at inflated input prices | Horizon demand × unit cost × expected price change % |

Holding cost default: **20 %/year** (pharma/chemical lower bound; APQC/nVentic benchmark range 20–30 %).
    """)

template_csv = build_template().to_csv(index=False).encode("utf-8")
st.download_button("Download CSV template", template_csv, "ivar_template.csv", "text/csv")
st.markdown("---")


# =============================================================================
# LOAD MAIN FILE
# =============================================================================

if main_uploaded is None:
    st.info("No file uploaded — showing sample data (5 chemical / pharma materials).")
    df = build_template()
else:
    try:
        df = _std_cols(_read_file(main_uploaded))
        df.rename(columns=RENAME_CORE, inplace=True)
        missing = [c for c in REQUIRED_CORE if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}. Showing sample data.")
            df = build_template()
        else:
            df["material"] = normalize_material(df["material"])
            st.success(f"Material data loaded — {len(df):,} materials")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = build_template()

# Coerce numerics / fill defaults
for col, default in {
    "unit_cost_eur": 1.0, "margin_pct": 0.0,
    "inventory_on_hand": 0.0, "safety_stock": 0.0,
    "avg_daily_demand": 0.0, "lead_time_days": 14.0,
    "shelf_life_days": 0.0, "days_on_hand": 0.0,
    "expected_price_change_pct": 0.0,
}.items():
    if col not in df.columns:
        df[col] = default
    else:
        df[col] = pd.to_numeric(clean_number(df[col]), errors="coerce").fillna(default)

for col in ["supplier", "sole_source", "country_of_origin", "description"]:
    if col not in df.columns:
        df[col] = ""


# =============================================================================
# LOAD LEAD TIME HISTORY
# =============================================================================

lt_cv_map = {}

if lt_history_uploaded is not None:
    try:
        lt_h = _std_cols(_read_file(lt_history_uploaded))
        lt_h.rename(columns={
            "sku": "material", "product_code": "material", "item": "material",
            "lead_time": "lead_time_days", "leadtime": "lead_time_days", "lt": "lead_time_days",
        }, inplace=True)
        if {"material", "lead_time_days"}.issubset(lt_h.columns):
            lt_h["material"]       = normalize_material(lt_h["material"])
            lt_h["lead_time_days"] = pd.to_numeric(lt_h["lead_time_days"], errors="coerce")
            lt_cv_map = compute_lt_cv_map(lt_h)
            st.success(f"Lead time history loaded — {len(lt_cv_map)} materials with volatility data")
        else:
            st.warning("Lead time history must contain: material, lead_time_days.")
    except Exception as e:
        st.error(f"Lead time history error: {e}")


# =============================================================================
# COMPUTE IVaR
# =============================================================================

ivar_df = compute_ivar(df, params, lt_cv_map)


# =============================================================================
# PORTFOLIO SUMMARY
# =============================================================================

# Additive totals (sum into Total Portfolio IVaR)
total_ivar          = float(ivar_df["total_ivar"].sum())
total_understock    = float(ivar_df["understock_ivar"].sum())
total_overstock     = float(ivar_df["overstock_ivar"].sum())
total_lt_volatility = float(ivar_df["lt_volatility_ivar"].sum())
total_aging         = float(ivar_df["aging_ivar"].sum())
total_tariff        = float(ivar_df["tariff_ivar"].sum())
total_commodity     = float(ivar_df["commodity_ivar"].sum())

# Lens totals (surfaced separately, NOT summed into Total IVaR)
total_concentration = float(ivar_df["concentration_ivar"].sum())
total_margin_sens   = float(ivar_df["margin_critical_ivar"].sum())

# Portfolio metadata
total_materials   = len(ivar_df)
high_risk_count   = int((ivar_df["total_ivar"] >= ivar_df["total_ivar"].quantile(0.75)).sum())
sole_source_mask  = ivar_df["sole_source"].astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])
sole_source_count = int(sole_source_mask.sum())

# ─── EXPECTED LOSS (TOTAL IVaR) ───
with st.container(border=True):
    st.markdown("### Expected Loss — Total IVaR")
    st.caption(
        f"{total_materials} materials · "
        f"{horizon_days}-day forward horizon · "
        f"{holding_pct}% annual holding cost · "
        f"sum of six additive dimensions"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total IVaR", f"€ {total_ivar:,.0f}",
        help="Sum of the six additive expected-loss dimensions: Understock, Overstock, LT Volatility, Aging, Tariff, Commodity.",
    )
    c2.metric(
        "Understock", f"€ {total_understock:,.0f}",
        help="Lost production / missed sales where inventory cannot cover the next replenishment cycle.",
    )
    c3.metric(
        "Overstock", f"€ {total_overstock:,.0f}",
        help="Capital cost of excess inventory held above 1.5× safety stock over the forward horizon.",
    )
    c4.metric(
        "LT Volatility", f"€ {total_lt_volatility:,.0f}",
        help="Additional safety-stock holding cost driven by observed lead-time variability (requires LT history file).",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Aging / Expiry", f"€ {total_aging:,.0f}",
        help="Write-off risk from inventory nearing shelf-life expiry or aged beyond the obsolescence threshold.",
    )
    c6.metric(
        "Tariff / Country", f"€ {total_tariff:,.0f}",
        help=f"Inventory revaluation at {tariff_change_pct}% tariff change on high-risk countries of origin.",
    )
    c7.metric(
        "Commodity Price", f"€ {total_commodity:,.0f}",
        help="Replacement cost gap for horizon-period demand at expected input price changes.",
    )
    c8.metric(
        "High-Risk Materials", high_risk_count,
        delta="top quartile by Total IVaR", delta_color="inverse",
    )

# ─── STRESS / LENS METRICS ───
with st.container(border=True):
    st.markdown("### Stress & Lens Metrics")
    st.caption("Surfaced separately — *not* summed into Total IVaR. These represent stress scenarios and P&L lenses, not expected loss.")

    l1, l2, l3, l4 = st.columns(4)
    l1.metric(
        "Concentration (stress)", f"€ {total_concentration:,.0f}",
        delta=f"{sole_source_count} sole-sourced materials", delta_color="inverse",
        help="Worst-case disruption cost across all sole-sourced materials — full supplier outage scenario, not probability-weighted.",
    )
    l2.metric(
        "Margin Sensitivity (lens)", f"€ {total_margin_sens:,.0f}",
        help="P&L portion of Understock exposure on high-margin materials. Already inside Understock — surfaced separately for visibility.",
    )

st.markdown("---")


# =============================================================================
# RISK SOURCE BREAKDOWN — portfolio bar chart
# =============================================================================

with st.expander("📊 Portfolio IVaR — by Risk Source", expanded=True):
    st.markdown(
        '<div class="dim-note">'
        'Where the financial exposure is coming from across the full portfolio.'
        '</div>',
        unsafe_allow_html=True,
    )
    breakdown = (
        pd.DataFrame({
            "Risk Dimension": [RISK_LABELS[c] for c in RISK_COLS_ADDITIVE]
                              + [f"{RISK_LABELS[c]} (lens)" for c in RISK_COLS_LENS],
            "Exposure (€)":   [float(ivar_df[c].sum()) for c in RISK_COLS_ADDITIVE]
                              + [float(ivar_df[c].sum()) for c in RISK_COLS_LENS],
        })
        .sort_values("Exposure (€)", ascending=False)
        .query("`Exposure (€)` > 0")
    )
    if not breakdown.empty:
        # Sort ascending so largest bar lands at the top in Plotly horizontal layout
        breakdown_plot = breakdown.sort_values("Exposure (€)", ascending=True)
        max_val = breakdown_plot["Exposure (€)"].max()

        # Build a gradient — bars get darker as they get larger (brand red family)
        n = len(breakdown_plot)
        colors = [
            f"rgba(192, 58, 44, {0.30 + 0.65 * (i / max(n - 1, 1))})"
            for i in range(n)
        ]

        fig = go.Figure(go.Bar(
            x=breakdown_plot["Exposure (€)"],
            y=breakdown_plot["Risk Dimension"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.15)", width=0.5)),
            text=[f"€ {v:,.0f}" for v in breakdown_plot["Exposure (€)"]],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{y}</b><br>Exposure: €%{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=80, t=20, b=40),
            xaxis=dict(
                title="Exposure (€)",
                tickformat=",.0f",
                tickprefix="€ ",
                gridcolor="rgba(128,128,128,0.20)",
                range=[0, max_val * 1.18],
            ),
            yaxis=dict(title=None, automargin=True),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No exposure to display with current data and parameters.")


# =============================================================================
# RISK MATRIX — Understock vs Overstock scatter
# =============================================================================

with st.expander("📊 Risk Matrix — Understock vs Overstock", expanded=True):
    st.markdown(
        '<div class="dim-note">'
        'Each point = one material. '
        '<b>Right</b> = overstock dominates (cash trapped). '
        '<b>Up</b> = understock dominates (production / service at risk). '
        '<b>Top-right</b> = both. '
        '<b>Bottom-left</b> = controlled.'
        '</div>',
        unsafe_allow_html=True,
    )
# Pull all 6 additive dimensions so the hover can show a non-zero breakdown
    scatter_df = ivar_df[[
        "material", "description", "supplier",
        "understock_ivar", "overstock_ivar", "lt_volatility_ivar",
        "aging_ivar", "tariff_ivar", "commodity_ivar",
        "total_ivar",
    ]].copy()

    # Pretty labels for the dimensions that appear in the hover breakdown
    _hover_dim_labels = {
        "understock_ivar":     "Understock",
        "overstock_ivar":      "Overstock",
        "lt_volatility_ivar":  "LT Volatility",
        "aging_ivar":          "Aging / Expiry",
        "tariff_ivar":         "Tariff / Country",
        "commodity_ivar":      "Commodity Price",
    }

    def _fmt_eur_compact(v: float) -> str:
        """Compact EUR for hover: €12.28M / €330k / €450."""
        if v >= 1_000_000:
            return f"€{v/1_000_000:.2f}M"
        if v >= 1_000:
            return f"€{v/1_000:.0f}k"
        return f"€{v:.0f}"

    def _build_hover_html(row: pd.Series) -> str:
        # Main driver = highest non-zero additive dimension
        dim_values = {label: row[col] for col, label in _hover_dim_labels.items() if row[col] > 0}
        if dim_values:
            top_label, top_value = max(dim_values.items(), key=lambda x: x[1])
            total = row["total_ivar"]
            pct = (top_value / total * 100) if total > 0 else 0
            main_driver_line = f"<b>Main driver:</b> {top_label} ({pct:.0f}%)"
        else:
            main_driver_line = "<b>Main driver:</b> —"

        # Breakdown — only non-zero dimensions, sorted descending
        breakdown_rows = sorted(dim_values.items(), key=lambda x: x[1], reverse=True)
        breakdown_lines = "".join(
            f"<br>&nbsp;&nbsp;• {label}: {_fmt_eur_compact(v)}"
            for label, v in breakdown_rows
        )

        return (
            f"<b>{row['material']}</b> — {row['description']}<br>"
            f"<i>{row['supplier']}</i><br>"
            f"<b>Total IVaR:</b> {_fmt_eur_compact(row['total_ivar'])}<br>"
            f"{main_driver_line}"
            f"{breakdown_lines}"
            "<extra></extra>"
        )

    scatter_df["__hover"] = scatter_df.apply(_build_hover_html, axis=1)
    scatter_df = scatter_df.rename(columns={
        "understock_ivar": "Understock IVaR (€)",
        "overstock_ivar":  "Overstock IVaR (€)",
        "total_ivar":      "Total IVaR (€)",
    })

    # Quadrant dividers — set at the median of NON-ZERO values so the cross
    # lands in a sensible place even when most SKUs are at zero on one axis
    x_vals = scatter_df["Overstock IVaR (€)"]
    y_vals = scatter_df["Understock IVaR (€)"]
    x_med = float(x_vals[x_vals > 0].median()) if (x_vals > 0).any() else 0.0
    y_med = float(y_vals[y_vals > 0].median()) if (y_vals > 0).any() else 0.0

    fig = px.scatter(
        scatter_df,
        x="Overstock IVaR (€)",
        y="Understock IVaR (€)",
        size="Total IVaR (€)",
        color="Total IVaR (€)",
        color_continuous_scale="Plasma",
        size_max=30,
        custom_data=["__hover"],
    )

    # Apply the custom rich hover and dot styling
    fig.update_traces(
        marker=dict(
            line=dict(width=0.8, color="rgba(128,128,128,0.7)"),
            opacity=0.85,
        ),
        hovertemplate="%{customdata[0]}",
    )

    # Quadrant divider lines (stay in data coordinates — they're anchored to the medians)
    fig.add_hline(y=y_med, line_dash="dot", line_color="rgba(128,128,128,0.50)", line_width=1)
    fig.add_vline(x=x_med, line_dash="dot", line_color="rgba(128,128,128,0.50)", line_width=1)

    # Quadrant labels — paper-anchored (xref/yref = "paper") so they stay in
    # the chart corners during zoom and pan. 0,0 = bottom-left; 1,1 = top-right.
    annotations = [
        dict(
            x=0.99, y=0.99, xref="paper", yref="paper",
            text="<b>Both</b><br><span style='font-size:10px;color:#999'>Critical</span>",
            showarrow=False, font=dict(size=12, color="#C03A2C"),
            align="right", xanchor="right", yanchor="top",
        ),
        dict(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text="<b>Production at risk</b><br><span style='font-size:10px;color:#999'>Understock dominates</span>",
            showarrow=False, font=dict(size=12, color="#6A1B9A"),
            align="left", xanchor="left", yanchor="top",
        ),
        dict(
            x=0.99, y=0.04, xref="paper", yref="paper",
            text="<b>Cash trapped</b><br><span style='font-size:10px;color:#999'>Overstock dominates</span>",
            showarrow=False, font=dict(size=12, color="#1565C0"),
            align="right", xanchor="right", yanchor="bottom",
        ),
        dict(
            x=0.01, y=0.04, xref="paper", yref="paper",
            text="<b>Controlled</b><br><span style='font-size:10px;color:#999'>Low on both axes</span>",
            showarrow=False, font=dict(size=12, color="#2E7D32"),
            align="left", xanchor="left", yanchor="bottom",
        ),
    ]

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=40),
        annotations=annotations,
        xaxis=dict(
            tickformat=",.0f", tickprefix="€ ",
            gridcolor="rgba(128,128,128,0.20)", zerolinecolor="rgba(128,128,128,0.35)",
        ),
        yaxis=dict(
            tickformat=",.0f", tickprefix="€ ",
            gridcolor="rgba(128,128,128,0.20)", zerolinecolor="rgba(128,128,128,0.35)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=dict(text="Total IVaR (€)", side="right"),
            tickformat=",.0f", tickprefix="€ ",
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(128,128,128,0.4)",
            font=dict(size=12, color="#222"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# =============================================================================
# MATERIAL IVaR DECOMPOSITION TABLE
# =============================================================================

st.subheader("Material IVaR — Full Decomposition")
st.markdown(
    '<div class="dim-note">'
    'Every EUR figure is independently auditable — formula and assumptions visible in the '
    'expander above. Hover any column header for a one-line methodology note. '
    'Ranked by Total IVaR.'
    '</div>',
    unsafe_allow_html=True,
)

max_n   = len(ivar_df)
if max_n <= 5:
    top_n = max_n
    st.caption(f"Showing all {max_n} materials")
else:
    top_n = st.slider(
        "Materials to display",
        min_value=5,
        max_value=max_n,
        value=min(20, max_n),
    )

# Build display columns
base_cols = ["material"]
if ivar_df["description"].astype(str).str.strip().replace("", np.nan).notna().any():
    base_cols.append("description")
if ivar_df["supplier"].astype(str).str.strip().replace("", np.nan).notna().any():
    base_cols.append("supplier")

table_df = ivar_df.head(top_n)[base_cols + RISK_COLS + ["total_ivar"]].copy()
table_df.rename(columns={
    **RISK_LABELS,
    "material":    "Material",
    "description": "Description",
    "supplier":    "Supplier",
}, inplace=True)

for c in [v for v in RISK_LABELS.values() if v in table_df.columns]:
    table_df[c] = table_df[c].round(0)

# Per-column heatmap coloring. Each numeric column scaled to its own max so colors
# stay meaningful within-column instead of being flattened by Total IVaR's larger range.
_numeric_cols = [v for v in RISK_LABELS.values() if v in table_df.columns]

def _style_numeric_cols(df: pd.DataFrame):
    styler = df.style
    for col in _numeric_cols:
        col_max = df[col].max()
        if col_max and col_max > 0:
            # Total IVaR uses deeper Reds to anchor it as the headline column.
            # Dimension columns use lighter OrRd to read as secondary.
            cmap = "Reds" if col == "Total IVaR (€)" else "OrRd"
            styler = styler.background_gradient(
                subset=[col],
                cmap=cmap,
                vmin=0,
                vmax=col_max,
            )
    # EUR formatting applied via Styler so it survives alongside the gradient.
    fmt = {c: "€{:,.0f}" for c in _numeric_cols}
    styler = styler.format(fmt)
    return styler

st.dataframe(
    _style_numeric_cols(table_df),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Understock (€)":          st.column_config.NumberColumn(help="Days short of coverage × daily demand × (unit cost + margin)."),
        "Overstock (€)":           st.column_config.NumberColumn(help="Excess inventory above 1.5× safety stock × unit cost × holding rate × horizon / 365."),
        "Concentration (€)":       st.column_config.NumberColumn(help="Worst-case sole-source outage: LT × factor × daily demand × (unit cost + margin). Stress scenario."),
        "LT Volatility (€)":       st.column_config.NumberColumn(help="z × CV(lead time) × LT × daily demand × unit cost × holding rate × horizon / 365. Requires LT history file."),
        "Margin Sensitivity (€)":  st.column_config.NumberColumn(help="P&L portion of Understock exposure on materials above the margin threshold. Lens — already inside Understock."),
        "Aging / Expiry (€)":      st.column_config.NumberColumn(help="Inventory value × obsolescence rate. Uses shelf-life data if provided, else aging-bucket method."),
        "Tariff / Country (€)":    st.column_config.NumberColumn(help="Inventory × unit cost × expected tariff change %. Applied to high-risk countries of origin by default."),
        "Commodity Price (€)":     st.column_config.NumberColumn(help="Horizon demand × unit cost × expected price change %. Replacement cost basis."),
        "Total IVaR (€)":          st.column_config.NumberColumn(help="Sum of all eight risk dimensions. Total financial exposure for this material."),
    },
)

st.markdown("---")


# =============================================================================
# CONCENTRATION RISK PANEL
# =============================================================================

sole_df = ivar_df[sole_source_mask].copy()

if not sole_df.empty:
    st.subheader("Concentration Risk — Sole-Source Materials")
    st.markdown(
        f'<div class="dim-note">'
        f'Worst-case disruption scenario: full supplier outage lasting '
        f'<b>{sole_source_lt_factor:.1f}× the normal lead time</b>. '
        f'This is a stress test — not a probability-weighted expected loss. '
        f'Use it to size buffer stock or dual-sourcing decisions.'
        f'</div>',
        unsafe_allow_html=True,
    )

    conc_cols = ["material"] + [c for c in ["description", "supplier", "country_of_origin"] if c in sole_df.columns]
    conc_cols += ["lead_time_days", "inventory_on_hand", "unit_cost_eur", "concentration_ivar", "total_ivar"]
    conc_display = sole_df[conc_cols].copy()
    conc_display["inventory_value_eur"] = (
        conc_display["inventory_on_hand"] * conc_display["unit_cost_eur"]
    ).round(0)
    conc_display[["concentration_ivar", "total_ivar"]] = conc_display[["concentration_ivar", "total_ivar"]].round(0)

    conc_display.rename(columns={
        "material": "Material", "description": "Description",
        "supplier": "Supplier", "country_of_origin": "Country",
        "lead_time_days": "Lead Time (days)", "inventory_on_hand": "Inventory (units)",
        "unit_cost_eur": "Unit Cost (€)", "inventory_value_eur": "Inventory Value (€)",
        "concentration_ivar": "Concentration IVaR (€)", "total_ivar": "Total IVaR (€)",
    }, inplace=True)

    st.dataframe(
        conc_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Concentration IVaR (€)": st.column_config.NumberColumn(format="€%,.0f"),
            "Inventory Value (€)":    st.column_config.NumberColumn(format="€%,.0f"),
            "Total IVaR (€)":         st.column_config.NumberColumn(format="€%,.0f"),
            "Unit Cost (€)":          st.column_config.NumberColumn(format="€%.2f"),
        },
    )
    st.markdown("---")


# =============================================================================
# MATERIAL DRILLDOWN
# =============================================================================

st.subheader("Material Drilldown")

selected_mat = st.selectbox("Select material", ivar_df["material"].tolist())
sel = ivar_df[ivar_df["material"] == selected_mat].iloc[0]

desc_val = str(sel.get("description", "")).strip()
title_str = f"**{selected_mat}**" + (f" — {desc_val}" if desc_val and desc_val.lower() != "nan" else "")
st.markdown(title_str)

# KPI row 1
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total IVaR",       f"€ {_safe(sel['total_ivar']):,.0f}")
k2.metric("Understock",       f"€ {_safe(sel['understock_ivar']):,.0f}")
k3.metric("Overstock",        f"€ {_safe(sel['overstock_ivar']):,.0f}")
k4.metric("Concentration",    f"€ {_safe(sel['concentration_ivar']):,.0f}")

# KPI row 2
k5, k6, k7, k8 = st.columns(4)
k5.metric("Aging / Expiry",   f"€ {_safe(sel['aging_ivar']):,.0f}")
k6.metric("Tariff / Country", f"€ {_safe(sel['tariff_ivar']):,.0f}")
k7.metric("Commodity Price",  f"€ {_safe(sel['commodity_ivar']):,.0f}")
k8.metric("LT Volatility",    f"€ {_safe(sel['lt_volatility_ivar']):,.0f}")

# Decomposition bar
wf = pd.DataFrame({
    "Risk Dimension": [RISK_LABELS[c] for c in RISK_COLS],
    "Exposure (€)":   [_safe(sel[c]) for c in RISK_COLS],
}).query("`Exposure (€)` > 0").sort_values("Exposure (€)", ascending=False)

if not wf.empty:
    st.markdown("#### IVaR Decomposition")
    wf_plot = wf.sort_values("Exposure (€)", ascending=True)
    n = len(wf_plot)
    max_val = wf_plot["Exposure (€)"].max()
    colors = [
        f"rgba(192, 58, 44, {0.30 + 0.65 * (i / max(n - 1, 1))})"
        for i in range(n)
    ]
    drill_fig = go.Figure(go.Bar(
        x=wf_plot["Exposure (€)"],
        y=wf_plot["Risk Dimension"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.15)", width=0.5)),
        text=[f"€ {v:,.0f}" for v in wf_plot["Exposure (€)"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>Exposure: €%{x:,.0f}<extra></extra>",
    ))
    drill_fig.update_layout(
        height=320,
        margin=dict(l=10, r=80, t=10, b=40),
        dragmode=False,
        xaxis=dict(
            title="Exposure (€)",
            tickformat=",.0f", tickprefix="€ ",
            gridcolor="rgba(128,128,128,0.20)",
            range=[0, max_val * 1.18] if max_val > 0 else None,
        ),
        yaxis=dict(title=None, automargin=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    drill_fig.update_xaxes(fixedrange=True)
    drill_fig.update_yaxes(fixedrange=True)
    st.plotly_chart(drill_fig, use_container_width=True)
else:
    st.success("No material IVaR exposure detected under current parameters.")

# Inputs & assumptions
with st.expander("📋 Material inputs & model assumptions", expanded=False):
    demand_val = _safe(sel.get("avg_daily_demand"), 1)
    coverage   = _safe(sel.get("inventory_on_hand")) / demand_val if demand_val > 0 else 0

# Conditional aging tags — surface whether the figure is precise or fallback
    has_shelf_life = _safe(sel.get("shelf_life_days")) > 0
    shelf_life_tag = "Aging / Expiry IVaR (precise)" if has_shelf_life else "Not used (no shelf life)"
    days_oh_tag    = "Aging / Expiry IVaR (precise)" if has_shelf_life else "Aging / Expiry IVaR (fallback — no shelf life)"

    detail_df = pd.DataFrame({
        "Field": [
            "Unit Cost (€)", "Gross Margin (%)", "Inventory on Hand (units)",
            "Safety Stock (units)", "Avg Daily Demand (units/day)", "Lead Time (days)",
            "Coverage (days)", "Supplier", "Sole Source",
            "Country of Origin", "Shelf Life (days)", "Days on Hand",
            "Expected Price Change (%)", "LT Volatility CV",
        ],
        "Value": [
            f"€ {_safe(sel.get('unit_cost_eur')):.2f}",
            f"{_safe(sel.get('margin_pct')):.1f}%",
            f"{_safe(sel.get('inventory_on_hand')):,.0f}",
            f"{_safe(sel.get('safety_stock')):,.0f}",
            f"{_safe(sel.get('avg_daily_demand')):.1f}",
            f"{_safe(sel.get('lead_time_days')):.0f}",
            f"{coverage:.1f}",
            str(sel.get("supplier", "—")),
            str(sel.get("sole_source", "—")),
            str(sel.get("country_of_origin", "—")),
            f"{_safe(sel.get('shelf_life_days')):.0f}" if _safe(sel.get("shelf_life_days")) > 0 else "Not provided",
            f"{_safe(sel.get('days_on_hand')):.0f}" if _safe(sel.get("days_on_hand")) > 0 else "Not provided",
            f"{_safe(sel.get('expected_price_change_pct')):.1f}%",
            f"{lt_cv_map.get(selected_mat, 0):.3f}" if lt_cv_map.get(selected_mat) else "No history uploaded",
        ],
        "Used in": [
            "All dimensions", "Understock, Concentration, Margin Sensitivity",
            "All dimensions", "Overstock", "Understock, Concentration, Margin Sensitivity, Commodity",
            "Understock, Concentration, LT Volatility", "Derived",
            "Display only", "Concentration IVaR", "Tariff / Country IVaR",
             shelf_life_tag, days_oh_tag,
            "Commodity Price IVaR", "LT Volatility IVaR",
        ],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)


st.markdown("---")


# =============================================================================
# EXPORT
# =============================================================================

excel_bytes = build_excel_export(ivar_df, params)
st.download_button(
    "Export IVaR Report (Excel)",
    data=excel_bytes,
    file_name=f"ivar_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    f"PlanSignal IVaR v{APP_VERSION} · "
    f"Pharma / Chemical · "
    f"Holding: {holding_pct}%/yr · "
    f"Horizon: {horizon_days}d · "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
