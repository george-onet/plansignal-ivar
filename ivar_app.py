# =============================================================================
# PlanSignal IVaR — Inventory Value at Risk
# =============================================================================
# Target   : Mid-market chemical and pharmaceutical manufacturers (€50M–€500M)
# Purpose  : Financial risk quantification per material — NOT a reorder tool.
#
# Eight risk dimensions, each expressed in EUR:
#   1. Understock          — lost production / missed sales risk
#   2. Overstock           — cash trapped / holding cost
#   3. Supply Continuity   — worst-case single-source disruption scenario
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

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

# =============================================================================
# CONSTANTS
# =============================================================================

APP_VERSION = "2.0"
LOG_FILE    = "ivar_usage_log.csv"
TRUE_FLAG_VALUES = {"Y", "YES", "TRUE", "1", "X"}
FALSE_FLAG_VALUES = {"N", "NO", "FALSE", "0", ""}

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
# (Supply Continuity is a stress scenario; Margin Sensitivity is a P&L lens
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
    "concentration_ivar":   "Supply Continuity (€)",
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
    inventory_target_eur:  float = 0.0    # Finance-set ceiling; 0 = auto-default to current portfolio value
    include_in_transit:    bool  = True   # treat in-transit (buyer's title) as part of effective inventory per dimension rules


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

def normalize_bool_flag(value) -> Optional[bool]:
    """
    Normalize ERP-style boolean flags.

    True examples:
    Y, YES, TRUE, 1, X

    False examples:
    N, NO, FALSE, 0, blank

    Unknown values return None so they can be flagged in the Data Quality panel.
    """
    raw = str(value).strip().upper()

    if raw in TRUE_FLAG_VALUES:
        return True

    if raw in FALSE_FLAG_VALUES or raw in {"NAN", "NONE", "NULL"}:
        return False

    return None


def is_true_flag(value) -> bool:
    """Safe boolean check used by risk logic."""
    return normalize_bool_flag(value) is True

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

def _inventory_with_transit(row: pd.Series, p: IVaRParams, ignore_horizon: bool = False) -> float:
    """
    Effective inventory in units = on-hand + applicable in-transit (buyer's title).
    
    In-transit treatment rules:
    - If params.include_in_transit is False → in-transit excluded everywhere
      (Finance view: only on-hand stock counts toward IVaR)
    - If in_transit_arrival_days > horizon AND ignore_horizon=False →
      in-transit excluded for this dimension (arrives outside analysis window)
    - ignore_horizon=True is used by Supply Continuity and Tariff: the stock
      is on our books from origin under buyer's title regardless of arrival
      timing, so supplier failure or tariff change exposes it whether it
      arrives in 30 days or 300
    
    Returns units (not EUR). Caller multiplies by unit_cost where needed.
    """
    on_hand = _safe(row.get("inventory_on_hand"))
    if not p.include_in_transit:
        return on_hand
    transit = _safe(row.get("in_transit_units"))
    if transit <= 0:
        return on_hand
    if not ignore_horizon:
        arrival = _safe(row.get("in_transit_arrival_days"))
        if arrival > p.horizon_days:
            return on_hand
    return on_hand + transit

def _understock(row: pd.Series, p: IVaRParams) -> float:
    """
    Inventory exposure from coverage shortfall.
    
    When stock is below the level needed to cover lead time, remaining inventory
    is effectively locked — committed to firm orders/production runs and
    unavailable for reduction. Exposure scales with shortfall severity:
    the closer to zero coverage, the larger the locked share.
    """
    demand = _safe(row.get("avg_daily_demand"))
    if demand <= 0:
        return 0.0
    inventory = _inventory_with_transit(row, p)
    lt        = _safe(row.get("lead_time_days"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)
    if lt <= 0:
        return 0.0
    
    coverage   = inventory / demand
    if coverage >= lt:
        return 0.0   # adequately covered — no inventory locked by shortfall
    
    locked_fraction = (lt - coverage) / lt   # 0 at full coverage, 1 at zero coverage
    inventory_value = inventory * cost
    return inventory_value * locked_fraction


def _overstock(row: pd.Series, p: IVaRParams) -> float:
    """Holding cost on capital trapped in inventory above 1.5 × safety stock."""
    inventory = _safe(row.get("inventory_on_hand"))
    ss        = _safe(row.get("safety_stock"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)

    excess  = max(0.0, inventory - ss * 1.5)
    capital = excess * cost
    return capital * (p.holding_cost_rate / 365) * p.horizon_days


def _concentration(row: pd.Series, p: IVaRParams) -> float:
    """
    Inventory exposure from sole-source supplier failure (stress scenario).
    
    EUR of inventory tied up in sole-sourced materials that is exposed to loss
    if the supplier fails — stock becomes either stranded (waiting on
    qualification of an alternative source) or written off (if requalification
    requires reformulation). Flat full exposure: if you're sole-sourced and
    supplier fails, the inventory value is the exposure.
    
    The sole_source_lt_factor slider governs which materials get flagged into
    the Action List (longer outage scenarios surface more borderline cases),
    not the severity per material.
    """
    if not is_true_flag(row.get("sole_source", "N")):
        return 0.0
    
    cost = _safe(row.get("unit_cost_eur"), 1.0)
    # ignore_horizon=True: in-transit under buyer's title is on our books from
    # origin, so a supplier failure or contract default exposes it regardless
    # of when the shipment would have arrived
    effective_units = _inventory_with_transit(row, p, ignore_horizon=True)
    return effective_units * cost


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
    """
    Margin-weighted inventory exposure on high-margin materials.
    
    Lens on Understock: high-margin materials carry amplified P&L weight per
    EUR of locked inventory. If understock-locked stock has to be discounted
    or written off, the margin loss is the additional P&L impact on top of
    the inventory exposure already captured in Understock.
    """
    margin = _safe(row.get("margin_pct"))
    if margin < p.margin_critical_pct:
        return 0.0
    
    demand = _safe(row.get("avg_daily_demand"))
    if demand <= 0:
        return 0.0
    
    inventory = _inventory_with_transit(row, p)
    lt        = _safe(row.get("lead_time_days"))
    cost      = _safe(row.get("unit_cost_eur"), 1.0)
    if lt <= 0:
        return 0.0
    
    coverage = inventory / demand
    if coverage >= lt:
        return 0.0
    
    locked_fraction = (lt - coverage) / lt
    inventory_value = inventory * cost
    return inventory_value * locked_fraction * (margin / 100)


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

    # ignore_horizon=True: tariff exposure applies on arrival regardless of
    # when stock lands. In-transit at origin under buyer's title is already
    # exposed to the tariff change because revaluation hits at customs clearance
    effective_units = _inventory_with_transit(row, p, ignore_horizon=True)
    cost = _safe(row.get("unit_cost_eur"), 1.0)
    return effective_units * cost * (rate / 100)


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
# ACTION LIST — categorize SKUs by joint Procurement+Finance decision type
# =============================================================================

ACTION_CATEGORIES = {
    "drawdown": {
        "label":    "Trapped Working Capital",
        "color":    "#C03A2C",  # red — highest-priority lever to close gap
        "emoji":    "🟥",
        "decision": "Procurement + Finance: write-down, repurpose, divest, or hold?",
        "rank":     1,
    },
    "writeoff": {
        "label":    "Shelf-life risk",
        "color":    "#E67E22",  # orange — P&L timing decision
        "emoji":    "🟧",
        "decision": "Procurement + Finance: when to book the loss, and how much is recoverable?",
        "rank":     2,
    },
    "structural": {
        "label":    "Single-source liability",
        "color":    "#F1C40F",  # yellow — informational, not in-period lever
        "emoji":    "🟨",
        "decision": "Procurement + Finance: dual-source investment, contingency reserve, or risk acceptance?",
        "rank":     3,
    },
}


def categorize_action(row: pd.Series, params: IVaRParams, materiality_eur: float = 50_000.0) -> tuple[Optional[str], str]:
    """
    Assign ONE action category per SKU, based on the most pressing joint
    Procurement+Finance decision. Returns (category_key, trigger_reason).
    Precedence: writeoff > drawdown > structural. Higher-precedence categories
    absorb SKUs that would otherwise also qualify for lower bands — keeps the
    action list clean (single primary category per SKU, per Decision 1).
    Returns (None, "") for SKUs that don't trigger any band.
    """
    inventory   = _safe(row.get("inventory_on_hand"))
    cost        = _safe(row.get("unit_cost_eur"), 1.0)
    demand      = _safe(row.get("avg_daily_demand"))
    lt          = _safe(row.get("lead_time_days"))
    shelf_life  = _safe(row.get("shelf_life_days"))
    days_oh     = _safe(row.get("days_on_hand"))
    sole_source = is_true_flag(row.get("sole_source", "N"))
    inventory_value = inventory * cost

    # ── 1. FORCED WRITE-OFF (highest precedence) ──
    if shelf_life > 0 and days_oh > shelf_life:
        return "writeoff", f"Past expiry ({days_oh - shelf_life:.0f} days over shelf life)"
    if shelf_life > 0 and (shelf_life - days_oh) <= 30 and inventory_value >= materiality_eur:
        return "writeoff", f"{shelf_life - days_oh:.0f} days of shelf life remaining"

    # ── 2. DRAWDOWN CANDIDATES ──
    if demand > 0 and lt > 0:
        coverage_days = inventory / demand
        if coverage_days >= 3 * lt and inventory_value >= materiality_eur:
            return "drawdown", f"{coverage_days:.0f} days coverage vs {lt:.0f}-day lead time"

    if demand == 0 and inventory_value >= materiality_eur:
        return "drawdown", f"€{inventory_value:,.0f} held with no recorded demand"

    # ── 3. STRUCTURAL COMMITMENTS (lowest precedence) ──
    if sole_source and lt >= 60 and inventory_value >= materiality_eur * 2:
        return "structural", f"Sole-source, {lt:.0f}-day lead time, €{inventory_value:,.0f} committed"

    return None, ""


def build_action_list(ivar_df: pd.DataFrame, params: IVaRParams) -> pd.DataFrame:
    """
    Add three columns to the IVaR-enriched DataFrame:
    - action_category: 'drawdown' | 'writeoff' | 'structural' | None
    - action_trigger: human-readable reason this SKU was flagged
    - action_value_eur: inventory book value (the EUR figure both functions discuss)
    """
    work = ivar_df.copy()
    cats_triggers = work.apply(lambda r: categorize_action(r, params), axis=1)
    work["action_category"] = cats_triggers.apply(lambda t: t[0])
    work["action_trigger"]  = cats_triggers.apply(lambda t: t[1])
    work["action_value_eur"] = (
        pd.to_numeric(work["inventory_on_hand"], errors="coerce").fillna(0)
        * pd.to_numeric(work["unit_cost_eur"], errors="coerce").fillna(0)
    )
    return work

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
        "concentration_ivar":    "Supply Continuity (€)",
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

    st.subheader("Supply Continuity Risk")
    sole_source_lt_factor = st.slider(
        "Worst-case outage (× lead time)",
        min_value=1.0, max_value=5.0, value=2.0, step=0.5,
        help=(
            "Outage assumption used for Action List flagging severity — longer "
            "outage scenarios surface more borderline sole-source materials into "
            "the 'Single-source liability' band. Note: Supply Continuity exposure "
            "itself is flat full inventory value at risk per sole-sourced material "
            "(not scaled by this slider) — the slider controls which materials get "
            "flagged, not the per-material number."
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
    st.subheader("Finance Target")
    inventory_target_eur = st.number_input(
        "Inventory target (€)",
        min_value=0.0,
        value=0.0,
        step=100_000.0,
        format="%.0f",
        help="Finance-set inventory value ceiling. Used to compute Gap to Target. Set to 0 to auto-default to current portfolio value.",
    )

    st.subheader("In-Transit Treatment")
    include_in_transit = st.toggle(
        "Include in-transit inventory in IVaR",
        value=True,
        help=(
            "When ON: in-transit stock under buyer's title (FCA, EXW, FOB, CIF, "
            "CFR, CIP) is counted as effective inventory per dimension rules. "
            "Understock, Margin Sensitivity → counted if arriving within horizon. "
            "Supply Continuity, Tariff → counted regardless of arrival timing. "
            "Overstock, Aging, Commodity → never counted (off balance sheet or not aging). "
            "When OFF: only on-hand stock is used everywhere — useful for Finance "
            "to see IVaR as it appears on the balance sheet today."
        ),
    )

    params = IVaRParams(
        holding_cost_rate     = holding_pct / 100,
        horizon_days          = horizon_days,
        margin_critical_pct   = margin_critical_pct,
        sole_source_lt_factor = sole_source_lt_factor,
        tariff_change_pct     = tariff_change_pct,
        obsolescence_days     = obsolescence_days,
        inventory_target_eur  = inventory_target_eur,
        include_in_transit    = include_in_transit,
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
    'Inventory carrying cost and value at risk — quantified per material.'
    '</div>'
    '<div style="font-size: 0.95rem; color: #777; font-style: italic; margin-top: 2px; margin-bottom: 8px;">'
    'Two sides of the same EBITDA coin: where to reduce inventory, and where reductions would create new risk.'
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
- **Supply Continuity** is a stress test (worst-case sole-source outage), not a probability-weighted loss
- **Margin Sensitivity** highlights the P&L portion of Understock on high-margin materials (already inside Understock, called out for visibility)

This separation keeps Total IVaR comparable to financial VaR concepts: expected loss over a horizon, with stress and amplification surfaced alongside.

---

**The more material data you provide, the more dimensions IVaR quantifies.**

4 dimensions compute from any portfolio: **Understock**, **Overstock**, **Supply Continuity**, **LT Volatility**.

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
| **Supply Continuity** *(lens)* | *Stress scenario* — full outage of sole-source supplier lasting LT × factor days | Outage days × daily demand × (unit cost + margin) |
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
ivar_df = build_action_list(ivar_df, params)

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
sole_source_mask = ivar_df["sole_source"].apply(is_true_flag)
sole_source_count = int(sole_source_mask.sum())

# Portfolio inventory value — Finance KPI denominator
total_inventory_value = float(
    (pd.to_numeric(ivar_df["inventory_on_hand"], errors="coerce").fillna(0)
     * pd.to_numeric(ivar_df["unit_cost_eur"], errors="coerce").fillna(0)).sum()
)
# Resolve target: 0 = auto-default to current portfolio value (neutral starting point)
effective_target = params.inventory_target_eur if params.inventory_target_eur > 0 else total_inventory_value
gap_to_target = total_inventory_value - effective_target  # positive = over-target (red); negative = under (green)

# ─── HEADLINE: CURRENT INVENTORY VALUE vs TARGET ───
with st.container(border=True):
    st.markdown("### Current Inventory Value vs. Target")
    if params.inventory_target_eur > 0:
        st.caption(
            f"{total_materials} materials · "
            f"Target set by Finance: € {params.inventory_target_eur:,.0f}"
        )
    else:
        st.caption(
            f"{total_materials} materials · "
            "No Finance target set — defaulting to current portfolio value. "
            "Enter a target in the sidebar to see Gap to Target."
        )

    h1, h2, h3 = st.columns(3)
    h1.metric(
        "Current inventory value",
        f"€ {total_inventory_value:,.0f}",
        help="Sum of (inventory on hand × unit cost) across all materials in the portfolio. Matches the Finance balance-sheet view of inventory.",
    )
    h2.metric(
        "Finance target",
        f"€ {effective_target:,.0f}",
        help="Inventory ceiling set in the sidebar. Defaults to current portfolio value when no target is entered.",
    )

    # Gap to target — labeled framing per Decision 2
    if params.inventory_target_eur > 0:
        if gap_to_target > 0:
            h3.metric(
                "Gap to target",
                f"Over by € {gap_to_target:,.0f}",
                delta=f"{(gap_to_target / effective_target * 100):.1f}% above target",
                delta_color="inverse",
                help="Current inventory value exceeds the Finance target. Use the Action List below to identify drawdown candidates.",
            )
        elif gap_to_target < 0:
            h3.metric(
                "Gap to target",
                f"Under by € {abs(gap_to_target):,.0f}",
                delta=f"{(abs(gap_to_target) / effective_target * 100):.1f}% below target",
                delta_color="normal",
                help="Current inventory value is below the Finance target. Headroom available.",
            )
        else:
            h3.metric(
                "Gap to target",
                "On target",
                help="Current inventory value matches the Finance target exactly.",
            )
    else:
        h3.metric(
            "Gap to target",
            "—",
            help="Set a Finance target in the sidebar to see Gap to Target.",
        )

# ─── ACTION LIST ───
with st.container(border=True):
    st.markdown("### 📋 Action List")
    st.caption(
        "SKUs surfaced for joint Procurement+Finance decisions. "
        "Each SKU appears in **one** category — the most pressing decision required. "
        "Ranked by inventory value within each band."
    )

    items_per_band = st.slider(
        "Items per band",
        min_value=3, max_value=10, value=3, step=1,
        help="How many SKUs to show in each action category. Default 3 (most pressing only).",
        key="action_items_per_band",
    )

    # Action list helper — render a single category block
    def _render_action_band(category_key: str, ivar_df_with_actions: pd.DataFrame, n: int) -> None:
        cfg = ACTION_CATEGORIES[category_key]
        band_df = (
            ivar_df_with_actions[ivar_df_with_actions["action_category"] == category_key]
            .sort_values("action_value_eur", ascending=False)
            .head(n)
        )

        total_in_band = int((ivar_df_with_actions["action_category"] == category_key).sum())
        eur_in_band = float(
            ivar_df_with_actions.loc[ivar_df_with_actions["action_category"] == category_key, "action_value_eur"].sum()
        )

        st.markdown(
            f"<div style='margin-top:1rem;'>"
            f"<span style='font-size:1.05rem;font-weight:600;color:{cfg['color']};'>{cfg['emoji']} {cfg['label']}</span>"
            f"&nbsp;&nbsp;<span style='color:#888;font-size:0.85rem;'>({total_in_band} SKU{'s' if total_in_band != 1 else ''} flagged · "
            f"€ {eur_in_band:,.0f} total)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#666;font-size:0.85rem;font-style:italic;margin-bottom:0.5rem;'>"
            f"{cfg['decision']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if band_df.empty:
            st.markdown(
                "<div style='color:#888;font-size:0.85rem;padding:0.5rem 1rem;'>"
                "No SKUs flagged — clean exposure in this category."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        # Build a compact display table for this band
        display = pd.DataFrame({
            "Material":      band_df["material"].astype(str),
            "Description":   band_df.get("description", pd.Series([""] * len(band_df))).astype(str),
            "Supplier":      band_df.get("supplier",    pd.Series([""] * len(band_df))).astype(str),
            "Inventory (€)": band_df["action_value_eur"].round(0),
            "Trigger":       band_df["action_trigger"].astype(str),
        })
        st.dataframe(
            display.style.format({"Inventory (€)": "€{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Inventory (€)": st.column_config.NumberColumn(help="Inventory book value: inventory_on_hand × unit_cost_eur"),
                "Trigger":       st.column_config.TextColumn(width="large"),
            },
        )

    # Render the three bands in precedence order (rank: drawdown=1, writeoff=2, structural=3)
    for cat_key in sorted(ACTION_CATEGORIES.keys(), key=lambda k: ACTION_CATEGORIES[k]["rank"]):
        _render_action_band(cat_key, ivar_df, items_per_band)

# ─── EXPECTED LOSS (TOTAL IVaR) ───
with st.container(border=True):
    st.markdown("### Total IVaR — EBITDA at Risk")
    st.caption(
        f"{total_materials} materials · "
        f"{horizon_days}-day forward horizon · "
        f"{holding_pct}% annual holding cost · "
        f"sum of six additive dimensions — the EBITDA at risk in inventory over the horizon"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total IVaR", f"€ {total_ivar:,.0f}",
        help=(
            "EBITDA at risk over the forward horizon — sum of six additive dimensions: "
            "Overstock (carrying cost on excess), LT Volatility (extra safety stock cost), "
            "Aging (write-off risk), Tariff (revaluation), Commodity (forward margin), "
            "Understock (locked working capital). Reduce inventory where these are largest = recover EBITDA."
        ),
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
    st.caption(
        "Surfaced separately — *not* summed into Total IVaR. "
        "These don't quantify recoverable EBITDA; they warn where reductions would create new risk. "
        "Supply Continuity flags exposure if sole-source suppliers fail. "
        "Margin Sensitivity flags amplified P&L impact on high-margin SKUs."
    )

    l1, l2, l3, l4 = st.columns(4)
    l1.metric(
        "Supply Continuity (stress)", f"€ {total_concentration:,.0f}",
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
# % of Total IVaR — uses full portfolio denominator, not just the displayed top_n
portfolio_total_ivar = ivar_df["total_ivar"].sum()
if portfolio_total_ivar > 0:
    table_df["% of Total"] = (table_df["Total IVaR (€)"] / portfolio_total_ivar * 100).round(1)
else:
    table_df["% of Total"] = 0.0
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
    if "% of Total" in df.columns:
        fmt["% of Total"] = "{:.1f}%"
    styler = styler.format(fmt)
    return styler

st.dataframe(
    _style_numeric_cols(table_df),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Understock (€)":          st.column_config.NumberColumn(help="Days short of coverage × daily demand × (unit cost + margin)."),
        "Overstock (€)":           st.column_config.NumberColumn(help="Excess inventory above 1.5× safety stock × unit cost × holding rate × horizon / 365."),
        "Supply Continuity (€)":   st.column_config.NumberColumn(help="Worst-case sole-source outage: lead time × factor × daily demand × (unit cost + margin). Stress scenario, not expected loss."),
        "LT Volatility (€)":       st.column_config.NumberColumn(help="z × CV(lead time) × LT × daily demand × unit cost × holding rate × horizon / 365. Requires LT history file."),
        "Margin Sensitivity (€)":  st.column_config.NumberColumn(help="P&L portion of Understock exposure on materials above the margin threshold. Lens — already inside Understock."),
        "Aging / Expiry (€)":      st.column_config.NumberColumn(help="Inventory value × obsolescence rate. Uses shelf-life data if provided, else aging-bucket method."),
        "Tariff / Country (€)":    st.column_config.NumberColumn(help="Inventory × unit cost × expected tariff change %. Applied to high-risk countries of origin by default."),
        "Commodity Price (€)":     st.column_config.NumberColumn(help="Horizon demand × unit cost × expected price change %. Replacement cost basis."),
        "Total IVaR (€)":          st.column_config.NumberColumn(help="Sum of the six additive IVaR dimensions. Supply Continuity and Margin Sensitivity are shown as lenses and are not included in Total IVaR."),
        "% of Total":              st.column_config.NumberColumn(help="This material's Total IVaR as a percentage of the full portfolio Total IVaR (sum across all 500 materials, not just those displayed)."),
    },
)

st.markdown("---")


# =============================================================================
# CONCENTRATION RISK PANEL
# =============================================================================

sole_df = ivar_df[sole_source_mask].copy()

if not sole_df.empty:
    st.subheader("Supply Continuity — Sole-Source Materials")
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
        "concentration_ivar": "Supply Continuity IVaR (€)", "total_ivar": "Total IVaR (€)",
    }, inplace=True)

    st.dataframe(
        conc_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Supply Continuity IVaR (€)": st.column_config.NumberColumn(format="€%,.0f"),
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
k4.metric("Supply Continuity", f"€ {_safe(sel['concentration_ivar']):,.0f}")

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
            "All dimensions", "Understock, Supply Continuity, Margin Sensitivity",
            "All dimensions", "Overstock", "Understock, Supply Continuity, Margin Sensitivity, Commodity",
            "Understock, Supply Continuity, LT Volatility", "Derived",
            "Display only", "Supply Continuity IVaR", "Tariff / Country IVaR",
             shelf_life_tag, days_oh_tag,
            "Commodity Price IVaR", "LT Volatility IVaR",
        ],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)


st.markdown("---")

# =============================================================================
# PDF S&OP AGENDA
# =============================================================================

def build_sop_pdf(
    ivar_df: pd.DataFrame,
    params: IVaRParams,
    items_per_band: int = 3,
) -> bytes:
    """
    Generate a one-page S&amp;OP Action Agenda PDF.
    
    Header   : portfolio snapshot (inventory, target, gap, EBITDA at Risk)
    Body     : three action category blocks, top N SKUs each
    Footer   : model parameters + timestamp
    
    Designed to be the agenda sheet for an S&amp;OP meeting where Procurement
    and Finance need to walk in with the same decisions in front of them.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.6*cm, leftMargin=1.6*cm,
        topMargin=1.4*cm, bottomMargin=1.2*cm,
    )
    
    # ─── STYLES ───
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=16, textColor=rl_colors.HexColor("#222"),
        spaceAfter=4, alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=9, textColor=rl_colors.HexColor("#666"),
        spaceAfter=14, italic=True,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, spaceBefore=10, spaceAfter=4,
    )
    decision_style = ParagraphStyle(
        "Decision", parent=styles["Normal"],
        fontSize=8.5, textColor=rl_colors.HexColor("#555"),
        italic=True, spaceAfter=6,
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=7.5, textColor=rl_colors.HexColor("#888"),
        spaceBefore=14,
    )
    
    # ─── COMPUTE PORTFOLIO HEADLINE NUMBERS ───
    inv_value = (ivar_df["inventory_on_hand"] * ivar_df["unit_cost_eur"]).sum()
    target    = params.inventory_target_eur if params.inventory_target_eur > 0 else inv_value
    gap       = inv_value - target
    total_ivar = ivar_df[RISK_COLS_ADDITIVE].sum().sum()
    
    elements = []
    
    # ─── HEADER ───
    elements.append(Paragraph("PlanSignal IVaR — S&amp;OP Action Agenda", title_style))
    elements.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{len(ivar_df)} materials · "
        f"{params.horizon_days}-day forward horizon",
        subtitle_style,
    ))
    
    # ─── PORTFOLIO SNAPSHOT TABLE ───
    snapshot_data = [
        ["Current inventory value", f"€ {inv_value:,.0f}"],
        ["Finance target",          f"€ {target:,.0f}"],
        ["Gap to target",           f"€ {gap:+,.0f}" + (" over" if gap > 0 else " under" if gap < 0 else "")],
        ["EBITDA at Risk (Total IVaR)", f"€ {total_ivar:,.0f}"],
    ]
    snapshot = Table(snapshot_data, colWidths=[7*cm, 6*cm])
    snapshot.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#F5F5F5")),
        ("BACKGROUND", (0,2), (-1,2), rl_colors.HexColor("#FDF0EE")),
        ("BACKGROUND", (0,3), (-1,3), rl_colors.HexColor("#FDF0EE")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("FONTNAME", (1,0), (1,-1), "Helvetica-Bold"),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,-1), (-1,-1), 0.4, rl_colors.HexColor("#CCC")),
    ]))
    elements.append(snapshot)
    elements.append(Spacer(1, 0.4*cm))
    
    # ─── ACTION CATEGORY BLOCKS ───
    for cat_key in sorted(ACTION_CATEGORIES.keys(), key=lambda k: ACTION_CATEGORIES[k]["rank"]):
        cfg = ACTION_CATEGORIES[cat_key]
        band_df = (
            ivar_df[ivar_df["action_category"] == cat_key]
            .sort_values("action_value_eur", ascending=False)
            .head(items_per_band)
        )
        total_in_band = int((ivar_df["action_category"] == cat_key).sum())
        eur_in_band = float(
            ivar_df.loc[ivar_df["action_category"] == cat_key, "action_value_eur"].sum()
        )
        
        # Section heading with color band
        heading = Paragraph(
            f'<font color="{cfg["color"]}"><b>■</b></font> &nbsp;'
            f'<b>{cfg["label"]}</b> &nbsp;'
            f'<font color="#888" size="9">'
            f'({total_in_band} SKU{"s" if total_in_band != 1 else ""} · € {eur_in_band:,.0f} total)'
            f'</font>',
            section_style,
        )
        elements.append(heading)
        elements.append(Paragraph(cfg["decision"], decision_style))
        
        if band_df.empty:
            elements.append(Paragraph(
                '<font color="#888" size="8.5"><i>No SKUs flagged in this band.</i></font>',
                styles["Normal"],
            ))
            elements.append(Spacer(1, 0.3*cm))
            continue
        
        # SKU table
        table_data = [["Material", "Description", "Supplier", "Inventory €", "Trigger"]]
        for _, row in band_df.iterrows():
            table_data.append([
                str(row.get("material", ""))[:14],
                str(row.get("description", ""))[:22],
                str(row.get("supplier", ""))[:20],
                f"€ {row['action_value_eur']:,.0f}",
                str(row.get("action_trigger", ""))[:38],
            ])
        
        skus = Table(
            table_data,
            colWidths=[2.2*cm, 3.4*cm, 3.4*cm, 2.6*cm, 5.4*cm],
            repeatRows=1,
        )
        skus.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor(cfg["color"])),
            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8.5),
            ("FONTNAME",   (0,1), (-1,-1), "Helvetica"),
            ("ALIGN",      (3,0), (3,-1), "RIGHT"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("GRID", (0,0), (-1,-1), 0.3, rl_colors.HexColor("#DDD")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#FAFAFA")]),
        ]))
        elements.append(skus)
        elements.append(Spacer(1, 0.3*cm))
    
    # ─── FOOTER ───
    elements.append(Paragraph(
        f"<b>Model parameters:</b> &nbsp;"
        f"Holding cost {params.holding_cost_rate*100:.0f}%/yr · "
        f"Horizon {params.horizon_days}d · "
        f"Margin threshold {params.margin_critical_pct:.0f}% · "
        f"Sole-source LT factor {params.sole_source_lt_factor:.1f}× · "
        f"Tariff change {params.tariff_change_pct:.0f}% · "
        f"In-transit {'included' if params.include_in_transit else 'excluded'}",
        footer_style,
    ))
    elements.append(Paragraph(
        f"PlanSignal IVaR v{APP_VERSION} · plansignal.streamlit.app",
        footer_style,
    ))
    
    doc.build(elements)
    return buf.getvalue()

# =============================================================================
# EXPORT
# =============================================================================

col_export_xlsx, col_export_pdf = st.columns(2)

with col_export_xlsx:
    excel_bytes = build_excel_export(ivar_df, params)
    st.download_button(
        "📊 Export IVaR Report (Excel)",
        data=excel_bytes,
        file_name=f"ivar_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with col_export_pdf:
    # Use the same items_per_band slider value that drives the on-screen Action List
    # so what the user prints matches what they're looking at.
    pdf_bytes = build_sop_pdf(ivar_df, params, items_per_band=items_per_band)
    st.download_button(
        "📄 Export S&amp;OP Agenda (PDF)",
        data=pdf_bytes,
        file_name=f"sop_agenda_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
        help="One-page S&amp;OP action agenda — portfolio snapshot + top SKUs per category.",
    )

st.caption(
    f"PlanSignal IVaR v{APP_VERSION} · "
    f"Pharma / Chemical · "
    f"Holding: {holding_pct}%/yr · "
    f"Horizon: {horizon_days}d · "
    f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
