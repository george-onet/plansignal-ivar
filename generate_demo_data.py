"""
Generate realistic demo data for PlanSignal IVaR.

Produces a 500-material portfolio for a mid-market pharma/chemical manufacturer.
Calibrated to land Total IVaR in the €10-15M band and flag 25-35 SKUs across
the three Action List categories (Trapped Working Capital, Shelf-life risk,
Single-source liability).

Includes incoterm + in-transit columns for the v2.2 Supply Continuity model.

Run: python generate_demo_data.py
Output: demo_data_realistic.csv
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------------------------------------------------------
# Portfolio composition: 500 materials, 40/30/20/10 split
# -----------------------------------------------------------------------------
COMPOSITION = {
    "API":   {"count": 200, "cost_range": (50, 500),  "demand_range": (1, 20),    "ss_days": (30, 90)},
    "INT":   {"count": 150, "cost_range": (30, 250),  "demand_range": (2, 30),    "ss_days": (20, 60)},
    "EXCIP": {"count": 100, "cost_range": (5,  80),   "demand_range": (10, 150),  "ss_days": (15, 45)},
    "CHEM":  {"count": 50,  "cost_range": (2,  40),   "demand_range": (5, 80),    "ss_days": (10, 30)},
}

DESCRIPTIONS = {
    "API":   ["Active Ingredient", "Bulk API", "Intermediate API", "Crystalline Form",
              "Micronized API", "Sterile API"],
    "INT":   ["Reaction Intermediate", "Catalyst Complex", "Synthesis Intermediate",
              "Purification Intermediate", "Stabilized Intermediate"],
    "EXCIP": ["Microcrystalline Cellulose", "Lactose Monohydrate", "Magnesium Stearate",
              "Croscarmellose Sodium", "Hypromellose", "Mannitol"],
    "CHEM":  ["Sodium Chloride", "Potassium Carbonate", "Citric Acid",
              "Phosphate Buffer", "Solvent A", "Solvent B"],
}

# Country distribution drives incoterm distribution
COUNTRIES = {
    # (country, weight, default_incoterm_pool)
    "DE": (0.18, ["FCA", "EXW", "DAP"]),
    "NL": (0.12, ["FCA", "EXW", "DAP"]),
    "BE": (0.08, ["FCA", "EXW", "DAP"]),
    "ES": (0.07, ["FCA", "DAP"]),
    "IT": (0.06, ["FCA", "DAP"]),
    "GB": (0.05, ["FCA", "DAP", "DDP"]),
    "IN": (0.15, ["CIF", "CFR", "FOB"]),
    "CN": (0.12, ["CIF", "CFR", "FOB"]),
    "US": (0.08, ["FCA", "DAP", "DDP"]),
    "JP": (0.03, ["CIF", "DAP"]),
    "KR": (0.03, ["CIF", "CFR", "FOB"]),
    "MX": (0.02, ["FCA", "DAP"]),
    "TW": (0.01, ["CIF", "FOB"]),
}

SUPPLIERS_BY_COUNTRY = {
    "DE": ["BASF SE", "Bayer", "Merck KGaA", "Evonik", "DSM"],
    "NL": ["DSM", "AkzoNobel", "Caldic BV"],
    "BE": ["Solvay", "Janssen", "UCB"],
    "ES": ["Ercros", "Esteve"],
    "IT": ["Olon SpA", "Flamma"],
    "GB": ["Johnson Matthey", "Croda International"],
    "IN": ["Aurobindo Pharma", "Dr Reddy's", "Cipla Bulk", "Divi's Labs", "Zhejiang Hisun"],
    "CN": ["Zhejiang Huahai", "Lupin China", "Hutchison MediPharma"],
    "US": ["DuPont", "Pfizer CentreOne", "Eternal Materials"],
    "JP": ["Mitsui Chemicals", "Sumitomo"],
    "KR": ["LG Chem", "SK Bioscience"],
    "MX": ["Alpek Polyester"],
    "TW": ["Formosa Plastics"],
}


def pick_country():
    countries = list(COUNTRIES.keys())
    weights = [COUNTRIES[c][0] for c in countries]
    return rng.choice(countries, p=np.array(weights) / sum(weights))


def pick_incoterm(country):
    pool = COUNTRIES[country][1]
    return rng.choice(pool)


# -----------------------------------------------------------------------------
# Build base portfolio
# -----------------------------------------------------------------------------
rows = []
mat_idx = 1
for category, cfg in COMPOSITION.items():
    for _ in range(cfg["count"]):
        country = pick_country()
        supplier = rng.choice(SUPPLIERS_BY_COUNTRY[country])
        incoterm = pick_incoterm(country)

        unit_cost = round(rng.uniform(*cfg["cost_range"]), 2)
        margin_pct = round(rng.uniform(15, 65), 1)
        avg_daily_demand = round(rng.uniform(*cfg["demand_range"]), 1)
        lead_time_days = int(rng.uniform(14, 90))
        ss_days = rng.uniform(*cfg["ss_days"])
        safety_stock = int(avg_daily_demand * ss_days)

        # Healthy default: ~60-90 days of cover, normal-ish around target
        days_on_hand = max(5, rng.normal(75, 25))
        inventory_on_hand = int(avg_daily_demand * days_on_hand)

        # Shelf life: APIs longest, chemicals shortest
        shelf_life_map = {"API": (540, 1080), "INT": (360, 720),
                          "EXCIP": (720, 1440), "CHEM": (180, 540)}
        shelf_life_days = int(rng.uniform(*shelf_life_map[category]))

        # Sole source: ~18% portfolio-wide (realistic for pharma)
        sole_source = "Y" if rng.random() < 0.18 else "N"

        # Tariff exposure: countries with active trade-policy noise
        if country in ("CN", "IN", "US", "KR"):
            expected_price_change_pct = round(rng.uniform(2, 12), 1)
        else:
            expected_price_change_pct = round(rng.uniform(-2, 4), 1)

        # In-transit: depends on incoterm + origin
        if incoterm in ("FCA", "EXW", "FOB"):
            # Title at origin — in-transit is OURS
            transit_share = rng.uniform(0.10, 0.35)
        elif incoterm in ("CIF", "CFR", "CIP"):
            # Title at loading — also OURS in transit
            transit_share = rng.uniform(0.08, 0.25)
        else:
            # DAP / DDP / DPU — seller's risk in transit, zero from buyer's books
            transit_share = 0

        if transit_share > 0:
            in_transit_units = int(avg_daily_demand * lead_time_days * transit_share)
            in_transit_arrival_days = int(rng.uniform(5, lead_time_days))
        else:
            in_transit_units = 0
            in_transit_arrival_days = 0

        rows.append({
            "material":                  f"{category}{mat_idx:03d}",
            "description":               rng.choice(DESCRIPTIONS[category]),
            "supplier":                  supplier,
            "country_of_origin":         country,
            "unit_cost_eur":             unit_cost,
            "margin_pct":                margin_pct,
            "inventory_on_hand":         inventory_on_hand,
            "safety_stock":              safety_stock,
            "avg_daily_demand":          avg_daily_demand,
            "lead_time_days":            lead_time_days,
            "shelf_life_days":           shelf_life_days,
            "days_on_hand":              round(days_on_hand, 1),
            "expected_price_change_pct": expected_price_change_pct,
            "sole_source":               sole_source,
            "incoterm":                  incoterm,
            "in_transit_units":          in_transit_units,
            "in_transit_arrival_days":   in_transit_arrival_days,
        })
        mat_idx += 1

df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Inject realistic risk pockets (calibrated for 25-35 flagged SKUs total)
# -----------------------------------------------------------------------------

# Trapped Working Capital: 15-20 materials with severe overstock (>180 days cover)
trapped_indices = rng.choice(df.index, size=18, replace=False)
for idx in trapped_indices:
    excess_days = rng.uniform(180, 400)
    df.at[idx, "days_on_hand"] = round(excess_days, 1)
    df.at[idx, "inventory_on_hand"] = int(df.at[idx, "avg_daily_demand"] * excess_days)

# Shelf-life risk: 8-12 materials approaching or past expiry
remaining = df.index.difference(trapped_indices)
shelf_life_indices = rng.choice(remaining, size=10, replace=False)
for idx in shelf_life_indices:
    # Material is old: days_on_hand > shelf_life_days remaining
    df.at[idx, "shelf_life_days"] = int(rng.uniform(-5, 30))

# Single-source liability: tighten lead-time × sole-source on high-value materials
# (already partially seeded by 18% sole-source flag; calibration below)
remaining = remaining.difference(shelf_life_indices)
single_source_indices = rng.choice(remaining, size=8, replace=False)
for idx in single_source_indices:
    df.at[idx, "sole_source"] = "Y"
    df.at[idx, "lead_time_days"] = int(rng.uniform(60, 120))

# Light understock cases (low days_on_hand vs lead time) — for understock dimension
understock_indices = rng.choice(
    df.index.difference(trapped_indices),
    size=12,
    replace=False,
)
for idx in understock_indices:
    df.at[idx, "days_on_hand"] = round(rng.uniform(3, 20), 1)
    df.at[idx, "inventory_on_hand"] = int(
        df.at[idx, "avg_daily_demand"] * df.at[idx, "days_on_hand"]
    )

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
df.to_csv("demo_data_realistic.csv", index=False)

# -----------------------------------------------------------------------------
# Generate matching LT history file
# -----------------------------------------------------------------------------
# Per material: 6-12 historical lead-time observations, distributed around the
# material's nominal lead_time_days with a country-influenced CV.
# Asia-sourced materials get higher variability (CV 15-30%), EU materials lower
# (CV 5-15%), reflecting real-world freight volatility.

LT_HISTORY_PATH = "lt_history_realistic.xlsx"

lt_rows = []
for _, mat_row in df.iterrows():
    nominal_lt = mat_row["lead_time_days"]
    country    = mat_row["country_of_origin"]
    
    # CV by region: longer/riskier lanes have higher variance
    if country in ("CN", "IN", "KR", "JP", "TW", "MX"):
        cv_target = rng.uniform(0.15, 0.30)
    elif country in ("US",):
        cv_target = rng.uniform(0.10, 0.20)
    else:
        cv_target = rng.uniform(0.05, 0.15)
    
    n_observations = int(rng.integers(6, 13))   # 6-12 historical shipments
    std_dev = nominal_lt * cv_target
    
    for _ in range(n_observations):
        observed_lt = max(1, int(rng.normal(nominal_lt, std_dev)))
        lt_rows.append({
            "material":       mat_row["material"],
            "lead_time_days": observed_lt,
        })

lt_df = pd.DataFrame(lt_rows)
lt_df.to_excel(LT_HISTORY_PATH, index=False)

print(f"\nGenerated LT history -> {LT_HISTORY_PATH}")
print(f"  {len(lt_df)} observations across {lt_df['material'].nunique()} materials")
print(f"  Mean observations per material: {len(lt_df)/lt_df['material'].nunique():.1f}")

# Summary for sanity-check
inv_value = (df["inventory_on_hand"] * df["unit_cost_eur"]).sum()
in_transit_value = (df["in_transit_units"] * df["unit_cost_eur"]).sum()
sole_count = (df["sole_source"] == "Y").sum()

print(f"Generated {len(df)} materials -> demo_data_realistic.csv")
print(f"Portfolio inventory value:   € {inv_value:>15,.0f}")
print(f"In-transit value (buyer's):  € {in_transit_value:>15,.0f}")
print(f"Sole-sourced materials:        {sole_count}")
print(f"Incoterm distribution:")
print(df["incoterm"].value_counts().to_string())
print(f"Country distribution (top 5):")
print(df["country_of_origin"].value_counts().head().to_string())