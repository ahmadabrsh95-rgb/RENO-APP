import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Renovation Scenario Recommender",
    page_icon="🏗️",
    layout="wide"
)

# =========================================================
# SETTINGS
# =========================================================
FILE_PATH = "trade_off_summary.xlsx"
SHEET_NAME = "TradeOff_Summary"

SCENARIOS = ["S01", "S02", "S03"]
BASE_CASE = "BC"
OVERHEATING_OK_THRESHOLD = 10.0  # %

BUILDINGS = {
    "High-rise building (HRB)": "HRB",
    "Low-rise building (LRB)": "LRB",
}

# ---------------------------------------------------------
# Weighted decision indicators
# ---------------------------------------------------------
@dataclass
class IndicatorConfig:
    key: str
    label: str
    unit: str
    direction: str
    alternatives: List[str]
    description: str


WEIGHTED_INDICATORS: Dict[str, IndicatorConfig] = {
    "energy": IndicatorConfig(
        key="energy",
        label="Total energy",
        unit="kWh/m²·yr",
        direction="lower_better",
        alternatives=["Total energy"],
        description="Lower total delivered energy is preferred."
    ),
    "gwp": IndicatorConfig(
        key="gwp",
        label="Total GWP",
        unit="kgCO₂e/m²",
        direction="lower_better",
        alternatives=["Total GWP"],
        description="Lower total climate impact is preferred."
    ),
    "overheating": IndicatorConfig(
        key="overheating",
        label="Mean Overheating [% of Apr-Sep hours]",
        unit="% of Apr-Sep hours",
        direction="lower_better",
        alternatives=[
            "Mean Overheating [% of Apr-Sep hours]",
            "Mean Overheating [% of Apr–Sep hours]",
            "Mean overheating [% of Apr-Sep hours]",
            "Mean overheating [% of Apr–Sep hours]",
        ],
        description="Lower overheating frequency is preferred. Values below 10% are generally acceptable."
    ),
    "circularity": IndicatorConfig(
        key="circularity",
        label="Circularity score",
        unit="score",
        direction="higher_better",
        alternatives=["Circularity score"],
        description="Higher circularity score is preferred."
    ),
}

# ---------------------------------------------------------
# Additional indicators shown in results
# ---------------------------------------------------------
RESULT_INDICATORS = {
    "heating_demand": {
        "label": "Heating demand",
        "unit": "kWh/m²·yr",
        "alternatives": ["Heating demand"],
    },
    "heating_reduction": {
        "label": "Heating reduction",
        "unit": "%",
        "alternatives": ["Heating reduction"],
    },
    "carbon_payback": {
        "label": "Carbon payback time",
        "unit": "years",
        "alternatives": ["Carbon payback time"],
    },
    "summer_temp": {
        "label": "Mean summer indoor temperature",
        "unit": "°C",
        "alternatives": ["Mean summer indoor temperature (Apr–Sep)"],
    },
    "winter_temp": {
        "label": "Mean winter indoor temperature",
        "unit": "°C",
        "alternatives": ["Mean winter indoor temperature (Oct–Mar)"],
    },
    "embodied_gwp": {
        "label": "Embodied GWP",
        "unit": "kgCO₂e/m²",
        "alternatives": ["Embodied GWP"],
    },
    "operational_gwp": {
        "label": "Operational GWP",
        "unit": "kgCO₂e/m²",
        "alternatives": ["Operational GWP"],
    },
    "overheating_hours": {
        "label": "Mean Overheating [h >26°C]",
        "unit": "h",
        "alternatives": ["Mean Overheating [h >26°C]"],
    },
}

DEFAULT_PRESETS = {
    "Balanced": {
        "energy": 30,
        "gwp": 30,
        "overheating": 20,
        "circularity": 20,
    },
    "Energy-focused": {
        "energy": 50,
        "gwp": 20,
        "overheating": 20,
        "circularity": 10,
    },
    "Climate-focused": {
        "energy": 20,
        "gwp": 50,
        "overheating": 15,
        "circularity": 15,
    },
    "Comfort-focused": {
        "energy": 20,
        "gwp": 15,
        "overheating": 50,
        "circularity": 15,
    },
    "Circularity-focused": {
        "energy": 15,
        "gwp": 20,
        "overheating": 15,
        "circularity": 50,
    },
}

SCENARIO_PACKAGES = {
    "S01": {
        "title": "Scenario 01: Low-intervention renovation",
        "summary": "This scenario includes limited energy-saving measures with low material input and minimal disturbance.",
        "items": [
            "Improved airtightness (e.g., sealing joints and identified leakage paths)",
            "Minor window refurbishment (e.g., new seals and adjustments)",
            "Additional attic insulation: 400 mm mineral wool",
        ],
    },
    "S02": {
        "title": "Scenario 02: Medium renovation (Vidingehem strategy)",
        "summary": "This scenario represents a balanced set of measures targeting both energy efficiency and indoor comfort.",
        "items": [
            "Window replacement with improved thermal performance (U-value = 0.9 W/m²K)",
            "Additional insulation at façade sections below windows (internal insulation): 70 mm mineral wool",
            "Additional attic insulation: 400 mm mineral wool",
            "Upgrade from exhaust ventilation (FX) to balanced ventilation with heat recovery (FTX), heat recovery efficiency: 82%, SFP: 1.5 kW/(m³/s)",
            "Improved airtightness",
        ],
    },
    "S03": {
        "title": "Scenario 03: Deep renovation",
        "summary": "This scenario represents a high-performance renovation pathway with higher material input.",
        "items": [
            "High-performance windows (U-value = 0.9 W/m²K)",
            "Extensive façade insulation (external): 200 mm insulation",
            "Extensive roof insulation: 400 mm insulation",
            "Advanced airtightness improvements",
            "Optimised balanced ventilation with heat recovery (FTX): heat recovery efficiency 85%, SFP: 1.2 kW/(m³/s)",
        ],
    },
}

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.metric-card {
    background-color: #f6f8fb;
    padding: 1rem 1.1rem;
    border-radius: 0.9rem;
    border: 1px solid #e6ebf2;
    min-height: 120px;
}
.scenario-card {
    background-color: #f9fbfd;
    padding: 1rem 1.1rem;
    border-radius: 1rem;
    border: 1px solid #e6ebf2;
    margin-bottom: 1rem;
}
.small-note {
    color: #5f6b7a;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def normalize_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


@st.cache_data
def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    if "Indicator" not in df.columns:
        raise ValueError("The Excel sheet must contain a column named 'Indicator'.")

    df["Indicator"] = df["Indicator"].astype(str).str.strip()
    df["Indicator_clean"] = df["Indicator"].apply(normalize_text)
    return df


def find_indicator_row(df: pd.DataFrame, possible_names: List[str]) -> pd.Series:
    candidates = [normalize_text(x) for x in possible_names]

    for cand in candidates:
        exact = df[df["Indicator_clean"] == cand]
        if not exact.empty:
            return exact.iloc[0]

    for cand in candidates:
        contains = df[df["Indicator_clean"].str.contains(re.escape(cand), na=False)]
        if not contains.empty:
            return contains.iloc[0]

    if any("overheating" in cand for cand in candidates):
        mask = (
            df["Indicator_clean"].str.contains("overheating", na=False)
            & (
                df["Indicator_clean"].str.contains("apr", na=False)
                | df["Indicator_clean"].str.contains("sep", na=False)
            )
        )
        row = df.loc[mask]
        if not row.empty:
            return row.iloc[0]

    raise ValueError(f"Could not find indicator row for any of: {possible_names}")


def min_max_score(values: np.ndarray, direction: str) -> np.ndarray:
    values = np.array(values, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values, dtype=float)

    if direction == "lower_better":
        return (vmax - values) / (vmax - vmin)
    elif direction == "higher_better":
        return (values - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    weight_sum = sum(raw_weights.values())
    if weight_sum == 0:
        raise ValueError("At least one weighting factor must be above zero.")
    return {k: v / weight_sum for k, v in raw_weights.items()}


def extract_indicator_values(
    df: pd.DataFrame,
    building_code: str,
    alternatives: List[str]
) -> Tuple[float, np.ndarray]:
    row = find_indicator_row(df, alternatives)

    bc_col = f"{building_code}-{BASE_CASE}"
    bc_value = float(row[bc_col])

    scenario_values = []
    for scenario in SCENARIOS:
        col = f"{building_code}-{scenario}"
        scenario_values.append(float(row[col]))

    return bc_value, np.array(scenario_values, dtype=float)


def compute_improvement_vs_bc(
    bc_value: float,
    scenario_values: np.ndarray,
    direction: str
) -> np.ndarray:
    if bc_value == 0:
        return np.zeros_like(scenario_values)

    if direction == "lower_better":
        return ((bc_value - scenario_values) / bc_value) * 100
    elif direction == "higher_better":
        return ((scenario_values - bc_value) / bc_value) * 100
    else:
        raise ValueError(f"Unknown direction: {direction}")


def build_result_table(df: pd.DataFrame, building_code: str, weights: Dict[str, float]) -> pd.DataFrame:
    weighted_raw = {}
    weighted_scores = {}
    weighted_improvements = {}
    bc_reference = {}

    for key, config in WEIGHTED_INDICATORS.items():
        bc, vals = extract_indicator_values(df, building_code, config.alternatives)
        bc_reference[key] = bc
        weighted_raw[key] = vals
        weighted_scores[key] = min_max_score(vals, config.direction)
        weighted_improvements[key] = compute_improvement_vs_bc(bc, vals, config.direction)

    total_scores = np.zeros(len(SCENARIOS), dtype=float)
    for key in WEIGHTED_INDICATORS:
        total_scores += weights[key] * weighted_scores[key]

    result_df = pd.DataFrame({
        "Scenario": SCENARIOS,

        "Total energy": weighted_raw["energy"],
        "Total GWP": weighted_raw["gwp"],
        "Overheating [%]": weighted_raw["overheating"],
        "Circularity score": weighted_raw["circularity"],

        "Energy improvement vs BC [%]": weighted_improvements["energy"],
        "GWP improvement vs BC [%]": weighted_improvements["gwp"],
        "Overheating improvement vs BC [%]": weighted_improvements["overheating"],
        "Circularity improvement vs BC [%]": weighted_improvements["circularity"],

        "Energy score": weighted_scores["energy"],
        "GWP score": weighted_scores["gwp"],
        "Overheating score": weighted_scores["overheating"],
        "Circularity score norm": weighted_scores["circularity"],

        "Weighted total score": total_scores,
    })

    # Add additional result indicators
    for key, meta in RESULT_INDICATORS.items():
        _, vals = extract_indicator_values(df, building_code, meta["alternatives"])
        result_df[meta["label"]] = vals

    result_df["Overheating status"] = np.where(
        result_df["Overheating [%]"] < OVERHEATING_OK_THRESHOLD,
        "Acceptable",
        "Above threshold"
    )

    result_df = result_df.sort_values("Weighted total score", ascending=False).reset_index(drop=True)
    return result_df


def build_base_case_table(df: pd.DataFrame, building_code: str) -> pd.DataFrame:
    rows = []

    for config in WEIGHTED_INDICATORS.values():
        bc, _ = extract_indicator_values(df, building_code, config.alternatives)
        rows.append({
            "Indicator": config.label,
            "Base case value": bc,
            "Unit": config.unit
        })

    for meta in RESULT_INDICATORS.values():
        bc, _ = extract_indicator_values(df, building_code, meta["alternatives"])
        rows.append({
            "Indicator": meta["label"],
            "Base case value": bc,
            "Unit": meta["unit"]
        })

    return pd.DataFrame(rows)


def scenario_explanation(best_row: pd.Series) -> str:
    strengths = []

    if best_row["Energy score"] >= 0.7:
        strengths.append("strong energy performance")
    if best_row["GWP score"] >= 0.7:
        strengths.append("low total climate impact")
    if best_row["Overheating score"] >= 0.7:
        strengths.append("good overheating performance")
    if best_row["Circularity score norm"] >= 0.7:
        strengths.append("strong circularity performance")

    if best_row["Overheating [%]"] < OVERHEATING_OK_THRESHOLD:
        strengths.append("acceptable overheating level")

    if not strengths:
        return (
            "This scenario ranks highest under the selected weighting, "
            "but the trade-offs remain important and should still be reviewed."
        )

    if len(strengths) == 1:
        return f"This scenario is recommended mainly because it shows {strengths[0]}."
    if len(strengths) == 2:
        return f"This scenario is recommended because it combines {strengths[0]} and {strengths[1]}."
    return "This scenario is recommended because it combines " + ", ".join(strengths[:-1]) + f", and {strengths[-1]}."


def make_total_score_plot(result_df: pd.DataFrame, building_label: str):
    plot_df = result_df.sort_values("Scenario").copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(plot_df["Scenario"], plot_df["Weighted total score"])

    ax.set_ylabel("Weighted total score")
    ax.set_title(f"Scenario ranking - {building_label}")
    ax.set_ylim(0, max(1.0, plot_df["Weighted total score"].max() * 1.18))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    for bar, val in zip(bars, plot_df["Weighted total score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()
    return fig


def make_indicator_score_plot(result_df: pd.DataFrame, building_label: str):
    plot_df = result_df.sort_values("Scenario").copy()
    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x - 1.5 * width, plot_df["Energy score"], width, label="Energy")
    ax.bar(x - 0.5 * width, plot_df["GWP score"], width, label="GWP")
    ax.bar(x + 0.5 * width, plot_df["Overheating score"], width, label="Overheating")
    ax.bar(x + 1.5 * width, plot_df["Circularity score norm"], width, label="Circularity")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Scenario"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Normalized score")
    ax.set_title(f"Indicator scores by scenario - {building_label}")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig


def make_full_results_plot(result_df: pd.DataFrame, building_label: str):
    plot_df = result_df.sort_values("Scenario").copy()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(plot_df))
    width = 0.25

    ax.bar(x - width, plot_df["Heating reduction"], width, label="Heating reduction [%]")
    ax.bar(x, plot_df["Circularity score"], width, label="Circularity score")
    ax.bar(x + width, plot_df["Overheating [%]"], width, label="Overheating [%]")

    ax.axhline(OVERHEATING_OK_THRESHOLD, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Scenario"])
    ax.set_title(f"Selected outcome indicators - {building_label}")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig


def make_temperature_plot(result_df: pd.DataFrame, building_label: str):
    plot_df = result_df.sort_values("Scenario").copy()

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(len(plot_df))
    width = 0.32

    ax.bar(x - width/2, plot_df["Mean summer indoor temperature"], width, label="Summer mean temperature")
    ax.bar(x + width/2, plot_df["Mean winter indoor temperature"], width, label="Winter mean temperature")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Scenario"])
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Mean indoor temperature by scenario - {building_label}")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig


# =========================================================
# LOAD DATA
# =========================================================
try:
    df = load_data(FILE_PATH, SHEET_NAME)
except Exception as e:
    st.error(f"Could not load Excel file: {e}")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.title("🏗️ Renovation Scenario Recommender")
st.write(
    "An interactive decision-support app for comparing renovation scenarios based on "
    "energy use, total GWP, overheating, circularity, and broader expected performance outcomes."
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Inputs")

building_label = st.sidebar.selectbox("Building type", list(BUILDINGS.keys()))
building_code = BUILDINGS[building_label]

preset_name = st.sidebar.selectbox("Weighting preset", list(DEFAULT_PRESETS.keys()))
use_custom_weights = st.sidebar.checkbox("Adjust weights manually", value=False)

preset_weights = DEFAULT_PRESETS[preset_name].copy()

st.sidebar.markdown("### Weighting criteria")

if use_custom_weights:
    w_energy = st.sidebar.slider("Total energy", 0, 100, int(preset_weights["energy"]), 1)
    w_gwp = st.sidebar.slider("Total GWP", 0, 100, int(preset_weights["gwp"]), 1)
    w_over = st.sidebar.slider("Overheating", 0, 100, int(preset_weights["overheating"]), 1)
    w_circ = st.sidebar.slider("Circularity", 0, 100, int(preset_weights["circularity"]), 1)
else:
    w_energy = preset_weights["energy"]
    w_gwp = preset_weights["gwp"]
    w_over = preset_weights["overheating"]
    w_circ = preset_weights["circularity"]

    st.sidebar.write(f"Total energy: **{w_energy}**")
    st.sidebar.write(f"Total GWP: **{w_gwp}**")
    st.sidebar.write(f"Overheating: **{w_over}**")
    st.sidebar.write(f"Circularity: **{w_circ}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### Decision logic")
st.sidebar.caption(
    "Users assign weights only to the core decision criteria. "
    "The results page then shows the broader expected performance of each scenario."
)

raw_weights = {
    "energy": float(w_energy),
    "gwp": float(w_gwp),
    "overheating": float(w_over),
    "circularity": float(w_circ),
}

try:
    weights = normalize_weights(raw_weights)
except ValueError as e:
    st.warning(str(e))
    st.stop()

# =========================================================
# CALCULATIONS
# =========================================================
result_df = build_result_table(df, building_code, weights)
base_case_df = build_base_case_table(df, building_code)

best_row = result_df.iloc[0]
recommended = best_row["Scenario"]
runner_up_score = result_df.iloc[1]["Weighted total score"]
score_gap = best_row["Weighted total score"] - runner_up_score

# =========================================================
# TOP SUMMARY
# =========================================================
top1, top2, top3 = st.columns([1.3, 1.0, 1.0])

with top1:
    st.subheader("Recommendation")
    st.success(f"Recommended renovation scenario: **{recommended}**")
    st.write(scenario_explanation(best_row))

    if score_gap < 0.05:
        st.warning(
            "The top scenarios are close to each other, so the recommendation is somewhat sensitive to the chosen weighting."
        )

    if best_row["Overheating [%]"] >= OVERHEATING_OK_THRESHOLD:
        st.error(
            f"{recommended} exceeds the overheating reference threshold of {OVERHEATING_OK_THRESHOLD:.0f}%."
        )
    else:
        st.info(
            f"{recommended} remains below the overheating reference threshold of {OVERHEATING_OK_THRESHOLD:.0f}%."
        )

with top2:
    st.subheader("Selected priorities")
    st.write(f"**Preset:** {preset_name}")
    st.write(f"- Total energy: {weights['energy']:.3f}")
    st.write(f"- Total GWP: {weights['gwp']:.3f}")
    st.write(f"- Overheating: {weights['overheating']:.3f}")
    st.write(f"- Circularity: {weights['circularity']:.3f}")

with top3:
    st.subheader("Ranking strength")
    st.metric(
        "Weighted total score",
        f"{best_row['Weighted total score']:.3f}",
        delta=f"{score_gap:.3f} vs next best"
    )

# =========================================================
# QUICK PERFORMANCE CARDS
# =========================================================
st.subheader("Expected performance of recommended scenario")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Energy reduction",
        f"{best_row['Energy improvement vs BC [%]']:.1f}%",
        f"{best_row['Total energy']:.1f} kWh/m²·yr"
    )

with c2:
    st.metric(
        "Carbon payback time",
        f"{best_row['Carbon payback time']:.1f} years"
    )

with c3:
    st.metric(
        "Mean summer temperature",
        f"{best_row['Mean summer indoor temperature']:.2f} °C",
        f"{best_row['Overheating [%]']:.1f}% overheating"
    )

with c4:
    st.metric(
        "Mean winter temperature",
        f"{best_row['Mean winter indoor temperature']:.2f} °C"
    )

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ranking",
    "Full expected results",
    "Temperatures and overheating",
    "Scenario packages",
    "Base case and method"
])

with tab1:
    st.subheader("Scenario ranking")
    ranking_cols = [
        "Scenario",
        "Weighted total score",
        "Total energy",
        "Total GWP",
        "Overheating [%]",
        "Circularity score",
        "Overheating status",
    ]
    st.dataframe(
        result_df[ranking_cols].style.format({
            "Weighted total score": "{:.3f}",
            "Total energy": "{:.2f}",
            "Total GWP": "{:.2f}",
            "Overheating [%]": "{:.2f}",
            "Circularity score": "{:.2f}",
        }),
        use_container_width=True
    )

    fig_total = make_total_score_plot(result_df, building_label)
    st.pyplot(fig_total)

    fig_indicator = make_indicator_score_plot(result_df, building_label)
    st.pyplot(fig_indicator)

with tab2:
    st.subheader("Broader expected performance by scenario")
    st.write(
        "These indicators are shown for interpretation of the full renovation outcome, "
        "not only for the weighted ranking."
    )

    full_cols = [
        "Scenario",
        "Heating demand",
        "Heating reduction",
        "Total energy",
        "Embodied GWP",
        "Operational GWP",
        "Total GWP",
        "Carbon payback time",
        "Circularity score",
        "Mean summer indoor temperature",
        "Mean winter indoor temperature",
        "Mean Overheating [h >26°C]",
        "Overheating [%]",
        "Overheating status",
    ]
    st.dataframe(
        result_df[full_cols].sort_values("Scenario").style.format({
            "Heating demand": "{:.2f}",
            "Heating reduction": "{:.1f}",
            "Total energy": "{:.2f}",
            "Embodied GWP": "{:.2f}",
            "Operational GWP": "{:.2f}",
            "Total GWP": "{:.2f}",
            "Carbon payback time": "{:.2f}",
            "Circularity score": "{:.2f}",
            "Mean summer indoor temperature": "{:.2f}",
            "Mean winter indoor temperature": "{:.2f}",
            "Mean Overheating [h >26°C]": "{:.1f}",
            "Overheating [%]": "{:.2f}",
        }),
        use_container_width=True
    )

    fig_full = make_full_results_plot(result_df, building_label)
    st.pyplot(fig_full)

with tab3:
    st.subheader("Indoor temperatures and overheating")
    st.write(
        f"The dashed overheating reference in the app is {OVERHEATING_OK_THRESHOLD:.0f}% of Apr-Sep hours."
    )

    fig_temp = make_temperature_plot(result_df, building_label)
    st.pyplot(fig_temp)

    temp_cols = [
        "Scenario",
        "Mean summer indoor temperature",
        "Mean winter indoor temperature",
        "Mean Overheating [h >26°C]",
        "Overheating [%]",
        "Overheating status",
    ]
    st.dataframe(
        result_df[temp_cols].sort_values("Scenario").style.format({
            "Mean summer indoor temperature": "{:.2f}",
            "Mean winter indoor temperature": "{:.2f}",
            "Mean Overheating [h >26°C]": "{:.1f}",
            "Overheating [%]": "{:.2f}",
        }),
        use_container_width=True
    )

with tab4:
    st.subheader("What is included in each renovation package?")
    st.write(
        "These descriptions explain the technical content of each scenario so the user can interpret the ranking together with the actual renovation measures."
    )

    for scenario in SCENARIOS:
        pkg = SCENARIO_PACKAGES[scenario]
        with st.expander(f"{scenario} - {pkg['title']}", expanded=(scenario == recommended)):
            st.markdown(f"**{pkg['title']}**")
            st.write(pkg["summary"])
            for item in pkg["items"]:
                st.markdown(f"- {item}")

            matching_row = result_df[result_df["Scenario"] == scenario].iloc[0]
            st.markdown("**Expected key results**")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Weighted score", f"{matching_row['Weighted total score']:.3f}")
            with k2:
                st.metric("Energy reduction", f"{matching_row['Energy improvement vs BC [%]']:.1f}%")
            with k3:
                st.metric("Carbon payback", f"{matching_row['Carbon payback time']:.1f} years")
            with k4:
                st.metric("Summer mean temp", f"{matching_row['Mean summer indoor temperature']:.2f} °C")

with tab5:
    st.subheader("Base case values")
    st.dataframe(
        base_case_df.style.format({
            "Base case value": "{:.2f}"
        }),
        use_container_width=True
    )

    st.subheader("Method")
    st.info(
        "The recommendation is based on a weighted sum of normalized scores for four decision criteria: "
        "total energy, total GWP, overheating, and circularity. "
        "Other indicators such as carbon payback time, seasonal indoor temperatures, heating reduction, "
        "and embodied versus operational GWP are shown to support interpretation of the broader renovation outcome."
    )

    with st.expander("Interpretation notes"):
        st.markdown(
            f"""
- For **Total energy**, **Total GWP**, and **Overheating**, lower values are better.
- For **Circularity score**, higher values are better.
- **Overheating below {OVERHEATING_OK_THRESHOLD:.0f}%** is treated here as an acceptable reference level.
- The final recommendation depends on the chosen weighting and should be read together with the broader scenario consequences shown in the app.
            """
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "This tool recommends the most suitable scenario under the selected priorities, while also showing the broader expected performance and the renovation content of each package."
)