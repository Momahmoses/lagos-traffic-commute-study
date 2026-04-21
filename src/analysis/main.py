"""
Main analysis runner: runs all statistical tests, PCA, and generates the evidence report.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings("ignore")

from src.analysis.stats import (
    test_rainfall_impact, test_brt_impact,
    correlation_analysis, decompose_traffic_series, cohort_analysis_brt
)

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/charts", exist_ok=True)


def load_data():
    try:
        return pd.read_csv("data/traffic_data.csv")
    except FileNotFoundError:
        print("Data not found. Run generate_data.py first.")
        from src.analysis.generate_data import generate_traffic_data
        return generate_traffic_data(365)


def run_pca(df: pd.DataFrame) -> dict:
    road_features = [
        "road_width_m", "lane_count", "traffic_lights_per_km",
        "market_proximity_km", "accident_rate",
    ]
    available = [c for c in road_features if c in df.columns]
    road_df = df.groupby("road_name")[available].mean()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(road_df)
    pca = PCA(n_components=min(3, len(available)))
    components = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    return {
        "n_components": len(explained),
        "explained_variance": [round(e * 100, 1) for e in explained],
        "total_explained": round(sum(explained) * 100, 1),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=available,
            columns=[f"PC{i+1}" for i in range(len(explained))]
        ).round(3),
    }


def plot_commute_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(values="commute_time_min", index="road_name", columns="hour", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}:00" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Average Commute Time (minutes) by Road and Hour", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Minutes")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/charts/commute_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_rainfall_effect(df: pd.DataFrame):
    bins = [0, 1, 10, 30, 200]
    labels = ["No rain", "Light", "Moderate", "Heavy"]
    df["rain_category"] = pd.cut(df["rainfall_mm"], bins=bins, labels=labels)
    means = df.groupby("rain_category", observed=True)["commute_time_min"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(means.index, means.values, color=["#22c55e", "#84cc16", "#f59e0b", "#ef4444"], alpha=0.85)
    ax.set_ylabel("Average Commute Time (minutes)")
    ax.set_title("Rainfall Impact on Lagos Commute Times", fontsize=13, fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:.0f} min" for v in means.values], padding=4)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/charts/rainfall_effect.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_full_report(df: pd.DataFrame):
    print("=" * 65)
    print("LAGOS TRAFFIC & COMMUTE PATTERN STUDY — EVIDENCE REPORT")
    print("=" * 65)

    print(f"\nDataset: {len(df):,} records | {df['road_name'].nunique()} roads | {df['date'].nunique()} days")

    print("\n" + "─" * 40)
    print("1. RAINFALL IMPACT HYPOTHESIS TEST")
    print("─" * 40)
    rainfall_result = test_rainfall_impact(df)
    for k, v in rainfall_result.items():
        print(f"   {k}: {v}")

    print("\n" + "─" * 40)
    print("2. BRT EXPANSION IMPACT TESTS")
    print("─" * 40)
    for road in ["Ikorodu Road", "Lekki-Epe Expressway"]:
        result = test_brt_impact(df, road)
        print(f"   {result.get('conclusion', result.get('error', 'N/A'))}")

    print("\n" + "─" * 40)
    print("3. PCA — ROAD CHARACTERISTIC COMPONENTS")
    print("─" * 40)
    pca_result = run_pca(df)
    print(f"   Components explain: {pca_result['explained_variance']}% variance")
    print(f"   Total explained: {pca_result['total_explained']}%")
    print(f"   PC1 loadings:\n{pca_result['loadings']['PC1'].to_string()}")

    print("\n" + "─" * 40)
    print("4. TRAFFIC DECOMPOSITION (All Roads)")
    print("─" * 40)
    decomp = decompose_traffic_series(df)
    for k, v in decomp.items():
        print(f"   {k}: {v}")

    print("\n" + "─" * 40)
    print("5. CORRELATION MATRIX (Spearman)")
    print("─" * 40)
    corr = correlation_analysis(df)
    if "commute_time_min" in corr.columns:
        commute_corr = corr["commute_time_min"].drop("commute_time_min").sort_values(key=abs, ascending=False)
        print("   Correlation with commute_time_min:")
        print(commute_corr.to_string())

    print("\n" + "─" * 40)
    print("6. GENERATING CHARTS")
    print("─" * 40)
    plot_commute_heatmap(df)
    plot_rainfall_effect(df)

    print("\n" + "=" * 65)
    print("REPORT COMPLETE — Charts saved to reports/charts/")
    print("=" * 65)


if __name__ == "__main__":
    df = load_data()
    generate_full_report(df)
