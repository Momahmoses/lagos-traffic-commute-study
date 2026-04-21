"""
Statistical analysis of Lagos traffic patterns.
Hypothesis testing, correlation analysis, time series decomposition.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.weightstats import ttest_ind
import warnings
warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)


def test_rainfall_impact(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H0: Rainfall does NOT increase commute time by >30%.
    H1: Rainfall increases average commute time by >30%.
    """
    rainy = df[df["rainfall_mm"] > 5]["commute_time_min"].dropna()
    dry = df[df["rainfall_mm"] <= 5]["commute_time_min"].dropna()

    t_stat, p_value = stats.mannwhitneyu(rainy, dry, alternative="greater")
    rainy_mean = rainy.mean()
    dry_mean = dry.mean()
    pct_increase = (rainy_mean - dry_mean) / dry_mean * 100

    result = {
        "test": "Mann-Whitney U (one-sided)",
        "hypothesis": "Rainfall increases commute time by >30%",
        "rainy_mean_minutes": round(rainy_mean, 1),
        "dry_mean_minutes": round(dry_mean, 1),
        "pct_increase": round(pct_increase, 1),
        "u_statistic": round(t_stat, 2),
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "conclusion": (
            f"REJECT H0: Rainfall significantly increases commute time by {pct_increase:.1f}% (p={p_value:.4f})"
            if p_value < alpha else
            f"FAIL TO REJECT H0: No significant effect (p={p_value:.4f})"
        ),
    }
    logger.info(result["conclusion"])
    return result


def test_brt_impact(df: pd.DataFrame, route: str, alpha: float = 0.05) -> dict:
    """
    H0: BRT expansion did NOT reduce commute times.
    H1: BRT expansion reduced commute times.
    """
    route_df = df[df["route_name"] == route].copy()
    route_df["date"] = pd.to_datetime(route_df["date"])
    brt_date = route_df["brt_launch_date"].iloc[0] if "brt_launch_date" in route_df.columns else route_df["date"].median()
    brt_date = pd.to_datetime(brt_date)

    before = route_df[route_df["date"] < brt_date]["commute_time_min"].dropna()
    after = route_df[route_df["date"] >= brt_date]["commute_time_min"].dropna()

    if len(before) < 10 or len(after) < 10:
        return {"route": route, "error": "Insufficient data for test"}

    t_stat, p_value = stats.mannwhitneyu(after, before, alternative="less")
    reduction = before.mean() - after.mean()
    pct_reduction = reduction / before.mean() * 100

    return {
        "route": route,
        "before_mean": round(before.mean(), 1),
        "after_mean": round(after.mean(), 1),
        "minutes_saved": round(reduction, 1),
        "pct_reduction": round(pct_reduction, 1),
        "p_value": round(p_value, 4),
        "significant": p_value < alpha,
        "conclusion": (
            f"BRT on {route} reduced commutes by {pct_reduction:.1f}% ({reduction:.0f} min) — p={p_value:.4f}"
            if p_value < alpha else
            f"BRT on {route}: No significant reduction (p={p_value:.4f})"
        ),
    }


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    road_features = [
        "road_width_m", "lane_count", "traffic_lights_per_km",
        "market_proximity_km", "commute_time_min", "accident_rate",
        "congestion_score", "rainfall_mm",
    ]
    available = [c for c in road_features if c in df.columns]
    corr = df[available].corr(method="spearman")
    return corr


def decompose_traffic_series(df: pd.DataFrame, road: str = None) -> dict:
    if road:
        series_df = df[df["road_name"] == road].copy()
    else:
        series_df = df.copy()

    series_df["date"] = pd.to_datetime(series_df["date"])
    daily = series_df.groupby("date")["commute_time_min"].mean().sort_index()

    if len(daily) < 14:
        return {"error": "Insufficient data for decomposition (need ≥14 days)"}

    result = seasonal_decompose(daily, model="additive", period=7, extrapolate_trend="freq")

    return {
        "road": road or "All roads",
        "trend_slope_min_per_month": round(
            (result.trend.iloc[-1] - result.trend.iloc[0]) / (len(result.trend) / 30), 2
        ),
        "seasonality_amplitude": round(result.seasonal.max() - result.seasonal.min(), 2),
        "residual_std": round(result.resid.std(), 2),
        "peak_day": result.seasonal.idxmax().strftime("%A") if hasattr(result.seasonal.idxmax(), "strftime") else str(result.seasonal.idxmax()),
    }


def cohort_analysis_brt(df: pd.DataFrame) -> pd.DataFrame:
    if "brt_route" not in df.columns:
        return pd.DataFrame({"message": ["Column 'brt_route' not found"]})

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    results = []
    for route in df["brt_route"].dropna().unique():
        route_df = df[df["brt_route"] == route]
        monthly = route_df.groupby("month")["commute_time_min"].mean()
        results.append({
            "brt_route": route,
            "months_tracked": len(monthly),
            "initial_avg_min": round(monthly.iloc[0], 1) if len(monthly) > 0 else None,
            "latest_avg_min": round(monthly.iloc[-1], 1) if len(monthly) > 0 else None,
            "improvement_min": round(monthly.iloc[0] - monthly.iloc[-1], 1) if len(monthly) > 1 else 0,
        })
    return pd.DataFrame(results)
