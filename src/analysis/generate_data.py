"""
Generates synthetic Lagos traffic GPS probe data for 2 years.
Simulates realistic traffic patterns, seasonality, BRT effect, and rainfall impact.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

ROADS = [
    "Lagos-Ibadan Expressway", "Third Mainland Bridge", "Eko Bridge",
    "Apapa-Oshodi Expressway", "Lekki-Epe Expressway", "Ikorodu Road",
    "Lagos-Badagry Expressway", "Agege Motor Road", "Broad Street",
    "Victoria Island Ring Road",
]

BRT_ROUTES = {
    "Ikorodu Road": datetime(2023, 3, 1),
    "Lekki-Epe Expressway": datetime(2023, 9, 1),
}

ROAD_PROPERTIES = {
    road: {
        "road_width_m": np.random.uniform(10, 40),
        "lane_count": np.random.randint(2, 8),
        "traffic_lights_per_km": np.random.uniform(0.5, 3.0),
        "market_proximity_km": np.random.uniform(0.1, 5.0),
        "accident_rate": np.random.uniform(0.5, 8.0),
    }
    for road in ROADS
}

BASE_COMMUTE = {road: np.random.uniform(20, 60) for road in ROADS}


def hour_factor(hour: int) -> float:
    if 7 <= hour <= 9:
        return 1.8
    elif 12 <= hour <= 14:
        return 1.3
    elif 17 <= hour <= 20:
        return 2.0
    elif 0 <= hour <= 5:
        return 0.5
    return 1.0


def rainfall_factor(rainfall_mm: float) -> float:
    if rainfall_mm < 1:
        return 1.0
    elif rainfall_mm < 10:
        return 1.15
    elif rainfall_mm < 30:
        return 1.35
    else:
        return 1.55


def generate_traffic_data(days: int = 730) -> pd.DataFrame:
    records = []
    start = datetime(2023, 1, 1)

    for day_offset in range(days):
        current_date = start + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5
        month = current_date.month
        is_rainy_season = month in [4, 5, 6, 7, 8, 9, 10]
        rainfall_mm = (
            np.random.exponential(15) if is_rainy_season and np.random.random() < 0.35
            else np.random.exponential(2) if np.random.random() < 0.1
            else 0
        )

        for hour in [7, 8, 9, 12, 17, 18, 19]:
            for road in ROADS:
                base = BASE_COMMUTE[road]
                h_factor = hour_factor(hour) * (0.6 if is_weekend else 1.0)
                r_factor = rainfall_factor(rainfall_mm)

                brt_launch = BRT_ROUTES.get(road)
                brt_reduction = 1.0
                if brt_launch and current_date >= brt_launch:
                    months_after = (current_date - brt_launch).days / 30
                    brt_reduction = max(0.7, 1.0 - 0.25 * min(1.0, months_after / 6))

                commute_time = base * h_factor * r_factor * brt_reduction * np.random.lognormal(0, 0.1)
                props = ROAD_PROPERTIES[road]

                records.append({
                    "date": current_date.date().isoformat(),
                    "hour": hour,
                    "road_name": road,
                    "commute_time_min": round(commute_time, 1),
                    "congestion_score": round(min(10, commute_time / base * 5), 1),
                    "rainfall_mm": round(rainfall_mm, 1),
                    "is_weekend": int(is_weekend),
                    "is_rainy_season": int(is_rainy_season),
                    "brt_route": road if road in BRT_ROUTES else None,
                    "brt_launch_date": BRT_ROUTES[road].date().isoformat() if road in BRT_ROUTES else None,
                    **{k: round(v, 2) for k, v in props.items()},
                })

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/traffic_data.csv", index=False)
    df.to_parquet("data/traffic_data.parquet", index=False)
    print(f"Generated {len(df):,} traffic records → data/traffic_data.csv")
    return df


if __name__ == "__main__":
    df = generate_traffic_data(730)
    print(f"\nRoads: {df['road_name'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nAverage commute by road (minutes):")
    print(df.groupby("road_name")["commute_time_min"].mean().sort_values(ascending=False).round(1).to_string())
