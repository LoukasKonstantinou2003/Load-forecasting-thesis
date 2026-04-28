import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# File paths (adjust if needed)
Residential_file_path = "Residential_Energy_Dataset_UK- 2014-2020.csv"
Commercial_file_path = "PC_LARGE_COMMERCIAL (SHOPS&STORES).csv"
Industrial_file_path = "PC_LARGE_INDUSTRIAL.csv"
Meteo_data_file_path = "meteo_data2022-2023.csv"

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_residential(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], dayfirst=True, errors="coerce")
    mask = df["utc_timestamp"].dt.minute.isin([0, 30])
    df = df.loc[mask].copy()
    df = df.sort_values("utc_timestamp")
    df = df.rename(columns={
        "utc_timestamp": "timestamp",
        "Electricity_load": "load_residential",
        "Temperature": "temp_residential",
        "Relative Humidity": "rh_residential"
    })
    return df

def parse_industrial_or_commercial(fp: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(fp, dtype=str)
    if df.shape[1] == 1:
        single_col = df.columns[0]
        df = df[single_col].str.split(",", expand=True)
        df.columns = ["Date_Time", "Reading", "Value", "Condition"]
    else:
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]
        if set(["Date_Time", "Reading", "Value", "Condition"]).intersection(df.columns) != set(["Date_Time", "Reading", "Value", "Condition"]):
            if any("Date Time" in c for c in df.columns):
                df = df[df.columns[0]].str.split(",", expand=True)
                df.columns = ["Date_Time", "Reading", "Value", "Condition"]

    if "Value" not in df.columns:
        raise ValueError(f"Value column not found in {fp}. Columns: {df.columns.tolist()}")

    df = df.drop(columns=[c for c in ["Reading", "Condition"] if c in df.columns])

    def _parse_ts(x: str):
        x = str(x)
        try:
            parts = x.split("-")
            if len(parts) < 4:
                return pd.NaT
            date_part = "-".join(parts[:3])
            time_part = parts[3]
            t_frag = time_part.split(".")
            if len(t_frag) == 3:
                hh, mm, ss = t_frag
            elif len(t_frag) == 2:
                hh, mm = t_frag
                ss = "00"
            else:
                return pd.NaT
            return pd.to_datetime(f"{date_part} {hh}:{mm}:{ss}", errors="coerce")
        except Exception:
            return pd.NaT

    df["timestamp"] = df["Date_Time"].apply(_parse_ts)
    df = df.drop(columns=["Date_Time"])
    df = df.dropna(subset=["timestamp"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.rename(columns={"Value": f"load_{kind}"})
    df = df[["timestamp", f"load_{kind}"]].sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"].dt.minute.isin([0, 30])]
    return df

def parse_meteo(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["Time"] = pd.to_datetime(df["Time"], dayfirst=True, errors="coerce")
    df = df.rename(columns={
        "Time": "timestamp",
        "G_pyrano1_15m": "solar_rad",
        "T_ambient_15m": "ambient_temp",
        "R_humidity_15m": "ambient_rh"
    }).sort_values("timestamp")
    freq = df["timestamp"].diff().median()
    if pd.notna(freq) and freq < pd.Timedelta("30min"):
        df = (df.set_index("timestamp")
                .resample("30min")
                .mean()
                .reset_index())
    return df

def unify_frequency(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    return df.reset_index()

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    row_missing_counts = df[numeric_cols].isna().sum(axis=1)
    multi_missing_idx = df.index[row_missing_counts > 1]
    single_missing_idx = df.index[row_missing_counts == 1]

    if single_missing_idx.any():
        df[numeric_cols] = df[numeric_cols].apply(lambda col: col.interpolate(limit_direction="both"))
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    max_drop = int(0.05 * len(df))
    if len(multi_missing_idx) and len(multi_missing_idx) <= max_drop:
        df = df.drop(index=multi_missing_idx)
    elif len(multi_missing_idx) > max_drop:
        df[numeric_cols] = df[numeric_cols].apply(lambda col: col.interpolate(limit_direction="both"))
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)

try:
    import holidays
    _UK_HOLIDAYS_CACHE = {}
    _CY_HOLIDAYS_CACHE = {}
    def is_holiday(ts, region):
        year = ts.year
        if region == "UK":
            if year not in _UK_HOLIDAYS_CACHE:
                _UK_HOLIDAYS_CACHE[year] = holidays.country_holidays("GB", subdiv="England", years=year)
            return ts.date() in _UK_HOLIDAYS_CACHE[year]
        elif region == "CY":
            if year not in _CY_HOLIDAYS_CACHE:
                _CY_HOLIDAYS_CACHE[year] = holidays.country_holidays("CY", years=year)
            return ts.date() in _CY_HOLIDAYS_CACHE[year]
        return False
except ImportError:
    def is_holiday(ts, region):
        return False

def add_forecast_features(df: pd.DataFrame, main_load_col: str, region: str) -> pd.DataFrame:
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = df["timestamp"].apply(lambda x: int(is_holiday(x, "UK" if region == "residential" else "CY")))
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)

    # Load lags — all shifted so only past data is used
    for lag in [48, 96, 144, 336, 672]:
        df[f"{main_load_col}_lag_{lag}"] = df[main_load_col].shift(lag)

    # Rolling features — all anchored at lag 48 (yesterday), no leakage
    df[f"{main_load_col}_roll_mean_6"]  = df[main_load_col].shift(48).rolling(window=6,  min_periods=1).mean()
    df[f"{main_load_col}_roll_mean_48"] = df[main_load_col].shift(48).rolling(window=48, min_periods=1).mean()
    df[f"{main_load_col}_roll_mean_96"] = df[main_load_col].shift(48).rolling(window=96, min_periods=1).mean()
    df[f"{main_load_col}_roll_std_48"]  = df[main_load_col].shift(48).rolling(window=48, min_periods=1).std()

    # Difference feature — both terms are in the past
    df[f"{main_load_col}_diff_48"] = df[main_load_col].shift(48) - df[main_load_col].shift(96)

    # Temperature lag features (residential)
    if "temp_residential" in df.columns:
        df["temp_lag_48"]       = df["temp_residential"].shift(48)
        df["temp_roll_mean_48"] = df["temp_residential"].shift(48).rolling(window=48, min_periods=1).mean()
        df["temp_hour_interaction"] = df["temp_residential"].shift(48) * df["hour_sin"]

    # Temperature lag features (commercial / industrial)
    if "ambient_temp" in df.columns:
        df["temp_lag_48"]       = df["ambient_temp"].shift(48)
        df["temp_roll_mean_48"] = df["ambient_temp"].shift(48).rolling(window=48, min_periods=1).mean()
        df["temp_hour_interaction"] = df["ambient_temp"].shift(48) * df["hour_sin"]

    # Hour-weekend interaction
    df["hour_weekend_interaction"] = df["hour_sin"] * df["is_weekend"]

    # Target: next-day load (48 steps ahead)
    df[f"{main_load_col}_t_plus_48"] = df[main_load_col].shift(-48)

    return df

def merge_with_meteo(load_df: pd.DataFrame, meteo_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(load_df, meteo_df, on="timestamp", how="left")

def clean_meteo_for_load(meteo_df: pd.DataFrame) -> pd.DataFrame:
    df = meteo_df.copy()
    daily_max = df.groupby(df["timestamp"].dt.date)["solar_rad"].max()
    bad_days = daily_max[daily_max == 0].index
    df = df[~df["timestamp"].dt.date.isin(bad_days)]

    window_mask = df["timestamp"].dt.hour.between(10, 14)
    non_zero_in_window = df[window_mask & (df["solar_rad"] > 0)]
    if not non_zero_in_window.empty:
        last_good_time = non_zero_in_window["timestamp"].max()
        future = df[df["timestamp"] > last_good_time]
        future_window_non_zero = future[future["timestamp"].dt.hour.between(10, 14) & (future["solar_rad"] > 0)]
        if future_window_non_zero.empty and not future.empty:
            df = df[df["timestamp"] <= last_good_time]

    return df.reset_index(drop=True)

def analyze_features(df: pd.DataFrame, dataset_name: str, main_load_col: str) -> None:
    target = f"{main_load_col}_t_plus_48"
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if target not in numeric_df.columns:
        print(f"Target {target} missing. Columns: {numeric_df.columns.tolist()}")
        return

    # ── FIX: exclude the raw (unlagged) load column from the feature set ──
    # At prediction time the current-period load is not yet known, so
    # including it would constitute data leakage.  Only the properly
    # lagged / rolled versions (all shifted by ≥ 48 steps) are valid inputs.
    cols_to_exclude = [target, main_load_col]
    feature_cols = [c for c in numeric_df.columns if c not in cols_to_exclude]
    # ──────────────────────────────────────────────────────────────────────

    model_df = numeric_df.dropna(subset=[target] + feature_cols)
    X = model_df[feature_cols]
    y = model_df[target]

    # 1. Feature-to-feature Pearson correlation matrix
    corr_matrix = X.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title(f"{dataset_name} – Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dataset_name}_feature_correlation_matrix.svg", dpi=150)
    plt.show()

    # 2. Pearson correlation of each feature with target
    corr_with_target = X.join(y).corr()[target].drop(target).sort_values(ascending=False)
    plt.figure(figsize=(5, max(6, 0.35 * len(corr_with_target))))
    sns.heatmap(corr_with_target.to_frame(name="Pearson"), annot=True, cmap="coolwarm", center=0)
    plt.title(f"{dataset_name} – Pearson Correlation with Target")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dataset_name}_pearson_correlation.svg", dpi=150)
    plt.show()

    # 3. Mutual Information of each feature with target
    discrete_set = {"is_weekend", "is_holiday"}
    mi_with_target = mutual_info_regression(
        X, y,
        discrete_features=[c in discrete_set for c in feature_cols],
        random_state=42
    )
    mi_series = pd.Series(mi_with_target, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(5, max(6, 0.35 * len(mi_series))))
    sns.heatmap(mi_series.to_frame(name="MI"), annot=True, cmap="viridis")
    plt.title(f"{dataset_name} – Mutual Information with Target")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{dataset_name}_mutual_information.svg", dpi=150)
    plt.show()

    print(f"\n[{dataset_name}] Top 10 features by Pearson |r| with target:")
    print(corr_with_target.abs().sort_values(ascending=False).head(10).to_string())
    print(f"\n[{dataset_name}] Top 10 features by Mutual Information with target:")
    print(mi_series.head(10).to_string())


def main():
    residential = parse_residential(Residential_file_path)
    commercial  = parse_industrial_or_commercial(Commercial_file_path, "commercial")
    industrial  = parse_industrial_or_commercial(Industrial_file_path, "industrial")
    meteo       = parse_meteo(Meteo_data_file_path)

    residential = unify_frequency(residential)
    commercial  = unify_frequency(commercial)
    industrial  = unify_frequency(industrial)
    meteo       = unify_frequency(meteo)

    meteo_clean = clean_meteo_for_load(meteo)

    commercial = merge_with_meteo(commercial, meteo_clean)
    industrial = merge_with_meteo(industrial, meteo_clean)

    residential = handle_missing(residential)
    commercial  = handle_missing(commercial)
    industrial  = handle_missing(industrial)

    residential = add_forecast_features(residential, "load_residential", region="residential")
    commercial  = add_forecast_features(commercial,  "load_commercial",  region="cyprus")
    industrial  = add_forecast_features(industrial,  "load_industrial",  region="cyprus")

    # Scaled residential (electricity columns ×1000)
    residential_scaled = residential.copy()
    electricity_cols = [c for c in residential_scaled.columns if "load_residential" in c]
    if electricity_cols:
        residential_scaled[electricity_cols] = residential_scaled[electricity_cols] * 1000

    # Save processed datasets
    residential.to_csv(OUTPUT_DIR / "residential_processed.csv", index=False)
    residential_scaled.to_csv(OUTPUT_DIR / "residential_processed_x1000.csv", index=False)
    commercial.to_csv(OUTPUT_DIR / "commercial_processed.csv", index=False)
    industrial.to_csv(OUTPUT_DIR / "industrial_processed.csv", index=False)

    print("Processed datasets saved to: processed/")

    # Feature analysis — raw load column excluded from all matrices
    analyze_features(residential, "residential", "load_residential")
    analyze_features(commercial,  "commercial",  "load_commercial")
    analyze_features(industrial,  "industrial",  "load_industrial")


if __name__ == "__main__":
    main()