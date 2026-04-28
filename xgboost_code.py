from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ── Okabe-Ito Palette ──────────────────────────────────────────────
C_HISTORY   = "#cccccc"   # light gray  – whole dataset context
C_ACTUAL    = "#0072B2"   # Okabe-Ito blue  – actual test window
C_PREDICTED = "#D55E00"   # Okabe-Ito vermillion – predicted

# ── Global style (apply once at top of script) ─────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.5,
    "grid.linestyle":    "--",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,      # publication quality on save
    "savefig.bbox":      "tight",
    "font.family":       "serif",  # matches Word/thesis font
})

Residential_data_path = "processed/residential_processed.csv"
Commercial_data_path = "processed/commercial_processed.csv"
Industrial_data_path = "processed/industrial_processed.csv"

METRICS_DIR = Path("results")
METRICS_DIR.mkdir(exist_ok=True)
METRICS_FILE = METRICS_DIR / "xgboost_70_30_metrics.csv"

SVG_DIR = Path("xgboost_svg")
SVG_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Feature lists (edit manually)
# --------------------------------------------------------------------------- #
RESIDENTIAL_FEATURES = [
    "hour", "minute", "minute_of_day",
    "dayofweek", "dayofyear", "month",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "temp_residential", "rh_residential",
    "lag_48", "lag_96", "lag_144", "lag_336",
]

COMMERCIAL_FEATURES = [
    "hour", "minute", "minute_of_day",
    "dayofweek", "dayofyear", "month",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "solar_rad", "ambient_temp", "ambient_rh",
    "lag_48", "lag_96", "lag_144", "lag_336",
]

INDUSTRIAL_FEATURES = [
    "hour", "minute", "minute_of_day",
    "dayofweek", "dayofyear", "month",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "solar_rad", "ambient_temp", "ambient_rh",
    "lag_48", "lag_96", "lag_144", "lag_336",
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def plot_load_timeseries(
    df: pd.DataFrame,
    load_col: str,
    dataset_name: str,
    split_pct: int | None = None
) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df[load_col], linewidth=0.5, color=C_ACTUAL)

    if split_pct is not None:
        split_idx = int(len(df) * split_pct / 100)
        split_ts = df["timestamp"].iloc[split_idx]
        plt.axvline(split_ts, color="red", linestyle=":", linewidth=1.5, label=f"{split_pct}%")
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.title(f"XGBoost: {dataset_name} – Load over Time")
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_load_timeseries.svg", format="svg", bbox_inches="tight")
    plt.close()


def resolve_features(df: pd.DataFrame, load_col: str, feature_list: list[str]) -> tuple[list[str], str]:
    target_col = f"{load_col}_t_plus_48"
    exclude = {"timestamp", target_col, load_col}
    roll_cols = {f"{load_col}_roll_mean_6", f"{load_col}_roll_mean_48", f"{load_col}_roll_mean_96"}
    exclude |= roll_cols

    lag_map = {
        "lag_48": f"{load_col}_lag_48",
        "lag_96": f"{load_col}_lag_96",
        "lag_144": f"{load_col}_lag_144",
        "lag_336": f"{load_col}_lag_336",
    }

    resolved = []
    for name in feature_list:
        col = lag_map.get(name, name)
        if col in df.columns and col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            resolved.append(col)
    return resolved, target_col


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE, nRMSE."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_val = np.mean(y_true)
    nrmse = rmse / mean_val if mean_val != 0 else np.nan

    # Avoid division by zero in MAPE
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "nRMSE": nrmse}


def train_size_experiment(
    df: pd.DataFrame,
    load_col: str,
    dataset_name: str,
    feature_list: list[str],
    train_pcts: list[int] | None = None,
    test_pct: int = 30
) -> pd.DataFrame:
    if train_pcts is None:
        train_pcts = [10, 20, 30, 40, 50, 60, 70]

    feature_cols, target_col = resolve_features(df, load_col, feature_list)
    print(f"\n{dataset_name} features ({len(feature_cols)}): {feature_cols}")

    model_df = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    n = len(model_df)

    test_start = int(n * (1 - test_pct / 100))
    test_df = model_df.iloc[test_start:]
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    results = []
    for pct in train_pcts:
        train_end = min(int(n * pct / 100), test_start)
        train_df = model_df.iloc[:train_end]
        if len(train_df) < 100:
            print(f"Skipping {pct}% (only {len(train_df)} rows)")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        results.append({"train_pct": pct, "nrmse": metrics["nRMSE"]})
        print(
            f"{dataset_name} | Train {pct}% ({len(train_df):,}) | Test {test_pct}% ({len(test_df):,}) | "
            f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
            f"MAPE={metrics['MAPE']:.2f}% | nRMSE={metrics['nRMSE']:.4f}"
        )

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["train_pct"], results_df["nrmse"], marker="o", color=C_ACTUAL)
    plt.xlabel("Training Data (%)")
    plt.ylabel("nRMSE")
    plt.title(f"XGBoost: {dataset_name} – nRMSE vs Training Size (Test={test_pct}%)")
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_train_size_experiment.svg", format="svg", bbox_inches="tight")
    plt.close()
    return results_df


def save_split_metrics(dataset_name: str, feature_cols: list[str], metrics: dict) -> None:
    row = {
        "dataset": dataset_name,
        "split": "70:30",
        "features_used": ", ".join(feature_cols),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "nRMSE": metrics["nRMSE"],
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(
        METRICS_FILE,
        mode="a",
        header=not METRICS_FILE.exists(),
        index=False
    )


def train_model_70_30(
    df: pd.DataFrame,
    load_col: str,
    dataset_name: str,
    feature_list: list[str]
) -> XGBRegressor:
    feature_cols, target_col = resolve_features(df, load_col, feature_list)
    print(f"\n{dataset_name} features ({len(feature_cols)}): {feature_cols}")

    model_df = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    n = len(model_df)

    train_end = int(n * 0.70)
    train_df = model_df.iloc[:train_end]
    test_df = model_df.iloc[train_end:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    print(
        f"{dataset_name} | 70:30 Split | "
        f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
        f"MAPE={metrics['MAPE']:.2f}% | nRMSE={metrics['nRMSE']:.4f}"
    )

    save_split_metrics(dataset_name, feature_cols, metrics)

    plt.figure(figsize=(14, 5))
    plt.plot(test_df["timestamp"], y_test, label="Actual", linewidth=0.8, color=C_ACTUAL)
    plt.plot(test_df["timestamp"], y_pred, label="Predicted", linewidth=0.8, color=C_PREDICTED, linestyle="-", alpha=0.9)
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.title(f"XGBoost: {dataset_name} – Actual vs Predicted (70:30)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_actual_vs_predicted.svg", format="svg", bbox_inches="tight")
    plt.close()
    return model


def run_analysis(filepath: str, load_col: str, dataset_name: str, feature_list: list[str]) -> None:
    df = load_data(filepath)
    if dataset_name == "Commercial":
        plot_load_timeseries(df, load_col, dataset_name, split_pct=30)
        train_pcts = [10, 20, 30, 35, 40, 50, 60, 70]
        train_size_experiment(df, load_col, dataset_name, feature_list, train_pcts=train_pcts)
    else:
        plot_load_timeseries(df, load_col, dataset_name)
        train_size_experiment(df, load_col, dataset_name, feature_list)
    train_model_70_30(df, load_col, dataset_name, feature_list)


def main():
    run_analysis(Residential_data_path, "load_residential", "Residential", RESIDENTIAL_FEATURES)
    run_analysis(Commercial_data_path, "load_commercial", "Commercial", COMMERCIAL_FEATURES)
    run_analysis(Industrial_data_path, "load_industrial", "Industrial", INDUSTRIAL_FEATURES)



if __name__ == "__main__":
    main()