from pathlib import Path
import json
import traceback
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lightgbm import LGBMRegressor, early_stopping
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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

RUN_TUNING = False  # Set to True to run safe tuning, False to use cached/baseline


def mape_scorer(y_true, y_pred):
    """MAPE scorer for sklearn (lower is better, so we negate)."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return -mape  # Negative because sklearn maximizes scores  # True: re-run tuning and overwrite cache | False: use cached params when available

Residential_data_path = "processed/residential_processed.csv"
Commercial_data_path = "processed/commercial_processed.csv"
Industrial_data_path = "processed/industrial_processed.csv"

METRICS_DIR = Path("results")
METRICS_DIR.mkdir(exist_ok=True)
METRICS_FILE = METRICS_DIR / "lightgbm_70_30_metrics.csv"
TUNED_PARAMS_DIR = METRICS_DIR / "tuned_params"
TUNED_PARAMS_DIR.mkdir(exist_ok=True)
ERROR_LOG_FILE = METRICS_DIR / "lightgbm_errors.log"

SVG_DIR = Path("LightGBM_svg")
SVG_DIR.mkdir(exist_ok=True)

# Feature lists - using all available features (feature selection didn't help)
RESIDENTIAL_FEATURES = [
    "hour", "minute", "minute_of_day",
    "dayofweek", "dayofyear", "month",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "temp_residential", "rh_residential",
    "lag_48", "lag_96", "lag_144", "lag_336", "lag_672",
    "roll_mean_6", "roll_mean_48", "roll_mean_96", "roll_std_48",
    "diff_48",
    "temp_lag_48", "temp_roll_mean_48", "temp_hour_interaction",
    "hour_weekend_interaction",
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
    "roll_mean_6", "roll_mean_48", "roll_mean_96",
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

    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.title(f"{dataset_name} – Load over Time")
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_load_timeseries.svg", format="svg", bbox_inches="tight")
    plt.show()

# ── Forecast plot function ─────────────────────────────────────────
def plot_forecast(x_full, y_full, x_test, y_test, y_pred, title, ylabel="Load", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    # 1. Whole dataset – Okabe-Ito blue
    ax.plot(x_full, y_full,
            color=C_ACTUAL, linewidth=0.6, alpha=0.55, zorder=1)

    # 2. Actual test window – Okabe-Ito blue
    ax.plot(x_test, y_test,
            color=C_ACTUAL, linewidth=0.8, alpha=0.90, zorder=2)

    # 3. Predicted – Okabe-Ito vermillion, solid
    ax.plot(x_test, y_pred,
            color=C_PREDICTED, linewidth=0.8, linestyle="-",
            alpha=0.90, zorder=3)

    # Legend
    handles = [
        mpatches.Patch(color=C_ACTUAL,    label="Actual (full dataset)"),
        mpatches.Patch(color=C_ACTUAL,    label="Actual (test window)"),
        mpatches.Patch(color=C_PREDICTED, label="Predicted"),
    ]
    ax.legend(handles=handles, loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()


# Replace resolve_features() with this version that INCLUDES rolling stats
def resolve_features(df: pd.DataFrame, load_col: str, feature_list: list[str]) -> tuple[list[str], str]:
    target_col = f"{load_col}_t_plus_48"
    exclude = {"timestamp", target_col, load_col}

    lag_map = {
        "lag_48": f"{load_col}_lag_48",
        "lag_96": f"{load_col}_lag_96",
        "lag_144": f"{load_col}_lag_144",
        "lag_336": f"{load_col}_lag_336",
        "lag_672": f"{load_col}_lag_672",
    }
    
    roll_map = {
        "roll_mean_6": f"{load_col}_roll_mean_6",
        "roll_mean_48": f"{load_col}_roll_mean_48",
        "roll_mean_96": f"{load_col}_roll_mean_96",
        "roll_std_48": f"{load_col}_roll_std_48",
        "diff_48": f"{load_col}_diff_48",
    }

    resolved = []
    for name in feature_list:
        col = lag_map.get(name, roll_map.get(name, name))
        if col in df.columns and col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            resolved.append(col)
    return resolved, target_col


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_val = np.mean(y_true)
    nrmse = rmse / mean_val if mean_val != 0 else np.nan

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "nRMSE": nrmse}


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="backslashreplace").decode("ascii"))


def append_metrics_row_with_retry(row: dict, retries: int = 3, delay_sec: float = 0.6) -> None:
    df_row = pd.DataFrame([row])

    for attempt in range(1, retries + 1):
        try:
            df_row.to_csv(
                METRICS_FILE,
                mode="a",
                header=not METRICS_FILE.exists(),
                index=False
            )
            return
        except PermissionError:
            if attempt < retries:
                safe_print(
                    f"[WARN] Could not write to {METRICS_FILE} (attempt {attempt}/{retries}). "
                    "File may be open in Excel. Retrying..."
                )
                time.sleep(delay_sec * attempt)
            else:
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                fallback_file = METRICS_DIR / f"lightgbm_70_30_metrics_fallback_{ts}.csv"
                df_row.to_csv(
                    fallback_file,
                    mode="a",
                    header=not fallback_file.exists(),
                    index=False
                )
                safe_print(f"[WARN] Metrics file locked. Wrote to fallback: {fallback_file}")


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
    append_metrics_row_with_retry(row)


def get_params_cache_path(dataset_name: str) -> Path:
    return TUNED_PARAMS_DIR / f"{dataset_name.lower().replace(' ', '_')}_params.json"


def load_cached_params(dataset_name: str) -> dict | None:
    path = get_params_cache_path(dataset_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def save_cached_params(dataset_name: str, params: dict) -> None:
    get_params_cache_path(dataset_name).write_text(
        json.dumps(params, indent=2, sort_keys=True),
        encoding="utf-8"
    )


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
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        metrics = compute_metrics(y_test, y_pred)
        results.append({"train_pct": pct, "nrmse": metrics["nRMSE"]})
        print(
            f"{dataset_name} | Train {pct}% ({len(train_df):,}) | Test {test_pct}% ({len(test_df):,}) | "
            f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
            f"MAPE={metrics['MAPE']:.2f}% | nRMSE={metrics['nRMSE']:.4f}"
        )

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["train_pct"], results_df["nrmse"], marker="o")
    plt.xlabel("Training Data (%)")
    plt.ylabel("nRMSE")
    plt.title(f"LightGBM: {dataset_name} – nRMSE vs Training Size (Test={test_pct}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_train_size_experiment.svg", format="svg", bbox_inches="tight")
    plt.show()
    return results_df


def tune_model_safe(
    df: pd.DataFrame,
    load_col: str,
    dataset_name: str,
    feature_list: list[str],
    n_splits: int = 5,
    n_iter: int = 50,
    random_state: int = 42
) -> dict | None:
    """
    Safe tuning: only returns params if they beat baseline on validation set.
    Returns None if baseline is better (so we keep using baseline).
    """
    feature_cols, target_col = resolve_features(df, load_col, feature_list)
    model_df = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)

    # Split: 60% train, 10% validation (for comparing), 30% test (held out)
    n = len(model_df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.70)
    
    train_df = model_df.iloc[:train_end]
    val_df = model_df.iloc[train_end:val_end]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Baseline params
    baseline_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }

    # Train baseline and get validation MAPE
    baseline_model = LGBMRegressor(**baseline_params)
    baseline_model.fit(X_train_scaled, y_train)
    baseline_pred = baseline_model.predict(X_val_scaled)
    baseline_metrics = compute_metrics(y_val, baseline_pred)
    baseline_mape = baseline_metrics["MAPE"]
    print(f"{dataset_name} – Baseline validation MAPE: {baseline_mape:.4f}%")

    # Conservative search space centered around baseline
    param_distributions = {
        "n_estimators": [300, 400, 500, 600],
        "max_depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.04, 0.05, 0.06, 0.07],
        "num_leaves": [25, 31, 40, 50],
        "min_child_samples": [15, 20, 25, 30],
        "subsample": [0.7, 0.8, 0.85, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.85, 0.9],
        "reg_alpha": [0.0, 0.01, 0.05],
        "reg_lambda": [0.0, 0.01, 0.05],
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    base_model = LGBMRegressor(
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )

    mape_scoring = make_scorer(mape_scorer, greater_is_better=True)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=mape_scoring,
        cv=tscv,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
        error_score="raise"
    )

    print(f"\n{dataset_name} – Running safe tuning (n_iter={n_iter})")
    try:
        search.fit(X_train_scaled, y_train)
    except Exception as exc:
        print(f"{dataset_name} – Tuning failed: {exc}")
        return None

    tuned_params = search.best_params_
    print(f"{dataset_name} – Best CV MAPE: {-search.best_score_:.4f}%")

    # Train with tuned params and compare on validation set
    tuned_params_full = {**tuned_params, "random_state": 42, "n_jobs": -1, "verbosity": -1, "force_col_wise": True}
    tuned_model = LGBMRegressor(**tuned_params_full)
    tuned_model.fit(X_train_scaled, y_train)
    tuned_pred = tuned_model.predict(X_val_scaled)
    tuned_metrics = compute_metrics(y_val, tuned_pred)
    tuned_mape = tuned_metrics["MAPE"]
    print(f"{dataset_name} – Tuned validation MAPE: {tuned_mape:.4f}%")

    # Only use tuned params if they beat baseline
    improvement = baseline_mape - tuned_mape
    if improvement > 0:  # Accept ANY improvement
        print(f"{dataset_name} – Tuned params ACCEPTED (improvement: {improvement:.4f}%)")
        return tuned_params
    else:
        print(f"{dataset_name} – Tuned params REJECTED (no improvement)")
        return None


def train_model_70_30(
    df: pd.DataFrame,
    load_col: str,
    dataset_name: str,
    feature_list: list[str],
    model_params: dict | None = None
) -> LGBMRegressor:
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Baseline hyperparameters
    base_params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
    }
    
    # Use tuned params if provided, otherwise baseline
    if model_params:
        final_params = {**base_params, **model_params}
        final_params["random_state"] = 42
        final_params["n_jobs"] = -1
        final_params["verbosity"] = -1
        final_params["force_col_wise"] = True
        print(f"{dataset_name} – Using TUNED hyperparameters")
    else:
        final_params = base_params
        print(f"{dataset_name} – Using BASELINE hyperparameters")
    
    model = LGBMRegressor(**final_params)
    model.fit(X_train_scaled, y_train)
    print(f"{dataset_name} – Trained with {model.n_estimators} estimators")

    y_pred = model.predict(X_test_scaled)
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
    plt.title(f"LightGBM: {dataset_name} – Actual vs Predicted (70:30)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_actual_vs_predicted.svg", format="svg", bbox_inches="tight")
    plt.show()

    plot_forecast(
        x_full=model_df["timestamp"],
        y_full=model_df[target_col],
        x_test=test_df["timestamp"],
        y_test=y_test,
        y_pred=y_pred,
        title=f"{dataset_name} – Forecast Plot over Whole Dataset",
        save_path=SVG_DIR / f"{dataset_name}_forecast_plot_over_whole_dataset.svg"
    )

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

    # Safe tuning: only uses tuned params if they beat baseline on validation
    if RUN_TUNING:
        tuned_params = tune_model_safe(df, load_col, dataset_name, feature_list)
        if tuned_params:
            save_cached_params(dataset_name, tuned_params)
    else:
        tuned_params = load_cached_params(dataset_name)
        if tuned_params:
            print(f"{dataset_name} – Loaded cached tuned params")
        else:
            print(f"{dataset_name} – No cached params, using baseline")

    train_model_70_30(df, load_col, dataset_name, feature_list, model_params=tuned_params)


def main():
    run_analysis(Residential_data_path, "load_residential", "Residential", RESIDENTIAL_FEATURES)
    run_analysis(Commercial_data_path, "load_commercial", "Commercial", COMMERCIAL_FEATURES)
    run_analysis(Industrial_data_path, "load_industrial", "Industrial", INDUSTRIAL_FEATURES)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err_text = traceback.format_exc()
        safe_print("\nScript failed. Full traceback:\n")
        safe_print(err_text)
        with ERROR_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n[main] Unhandled exception\n")
            f.write(err_text)
            f.write("\n")
        raise