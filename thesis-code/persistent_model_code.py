from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "persistent_70_30_metrics.csv"

SVG_DIR = Path("persistent_svg")
SVG_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("Residential", Residential_data_path, "load_residential"),
    ("Commercial", Commercial_data_path, "load_commercial"),
    ("Industrial", Industrial_data_path, "load_industrial"),
]


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


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


def persistent_forecast(df: pd.DataFrame, load_col: str) -> pd.DataFrame:
    df = df.copy()
    # Average of the previous three days at the same timestamp
    lag_1d = df[load_col].shift(48)
    lag_2d = df[load_col].shift(96)
    lag_3d = df[load_col].shift(144)
    df["persistent_pred"] = (lag_1d + lag_2d + lag_3d) / 3.0
    return df


def evaluate_70_30(df: pd.DataFrame, load_col: str) -> tuple[dict, pd.DataFrame]:
    df = persistent_forecast(df, load_col)
    df = df.dropna(subset=[load_col, "persistent_pred"]).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    test_df = df.iloc[train_end:].copy()

    y_test = test_df[load_col].values
    y_pred = test_df["persistent_pred"].values
    return compute_metrics(y_test, y_pred), test_df


def plot_forecast(test_df: pd.DataFrame, load_col: str, dataset_name: str) -> None:
    plt.figure(figsize=(14, 5))
    plt.plot(test_df["timestamp"], test_df[load_col], label="Actual", linewidth=0.8, color=C_ACTUAL)
    plt.plot(test_df["timestamp"], test_df["persistent_pred"], label="Persistent Forecast", linewidth=0.8, color=C_PREDICTED, linestyle="-", alpha=0.9)
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.title(f"{dataset_name} – Persistent Forecast vs Actual (Test 30%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SVG_DIR / f"{dataset_name}_actual_vs_predicted.svg", format="svg", bbox_inches="tight")
    plt.close()


def main() -> None:
    rows = []
    for name, path, load_col in DATASETS:
        df = load_data(path)
        metrics, test_df = evaluate_70_30(df, load_col)
        rows.append({
            "dataset": name,
            "split": "70:30",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "nRMSE": metrics["nRMSE"],
        })
        print(
            f"{name} | 70:30 Persistent | "
            f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
            f"MAPE={metrics['MAPE']:.2f}% | nRMSE={metrics['nRMSE']:.4f}"
        )
        plot_forecast(test_df, load_col, name)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()