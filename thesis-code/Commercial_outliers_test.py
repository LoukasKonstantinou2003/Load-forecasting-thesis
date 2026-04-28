import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "Times New Roman"

DATA_PATH = "processed/commercial_processed.csv"
LOAD_COL = "load_commercial"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
SHIFT_CSV = RESULTS_DIR / "commercial_train_test_shift.csv"
SVG_DIR = Path("Commercia_dataset_anomaly_svg")
SVG_DIR.mkdir(exist_ok=True)

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

def iqr_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

def zscore_outliers(series: pd.Series, z: float = 3.0) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    zscores = (series - mean) / std
    return zscores.abs() > z

def seasonal_zscore_outliers(df: pd.DataFrame, load_col: str, z: float = 3.0) -> pd.Series:
    # Group by month + hour to avoid flagging seasonal peaks as outliers
    group_keys = [df["timestamp"].dt.month, df["timestamp"].dt.hour]
    group_mean = df.groupby(group_keys)[load_col].transform("mean")
    group_std = df.groupby(group_keys)[load_col].transform("std")
    zscores = (df[load_col] - group_mean) / group_std.replace(0, np.nan)
    return zscores.abs() > z

def summarize_outliers(df: pd.DataFrame, mask: pd.Series, label: str) -> None:
    count = mask.sum()
    pct = (count / len(df)) * 100
    print(f"{label}: {count} outliers ({pct:.2f}%)")

def split_stats(df: pd.DataFrame, load_col: str) -> dict:
    s = df[load_col]
    return {
        "mean": s.mean(),
        "std": s.std(),
        "q10": s.quantile(0.10),
        "q25": s.quantile(0.25),
        "q50": s.quantile(0.50),
        "q75": s.quantile(0.75),
        "q90": s.quantile(0.90),
    }

def train_test_shift_table(
    df: pd.DataFrame,
    load_col: str,
    train_pcts: list[float],
    test_pct: int = 30
) -> pd.DataFrame:
    n = len(df)
    test_start = int(n * (100 - test_pct) / 100)
    test_df = df.iloc[test_start:].copy()

    rows = []
    for pct in train_pcts:
        train_end = min(int(n * pct / 100), test_start)
        train_df = df.iloc[:train_end].copy()
        if len(train_df) == 0:
            continue

        train_stats = split_stats(train_df, load_col)
        test_stats = split_stats(test_df, load_col)

        mean_shift = (test_stats["mean"] - train_stats["mean"]) / train_stats["mean"] if train_stats["mean"] != 0 else np.nan
        std_ratio = test_stats["std"] / train_stats["std"] if train_stats["std"] != 0 else np.nan

        rows.append({
            "train_pct": pct,
            "train_start": train_df["timestamp"].iloc[0].date(),
            "train_end": train_df["timestamp"].iloc[-1].date(),
            "test_start": test_df["timestamp"].iloc[0].date(),
            "test_end": test_df["timestamp"].iloc[-1].date(),
            "train_mean": train_stats["mean"],
            "test_mean": test_stats["mean"],
            "mean_shift": mean_shift,
            "train_std": train_stats["std"],
            "test_std": test_stats["std"],
            "std_ratio": std_ratio,
            "train_q10": train_stats["q10"],
            "train_q25": train_stats["q25"],
            "train_q50": train_stats["q50"],
            "train_q75": train_stats["q75"],
            "train_q90": train_stats["q90"],
            "test_q10": test_stats["q10"],
            "test_q25": test_stats["q25"],
            "test_q50": test_stats["q50"],
            "test_q75": test_stats["q75"],
            "test_q90": test_stats["q90"],
        })

    return pd.DataFrame(rows)

def plot_shift_metrics(shift_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(shift_df["train_pct"], shift_df["mean_shift"], marker="o", label="Mean shift")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Train %")
    plt.ylabel("Mean shift (test vs train)")
    plt.title("Commercial: Mean Shift vs Train %")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SVG_DIR / "mean_shift.svg", format="svg")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(shift_df["train_pct"], shift_df["std_ratio"], marker="o", label="Std ratio")
    plt.axhline(1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Train %")
    plt.ylabel("Std ratio (test / train)")
    plt.title("Commercial: Std Ratio vs Train %")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SVG_DIR / "std_ratio.svg", format="svg")
    plt.close()

def main():
    df = load_data(DATA_PATH)

    # train/test distribution shift table
    train_pcts = [10, 20, 22.5, 25, 30, 35, 40, 50, 60, 70]
    shift_df = train_test_shift_table(df, LOAD_COL, train_pcts=train_pcts, test_pct=30)

    # plot shift metrics
    plot_shift_metrics(shift_df)

    iqr_mask = iqr_outliers(df[LOAD_COL], k=1.5)
    z_mask = zscore_outliers(df[LOAD_COL], z=3.0)
    seasonal_mask = seasonal_zscore_outliers(df, LOAD_COL, z=3.0)

    print("=== Full series outliers ===")
    summarize_outliers(df, iqr_mask, "IQR")
    summarize_outliers(df, z_mask, "Z-score (global)")
    summarize_outliers(df, seasonal_mask, "Z-score (month+hour)")

    # Focus on the 20–30% segment
    n = len(df)
    seg_start = int(n * 0.20)
    seg_end = int(n * 0.30)
    seg = df.iloc[seg_start:seg_end].copy()
    seg_iqr = iqr_outliers(seg[LOAD_COL], k=1.5)
    seg_z = zscore_outliers(seg[LOAD_COL], z=3.0)
    seg_seasonal = seasonal_zscore_outliers(seg, LOAD_COL, z=3.0)

    print("\n=== 20–30% segment outliers ===")
    summarize_outliers(seg, seg_iqr, "IQR")
    summarize_outliers(seg, seg_z, "Z-score (global)")
    summarize_outliers(seg, seg_seasonal, "Z-score (month+hour)")

    # Plot with initial (global) outliers highlighted
    combined_mask = iqr_mask | z_mask

    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df[LOAD_COL], linewidth=0.5, label="Load")
    plt.scatter(
        df.loc[combined_mask, "timestamp"],
        df.loc[combined_mask, LOAD_COL],
        color="red",
        s=10,
        label="Outliers (IQR + global z-score)"
    )
    plt.axvspan(
        df["timestamp"].iloc[seg_start],
        df["timestamp"].iloc[seg_end - 1],
        color="orange",
        alpha=0.15,
        label="20-30 %% segment"
    )
    plt.title("Commercial Load with Global Outliers Highlighted")
    plt.xlabel("Date")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SVG_DIR / "commercial_load_outliers.svg", format="svg")
    plt.close()

if __name__ == "__main__":
    main()