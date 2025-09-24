import os
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from typing import List
from zoneinfo import ZoneInfo
from matplotlib.dates import DateFormatter

####

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
from zoneinfo import ZoneInfo

####

DEFAULT_FEAT = ["HR", "HRV_MeanNN", "HRV_RMSSD", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn"]

# --- label 파싱: [1 0 0 0 0], [1,0,0,0,0], list/ndarray 모두 대응 ---
def _parse_label_cell(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        vals = [int(v) for v in x]
    else:
        # 문자열일 경우 0/1만 추출 (공백/콤마 상관없이)
        vals = [int(v) for v in re.findall(r'[01]', str(x))]
    # 길이 보정
    vals += [0] * max(0, 5 - len(vals))
    return vals[:5]

def _ensure_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    arr = np.array([_parse_label_cell(x) for x in df["label"]])
    df[["meal", "cigarette", "htp", "vape", "alcohol"]] = arr
    return df

def _intervals_from_binary(series: pd.Series):
    """0/1 시퀀스에서 (start_idx, end_idx) 구간 리스트 반환. 끝 열림 구간도 자동 종결."""
    x = series.astype(int).to_numpy()
    change = np.diff(np.r_[0, x, 0])         # 0→1: +1, 1→0: -1
    starts = np.where(change == 1)[0]
    ends   = np.where(change == -1)[0]
    return list(zip(starts, ends))           # 길이 동일 보장

def plot_columns_with_labels(
    df: pd.DataFrame,
    target_cols = ["HR"],
    mode: str = "span",   # "span" or "line",
    save_dir = None
):
    df = df.copy()

    # timestamp → KST (앞서 맞춰서 잘 나오는 방식 유지: UTC epoch(ms) → KST)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(ZoneInfo("Asia/Seoul"))

    # label 파생 컬럼 만들기
    _ensure_label_columns(df)

    # 그룹 정의
    groups = {
        "Meal": df["meal"].astype(int),
        "Smoke/Vape": df[["cigarette", "htp", "vape"]].max(axis=1).astype(int),
        "Alcohol": df["alcohol"].astype(int),
    }
    colors = {
        "Meal": "orange",
        "Smoke/Vape": "red",
        "Alcohol": "blue",
    }

    fig, axs = plt.subplots(len(target_cols), 1, figsize=(12, 4 * len(target_cols)))
    if len(target_cols) == 1:
        axs = [axs]

    for ax, col in zip(axs, target_cols):
        ax.plot(ts, df[col], marker="o", label=col)
        ax.set_title(col)
        ax.grid(True)

        # 그룹별 구간 표시
        for name, bin_series in groups.items():
            for s_idx, e_idx in _intervals_from_binary(bin_series):
                x1 = ts.iloc[s_idx]
                # e_idx는 상태가 0으로 바뀌는 "경계"이므로 ts.iloc[e_idx]가 존재하지 않으면 마지막 시점으로 클램프
                x2 = ts.iloc[min(e_idx, len(ts) - 1)]
                if mode == "span":
                    ax.axvspan(x1, x2, color=colors[name], alpha=0.18)
                else:  # "line"
                    ax.axvline(x1, linestyle="--", linewidth=1.3, color=colors[name], alpha=0.9)
                    ax.axvline(x2, linestyle=":",  linewidth=1.0, color=colors[name], alpha=0.6)

    # 범례(라벨 그룹)
    handles = [Patch(facecolor=colors[k], edgecolor="none", alpha=0.18, label=k) for k in groups.keys()]
    axs[0].legend(handles=handles, loc="upper right", title="Labels")

    # X축 포맷 (KST)
    axs[-1].xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M", tz=ZoneInfo("Asia/Seoul")))
    plt.xlabel("Timestamp (Asia/Seoul)")
    plt.tight_layout()
    if save_dir is not None:
        save_path = os.path.join(save_dir, "plot.png")
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def main(args):
    
    parquet_path = args.data_path
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    # hr filtering
    df = df[df["HR"] >= 40]
    
    target_cols = DEFAULT_FEAT if args.feat is None else DEFAULT_FEAT+args.feat.split(",")
    
    plot_columns_with_labels(
        df=df,
        target_cols=target_cols,
        save_dir=args.save_dir
    )
    

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="tmp1-3.parquet")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--feat", type=str, default=None)
    
    
    args = parser.parse_args()
    
    main(args)