import os
from tqdm import tqdm
from zoneinfo import ZoneInfo
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from scipy.signal import resample, butter, filtfilt

def to3(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        vv = list(v)
        return [vv[0] if len(vv)>0 else np.nan,
                vv[1] if len(vv)>1 else np.nan,
                vv[2] if len(vv)>2 else np.nan]
    return [np.nan, np.nan, np.nan]

def convert_date(date_str: str) -> str:
    year = "20" + date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]
    return f"{year}-{month}-{day}"

# ========================================================================
# 0. Data Parsing

## Label filtering
def label_filtering(label_datas, target_date_str):
    
    target_har_labels = {}
    
    # for label_data in tqdm(label_datas, desc="select target date in labeling"):
    for label_data in label_datas:
        date = label_data["timeString"].split(" ")[0]
        if date == target_date_str:
            timestamp = label_data["timeStamp"]
            dt_timestamp = datetime.fromtimestamp(timestamp/1000.0, tz=ZoneInfo("Asia/Seoul"))
            
            label_dict = label_data.copy()
            for del_key in ["timeStamp", "timeString"]:
                label_dict.pop(del_key)
            target_har_labels[dt_timestamp] = label_dict
            
    return target_har_labels

def extract_behavior_ranges(
    tag_dict: Dict[datetime, Dict[str, bool]]
) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    주어진 시간별 행동 태그(tag_dict)를 기반으로 각 행동의 (시작, 종료) 구간 추출
    - 병합 행동(meal, alcohol): 함께 True인 동안은 동일한 시작 시점에서 구간 기록
    - 단독 행동(cigarette, htp, vape): 독립적으로 True/False 기준으로 구간 기록

    Args:
        tag_dict (Dict[datetime, Dict[str, bool]]): 시간순 행동 상태 변화 기록

    Returns:
        Dict[str, List[Tuple[datetime, datetime]]]: 각 행동별 (start, end) 범위 리스트
    """
    
    behaviors = ["meal", "cigarette", "htp", "vape", "alcohol"]
    merge_behaviors = {"meal", "alcohol"}
    solo_behaviors = {"cigarette", "htp", "vape"}
    
    # 초기 상태
    prev_state = {b: False for b in behaviors}
    ongoing_ranges = {}  # 행동명 -> 시작시간
    result = defaultdict(list)
    
    # 시간 순 정렬
    sorted_items = sorted(tag_dict.items())
    
    for i, (ts, curr_state) in enumerate(sorted_items):
        # 병합 행동 처리
        prev_merge_active = {b for b in merge_behaviors if prev_state[b]}
        curr_merge_active = {b for b in merge_behaviors if curr_state[b]}
        
        # 병합 종료 조건: 기존 켜짐 집합 ⊄ 현재 켜짐 집합
        if prev_merge_active and not prev_merge_active.issubset(curr_merge_active):
            start_time = ongoing_ranges.get("__merge__")
            end_time = ts
            if start_time:
                for b in prev_merge_active:
                    result[b].append((start_time, end_time))
                ongoing_ranges.pop("__merge__", None)
                for b in prev_merge_active:
                    ongoing_ranges.pop(b, None)
        
        # 병합 시작 조건: 새롭게 병합 구간 진입
        if curr_merge_active and not prev_merge_active:
            # 이미 시작된 행동이 있으면 그 중 가장 과거 시점을 병합 시작 시점으로 사용
            existsing_starts = [ongoing_ranges.get(b) for b in curr_merge_active if b in ongoing_ranges]
            if existsing_starts:
                start_time = min(existsing_starts)
            else :
                start_time = ts

            ongoing_ranges["__merge__"] = start_time
            for b in curr_merge_active:
                ongoing_ranges[b] = start_time
        
        # 단독 행동 처리
        for b in solo_behaviors:
            prev = prev_state[b]
            curr = curr_state[b]
            if not prev and curr:
                ongoing_ranges[b] = ts
            elif prev and not curr:
                start_time = ongoing_ranges.pop(b, None)
                if start_time:
                    result[b].append((start_time, ts))
        
        # 상태 갱신
        prev_state = curr_state.copy()
    
    # 마지막 범위 정리
    if sorted_items:
        last_ts = sorted_items[-1][0]
        
        # 병합 그룹 종료 처리
        merge_active = {b for b in merge_behaviors if prev_state[b]}
        if merge_active and "__merge__" in ongoing_ranges:
            start_time = ongoing_ranges.pop("__merge__")
            for b in merge_active:
                result[b].append((start_time, last_ts))
                ongoing_ranges.pop(b, None)
        
        # 단독 행동 종료 처리
        for b in solo_behaviors:
            if prev_state[b] and b in ongoing_ranges:
                start_time = ongoing_ranges.pop(b)
                result[b].append((start_time, last_ts))
    
    return dict(result)

def label_window_from_ranges(
    window_start: datetime,
    window_end: datetime,
    behavior_ranges: Dict[str, List[Tuple[datetime, datetime]]],
    behavior_order: List[str] = ["meal", "cigarette", "htp", "vape", "alcohol"]
) -> List[int]:
    labels = []
    for behavior in behavior_order:
        has_overlap = any(
            (start < window_end and end > window_start)
            for (start, end) in behavior_ranges.get(behavior, [])
        )
        labels.append(1 if has_overlap else 0)
    return labels

def data_preprocessing(raw_parquet: pd.DataFrame, target_device):
    
    # ========================================================================
    # time filtering
    start_time = min(raw_parquet["collected_time"])
    end_time = max(raw_parquet["collected_time"])
    
    mask = (raw_parquet["collected_time"] >= start_time) & (raw_parquet["collected_time"] <= end_time)
    
    filtered_df = raw_parquet.loc[mask].reset_index(drop=True)
    
    # ========================================================================
    # preprocess dataframe for denoising (PPG + accelerate)
    # ppg
    ppg_df = filtered_df[filtered_df["sensor_type"] == "SAMSUNG_PPG"]
    ppg_df["ppg"] = pd.Series(ppg_df["data"].map(lambda x: x[2]).to_list(), index=ppg_df.index)
    
    # acc
    acc_df = filtered_df[filtered_df["sensor_type"] == "SAMSUNG_ACCE"]
    xyz_df = pd.DataFrame(
        acc_df["data"].map(to3).tolist(),
        index=acc_df.index,
        columns=["acc_x", "acc_y", "acc_z"]
    )
    
    processed_acc_df = pd.concat([acc_df["timestamp"], xyz_df], axis=1).reset_index(drop=True)
    processed_ppg_df = ppg_df[["timestamp", "ppg"]].reset_index(drop=True)
    
    # ppg + acc datafarme aligned with timestamp
    aligned_df = pd.merge_asof(
        processed_ppg_df.sort_values("timestamp"),
        processed_acc_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=80,
        suffixes=["-ppg", "-acc"]
    )
    # 결측치 처리
    if aligned_df.isna().values.any():
        aligned_df.loc[:, ["acc_x", "acc_y", "acc_z"]] = (
            aligned_df.loc[:, ["acc_x", "acc_y", "acc_z"]]
            .astype(float)
            .ffill()
            .bfill()
        )
        print("Nan in acc, apply ffill, bfill")
    
    # ========================================================================
    # transform struct, splited with windowsize (8 seconds)
    
    window_size = 8000 # ms
    
    t_min = int(aligned_df["timestamp"].min())
    t_max = int(aligned_df["timestamp"].max())
    
    bins = range(t_min, t_max + window_size, window_size)
    aligned_df["windowNumber"] = pd.cut(aligned_df["timestamp"], bins=bins, labels=False, right=False)
    
    groups = dict(tuple(aligned_df.groupby("windowNumber")))
    
    results = []
    for window_number, splited_df in groups.items():
        
        start_time = splited_df["timestamp"].min()
        end_time = splited_df["timestamp"].max()
        
        inner_ppg_list, inner_acc_list, inner_timestamp_list = [], [], []
        
        for _, row in splited_df.iterrows():
            ppg, x, y, z, timestamp = row[["ppg", "acc_x", "acc_y", "acc_z", "timestamp"]]
            
            inner_ppg_list.append(str(ppg))
            inner_acc_list.extend(list(map(str, [x, y, z])))
            inner_timestamp_list.append(str(timestamp))
            
        inner_result = {
            "device_id": target_device,
            "windowNumber": window_number,
            "startTime": start_time,
            "endTime": end_time,
            "galaxyPPG": ";".join(inner_ppg_list),
            "galaxyACC": ";".join(inner_acc_list),
            "timestamps": ";".join(inner_timestamp_list)
        }
        results.append(inner_result)
    
    result_df = pd.DataFrame(results)
    
    return result_df


# 1. Bandpass Filtering + Denoising

def denoising_data(wiener_filter, data_df):
    
    results = {
        "denoisedGalaxy": [None] * len(data_df),
        "estimated_BPM_Galaxy": [None] * len(data_df)
    }
    
    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        try:
            galaxy_ppg = - np.array([float(x) for x in row["galaxyPPG"].split(";") if x.strip()])
            galaxy_acc = np.array([float(x) for x in row["galaxyACC"].split(";") if x.strip()]).reshape(-1, 3)
            
            galaxy_denoised, galaxy_bpm = wiener_filter.process_galaxy(
                galaxy_ppg,
                galaxy_acc[:, 0],
                galaxy_acc[:, 1],
                galaxy_acc[:, 2]
            )
            
            # results["denoisedGalaxy"][i] = ";".join(map(str, (-galaxy_denoised).tolist()))
            results["denoisedGalaxy"][i] = ";".join(map(str, galaxy_denoised.tolist()))
            results["estimated_BPM_Galaxy"][i] = galaxy_bpm
        
        except Exception as e:
            print({str(e)})
            pass
    
    for col, values in results.items():
        data_df[col] = values
    
    subset_df = data_df[["timestamps", "denoisedGalaxy"]]
    
    splited_results = []
    for _, row in subset_df.iterrows():
        timestamps = list(map(float, row["timestamps"].split(";")))
        ppgs = list(map(float, row["denoisedGalaxy"].split(";")))
        for timestamp, ppg in zip(timestamps, ppgs):
            splited_results.append({
                "timestamp": timestamp,
                "ppg": ppg
            })
    
    splited_df = pd.DataFrame(splited_results)
    
    return splited_df


## 
def peak_detection(sampling_rate, clean_segments):
    
    total_peaks = []
    
    upsampling_rate = 2
    sampling_rate_new = sampling_rate * upsampling_rate
    
    for i in range(len(clean_segments)):
        # Normalize PPG Signal
        ppg_normed = normalize_data(clean_segments[i][1])
        
        # Upsampling the Signal
        resampled = signal.resample(ppg_normed, len(ppg_normed)*upsampling_rate)
        
        # Perform peak detction
        ppg_cleaned = nk.ppg_clean(resampled, sampling_rate=sampling_rate_new)
        info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate_new)
        peaks = info["PPG_Peaks"]
        
        # Update peak indices according to the original sampling rate
        inner_peaks = (peaks // upsampling_rate).astype(int)
        
        # Add peaks of the current segment to the total peaks
        total_peaks.append(inner_peaks)
    
    return total_peaks

# -*- coding: utf-8 -*-
'''
Miscellaneous functions

'''

def get_data(
        file_name: str,
        local_directory: str = "data",
        usecols: List[str] = ['ppg'],
        whitespace = False
) -> np.ndarray:
    """
    Import data (e.g., PPG signals)
    
    Args:
        file_name (str): Name of the input file
        local_directory (str): Data directory
        usecols (List[str]): The columns to read from the input file
    
    Return:
        sig (np.ndarray): the input signal (e.g., PPG)
    """
    try:
        # Construct the file path
        file_path = os.path.join(local_directory, file_name)
        # Load data from the specified CSV file
        input_data = pd.read_csv(
            file_path,
            delim_whitespace=whitespace,
            usecols=usecols)
        # Extract signal
        sig = input_data[usecols[0]].values
        return sig
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except pd.errors.EmptyDataError:
        print(f"Empty data in file: {file_name}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # Return None in case of an error
    return None


def normalize_data(sig: np.ndarray) -> np.ndarray:
    """
    Normalize the input signal between zero and one
    
    Args:
        sig (np.ndarray): PPG signal.
    
    Return:
        np.ndarray: Normalized signal
    """
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))


def resample_signal(
        sig: np.ndarray,
        fs_origin: int,
        fs_target: int = 20,
) -> np.ndarray:
    """
    Resample the signal

    Args:
        sig (np.ndarray): The input signal.
        fs_origin (int): The sampling frequency of the input signal.
        fs_target (int): The sampling frequency of the output signal.

    Return:
        sig_resampled (np.ndarray): The resampled signal.
    """
    # Exit if the sampling frequency already is 20 Hz (return the original signal)
    if fs_origin == fs_target:
        return sig
    # Calculate the resampling rate
    resampling_rate = fs_target/fs_origin
    # Resample the signal
    sig_resampled = resample(sig, int(len(sig)*resampling_rate))
    # Update the sampling frequency
    return sig_resampled


def bandpass_filter(
        sig: np.ndarray,
        fs: int,
        lowcut: float,
        highcut: float,
        order: int=2
) -> np.ndarray:
    """
    Apply a bandpass filter to the input signal.

    Args:
        sig (np.ndarray): The input signal.
        fs (int): The sampling frequency of the input signal.
        lowcut (float): The low cutoff frequency of the bandpass filter.
        highcut (float): The high cutoff frequency of the bandpass filter.

    Return:
        sig_filtered (np.ndarray): The filtered signal using a Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    sig_filtered = filtfilt(b, a, sig)
    return sig_filtered


def find_peaks(
        ppg: np.ndarray,
        sampling_rate: int,
        return_sig: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in PPG.

    Args:
        ppg (np.ndarray): The input PPG signal.
        sampling_rate (int): The sampling rate of the signal.
        return_sig (bool): If True, return the cleaned PPG
            signal along with the peak indices (default is False).

    Return:
        peaks (np.ndarray): An array containing the indices of
            the detected peaks in the PPG signal.
        ppg_cleaned (np.ndarray): The cleaned PPG signal, return if return_sig is True.

    """

    # Clean the PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)

    # Peak detection
    info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]

    # Return either just the peaks or both the cleaned signal and peaks
    if return_sig:
        return peaks, ppg_cleaned
    else:
        return peaks, None