import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

# Denoising
from wiener_denosing import WienerDenoising

# E2E-PPG
from ppg_sqa import sqa
from ppg_hrv_extraction import hrv_extraction
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction

from utils import convert_date, label_filtering, extract_behavior_ranges,\
    label_window_from_ranges, peak_detection, data_preprocessing, denoising_data

DATA_DIR = "/data3/watch_sensor_data/src/processed_data"
LABEL_DIR = "/data3/ppg_data/raw/"

def main(args):
    """
    0. Data 파싱..
        -> 8 seconds 단위의 ppg, acc(x, y, z) 값들
    ---
    1. Bandpass filtering + Denoising
    ---
    2. SQA
    3. Reconstruction
    4. clean-seg-extraction
    ---
    5. peak detection
    6. hrv extraction
    """
    
    # ========================================================================
    # 0. 데이터 파싱
    
    ## TODO: interation with target_device, each files
    # target_device = args.target_deivce
    # target_date = args.target_date
    target_device = "cf782c01_10c971c2"
    target_date = "250701"
    cvt_target_date = convert_date(target_date)
    
    tmp_data_path = os.path.join(DATA_DIR, target_device, f"{cvt_target_date}.parquet")
    tmp_label_path = os.path.join(LABEL_DIR, target_device, "har_label", f"250714_har.json")
    
    # label flitering
    with open(tmp_label_path, "r") as f:
        label_datas = json.load(f)
    
    target_har_label = label_filtering(label_datas, cvt_target_date)
    target_har_ranges = extract_behavior_ranges(target_har_label)
    
    # raw data preprocessing
    raw_parquet = pd.read_parquet(tmp_data_path, engine="pyarrow")
    df = data_preprocessing(raw_parquet, target_device)  # target_device 빼도 되긴 할텐데
    
    # ========================================================================
    # 1. Bandpass Filtering + Denoising
    wiener = WienerDenoising(
        # hz_hp=0.5,
        hz_hp=1,
        hz_lp=3
    )
    
    splited_df = denoising_data(wiener, df)
    
    # ========================================================================
    # 2. SQA
    
    print("SQA")
    sampling_rate = 25
    
    input_sig = splited_df["ppg"]
    
    clean_indices, noisy_indices = sqa(
        sig=input_sig,
        sampling_rate=sampling_rate,
        filter_signal=False
    )
    
    # ========================================================================
    # 3. Reconstruction
    print("reconstruction")
    sig_reconstructed, clean_indices, noisy_indices = reconstruction(
        sig=input_sig,
        clean_indices=clean_indices,
        noisy_indices=noisy_indices,
        sampling_rate=sampling_rate,
        filter_signal=False
    )
    
    # ========================================================================
    # 4. Segmentation
    
    print("segmentation")
    window_length_sec = 60 * 5 # -> Frequency Feature가 추출 될 수 있는 최소 시간
    window_length = window_length_sec * sampling_rate
    
    clean_segments = clean_seg_extraction(
        sig = sig_reconstructed,
        noisy_indices=noisy_indices,
        window_length=window_length
    )
    
    # seg_start_idx_list = [x[0] for x in clean_segments]
    
    # ========================================================================
    # 5. Peak Detection
    
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
        return None
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )
        
        ### Run PPG peak detection using nk
        total_peaks = peak_detection(sampling_rate, clean_segments)
        
        # ========================================================================
        # 6. HRV Extraction
        # Perform HR and HRV extraction
        hrv_data = hrv_extraction(
            clean_segments=clean_segments,
            peaks=total_peaks,
            sampling_rate=sampling_rate,
            window_length=window_length)
        print("HR and HRV parameters:")
        print(hrv_data)
        print('Done!')
        
        
        # add timestamp
        seg_start_idx_list = list(map(int, hrv_data["Start_idx"].values))
        seg_timestamp = splited_df.iloc[seg_start_idx_list]["timestamp"].values
        hrv_data["timestamp"] = seg_timestamp
        
        # add labels
        label_values = []
        for _, row in hrv_data.iterrows():
            window_start_timestamp = row["timestamp"]
            window_start = datetime.fromtimestamp(window_start_timestamp / 1000, tz=ZoneInfo("Asia/Seoul"))
            window_end = window_start + timedelta(minutes=5)
            labels = label_window_from_ranges(window_start, window_end, target_har_ranges)
            label_values.append(labels)
        
        hrv_data["label"] = label_values
        
        hrv_data.to_parquet("tmp.parquet", engine="pyarrow", index=False)
    
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cf782c01_10c971c2")
    parser.add_argument("--date", type=str, default="250701")
    
    args = parser.parse_args()
    
    main(args)