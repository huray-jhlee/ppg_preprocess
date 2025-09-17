import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import to3
from ppg_sqa import sqa
from ppg_hrv_extraction import hrv_extraction
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction

from wiener_denosing import WienerDenoising
from kazemi_peak_detection import ppg_peaks


DATA_DIR = "/data3/watch_sensor_data/src/processed_data"


def main():
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
    
    target_device = "cf782c01_10c971c2"
    tmp_data_path = os.path.join(DATA_DIR, target_device, "2025-07-01.parquet")
    
    
    ### input(parquet_file_path, start_time, end_time, window_size)
    raw_parquet = pd.read_parquet(tmp_data_path, engine="pyarrow")
    
    ## 시간 필터링을 걸..까?
    ## 다른 라벨에 대한 정답도 포함시켜서 파싱해야하지 않을까..?
    # 일단 시간 필터링
    
    start_time = pd.Timestamp("2025-07-01 11:00:00.000", tz="Asia/Seoul")
    end_time = pd.Timestamp("2025-07-01 13:00:00.000", tz="Asia/Seoul")
    
    mask = (raw_parquet["collected_time"] >= start_time) & (raw_parquet["collected_time"] <= end_time)
    
    filtered_df = raw_parquet.loc[mask].reset_index(drop=True)
    
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
    # 여기서 여전히 결측치가 존재, ppg와 acc가 매칭이 되지 않는 케이스가 분명 존재함
    # 하지만 wiener 필터를 지나면서 해당 부분이 에러문으로 넘어가길 바랬는데
    # Nan값이 그대로 넘어가고 있는 상황
    if aligned_df.isna().values.any():
        aligned_df.loc[:, ["acc_x", "acc_y", "acc_z"]] = (
            aligned_df.loc[:, ["acc_x", "acc_y", "acc_z"]]
            .astype(float)
            .ffill()
            .bfill()
        )
        print("Nan in acc, apply ffill, bfill")
    
    
    # transform struct, splited with windowsize (8 seconds)
    
    window_size = 8000 # ms
    
    t_min = int(aligned_df["timestamp"].min())
    t_max = int(aligned_df["timestamp"].max())
    
    bins = range(t_min, t_max + window_size, window_size)
    
    aligned_df["windowNumber"] = pd.cut(aligned_df["timestamp"], bins=bins, labels=False, right=False)
    groups = dict(tuple(aligned_df.groupby("windowNumber")))
    
    results = []
    for window_number, splited_df in tqdm(groups.items()):
        
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
    
    df = pd.DataFrame(results)
    
    # =============================================
    
    # 1. Bandpass Filtering + Denoising
    
    wiener = WienerDenoising()
    
    results = {
        "denoisedGalaxy": [None] * len(df),
        "estimated_BPM_Galaxy": [None] * len(df),
    }
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            galaxy_ppg = - np.array([float(x) for x in row["galaxyPPG"].split(";") if x.strip()])
            galaxy_acc = np.array([float(x) for x in row["galaxyACC"].split(";") if x.strip()]).reshape(-1, 3)
            
            galaxy_denoised, galaxy_bpm = wiener.process_galaxy(
                galaxy_ppg,
                galaxy_acc[:, 0],
                galaxy_acc[:, 1],
                galaxy_acc[:, 2]
            )
            
            results["denoisedGalaxy"][i] = ";".join(map(str, galaxy_denoised.tolist()))
            results["estimated_BPM_Galaxy"][i] = galaxy_bpm
        
        except Exception as e:
            print({str(e)})
            pass
    
    for col, values in results.items():
        df[col] = values
    
    subset_df = df[["timestamps", "denoisedGalaxy"]]
    
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
    
    # =============================================
    
    # 지금 앞서 ppg값이 비어 있는 케이스가 존재.
    
    
    
    # 2. SQA
    # 
    print("SQA")
    window_length_sec = 60*8 # 60초 * 8 = 8분
    sampling_rate = 25
    
    input_sig = splited_df["ppg"]
    
    clean_indices, noisy_indices = sqa(
        sig=input_sig,
        sampling_rate=sampling_rate,
        filter_signal=True
    )
    
    # 3. Reconstruction
    print("reconstruction")
    sig_reconstructed, clean_indices, noisy_indices = reconstruction(
        sig=input_sig,
        clean_indices=clean_indices,
        noisy_indices=noisy_indices,
        sampling_rate=sampling_rate,
        filter_signal=False
    )
    
    
    # 4. Segmentation
    print("segmentation")
    
    print(1)
    window_length = window_length_sec * sampling_rate
    
    clean_segments = clean_seg_extraction(
        sig = sig_reconstructed,
        noisy_indices=noisy_indices,
        window_length=window_length
    )
    
    print(1)
    
    #####
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
        return None
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )

        # Run PPG peak detection Using kazemi method
        # peaks = peak_detection(clean_segments=clean_segments, sampling_rate=sampling_rate, method=peak_detection_method)
        peaks = []
        
        for i in range(len(clean_segments)):
            inner_peaks = ppg_peaks(np.asarray(clean_segments[i][1]), sampling_rate, seconds=15, overlap=0, minlen=15)
            
            peaks.append(inner_peaks)
        
        print(1)
        

        # Perform HR and HRV extraction
        hrv_data = hrv_extraction(
            clean_segments=clean_segments,
            peaks=peaks,
            sampling_rate=sampling_rate,
            window_length=window_length)
        print("HR and HRV parameters:")
        print(hrv_data)
        print('Done!')

    print(1)

    
    
    pass

if __name__ == "__main__":
    main()