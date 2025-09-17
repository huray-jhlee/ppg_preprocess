# -*- coding: utf-8 -*-

import warnings
from utils import get_data
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction
import neurokit2 as nk
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

def hrv_parameters(
        peaks: np.ndarray,
        seg_start_idx: int,
        sampling_rate: int,
        window_length: int) -> pd.DataFrame:
    """
    Calculate HR and HRV parameters from a list of peak indices.

    Args:
        peaks (np.ndarray): Peak indices for a segment.
        seg_start_idx (int): Starting index of a segemnt.
        sampling_rate (int): Sampling rate of the signal.
        window_length (int): Desired window length for HR and HRV extraction in terms of samples.

    Return:
        HRV_indices (pandas.DataFrame): DataFrame containing HRV parameters.
        
    Reference:
        Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Chen, S. A. (2021).
        NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior research methods, 1-8.
    """
    
    # Calculate window length in minutes
    window_length_min = (window_length/sampling_rate)/60    
    
    # Calculate heart rate
    heart_rate = (len(peaks)/window_length_min)
    
    # Initialize a list to store Hr and HRV parameters
    HRV_indices = []
    
    # Compute time-domain HRV parameters
    hrv_times = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
    hrv_times = hrv_times[['HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI']]
    HRV_indices.append(hrv_times)

    # Computes frequency-domain HRV parameters
    hrv_freqs = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False, psd_method="welch")
    HRV_indices.append(hrv_freqs)
    
    # Computes nonlinear indices HRV parameters
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
    hrv_nonlinear = hrv_nonlinear[['HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S']]
    HRV_indices.append(hrv_nonlinear)

    # Concatenate HRV parameters into a DataFrame
    HRV_indices = pd.concat(HRV_indices, axis=1)
    
    # Add segment starting index and heart rate to the DataFrame
    HRV_indices.insert(0, 'Start_idx', seg_start_idx)
    HRV_indices.insert(1, 'HR', heart_rate)
    
    return HRV_indices



def hrv_extraction(
        clean_segments: list,
        peaks: list,
        sampling_rate: int,
        window_length: int) -> pd.DataFrame:
    
    """
    Calculate HR and HRV parameters from clean segments peak indices.

    Args:
        clean_segments (list): List of clean PPG segments and their starting index.
        peaks (list): List of lists, each containing the detected peaks for a corresponding clean segment.
        sampling_rate (int): Sampling rate of the PPG signal.
        window_length (int): Desired window length for HR and HRV extraction in terms of samples.

    Return:
    - hrv_data (pandas.DataFrame): DataFrame containing HRV parameters for all clean segments.
    """
    
    # Iterate through the segments to extract HRV parameters
    for i in range(len(peaks)):
        try:
            # Attempt to calculate HRV parameters for the current segment
            HRV_indices = hrv_parameters(peaks=peaks[i], seg_start_idx=clean_segments[i][0], sampling_rate=sampling_rate, window_length=window_length)
        except:
            # Handle exceptions and continue to the next iteration
            if i == 0:
                # If it's the first iteration and an exception occurred, initialize an empty DataFrame
                hrv_data = pd.DataFrame(columns=['Start_idx', 'HR', 'HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI','ULF','VLF','LF','HF','VHF','LFHF','LFn','HFn','LnHF','HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S'])                    
            continue
        
        if i == 0:
            # If it's the first iteration, assign the HRV parameters to the DataFrame
            hrv_data = HRV_indices
        else:
            # Append HRV parameters to the DataFrame
            hrv_data.loc[len(hrv_data.index)] = HRV_indices.values.tolist()[0]    
    return hrv_data


