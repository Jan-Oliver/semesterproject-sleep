import numpy as np
from KC_algorithm.utils import EpochData

def remove_steady_epochs(peaks, data, pre_peak, post_peak, Fs, threshold=0.5):
    """
    Remove epochs with steady data (low variance) around the detected peaks.

    Parameters:
    - peaks (ndarray): The onset times of the detected K-complexes.
    - data (ndarray): The EEG data.
    - pre_peak (float): The duration in seconds to include before the KC peak.
    - post_peak (float): The duration in seconds to include after the KC peak.
    - Fs (float): The sampling frequency of the EEG data.
    - threshold (float): The variance threshold below which an epoch is considered steady. Defaults to 0.5.

    Returns:
    - valid_peaks (list): The peaks that are not considered steady.
    - invalid_peaks (list): The peaks that are considered steady.
    - valid_indices (list): The indices of valid peaks.
    - invalid_indices (list): The indices of invalid peaks.
    """
    d = EpochData(peaks, data, post_peak, pre_peak, Fs)
    d = d * 10**6  # Convert to microvolts
    Fs = int(Fs)
    valid_indices = []
    invalid_indices = []
    window_size = int(0.05 * Fs)  # 50 ms window
    mean_squared = np.mean(window_size ** 2)
    
    for i in range(len(d)):
        epoch = d[i]
        # Create a 2D array of overlapping windows of 50 ms
        windows = np.lib.stride_tricks.sliding_window_view(epoch, window_size)
        # Calculate the variance for each window
        variances = np.var(windows, axis=1) / mean_squared
        # Check if any window has variance below the threshold
        if np.any(variances < threshold):
            invalid_indices.append(i)
        else:
            valid_indices.append(i)
    
    valid_peaks = [peaks[i] for i in valid_indices]
    invalid_peaks = [peaks[i] for i in invalid_indices]
    
    return valid_peaks, invalid_peaks, valid_indices, invalid_indices


def remove_standard_deviation_outliers(peaks, data, pre_peak, post_peak, Fs, std_threshold=2):
    """
    Remove all epochs that are 2 standard deviations away from the mean somewhere.

    Parameters:
    - peaks (ndarray): The onset times of the detected K-complexes.
    - data (ndarray): The EEG data.
    - pre_peak (float): The duration in seconds to include before the KC peak.
    - post_peak (float): The duration in seconds to include after the KC peak.
    - Fs (float): The sampling frequency of the EEG data.
    - std_threshold (float): The threshold in standard deviations to consider an epoch as an outlier. Defaults to 2.

    Returns:
    - valid_peaks (list): The peaks that are not considered outliers.
    - invalid_peaks (list): The peaks that are considered outliers.
    - valid_indices (list): The indices of valid peaks.
    - invalid_indices (list): The indices of invalid peaks.
    """
    d = EpochData(peaks, data, post_peak, pre_peak, Fs) * 10**6  # Convert to microvolts
    mean_epoch = np.mean(d, axis=0)
    std_epoch = np.std(d, axis=0)
    
    valid_indices = []
    invalid_indices = []
    
    for i in range(len(d)):
        epoch = d[i]
        if np.any(np.abs(epoch - mean_epoch) > std_threshold * std_epoch):
            invalid_indices.append(i)
        else:
            valid_indices.append(i)
    
    valid_peaks = [peaks[i] for i in valid_indices]
    invalid_peaks = [peaks[i] for i in invalid_indices]
    
    return valid_peaks, invalid_peaks, valid_indices, invalid_indices