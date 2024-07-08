import numpy as np

def calculate_kcs_per_minute(onsets, sampling_freq, total_duration):
    """
    Calculate the number of K-complexes (KCs) per minute over the total recording duration.

    Parameters:
    - onsets (ndarray): The onset times of the detected K-complexes.
    - sampling_freq (float): The sampling frequency of the EEG data.
    - total_duration (float): The total duration of the recording in seconds.

    Returns:
    - minutes (ndarray): The time points in minutes.
    - kcs_per_minute (ndarray): The number of KCs per minute at each time point.
    """
    minutes = np.arange(0, total_duration / 60, 1)
    kcs_per_minute = np.zeros(len(minutes))

    for onset in onsets:
        minute_index = int(onset / (sampling_freq * 60))
        if minute_index < len(kcs_per_minute):
            kcs_per_minute[minute_index] += 1
        else:
            print("minute index too big")

    return minutes, kcs_per_minute


def filter_kcs_by_sleep_stages(onsets, stages, valid_stages, sampling_freq):
    """
    Filter K-complex (KC) onsets by sleep stages.

    Parameters:
    - onsets (ndarray): The onset times of the detected K-complexes.
    - stages (pd.DataFrame): Dataframe with columns 'label', 'dur', and 'onset' for each sleep stage.
    - valid_stages (list): List of valid sleep stages for K-complex detection.
    - sampling_freq (float): The sampling frequency of the EEG data.

    Returns:
    - filtered_onsets (ndarray): The onsets of K-complexes within valid sleep stages.
    - filtered_out_onsets (ndarray): The onsets of K-complexes outside valid sleep stages.
    """
    filtered_onsets = []
    filtered_out_onsets = []
    for onset in onsets:
        onset_time = onset / sampling_freq  # Convert sample index to seconds
        is_valid = False
        for _, row in stages.iterrows():
            if row['label'] in valid_stages:
                stage_start = row['onset']
                stage_end = row['onset'] + row['dur']
                if stage_start <= onset_time <= stage_end:
                    filtered_onsets.append(onset)
                    is_valid = True
                    break
        if not is_valid:
            filtered_out_onsets.append(onset)
    return np.array(filtered_onsets), np.array(filtered_out_onsets)