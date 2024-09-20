import numpy as np
import matplotlib.pyplot as plt
from mne.stats.parametric import _parametric_ci
from KC_algorithm.utils import EpochData

def plot_mean_kc_with_std(epoch_data, Fs, post_peak, pre_peak, title="Mean K-Complex Signal"):
    """
    Plot the mean K-complex signal with 1 and 2 standard deviation intervals.

    Parameters:
    - epoch_data (ndarray): The epoch data containing the EEG signal. Shape should be (n_epochs, n_times).
    - Fs (float): The sampling frequency of the EEG data.
    - post_peak (float): Duration in seconds to include after the KC peak.
    - pre_peak (float): Duration in seconds to include before the KC peak.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    total_peak = post_peak + pre_peak
    half_peak = total_peak / 2.0

    # Convert to microvolts
    epoch_data_uv = epoch_data * 10**6

    # Calculate mean and standard deviation
    mean_kc = np.mean(epoch_data_uv, axis=0)
    std_kc = np.std(epoch_data_uv, axis=0)

    times = np.arange(0, total_peak, total_peak / len(mean_kc)) - half_peak

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean KC
    ax.plot(times, mean_kc, color='blue', label='Mean KC')

    # Plot 1 std interval
    ax.fill_between(times, mean_kc - std_kc, mean_kc + std_kc, color='blue', alpha=0.3, label='1 Std Dev')

    # Plot 2 std interval
    ax.fill_between(times, mean_kc - 2*std_kc, mean_kc + 2*std_kc, color='blue', alpha=0.1, label='2 Std Dev')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    
def KC_from_probas_epoch_data(epoch_data, 
                              onsets,
                              probas,
                              Fs, 
                              post_peak, 
                              pre_peak):
    """
    Plot average K-complexes (KCs) at time onsets for different probability thresholds.

    Parameters:
    - epoch_data (ndarray): The epoch data containing the EEG signal. Shape should be (n_epochs, n_times).
    - onsets (ndarray): The onset times of the detected K-complexes.
    - probas (ndarray): The probabilities associated with the detected K-complexes.
    - Fs (float): The sampling frequency of the EEG data.
    - post_peak (float): Duration in seconds to include after the KC peak. Default is 1.5 seconds.
    - pre_peak (float): Duration in seconds to include before the KC peak. Default is 1.5 seconds.

    Returns:
    - None
    """
    
    total_peak = post_peak + pre_peak
    half_peak = total_peak / 2.0

    colors = ['#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]


    for count,th in enumerate(thresholds):

        indexes_th = np.nonzero(np.bitwise_and(probas>th,probas<1.0))
        kc_onset_ths = onsets[indexes_th]
        epoch_data_ths = epoch_data[indexes_th]* 10**6

        times = np.arange(0, total_peak, total_peak / len(epoch_data_ths[0, :])) - half_peak

        ci_ = _parametric_ci(epoch_data_ths)
        av = epoch_data_ths.mean(axis=0)

        upper_bound = ci_[0].flatten()
        lower_bound = ci_[1].flatten()
        plt.plot(times, av, color=colors[count], label=str(th))
        plt.fill_between(times, upper_bound, lower_bound,
                         zorder=9, color=colors[count], alpha=.2,
                         clip_on=False)
    plt.legend()
    plt.show()

def plot_Kcs_in_single_chart(C3, 
                             onsets, 
                             probas, 
                             Fs, 
                             post_peak, 
                             pre_peak, 
                             ax=None, 
                             num_samples=500):
    """
    Plot individual K-complexes (at time onsets) for different probability thresholds and their averages.

    Parameters:
    - C3 (ndarray): The EEG data from the C3 channel.
    - onsets (ndarray): The onset times of the detected K-complexes.
    - probas (ndarray): The probabilities of the detected K-complexes.
    - Fs (float): The sampling frequency of the EEG data.
    - post_peak (float): The duration in seconds to include after the KC peak.
    - pre_peak (float): The duration in seconds to include before the KC peak.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): The axes to plot on. Creates new figure and axes if None.
    - num_samples (int, optional): The number of samples to plot. Defaults to 500.

    Returns:
    - None
    """
    total_peak = post_peak + pre_peak
    half_peak = total_peak / 2.0

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    kc_onset_ths = onsets

    # Randomly select a subset of samples
    if len(kc_onset_ths) > num_samples:
        selected_indices = np.random.choice(len(kc_onset_ths), num_samples, replace=False)
        kc_onset_ths = kc_onset_ths[selected_indices]

    d = EpochData(kc_onset_ths, C3, post_peak, pre_peak, Fs) * 10**6
    times = np.arange(0, total_peak, total_peak / len(d[0, :])) - half_peak

    # Plot each individual K-complex
    for i in range(d.shape[0]):
        ax.plot(times, d[i, :], color=colors[0], alpha=0.2)

    # Calculate and plot the average with confidence intervals
    av = d.mean(axis=0)
    ci_ = _parametric_ci(d)
    upper_bound = ci_[0].flatten()
    lower_bound = ci_[1].flatten()

    # Make the average line more visible
    ax.plot(times, av, color='black', linewidth=3, label='Average')
    ax.fill_between(times, upper_bound, lower_bound, color='black', alpha=0.1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
        
def plot_Kcs_in_single_chart_epoch_data(epoch_data, 
                                        onsets, 
                                        probas, 
                                        Fs, 
                                        post_peak, 
                                        pre_peak, 
                                        ax=None, 
                                        num_samples=500):
    """
    Plot individual K-complexes (at time onsets) for different probability thresholds and their averages.

    Parameters:
    - epoch_data (ndarray): The data epochs to be plotted.
    - onsets (ndarray): The onset times of the detected K-complexes.
    - probas (ndarray): The probabilities of the detected K-complexes.
    - Fs (float): The sampling frequency of the EEG data.
    - post_peak (float): The duration in seconds to include after the KC peak.
    - pre_peak (float): The duration in seconds to include before the KC peak.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): The axes to plot on. Creates new figure and axes if None.
    - num_samples (int, optional): The number of samples to plot. Defaults to 500.

    Returns:
    - None
    """
    total_peak = post_peak + pre_peak
    half_peak = total_peak / 2.0

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    kc_onset_ths = onsets

    # Randomly select a subset of samples
    if len(kc_onset_ths) > num_samples:
        selected_indices = np.random.choice(len(kc_onset_ths), num_samples, replace=False)
        kc_onset_ths = kc_onset_ths[selected_indices]

    d = epoch_data * 10**6
    times = np.arange(0, total_peak, total_peak / len(d[0, :])) - half_peak

    # Plot each individual K-complex
    for i in range(d.shape[0]):
        ax.plot(times, d[i, :], color=colors[0], alpha=0.2)

    # Calculate and plot the average with confidence intervals
    av = d.mean(axis=0)
    ci_ = _parametric_ci(d)
    upper_bound = ci_[0].flatten()
    lower_bound = ci_[1].flatten()

    # Make the average line more visible
    ax.plot(times, av, color='black', linewidth=3, label='Average')
    ax.fill_between(times, upper_bound, lower_bound, color='black', alpha=0.1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
        
def plot_kcs_per_minute_and_sleep_stages(minutes, kcs_per_minute, stages, n2_n3_same_color=True):
    """
    Plot the number of K-complexes (KCs) per minute and sleep stages over time.

    Parameters:
    - minutes (ndarray): The time points in minutes.
    - kcs_per_minute (ndarray): The number of KCs per minute at each time point.
    - stages (pd.DataFrame): Dataframe with columns 'label', 'dur', and 'onset' for each sleep stage.

    Returns:
    - None
    """
    if n2_n3_same_color:
        colors = {0: 'lightblue', 1: 'lightgreen', 2: 'lightsalmon', 3: 'lightsalmon', 5: 'lightyellow'}
    else:
        colors = {0: 'lightblue', 1: 'lightgreen', 2: 'lightsalmon', 3: 'lightcoral', 5: 'lightyellow'}

    stage_labels = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 5: 'REM'}

    fig, ax = plt.subplots(figsize=(30, 15))
    ax.plot(minutes, kcs_per_minute, marker='o', label='KCs per Minute')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('KCs per minute')
    ax.set_title('K-Complexes per Minute Over Time')
    ax.grid(True)

    for _, row in stages.iterrows():
        stage_start = row['onset'] / 60  # Convert seconds to minutes
        stage_end = (row['onset'] + row['dur']) / 60  # Convert seconds to minutes
        ax.axvspan(stage_start, stage_end, color=colors.get(row['label'], 'gray'), alpha=0.3, label=stage_labels.get(row['label'], 'Unknown'))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()