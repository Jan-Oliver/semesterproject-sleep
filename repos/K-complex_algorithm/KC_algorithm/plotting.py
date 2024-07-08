#Copyright (C) 2020 Bastien Lechat

#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
#published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
#of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.


import numpy as np
import matplotlib.pyplot as plt
from KC_algorithm.utils import EpochData
from mne.stats.parametric import _parametric_ci

def KC_from_probas(C3,onsets,probas,Fs):
    """
    PLot average K-complexes (at time onsets) for different probability thresholds
    """

    post_peak = 1.5  # [s]
    pre_peak = 1.5  # [s]

    colors = ['#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]


    for count,th in enumerate(thresholds):

        indexes_th = np.nonzero(np.bitwise_and(probas>th,probas<1.0))
        kc_onset_ths = onsets[indexes_th]

        d = EpochData(kc_onset_ths, C3, post_peak, pre_peak, Fs) * 10**6
        times = np.arange(0, 3, 3 / len(d[0,:])) - 1.5

        ci_ = _parametric_ci(d)
        av = d.mean(axis=0)

        upper_bound = ci_[0].flatten()
        lower_bound = ci_[1].flatten()
        plt.plot(times, av, color=colors[count], label=str(th))
        plt.fill_between(times, upper_bound, lower_bound,
                         zorder=9, color=colors[count], alpha=.2,
                         clip_on=False)
    plt.legend()
    plt.show()

def plot_all_Kcs(C3, onsets, probas, Fs, num_samples=100):
    """
    Plot individual K-complexes (at time onsets) for different probability thresholds and their averages.
    """

    post_peak = 1.5  # [s]
    pre_peak = 1.5  # [s]

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, axs = plt.subplots(len(thresholds) - 1, 1, figsize=(15, 5 * (len(thresholds) - 1)))

    for count, th in enumerate(thresholds[:-1]):
        next_th = thresholds[count + 1]
        indexes_th = np.nonzero(np.bitwise_and(probas > th, probas <= next_th))[0]
        kc_onset_ths = onsets[indexes_th]

        if len(kc_onset_ths) == 0:
            continue

        # Randomly select a subset of samples
        if len(kc_onset_ths) > num_samples:
            selected_indices = np.random.choice(len(kc_onset_ths), num_samples, replace=False)
            kc_onset_ths = kc_onset_ths[selected_indices]

        d = EpochData(kc_onset_ths, C3, post_peak, pre_peak, Fs) * 10**6
        times = np.arange(0, 3, 3 / len(d[0, :])) - 1.5

        # Plot each individual K-complex
        for i in range(d.shape[0]):
            axs[count].plot(times, d[i, :], color=colors[count], alpha=0.2)

        # Calculate and plot the average with confidence intervals
        av = d.mean(axis=0)
        ci_ = _parametric_ci(d)
        upper_bound = ci_[0].flatten()
        lower_bound = ci_[1].flatten()

        # Make the average line more visible
        axs[count].plot(times, av, color='black', linewidth=3, label='Average')
        axs[count].fill_between(times, upper_bound, lower_bound, color='black', alpha=0.1)

        axs[count].set_title(f'Probability Threshold: {th:.1f} - {next_th:.1f}')
        axs[count].set_xlabel('Time (s)')
        axs[count].set_ylabel('Amplitude (µV)')
        axs[count].legend()

    plt.tight_layout()
    plt.show()
    
def plot_Kcs_in_single_chart(C3, onsets, probas, Fs, num_samples=100):
    """
    Plot individual K-complexes (at time onsets) for different probability thresholds and their averages.
    """

    post_peak = 1.5  # [s]
    pre_peak = 1.5  # [s]

    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#e6ab02', '#66a61e', '#e41a1c', '#377eb8', '#ff7f00']

    fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    kc_onset_ths = onsets

    # Randomly select a subset of samples
    if len(kc_onset_ths) > num_samples:
        selected_indices = np.random.choice(len(kc_onset_ths), num_samples, replace=False)
        kc_onset_ths = kc_onset_ths[selected_indices]

    d = EpochData(kc_onset_ths, C3, post_peak, pre_peak, Fs) * 10**6
    times = np.arange(0, 3, 3 / len(d[0, :])) - 1.5

    # Plot each individual K-complex
    for i in range(d.shape[0]):
        axs[count].plot(times, d[i, :], color=colors[count], alpha=0.2)

    # Calculate and plot the average with confidence intervals
    av = d.mean(axis=0)
    ci_ = _parametric_ci(d)
    upper_bound = ci_[0].flatten()
    lower_bound = ci_[1].flatten()

    # Make the average line more visible
    axs[count].plot(times, av, color='black', linewidth=3, label='Average')
    axs[count].fill_between(times, upper_bound, lower_bound, color='black', alpha=0.1)

    axs[count].set_title(f'Probability Threshold: {th:.1f} - {next_th:.1f}')
    axs[count].set_xlabel('Time (s)')
    axs[count].set_ylabel('Amplitude (µV)')
    axs[count].legend()

    plt.tight_layout()
    plt.show()