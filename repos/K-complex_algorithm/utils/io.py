
import os
import mne
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from mne import create_info
from mne.io import RawArray

def load_raw_edf(edf_filename, sampling_freq, high_pass_filter_cutoff):
    """
    Load and preprocess an EDF file.

    Parameters:
    - edf_filename (str): The path to the EDF file to be loaded.
    - sampling_freq (float): The desired sampling frequency to resample the EEG data.
    - high_pass_filter_cutoff (float): The high-pass filter cutoff frequency.

    Returns:
    - raw (mne.io.Raw): The preprocessed raw EEG data.
    """
    # Load the original EDF file
    raw = mne.io.read_raw_edf(edf_filename, preload=True)
    # Set EEG reference
    # Data is already referenced in this NSRR file
    raw, _ = mne.set_eeg_reference(raw, [], verbose='warning')  
    # Resample the data
    raw.resample(sampling_freq)
    # Apply high-pass filter
    raw = raw.filter(high_pass_filter_cutoff, None)
    return raw

def create_edf_from_epochs(epoch_data, sfreq, ch_names, ch_types, filename):
    """
    Create an EDF file from epoch data.

    Parameters:
    - epoch_data (ndarray): The epoch data to be saved into the EDF file. Shape should be (n_channels, n_times).
    - sfreq (float): The sampling frequency of the EEG data.
    - ch_names (list of str): List of channel names.
    - ch_types (list of str): List of channel types (e.g., 'eeg', 'emg', etc.).
    - filename (str): The name of the output EDF file.

    Returns:
    - None
    """
    # Create MNE info structure
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    # Create a RawArray object
    raw = RawArray(epoch_data, info)
    # Export the data to an EDF file
    raw.export(filename, fmt='edf', overwrite=True)

def import_event_and_stages_SHHS(xml_file):
    """
    Import stages and events from an XML file of the Sleep Heart Health Study (SHHS).

    Parameters:
    - xml_file (str): The path to the XML file to be parsed.

    Returns:
    - events (pd.DataFrame): A dataframe with columns 'label', 'dur', and 'onset' for each event.
    - stages (pd.DataFrame): A dataframe with columns 'label', 'dur', and 'onset' for each sleep stage.
    """
    # Parse the XML string
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Initialize an empty list to store the stages
    stages = []
    events = []
    
    # Iterate over each ScoredEvent
    for event in root.findall("./ScoredEvents/ScoredEvent"):
        event_type = event.find('EventType').text
        event_concept = event.find('EventConcept').text
        start = float(event.find('Start').text)
        duration = float(event.find('Duration').text)
        
        # Check if the event concept indicates a sleep stage
        if event_type is not None and "Stages|Stages" in event_type:
            # Extract the sleep stage number from the event concept
            stage_number = event_concept.split('|')[1]
            stage = {
                'label': int(stage_number),
                'dur': duration,
                'onset': start
            }
            stages.append(stage)
        else:
            event = {
                'label': event_concept,
                'dur': duration,
                'onset': start
            }
            events.append(event)
    
    events = pd.DataFrame(events)
    stages = pd.DataFrame(stages)
    
    return events, stages

def get_total_recording_time(xml_annot_file):
    """
    Get the total recording time from an XML annotation file.

    Parameters:
    - xml_annot_file (str): The path to the XML annotation file.

    Returns:
    - duration (float): The total duration of the recording in seconds.
    """
    tree = ET.parse(xml_annot_file)
    root = tree.getroot()
    total_duration_event = root.findall("./ScoredEvents/ScoredEvent")[0]
    duration = float(total_duration_event.find('Duration').text)
    return duration

def store_kcs_data(base_dir, filename, onsets_valid, probas_valid, labels_valid, epochs_data_valid, onsets_invalid, probas_invalid, labels_invalid, epochs_data_invalid, sampling_freq):
    """
    Store K-complex (KC) data, including valid and invalid epochs, in specified directories and save them as EDF files.

    Parameters:
    - base_dir (str): The base directory where the data will be stored.
    - filename (str): The specific filename or folder name under the base directory.
    - onsets_valid (ndarray): The onset times of valid K-complexes.
    - probas_valid (ndarray): The probabilities associated with valid K-complexes.
    - labels_valid (ndarray): The labels for valid K-complexes.
    - epochs_data_valid (list of ndarray): The epoch data for valid K-complexes.
    - onsets_invalid (ndarray): The onset times of invalid K-complexes.
    - probas_invalid (ndarray): The probabilities associated with invalid K-complexes.
    - labels_invalid (ndarray): The labels for invalid K-complexes.
    - epochs_data_invalid (list of ndarray): The epoch data for invalid K-complexes.
    - sampling_freq (int): The sampling frequency

    Returns:
    - None
    """
    # Create directories for valid and invalid K-complexes
    os.makedirs(os.path.join(base_dir, filename, 'valid_kcs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, filename, 'invalid_kcs'), exist_ok=True)
    
    valid_kcs_dir = os.path.join(base_dir, filename, 'valid_kcs', 'edfs')
    invalid_kcs_dir = os.path.join(base_dir, filename, 'invalid_kcs', 'edfs')

    os.makedirs(valid_kcs_dir, exist_ok=True)
    os.makedirs(invalid_kcs_dir, exist_ok=True)

    # Save valid data
    np.save(os.path.join(base_dir, filename, 'valid_kcs', 'labels_valid.npy'), labels_valid)
    np.save(os.path.join(base_dir, filename, 'valid_kcs', 'onsets_valid.npy'), onsets_valid)
    np.save(os.path.join(base_dir, filename, 'valid_kcs', 'probas_valid.npy'), probas_valid)

    # Save invalid data
    np.save(os.path.join(base_dir, filename, 'invalid_kcs', 'labels_invalid.npy'), labels_invalid)
    np.save(os.path.join(base_dir, filename, 'invalid_kcs', 'onsets_invalid.npy'), onsets_invalid)
    np.save(os.path.join(base_dir, filename, 'invalid_kcs', 'probas_invalid.npy'), probas_invalid)

    # Save valid epochs as EDF files
    for i, epoch_data in enumerate(epochs_data_valid):
        edf_filename = os.path.join(valid_kcs_dir, f'kc_{i}.edf')
        create_edf_from_epochs(epoch_data.reshape(1, -1), sampling_freq, ['EEG'], ['eeg'], edf_filename)
    print(f"Saved {len(epochs_data_valid)} valid K-complex epochs as EDF files in {valid_kcs_dir}")

    # Save invalid epochs as EDF files
    for i, epoch_data in enumerate(epochs_data_invalid):
        edf_filename = os.path.join(invalid_kcs_dir, f'kc_{i}.edf')
        create_edf_from_epochs(epoch_data.reshape(1, -1), sampling_freq, ['EEG'], ['eeg'], edf_filename)
    print(f"Saved {len(epochs_data_invalid)} invalid K-complex epochs as EDF files in {invalid_kcs_dir}")
    
    
def load_kcs_edf_files(edf_dir):
    """
    Load all EDF files from the specified directory and reconstruct them into the format of valid_epochs_data.

    Parameters:
    - edf_dir (str): The directory containing the EDF files to be loaded.

    Returns:
    - all_epoch_data (ndarray): A numpy array containing the reconstructed epoch data from the EDF files.
    """
    # Get the list of EDF files in the directory and sort them
    edf_files = sorted([os.path.join(edf_dir, f) for f in os.listdir(edf_dir) if f.endswith('.edf')])

    # Initialize a list to store the epoch data
    all_epoch_data = []

    # Load each EDF file and extract the data
    for edf_file in edf_files:
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        data, times = raw[:]
        all_epoch_data.append(data)

    # Convert the list of epoch data to a numpy array and squeeze it
    all_epoch_data = np.array(all_epoch_data).squeeze()

    return all_epoch_data

def load_invalid_kc_metadata(kc_dir):
    """
    Load the metadata for invalid K-complexes (KCs) from the specified directory.

    Parameters:
    - kc_dir (str): The base directory containing the K-complex metadata.

    Returns:
    - labels (ndarray): The labels for the invalid K-complexes.
    - onsets (ndarray): The onset times of the invalid K-complexes.
    - probas (ndarray): The probabilities associated with the invalid K-complexes.
    """
    invalid_kc_dir = os.path.join(kc_dir, 'invalid_kcs')
    
    
    # Load the stored inference files for invalid KCs
    labels = np.load(os.path.join(invalid_kc_dir, 'labels_invalid.npy'))
    onsets = np.load(os.path.join(invalid_kc_dir, 'onsets_invalid.npy'))
    probas = np.load(os.path.join(invalid_kc_dir, 'probas_invalid.npy'))
    
    return labels, onsets, probas

def load_valid_kc_metadata(kc_dir):
    """
    Load the metadata for valid K-complexes (KCs) from the specified directory.

    Parameters:
    - kc_dir (str): The base directory containing the K-complex metadata.

    Returns:
    - labels (ndarray): The labels for the valid K-complexes.
    - onsets (ndarray): The onset times of the valid K-complexes.
    - probas (ndarray): The probabilities associated with the valid K-complexes.
    """
    valid_kc_dir = os.path.join(kc_dir, 'valid_kcs')

    # Load the stored inference files for valid KCs
    labels = np.load(os.path.join(valid_kc_dir, 'labels_valid.npy'))
    onsets = np.load(os.path.join(valid_kc_dir, 'onsets_valid.npy'))
    probas = np.load(os.path.join(valid_kc_dir, 'probas_valid.npy'))
    
    return labels, onsets, probas