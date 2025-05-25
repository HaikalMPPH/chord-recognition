#!../venv/bin/python3

import os
import glob
import sys

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage

OUTPUT_FEATURES_CSV = 'dataset.csv'
ROOT_DIR = '../datasets'
PITCH_CLASS_LABELS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
CHORD_LABEL_COLUMN = 'chord'
SEGMENT_DURATION_SEC = 0.1

# --- Feature Extraction Functions ---

def featurize_audio_segment(chroma_segment):
    """
    Calculates the mean of chroma features over a segment.
    (Keeping mean as it's generally more suitable for continuous chroma values)
    """
    # If chroma_segment is a 1D array (a single frame), np.mean(axis=1) won't work.
    # It should always be 2D (12, N_frames) if it represents multiple frames.
    # If it's a single 100ms frame, its mean is just itself.
    if chroma_segment.ndim == 1:
        return chroma_segment
    return np.mean(chroma_segment, axis=1)

def featurize_file(audio_filename, label_filename, segment_duration_sec):
    """
    Extracts chroma features for an audio file and aligns them with
    labels from a corresponding .txt file.
    Now subdivides each chord's duration into 100ms (SEGMENT_DURATION_SEC)
    splits and extracts a feature vector for each of those sub-splits.

    Args:
        audio_filename (str): Path to the audio file (.mp3).
        label_filename (str): Path to the corresponding label file (.txt).
        segment_duration_sec (float): The desired duration of each chroma analysis segment (e.g., 0.1s).

    Returns:
        pd.DataFrame: DataFrame containing extracted features and aligned labels.
                      Returns an empty DataFrame if an error occurs or no labels.
    """
    try:
        # Load labels from the .txt file (no header, comma-separated)
        # Columns are assumed to be: start_seconds, end_seconds, chord_label
        df_labels = pd.read_csv(label_filename, header=None, names=['start_seconds', 'end_seconds', 'chord'])

        y, sr = librosa.load(audio_filename)

        # Calculate hop_length based on the desired segment duration
        hop_length_samples = int(segment_duration_sec * sr)
        if hop_length_samples < 1:
            hop_length_samples = 1
            print(f"Warning: Segment duration {segment_duration_sec}s too short for sample rate {sr}. Using hop_length=1.")

        chroma = 0
        if False: # used when testing
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length_samples)
        else:
            y_harm = librosa.effects.harmonic(y=y, margin=8)
            chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length_samples)
            chroma_filter = np.minimum(
                chroma_harm,
                librosa.decompose.nn_filter(chroma_harm, aggregate=np.median)
            )
            chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
            chroma = chroma_smooth

        # The effective frame rate is now explicitly defined by segment_duration_sec
        # It's better to calculate this from the hop_length directly for consistency
        chroma_frames_per_second = sr / hop_length_samples

        features_list = []
        labels_list = []

        # Iterate through each chord label entry
        for idx, seconds_start, seconds_end, label in df_labels.itertuples():
            # Convert label timestamps to overall chroma frame indices
            overall_chroma_start_idx = int(np.round(seconds_start * chroma_frames_per_second))
            overall_chroma_end_idx = int(np.round(seconds_end * chroma_frames_per_second))

            # Ensure indices are within bounds of the chroma array
            overall_chroma_start_idx = max(0, overall_chroma_start_idx)
            overall_chroma_end_idx = min(chroma.shape[1], overall_chroma_end_idx)

            if overall_chroma_end_idx <= overall_chroma_start_idx:
                print(f"Warning: Skipping label '{label}' ({seconds_start}-{seconds_end}s) in {audio_filename} due to invalid/empty overall chroma range.")
                continue

            # Iterate through each 100ms frame within the current labeled chord duration
            for frame_idx in range(overall_chroma_start_idx, overall_chroma_end_idx):
                # Each 'frame_idx' corresponds to a 100ms segment (because of hop_length)
                chroma_100ms_segment = chroma[:, frame_idx]

                # featurize_audio_segment will just return this 12-dim vector
                features_list.append(featurize_audio_segment(chroma_100ms_segment))
                labels_list.append(label) # Assign the chord label to this 100ms sub-segment

        # Create DataFrame from extracted features and labels
        df_features = pd.DataFrame(features_list, columns=PITCH_CLASS_LABELS)
        df_features[CHORD_LABEL_COLUMN] = labels_list
        return df_features

    except Exception as e:
        print(f"Error featurizing '{audio_filename}': {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Main Execution (remains the same) ---

def main():
    # Find audio files in the hardcoded ROOT_DIR
    audio_filenames = glob.glob(os.path.join(ROOT_DIR, '*.mp3'))
    audio_filenames.sort() # Sort to ensure consistent order

    if not audio_filenames:
        print(f"No .mp3 files found in '{ROOT_DIR}'. Please check the ROOT_DIR configuration.")
        sys.exit(1)

    output_filename = os.path.abspath(OUTPUT_FEATURES_CSV)

    # Generate corresponding label filenames (assuming .txt extension)
    label_filenames = [
        f'{os.path.splitext(audio_filename)[0]}.txt'
        for audio_filename in audio_filenames
    ]

    # Check for missing label files
    missing_label_files = [
        label_filename
        for label_filename in label_filenames
        if not os.path.exists(label_filename)
    ]
    if missing_label_files:
        print('Error: expected the following annotation files but they were not found:')
        for missing_file in missing_label_files:
            print(missing_file)
        sys.exit(1)

    print(f'Collected {len(audio_filenames)} audio files and their corresponding label files.')
    dataframes = []

    for audio_filename, label_filename in zip(audio_filenames, label_filenames):
        print(f'Featurizing "{os.path.basename(audio_filename)}"...', end='', flush=True)
        #df_file_features = featurize_file(audio_filename, label_filename)
        # TODO
        #   data augmentation: do another loop and from 0 to 3 use the other
        #   types of chroma (stft, harm, filter, smooth)
        df_file_features = featurize_file(audio_filename, label_filename, SEGMENT_DURATION_SEC)
        if not df_file_features.empty:
            dataframes.append(df_file_features)
            print(' Done')
        else:
            print(' Failed (see error above)')


    if not dataframes:
        print("No dataframes were successfully featurized. Output CSV will not be created.")
        sys.exit(1)

    df_master = pd.concat(dataframes, axis=0)
    # Save to CSV without index, with header, and formatted float values
    df_master.to_csv(output_filename, header=True, index=None, float_format='%.6f')
    print(f'\nSaved combined features and labels to {output_filename}')
    print(f'Total segments featurized: {len(df_master)}')
    print(f'Unique chord labels in dataset: {df_master[CHORD_LABEL_COLUMN].unique()}')

if __name__ == "__main__":
    main()
