#!../venv/bin/python3
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import sys
import scipy.ndimage # For median_filter if using harmonic chroma
from collections import Counter
# from collections import Counter # Not needed without smoothing

# --- Configuration ---
# Make sure these paths are correct for your saved files
MODEL_PATH = 'model_rf.xz' # Path to your trained model
ENCODER_PATH = 'encoder.xz'   # Path to your saved LabelEncoder

SEGMENT_DURATION_SEC = 0.1                    # 100ms - MUST match training segment duration
INPUT_FILE = '../datasets/1.mp3'              # Your example input audio file path
#INPUT_FILE = '/home/haikal-mpph/Music/elijah-fox-ontario.mp3'              # Your example input audio file path


# --- Feature Extraction Function ---
def featurize(audio_filename, segment_duration_sec):
    """
    Extracts chroma features for an audio file at specified segment durations.
    This function mirrors the feature extraction used during training.

    Args:
        audio_filename (str): Path to the audio file.
        segment_duration_sec (float): The desired duration of each chroma analysis segment (e.g., 0.1s).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Extracted chroma feature vectors (shape: N_segments, 12).
            - np.ndarray: Corresponding start times for each segment.
            - np.ndarray: Corresponding end times for each segment.
            Returns empty arrays if an error occurs.
    """
    try:
        y, sr = librosa.load(audio_filename)
        file_duration = librosa.get_duration(y=y, sr=sr)

        hop_length = int(segment_duration_sec * sr)
        if hop_length < 1:
            hop_length = 1
            print(f"Warning: Segment duration {segment_duration_sec}s too short for sample rate {sr}. Using hop_length=1.")

        # --- IMPORTANT: Ensure this chroma calculation matches your training script ---
        # This part should be identical to the 'chroma = ...' logic in your featurize_file
        chroma = 0
        if False: # Change this to True if you used chroma_stft during training
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        else: # This path uses harmonic chroma with median filtering (as in your last featurize_file)
            y_harm = librosa.effects.harmonic(y=y, margin=8)
            chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
            chroma_filter = np.minimum(
                chroma_harm,
                librosa.decompose.nn_filter(chroma_harm, aggregate=np.median)
            )
            chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
            chroma = chroma_smooth
        # If you used chroma_cens, uncomment the line below and comment out the if/else block above:
        # chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        # --- END IMPORTANT SECTION ---

        # Each column in 'chroma' is a feature vector for a 'segment_duration_sec' interval
        # Transpose to get (N_segments, 12)
        features = chroma.T

        # Calculate start and end times for each 100ms segment
        num_segments = features.shape[0]
        start_times = np.arange(0, num_segments) * segment_duration_sec
        end_times = np.minimum(start_times + segment_duration_sec, file_duration)

        return features, start_times, end_times

    except Exception as e:
        print(f"Error processing audio '{audio_filename}': {e}")
        return np.array([]), np.array([]), np.array([])


# --- Merging Consecutive Identical Chords ---
def merge_consecutive_chords(df_predictions):
    """
    Merges consecutive rows in a DataFrame if their 'predicted_chord' is the same.
    Adjusts 'start_seconds' and 'end_seconds' accordingly.

    Args:
        df_predictions (pd.DataFrame): DataFrame with 'start_seconds', 'end_seconds', 'predicted_chord'.

    Returns:
        pd.DataFrame: A new DataFrame with merged chord segments.
    """
    if df_predictions.empty:
        return pd.DataFrame(columns=['start_seconds', 'end_seconds', 'predicted_chord'])

    merged_data = []
    current_chord = None
    current_start_time = None
    current_end_time = None

    for index, row in df_predictions.iterrows():
        chord = row['predicted_chord']
        start_time = row['start_seconds']
        end_time = row['end_seconds']

        if current_chord is None: # First segment
            current_chord = chord
            current_start_time = start_time
            current_end_time = end_time
        elif chord == current_chord: # Same chord, extend duration
            current_end_time = end_time
        else: # Different chord, save previous and start new
            merged_data.append({
                'start_seconds': current_start_time,
                'end_seconds': current_end_time,
                'predicted_chord': current_chord
            })
            current_chord = chord
            current_start_time = start_time
            current_end_time = end_time

    # Add the last accumulated segment after the loop
    if current_chord is not None:
        merged_data.append({
            'start_seconds': current_start_time,
            'end_seconds': current_end_time,
            'predicted_chord': current_chord
        })

    return pd.DataFrame(merged_data)


# --- Prediction Function ---
def predict():
    """
    Loads model and encoder, processes the input audio file,
    predicts chords, merges consecutive identical chords,
    and prints the results to the console.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input audio file not found at '{INPUT_FILE}'")
        sys.exit(1)

    # 1. Load the model and encoder
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Model and LabelEncoder loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model or encoder file not found. Make sure '{MODEL_PATH}' and '{ENCODER_PATH}' exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
        sys.exit(1)

    # 2-3. Process the audio file (featurize)
    features, start_times, end_times = featurize(INPUT_FILE, SEGMENT_DURATION_SEC)

    if features.size == 0:
        print("No features extracted from audio. Exiting prediction.")
        return

    # Create a DataFrame for raw predictions for easier processing
    raw_predictions_df = pd.DataFrame({
        'start_seconds': start_times,
        'end_seconds': end_times,
        'features': list(features) # Storing features temporarily if needed, though not directly for output
    })

    print(f"Extracted {len(raw_predictions_df)} raw 100ms segments. Making predictions...")

    # Perform predictions (model expects 2D array, features is already (N_segments, 12))
    numerical_predictions = model.predict(features)
    predicted_chords_decoded = encoder.inverse_transform(numerical_predictions)

    raw_predictions_df['predicted_chord'] = predicted_chords_decoded

    # --- Apply Merging of Consecutive Chords ---
    # Since smoothing is removed, we directly merge the raw predictions
    final_predictions_df = merge_consecutive_chords(raw_predictions_df)
    print(f"Original {len(raw_predictions_df)} raw segments processed.")
    print(f"Final output contains {len(final_predictions_df)} merged chord segments.")

    # 5. Output the results to the console
    print("\n--- Predicted Chords ---")
    for index, row in final_predictions_df.iterrows():
        # Using f-string for formatted output (e.g., 0.000 - 0.500: Cmaj)
        print(f"{row['start_seconds']:.3f} - {row['end_seconds']:.3f}: {row['predicted_chord']}")
    print("------------------------")

# --- Main execution point ---
predict()




