#!../Venv/bin/python

import os
import glob
import sys
import re
import gc

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage

OUTPUT_FILE = 'dataset.h5'
ROOT_DIR = '../Datasets'
CQT_COL = ['Cqt_C', 'Cqt_Db', 'Cqt_D', 'Cqt_Eb', 'Cqt_E', 'Cqt_F', 'Cqt_Gb', 'Cqt_G', 'Cqt_Ab', 'Cqt_A', 'Cqt_Bb', 'Cqt_B']
CENS_COL = ['Cens_C', 'Cens_Db', 'Cens_D', 'Cens_Eb', 'Cens_E', 'Cens_F', 'Cens_Gb', 'Cens_G', 'Cens_Ab', 'Cens_A', 'Cens_Bb', 'Cens_B']
SEGMENT_DURATION_SEC = 0.1 # 100 ms split

if __name__ == "__main__":
  print(":::::::::::::::::::: PREPARING ::::::::::::::::::::")

  # Input audio files
  in_files = glob.glob(os.path.join(ROOT_DIR, "*.mp3"))
  in_files.sort()
  if not in_files:
    print(f"[ERRO]: Cannot find audio files in {ROOT_DIR}")
    sys.exit(sys.EXIT_FAILURE)

  # The corresponding label files of the input audio files
  in_label_files = [f"{os.path.splitext(in_file)[0]}.txt" for in_file in in_files]

  # Output dataset CSV file
  out_files = os.path.abspath(OUTPUT_FILE)

  # If some corresponding label files is missing
  in_missing_label_files = [in_label_file for in_label_file in in_label_files if not os.path.exists(in_label_file)]
  if in_missing_label_files:
    print(f"[ERRO]: Missing corresponding label .txt files:")
    for file in in_missing_label_files:
      print(f"  {f}")
    sys.exit(sys.EXIT_FAILURE)
  
  # Featurizing the file
  dataset = []
  print(f"[INFO]: Featurizing {len(in_files)} files: ")
  for in_file, in_label_file in zip(in_files, in_label_files):
    print(f"  Featurizing {os.path.basename(in_file)} -> ", end='', flush=True)

    # Chord timestamps
    df_label = pd.read_csv(in_label_file, header=None, names=["start", "end", "chord"])

    # Loading the audio files
    y, sr = librosa.load(in_file)
    hop_length = int(SEGMENT_DURATION_SEC * sr)
     
    # :::::::: Data augmentation :::::::::
    augmented_y_and_labels = []
    pitches = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    # Function for changing the root after shifting
    def get_new_chord(label, shift):
      match = re.match(r"([A-G]b?)(.*)", label)
      root = match.group(1)
      suffix = match.group(2)
      root_idx = pitches.index(root)
      new_root_idx = (root_idx + shift) % 12
      return f"{pitches[new_root_idx]}{suffix}"

    # Pitch shift
    for i in range(-12, 13): # 12 semitones above and below
      label_aug_1 = df_label.copy()
      y_aug_1 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=i)
      label_aug_1["chord"] = label_aug_1["chord"].apply(
        lambda x: get_new_chord(x, i)
      )
      augmented_y_and_labels.append((y_aug_1, label_aug_1))
      del label_aug_1
      del y_aug_1
      gc.collect()
    
    # Extracting Audio Features
    for y_aug, labels_aug in augmented_y_and_labels:
      y_harm = librosa.effects.harmonic(y=y_aug)

      # CENS
      #chroma_cens = librosa.feature.chroma_cens(y=y_aug, sr=sr, hop_length=hop_length)
      chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)

      # Extracting the chroma from a given timestamps
      chroma_fps = sr / hop_length
      feature_list = []
      label_list = []
      for idx, start_lbl, end_lbl, chord_lbl in labels_aug.itertuples():
        # start & end index
        segment_start_idx = int(np.round(start_lbl * chroma_fps))
        segment_end_idx = int(np.round(end_lbl * chroma_fps))

        # bounds check the index
        segment_start_idx = max(0, segment_start_idx)
        segment_end_idx = min(chroma_cens.shape[1], segment_end_idx)

        for idx in range(segment_start_idx, segment_end_idx):
          segment = chroma_cens[:, idx]
          feature_list.append(segment)
          label_list.append(chord_lbl)

      # Current file features
      df_feature = pd.DataFrame(feature_list, columns=CENS_COL).astype(np.float32)
      df_feature["chord"] = label_list
      
      if not df_feature.empty:
        dataset.append(df_feature)
        print(".", end='', flush=True)
      else:
        print("ERROR")
      
      del y_aug
      del labels_aug
      del df_feature
      del feature_list
      del label_list
      #del chroma
      #del chroma_harm
      #del chroma_smooth
      #del chroma_filter
      del chroma_cens
      gc.collect()
    print("DONE")

  del augmented_y_and_labels
  gc.collect()

  #df_dataset = pd.concat(dataset, axis=0).drop_duplicates()
  df_dataset = pd.concat(dataset, axis=0)
  del dataset
  gc.collect()

  # normalizing tonnetz & save min & max value
  #tonnetz_min = df_dataset[TONNETZ_COL].min()
  #tonnetz_max = df_dataset[TONNETZ_COL].max()
  #np.savez("./tonnetz_min_max.npz", min=tonnetz_min, max=tonnetz_max)
  #df_dataset[TONNETZ_COL] = (df_dataset[TONNETZ_COL] - tonnetz_min) / (tonnetz_max - tonnetz_min + 1e-6)
  #df_dataset[TONNETZ_COL] = df_dataset[TONNETZ_COL].clip(0.0, 1.0)

  # TODO: replace this and actually split the train and test data here and save it as .npy
  #df_dataset.to_csv(out_files, header=True, index=None, float_format="%.6f")
  df_dataset.to_hdf(out_files, key="df", mode="w", complib="zlib", complevel=9)

  print(f"[INFO]: Total segments -> {len(df_dataset)}")
  print(f"[INFO]: Total chords   -> {len(df_dataset['chord'].unique())}")
  print(f"[INFO]: Chords labels {df_dataset['chord'].unique()}")

  print("\n\n\n")
