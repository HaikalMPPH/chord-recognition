#!../venv/bin/python3

import os
import glob
import sys

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage

OUTPUT_CSV = 'dataset.csv'
ROOT_DIR = '../datasets'
PITCH_CLASS_LABELS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
# per 100ms split
SEGMENT_DURATION_SEC = 0.1

if __name__ == "__main__":
        # Input audio files
        in_files = glob.glob(os.path.join(ROOT_DIR, "*.mp3"))
        in_files.sort()
        if not in_files:
                print(f"[ERRO]: Cannot find audio files in {ROOT_DIR}")
                sys.exit(sys.EXIT_FAILURE)

        # The corresponding label files of the input audio files
        in_label_files = [f"{os.path.splitext(in_file)[0]}.txt" for in_file in in_files]

        # Output dataset CSV file
        out_files = os.path.abspath(OUTPUT_CSV)

        # If some corresponding label files is missing
        in_missing_label_files = [in_label_file for in_label_file in in_label_files if not os.path.exists(in_label_file)]
        if in_missing_label_files:
                print(f"[ERRO]: Missing corresponding label .txt files:")
                for file in in_missing_label_files:
                        print(f"        {f}")
                sys.exit(sys.EXIT_FAILURE)

        # Featurizing the file
        print(f"[INFO]: Featurizing {len(in_files)} files: ")
        dataset = []
        for in_file, in_label_file in zip(in_files, in_label_files):
                print(f"        Featurizing {os.path.basename(in_file)} -> ", end='', flush=True)

                # Chord timestamps
                df_label = pd.read_csv(in_label_file, header=None, names=["start", "end", "chord"])

                # Loading the audio files
                y, sr = librosa.load(in_file)
                hop_length = int(SEGMENT_DURATION_SEC * sr)

                # Extracting chroma features
                y_harm = librosa.effects.harmonic(y=y, margin=10)
                chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
                chroma_filter = np.minimum(
                        chroma_harm,
                        librosa.decompose.nn_filter(chroma_harm, aggregate=np.median)
                )
                chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
                chroma = chroma_smooth

                # Extracting the chroma from a given timestamps
                chroma_fps = sr / hop_length
                feature_list = []
                label_list = []
                for idx, start_lbl, end_lbl, chord_lbl in df_label.itertuples():
                        # start & end index
                        segment_start_idx = int(np.round(start_lbl * chroma_fps))
                        segment_end_idx = int(np.round(end_lbl * chroma_fps))

                        # bounds check the index
                        segment_start_idx = max(0, segment_start_idx)
                        segment_end_idx = min(chroma.shape[1], segment_end_idx)

                        for idx in range(segment_start_idx, segment_end_idx):
                                segment = chroma[:, idx]
                                feature_list.append(segment)
                                label_list.append(chord_lbl)

                # Current file features
                df_feature = pd.DataFrame(feature_list, columns=PITCH_CLASS_LABELS)
                df_feature["chord"] = label_list
                
                if not df_feature.empty:
                        dataset.append(df_feature)
                        print("DONE")
                else:
                        print("ERROR")

        df_dataset = pd.concat(dataset, axis=0)
        df_dataset.to_csv(out_files, header=True, index=None, float_format="%.6f")
        print(f"[INFO]: Output dataset -> {out_files}")
        print(f"[INFO]: Total segments -> {len(df_dataset)}")
        print(f"[INFO]: Total chords   -> {len(df_dataset['chord'].unique())}")
        print(f"[INFO]: Chords labels {df_dataset['chord'].unique()}")
