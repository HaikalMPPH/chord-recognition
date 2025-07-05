#!../venv/bin/python

import streamlit as st
import joblib
import pandas as pd
import librosa
import numpy as np
import scipy
import os
import json
import io
import base64
import tensorflow as tf
import tempfile
import yt_dlp

CWD = os.getcwd()

MODEL = tf.keras.models.load_model("/mount/src/rechordnizer/Model/model_lstm_cens_32_32.keras")
ENCODER = joblib.load("/mount/src/rechordnizer/Model/encoder.xz")
SEGMENT_DURATION_SEC = 0.1
SEQ_LEN = 20 # 20 * 0.1 seconds

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = ""

if __name__ == "__main__":
  st.title("Rechordnizer")
  uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "ogg", "opus", "flac", "avi"])
  yt_url = st.text_input("Or youtube URL")

  audio_librosa_input = None
  audio_bytes = None
  if yt_url:
    st.session_state.yt_url = yt_url
    with st.spinner("Downloading audio..."):
      audio_path = os.path.join(tempfile.gettempdir(), "audio.mp3")
      yt_dlp_flags = {
          "format": "bestaudio/best",
          "quiet": False,
          "noplaylist": True,
          "outtmpl": os.path.join(tempfile.gettempdir(), "audio.%(ext)s"),
          "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
          }],
          'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
          },
      }
      
      with yt_dlp.YoutubeDL(yt_dlp_flags) as ytdl:
        ytdl.download([yt_url])
      
      audio_librosa_input = audio_path

      with open(audio_path, "rb") as f:
        audio_bytes = f.read()

  elif uploaded_file is not None:
    st.session_state.uploaded_file = None

    audio_bytes = uploaded_file.read()
    audio_librosa_input = io.BytesIO(audio_bytes)
  else:
    st.stop()
    
  with st.spinner("Processing audio..."):
    # :::::::: Featurizing files ::::::::
    y, sr = librosa.load(audio_librosa_input)
    hop_length = int(SEGMENT_DURATION_SEC * sr)
    file_duration = librosa.get_duration(y=y, sr=sr)

    # CENS
    y_harm = librosa.effects.harmonic(y=y)
    chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)

    features = [np.hstack(chroma_cens[:, idx]) for idx in range(chroma_cens.shape[1])]
    features = np.array(features)

    # Insert SEQ_LEN-1 zero vectors at the beginning
    zero_padding = np.zeros((SEQ_LEN - 1, features.shape[1]))
    features = np.vstack((zero_padding, features))

    num_segment = features.shape[0] - (SEQ_LEN - 1)
    start_time = np.arange(0, num_segment) * SEGMENT_DURATION_SEC
    end_time = np.minimum(start_time + SEGMENT_DURATION_SEC, file_duration)

    # Creating LSTM input sequences
    features_seq = [features[i:i + SEQ_LEN] for i in range(num_segment)]
    features_seq = np.array(features_seq)

    # Predict chords
    prediction = ENCODER.inverse_transform(
      np.argmax(MODEL.predict(features_seq), axis=1)
    )

    # Smooth prediction with 1-sec window
    window_size = 5
    smoothed_prediction = []
    for i in range(len(prediction)):
      start = max(0, i - window_size // 2)
      end = min(len(prediction), i + window_size // 2 + 1)
      window = prediction[start:end]
      chord_count = pd.Series(window).value_counts()
      smoothed_prediction.append(chord_count.idxmax())

    prediction_df = pd.DataFrame({
      "start": start_time,
      "end": end_time,
      "chord": smoothed_prediction,
      #"chord": prediction,
    })

    # Merge repeating chords
    merged_prediction = []
    current_start = prediction_df.iloc[0]["start"]
    current_end = prediction_df.iloc[0]["end"]
    current_chord = prediction_df.iloc[0]["chord"]
    for _, row in prediction_df.iloc[1:].iterrows():
      if current_chord == row["chord"] and np.isclose(current_end, row["start"], atol=1e-6):
        current_end = row["end"]
      else:
        merged_prediction.append({
          "start": current_start,
          "end": current_end,
          "chord": current_chord
        })
        current_start = row["start"]
        current_end = row["end"]
        current_chord = row["chord"]

    merged_prediction.append({
      "start": current_start,
      "end": current_end,
      "chord": current_chord
    })

    final_prediction_df = pd.DataFrame(merged_prediction)
    # ::::::::::::::::::::::::::::::::::::::::::

  # :::::::: Displaying custom HTML for audio ::::::::
  feature_df_json = json.dumps(
    final_prediction_df[["start", "end", "chord"]].to_dict(orient="records")
  )
  audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
  page = f"""
    <style>
      #chord_display {{
        color: rgb(250, 250, 250);
        font-family: 'Source Sans', sans-serif;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
      }}

      #audio_player {{
        width: 100%;
      }}
    </style>

    <div id="chord_display">Press Play</div>
    <audio controls id="audio_player" src="data:audio/mpeg;base64,{audio_base64}" style=""></audio>

    <script>
      let chord_display = document.getElementById("chord_display");
      let audio_player = document.getElementById("audio_player");
      let feature_json = {feature_df_json};

      // avoid constant DOM update
      let last_chord = null;

      // audio player update callback
      audio_player.addEventListener("timeupdate", () => {{
        let current_time = audio_player.currentTime;
        let current_chord = null;

        for (let i = 0; i < feature_json.length; ++i) {{
          let chord = feature_json[i];
          if (current_time >= chord["start"] && current_time <= chord["end"]) {{
            current_chord = chord["chord"];
            break;
          }}

        }}

        if (current_chord != last_chord) {{
          chord_display.textContent = current_chord;
          last_chord = current_chord;
        }}
      }});
    </script>
  """

  # display html
  #st.html(page)
  st.components.v1.html(page)
  st.session_state.yt_url = ""
  st.session_state.uploaded_file = None
  # ::::::::::::::::::::::::::::::::::::::::::::::::::

