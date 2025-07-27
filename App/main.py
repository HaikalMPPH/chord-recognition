import streamlit as st
import joblib
import pandas as pd
import librosa
import numpy as np
import os
import json
import io
import base64
import tensorflow as tf
import tempfile
import yt_dlp

ROOT = ""
CWD = os.getcwd()
MODEL = tf.keras.models.load_model(ROOT + "./Model/model_lstm_128_64.keras")
ENCODER = joblib.load(ROOT + "./Model/encoder.xz")
SEGMENT_DURATION_SEC = 0.1
SEQ_LEN = 20 # 20 * 0.1 seconds

# Default Config
st.set_page_config(
   layout="centered",
   page_title="Rechordnizer🎵🎶",
)
# Default state
if "page" not in st.session_state:
   st.session_state.page = "home"
if "page_chord_vars" not in st.session_state:
   st.session_state.page_chord_vars = {
      "msg": "This should not happen...",
      "audio_base64": None,
      "feature_json": None
   }

def page_home():
   st.title("Rechordnizer🎵🎶")
   st.write(
      "Automatically detect chords from audio recordings via file uploads or URLs "
      + "such as YouTube, Instagram, SoundCloud, or any site containing audio or video "
      + "of your interest!"
   )
   st.subheader("NOTE")
   st.markdown("- Rechordnizer can only classify **minor 7**, **major 7**, and **dominant 7** chords!")
   st.markdown("- Major and minor triads are classified as major 7 and minor 7 chords")
   
   uploaded_file = st.file_uploader(
      "Upload audio file", 
      type=["mp3", "wav", "ogg", "opus", "flac", "avi"]
   )
   _, c, _ = st.columns([2, 1, 1]) # hacky centering
   with c:
      st.subheader("or")
   uploaded_url = st.text_input("Or supported media URL (ex: https://youtu.be/dQw4w9WgXcQ)")

   audio_librosa_input = None
   audio_bytes = None

   def give_error(e: Exception):
      st.session_state.page = "chord"
      st.session_state.page_chord_vars["msg"] = f"Error: {e}"
      st.rerun()

   if uploaded_url:
      try:
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
               }]
            }
            
            with yt_dlp.YoutubeDL(yt_dlp_flags) as ytdl:
               ytdl.download([uploaded_url])
            
            audio_librosa_input = audio_path

            with open(audio_path, "rb") as f:
               audio_bytes = f.read()
      except Exception as e:
         give_error(e)
   elif uploaded_file is not None:
      audio_bytes = uploaded_file.read()
      audio_librosa_input = io.BytesIO(audio_bytes)
   else:
      st.stop()
      
   with st.spinner("Processing audio..."):
      # :::::::: Featurizing files ::::::::
      try:
         y, sr = librosa.load(audio_librosa_input)
      except Exception as e:
         give_error(e)
      hop_length = int(SEGMENT_DURATION_SEC * sr)
      file_duration = librosa.get_duration(y=y, sr=sr)

      # extracting CENS
      y_harm = librosa.effects.harmonic(y=y)
      chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length)
      features = [np.hstack(chroma_cens[:, idx]) for idx in range(chroma_cens.shape[1])]
      features = np.array(features)

      # Insert SEQ_LEN-1 zeros at the beginning
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
      prediction_df = pd.DataFrame({
         "start": start_time,
         "end": end_time,
         "chord": prediction,
      })

      # Merge repeating chords
      merged_prediction = []
      current_start = prediction_df.iloc[0]["start"]
      current_end = prediction_df.iloc[0]["end"]
      current_chord = prediction_df.iloc[0]["chord"]
      for _, row in prediction_df.iloc[1:].iterrows():
         if current_chord == row["chord"]:
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

   feature_json = json.dumps(
      final_prediction_df[["start", "end", "chord"]].to_dict(orient="records")
   )
   audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
   
   st.session_state.page = "chord"
   st.session_state.page_chord_vars["msg"] = None
   st.session_state.page_chord_vars["audio_base64"] = audio_base64
   st.session_state.page_chord_vars["feature_json"] = feature_json
   st.rerun()

def page_chord():
   def btn_home():
      l, c, r = st.columns([2, 2, 1]) # hacky centering
      with c:
         if st.button("Return to Home"):
            st.session_state.page = "home"
            st.rerun()

   msg = st.session_state.page_chord_vars["msg"]
   if msg:
      st.title("Error occured:")
      st.write(msg)
      btn_home()
      st.stop()

   st.title("Detected Chords")
   audio_base64 = st.session_state.page_chord_vars["audio_base64"]
   feature_json = st.session_state.page_chord_vars["feature_json"]
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
         let feature_json = {feature_json};

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

   st.components.v1.html(page)
   btn_home()

if __name__ == "__main__":
   match st.session_state.page:
      case "chord":
         page_chord()
      case "home":
         page_home()
