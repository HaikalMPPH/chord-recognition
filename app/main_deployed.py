# for the deployed version of streamlit

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

MODEL = joblib.load("/mount/src/rechordnizer/model/model_rf.xz")
ENCODER = joblib.load("/mount/src/rechordnizer/model/encoder.xz")
SEGMENT_DURATION_SEC = 0.1

if __name__ == "__main__":
        st.title("Rechordnizer")
        uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

        if uploaded_file is not None:
                #st.subheader("Transcribed chords")

                # retrieve audio bytes for processing
                audio_bytes = uploaded_file.read()
                audio_librosa_input = io.BytesIO(audio_bytes)
                
                with st.spinner("Processing audio..."):
                        # :::::::: Featurizing files ::::::::
                        y, sr = librosa.load(audio_librosa_input)
                        hop_length = int(SEGMENT_DURATION_SEC * sr)
                        file_duration = librosa.get_duration(y=y, sr=sr)

                        y_harm = librosa.effects.harmonic(y=y, margin=10)
                        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
                        chroma_filter = np.minimum(
                                chroma_harm,
                                librosa.decompose.nn_filter(chroma_harm, aggregate=np.median)
                        )
                        chroma_smooth = scipy.ndimage.median_filter(chroma_filter, size=(1, 9))
                        chroma = chroma_smooth

                        features = chroma.T
                        num_segment = features.shape[0]
                        start_time = np.arange(0, num_segment) * SEGMENT_DURATION_SEC
                        end_time = np.minimum(start_time + SEGMENT_DURATION_SEC, file_duration)
                        # :::::::::::::::::::::::::::::::::::

                        # Predict chords
                        prediction_df = pd.DataFrame({
                                "start": start_time,
                                "end": end_time,
                                "chord": ENCODER.inverse_transform(MODEL.predict(features))
                        })

                        # :::::::: Merging repeating chords ::::::::
                        merged_prediction = []
                        current_start = None
                        current_end = None
                        current_chord = None
                        for index, rows in prediction_df.iterrows():
                                start = rows["start"]
                                end = rows["end"]
                                chord = rows["chord"]

                                # if it's the beginning
                                if current_chord == None:
                                        current_start = start
                                        current_end = end
                                        current_chord = chord
                                # move the end forward if it's still the same chord.
                                elif current_chord == chord:
                                        current_end = end
                                # append the merged chord and reset the current value for the new ones.
                                else:
                                        merged_prediction.append({
                                                "start": current_start,
                                                "end": current_end,
                                                "chord": current_chord,
                                        })
                                        current_start = start
                                        current_end = end
                                        current_chord = chord

                        # if the last chord
                        if current_chord is not None:
                                merged_prediction.append({
                                        "start": current_start,
                                        "end": current_end,
                                        "chord": current_chord,
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
                # ::::::::::::::::::::::::::::::::::::::::::::::::::

