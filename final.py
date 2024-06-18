import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Define class names
class_names = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

# Function to preprocess audio file
def preprocess_audio(audio_file, sampling_rate=16000, duration=1):
    y, sr = librosa.load(audio_file, sr=sampling_rate, duration=duration, mono=True)
    y = librosa.util.normalize(y)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    resized_spectrogram = np.resize(mel_spectrogram, (8000, 1))
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=0)
    resized_spectrogram = resized_spectrogram.astype(np.float32)
    return resized_spectrogram

# Function to make predictions
def predict(audio_file):
    processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(processed_audio)
    return prediction

# Streamlit app
st.title("Audio Classification Demo")

# Load the model
model_path = "m3.h5"
try:
    model = load_model(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.write("Error loading model:", e)
    st.stop()



uploaded_file = st.file_uploader("Choose an audio file (.wav)", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Perform prediction when the user uploads an audio file
    try:
        prediction = predict(uploaded_file)
        flat_list = [item for sublist in prediction for item in sublist]
        max_index = flat_list.index(max(flat_list))
        st.write("Prediction:", class_names[max_index])
    except Exception as e:
        st.write("Error during prediction:", e)
