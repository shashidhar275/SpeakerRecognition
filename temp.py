import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras.models import load_model



class_names = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

def preprocess_audio(audio_file, sampling_rate=16000, duration=1):
    # Load audio file using librosa
    y, sr = librosa.load(audio_file, sr=sampling_rate, duration=duration, mono=True)

    # Normalize audio
    y = librosa.util.normalize(y)

    # Extract features (Mel spectrogram)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale

    # Resize spectrogram to match model input shape
    # Adjust these dimensions based on your model's input shape
    resized_spectrogram = np.resize(mel_spectrogram, (8000, 1))  # Example shape (height, width)

    # Add batch dimension
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=0)  # Shape (1, height, width)

    # Convert to float32
    resized_spectrogram = resized_spectrogram.astype(np.float32)

    return resized_spectrogram


# Function to make predictions
def predict(audio_file):
    # processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(audio_file)
    return prediction

# Load the model
try:
    model_path = "m3.h5"
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Path to a sample audio file for testing
sample_audio_path = "n2Magaret.wav"

# Make a prediction
try:
    prediction = predict(sample_audio_path)
    print("Prediction:", prediction)
    flat_list = [item for sublist in prediction for item in sublist]
    print(flat_list)
    max_index = flat_list.index(max(flat_list))
    print("Prediction:", class_names[max_index])

except Exception as e:
    print("Error during prediction:", e)
