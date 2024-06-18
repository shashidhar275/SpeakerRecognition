import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Define the class names in the order of your model's output
class_names = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

# Load the model outside of the prediction function
try:
    model_path = "m3.h5"
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Function to preprocess audio for prediction
def preprocess_audio(audio_file, sampling_rate=16000, duration=1):
    try:
        # Load audio file using librosa
        y, sr = librosa.load(audio_file, sr=sampling_rate, duration=duration, mono=True)

        # Normalize audio
        y = librosa.util.normalize(y)

        # Extract features (Mel spectrogram)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale

        # Resize spectrogram to match model input shape
        resized_spectrogram = np.resize(mel_spectrogram, (8000, 1))  # Resize to (height, width)

        # Add batch dimension
        resized_spectrogram = np.expand_dims(resized_spectrogram, axis=0)  # Shape (1, height, width)

        # Convert to float32
        resized_spectrogram = resized_spectrogram.astype(np.float32)

        return resized_spectrogram

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

# Function to make predictions
def predict(audio_file):
    try:
        processed_audio = preprocess_audio(audio_file)
        if processed_audio is None:
            return None
        
        # Make prediction
        prediction = model.predict(processed_audio)
        
        # Get predicted class index
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        
        return predicted_class
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Path to a sample audio file for testing
sample_audio_path = "running_tap.wav"

# Make a prediction
try:
    prediction = predict(sample_audio_path)
    if prediction is not None:
        print("Predicted Speaker:", prediction)
    else:
        print("Prediction failed.")
except Exception as e:
    print("Error during prediction:", e)
