import tensorflow as tf
import librosa
import numpy as np

# Define your class names
class_names = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

# Function to preprocess audio file (adjust based on your model input requirements)
def preprocess_audio(audio_file, sampling_rate=16000, duration=1):
    y, sr = librosa.load(audio_file, sr=sampling_rate, duration=duration, mono=True)
    y = librosa.util.normalize(y)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    resized_spectrogram = np.resize(mel_spectrogram, (128, 32))
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
    resized_spectrogram = resized_spectrogram.astype(np.float32)
    return resized_spectrogram

# Function to make predictions
def predict(audio_file):
    processed_audio = preprocess_audio(audio_file)
    prediction = model.predict(np.expand_dims(processed_audio, axis=0))
    return prediction[0]

# Load the model
model_path = "m3.h5"
model = tf.keras.models.load_model(model_path)

# Path to a sample audio file for testing
sample_audio_path = "4.wav"

# Make a prediction
try:
    prediction = predict(sample_audio_path)
    print("Prediction:", prediction)

    # Map predictions to class names
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[predicted_class_index]

    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence}")

except Exception as e:
    print("Error during prediction:", e)
