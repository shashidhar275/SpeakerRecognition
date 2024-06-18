import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

# Function to load and preprocess audio file
def preprocess_audio(audio_file):
    # Load audio file
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    # Resample if needed (your model expects 16000 Hz)
    if _ != 16000:
        audio = librosa.resample(audio, _, 16000)
    # Split audio into chunks of 1 second each
    audio_chunks = []
    for i in range(0, len(audio) - 16000 + 1, 16000):
        audio_chunks.append(audio[i:i+16000])
    return np.array(audio_chunks)

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Replace with your model path
    return model

def predict(audio_chunks, model):
    # Convert chunks to spectrograms
    ffts = []
    for chunk in audio_chunks:
        fft = np.abs(librosa.stft(chunk, n_fft=512, hop_length=160, win_length=400))
        # Reshape to match model input shape (8000, 1)
        fft_reshaped = np.expand_dims(np.transpose(fft[:, :8000]), axis=-1)
        ffts.append(fft_reshaped)
    ffts = np.array(ffts)
    # Predict using the model
    predictions = model.predict(ffts)
    predicted_labels = np.argmax(predictions, axis=-1)
    return predicted_labels



# Streamlit UI
def main():
    st.title('Speech Recognition Web App')
    st.markdown('Upload an audio file (.wav) for speaker recognition.')

    uploaded_file = st.file_uploader("Choose an audio file...", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button('Recognize Speaker'):
            # Preprocess the uploaded file
            audio_chunks = preprocess_audio(uploaded_file)
            # Load the model
            model = load_model()
            # Predict
            predicted_labels = predict(audio_chunks, model)
            # Show the prediction result
            st.write("Prediction:", predicted_labels)

if __name__ == '__main__':
    main()
