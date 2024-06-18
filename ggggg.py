from tensorflow.keras.models import load_model
model_path = "m3.h5"
model = load_model(model_path)
model.predict("benz.wav")