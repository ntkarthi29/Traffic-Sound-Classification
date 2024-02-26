import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Load the LSTM model
model = load_model('your_model.h5')

# Function to preprocess audio data
def preprocess_audio(audio_file):
    # Load audio file and extract features (e.g., MFCCs)
    # Example: Using Librosa to extract MFCCs
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Example with 20 MFCCs
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean across time
    
    # Reshape the data to match the input shape expected by the model
    mfccs_processed = mfccs_processed.reshape(1, -1)
    
    return mfccs_processed

# Function to make predictions
def predict(audio_data):
    # Perform classification using the loaded model
    prediction = model.predict(audio_data)
    return prediction

# Streamlit app
def main():
    st.title('Audio Classification Demo')
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        audio_data = preprocess_audio(uploaded_file)
        prediction = predict(audio_data)
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()

