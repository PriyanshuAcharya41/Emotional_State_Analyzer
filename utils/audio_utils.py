import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
import io
import matplotlib.pyplot as plt

import speech_recognition as sr


def transcribe_audio(buffer):
    r = sr.Recognizer()
    with sr.AudioFile(buffer) as source:
        audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "(unrecognized)"
        except sr.RequestError:
            return "(API unavailable)"

# Load model and label encoder
model = tf.keras.models.load_model("models/voice_emotion_model.h5")
with open("models/voice_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

#  Custom label mapping from RAVDESS to your categories
def remap_emotion(label):
    if label in ["Calm", "Neutral"]:
        return "Neutral"
    elif label in ["Disgust", "Fear"]:
        return "Confused"
    elif label in ["Happy", "Surprise"]:
        return "Excited"
    elif label == "Angry":
        return "Angry"
    elif label == "Sad":
        return "Sad"
    else:
        return "Unknown"

def extract_mfcc(audio, sr=22050, max_pad_len=174):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    mfcc = np.transpose(mfcc)
    return mfcc[np.newaxis, ...]

def record_and_predict(duration=3, sample_rate=22050):
    print(" Recording voice...")
    sd.default.device = [11, None]  # Set your working input device index
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    print(f" Recorded shape: {audio.shape}")
    print(" Sample values:", audio[:10].flatten())


#  Visualize audio waveform
    plt.figure(figsize=(10, 3))
    plt.plot(audio[:2000], color='purple')
    plt.title(" Mic Input Waveform (First 2000 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    # Playback
    print(" Playing back your voice...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

    # Save to buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)

    # Transcribe with Google
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(buffer) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f" Transcription: \"{text}\"")
    except Exception as e:
        print(f" Transcription error: {e}")
        text = ""

    # Predict
    buffer.seek(0)
    y, sample_rate_resolved = librosa.load(buffer, sr=sample_rate)
    features = extract_mfcc(y, sample_rate_resolved)



    prediction = model.predict(features, verbose=0)[0]
    original_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    mapped_label = remap_emotion(original_label)

    print(f"\n Original Emotion: {original_label}")
    print(f" Remapped Emotion: {mapped_label}")

    # Show Top 3 predictions (remapped)
    print("\n Top 3 Predicted Emotions (Remapped):")
    top_indices = prediction.argsort()[-3:][::-1]
    for idx in top_indices:
        label = label_encoder.inverse_transform([idx])[0]
        print(f"   • {remap_emotion(label)} (→ {label}) — {prediction[idx]*100:.1f}%")

    return mapped_label, text
