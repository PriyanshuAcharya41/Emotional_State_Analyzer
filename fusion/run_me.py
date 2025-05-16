print("💣 THIS IS RUNNING from: fusion_predict.py")

import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
import io
from utils.audio_utils import extract_mfcc
import traceback

print("🚀 fusion_predict.py is starting...")

# === Load models and assets ===
try:
    face_model = tf.keras.models.load_model("models/face_emotion_model.h5")
    voice_model = tf.keras.models.load_model("models/voice_emotion_model.h5")
    with open("models/voice_label_encoder.pkl", "rb") as f:
        voice_label_encoder = pickle.load(f)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Face labels from FER2013
face_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set default input device index for mic
sd.default.device = [11, None]  # Change index if needed

# === Unified Emotion Mapping ===
def map_emotion(label):
    mapping = {
        "Calm": "Neutral",
        "Neutral": "Neutral",
        "Happy": "Excited",
        "Surprise": "Excited",
        "Angry": "Angry",
        "Sad": "Sad",
        "Fear": "Confused",
        "Disgust": "Confused",
    }
    return mapping.get(label, "Unknown")

# === Emotion Detection Functions ===
def detect_face_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "No Face", 0
    (x, y, w, h) = faces[0]
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48)) / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    pred = face_model.predict(roi, verbose=0)[0]
    raw_label = face_labels[np.argmax(pred)]
    return map_emotion(raw_label), np.max(pred) * 100

def record_audio(duration=3, sr=22050):
    print("🎙️ Recording audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def predict_voice_emotion(audio, sr=22050):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    y, _ = librosa.load(buffer, sr=sr)
    mfcc = extract_mfcc(y, sr)
    pred = voice_model.predict(mfcc, verbose=0)[0]
    raw_label = voice_label_encoder.inverse_transform([np.argmax(pred)])[0]
    return map_emotion(raw_label), np.max(pred) * 100

# === Main Pipeline ===
def main():
    print("\n🚀 Fusion: Face + Voice Emotion Detection\n")

    print("📸 Capturing webcam image...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Webcam capture failed.")
        return

    print("✅ Image captured.")
    face_emotion, face_conf = detect_face_emotion(frame)
    print(f"🧠 Face Emotion: {face_emotion} ({face_conf:.1f}%)")

    audio = record_audio()
    print("✅ Audio recorded.")
    voice_emotion, voice_conf = predict_voice_emotion(audio)
    print(f"🔊 Voice Emotion: {voice_emotion} ({voice_conf:.1f}%)")

    print("\n🎯 Final Result (One-Shot Fusion)")
    print(f"→ Face  : {face_emotion}")
    print(f"→ Voice : {voice_emotion}")

if __name__ == "__main__":
    try:
        print("✅ Calling main()...")
        main()
    except Exception as e:
        print(f"❌ Exception in main(): {e}")
        traceback.print_exc()
