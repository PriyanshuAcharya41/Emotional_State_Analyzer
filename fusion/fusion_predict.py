# # # import os
# # # import sys
# # # import cv2
# # # import numpy as np
# # # import tensorflow as tf
# # # import soundfile as sf
# # # import librosa
# # # import pickle
# # # import io
# # # from datetime import datetime
# # # import speech_recognition as sr
# # # from transformers import pipeline

# # # print("fusion_predict_py Starting...")

# # # # === Load Models and Assets ===
# # # face_emotion_model = r"C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/face_emotion_model.h5"
# # # voice_emotion_model = r"C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_emotion_model.h5"
# # # voice_label_encoder_file = r"C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_label_encoder.pkl"

# # # try:
# # #     face_model = tf.keras.models.load_model(face_emotion_model)
# # #     voice_model = tf.keras.models.load_model(voice_emotion_model)
# # #     with open(voice_label_encoder_file, "rb") as f:
# # #         voice_label_encoder = pickle.load(f)
# # #     print("Models loaded successfully.")
# # # except Exception as e:
# # #     print(f" Error loading models: {e}")
# # #     exit()

# # # sentiment_pipeline = pipeline("sentiment-analysis")

# # # face_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# # # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # # # === Helper Functions ===

# # # def map_emotion(label):
# # #     mapping = {
# # #         "calm": "Neutral",
# # #         "neutral": "Neutral",
# # #         "happy": "Excited",
# # #         "surprised": "Excited",
# # #         "angry": "Angry",
# # #         "sad": "Sad",
# # #         "fearful": "Confused",
# # #         "disgust": "Confused",
# # #         "positive": "Excited",
# # #         "negative": "Frustrated"
# # #     }
# # #     return mapping.get(label.lower(), "Unknown")

# # # def extract_mfcc(y, sr=22050, n_mfcc=40, max_len=174):
# # #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
# # #     pad_width = max_len - mfcc.shape[1]
# # #     if pad_width > 0:
# # #         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
# # #     else:
# # #         mfcc = mfcc[:, :max_len]
# # #     return mfcc.T[np.newaxis, ...]

# # # def transcribe_audio(path):
# # #     r = sr.Recognizer()
# # #     with sr.AudioFile(path) as source:
# # #         audio = r.record(source)
# # #         try:
# # #             return r.recognize_google(audio)
# # #         except sr.UnknownValueError:
# # #             return "(unrecognized)"
# # #         except sr.RequestError:
# # #             return "(API unavailable)"

# # # def analyze_sentiment_bert(text):
# # #     if not text.strip():
# # #         return "Neutral"
# # #     result = sentiment_pipeline(text)[0]
# # #     label = result["label"]
# # #     if "POS" in label.upper():
# # #         return "Positive"
# # #     elif "NEG" in label.upper():
# # #         return "Negative"
# # #     else:
# # #         return "Neutral"

# # # # === Real-time Detection (from uploaded files) ===

# # # def main():
# # #     if len(sys.argv) != 3:
# # #         print("Usage: python fusion_predict.py <audio_path> <image_path>")
# # #         exit()

# # #     audio_path = sys.argv[1]
# # #     image_path = sys.argv[2]

# # #     print(f" Audio path: {audio_path}")
# # #     print(f" Image path: {image_path}")

# # #     # --- Face Detection ---
# # #     img = cv2.imread(image_path)
# # #     if img is None:
# # #         raise ValueError("Image could not be loaded.")
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# # #     if len(faces) == 0:
# # #         face_emotion, face_conf = "No Face", 0.0
# # #     else:
# # #         (x, y, w, h) = faces[0]
# # #         roi = gray[y:y+h, x:x+w]
# # #         roi = cv2.resize(roi, (48, 48)) / 255.0
# # #         roi = np.reshape(roi, (1, 48, 48, 1))
# # #         pred = face_model.predict(roi, verbose=0)[0]
# # #         raw_label = face_labels[np.argmax(pred)]
# # #         face_emotion = map_emotion(raw_label)
# # #         face_conf = np.max(pred) * 100

# # #     # --- Voice Emotion ---
# # #     y, sr = librosa.load(audio_path, sr=22050)
# # #     mfcc = extract_mfcc(y, sr)
# # #     pred = voice_model.predict(mfcc, verbose=0)[0]
# # #     raw_label = voice_label_encoder.inverse_transform([np.argmax(pred)])[0]
# # #     voice_emotion = map_emotion(raw_label)
# # #     voice_conf = np.max(pred) * 100

# # #     # --- Transcription + Text Sentiment ---
# # #     transcript = transcribe_audio(audio_path)
# # #     sentiment = analyze_sentiment_bert(transcript)
# # #     sentiment_emotion = map_emotion(sentiment)

# # #     # --- Output as JSON to stdout ---
# # #     result = {
# # #         "face": {
# # #             "label": face_emotion,
# # #             "confidence": round(face_conf / 100, 4)
# # #         },
# # #         "voice": {
# # #             "label": voice_emotion,
# # #             "confidence": round(voice_conf / 100, 4)
# # #         },
# # #         "text": {
# # #             "sentiment": sentiment_emotion,
# # #             "transcript": transcript
# # #         }
# # #     }

# # #     print(" FUSION COMPLETE")
# # #     print()
# # #     print(json.dumps(result))
# # #     return result

# # # if __name__ == "__main__":
# # #     import json
# # #     try:
# # #         main()
# # #     except Exception as e:
# # #         print(f" Error in main(): {e}")
# # #         import traceback
# # #         traceback.print_exc()
# # import os
# # import cv2
# # import numpy as np
# # import tensorflow as tf
# # import sounddevice as sd
# # import soundfile as sf
# # import librosa
# # import pickle
# # import io
# # import json
# # from datetime import datetime
# # import speech_recognition as sr
# # from transformers import pipeline
# # import traceback

# # print("[fusion_predict.py] Starting...")

# # # Load Models
# # face_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/face_emotion_model.h5")
# # voice_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_emotion_model.h5")
# # with open("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_label_encoder.pkl", "rb") as f:
# #     voice_label_encoder = pickle.load(f)

# # print("✅ Models loaded successfully.")

# # # Constants
# # face_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# # sentiment_pipeline = pipeline("sentiment-analysis")
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # def map_emotion(label):
# #     mapping = {
# #         "calm": "Neutral", "neutral": "Neutral", "happy": "Excited", "surprised": "Excited",
# #         "angry": "Angry", "sad": "Sad", "fearful": "Confused", "disgust": "Confused",
# #         "positive": "Excited", "negative": "Frustrated"
# #     }
# #     return mapping.get(label.lower(), "Unknown")

# # def extract_mfcc(y, sr=22050, n_mfcc=40, max_len=174):
# #     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
# #     pad_width = max_len - mfcc.shape[1]
# #     if pad_width > 0:
# #         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #     else:
# #         mfcc = mfcc[:, :max_len]
# #     return mfcc.T[np.newaxis, ...]

# # def transcribe_audio(path):
# #     r = sr.Recognizer()
# #     with sr.AudioFile(path) as source:
# #         audio = r.record(source)
# #         try:
# #             return r.recognize_google(audio)
# #         except sr.UnknownValueError:
# #             return "(unrecognized)"
# #         except sr.RequestError:
# #             return "(API unavailable)"

# # def analyze_sentiment_bert(text):
# #     if not text.strip():
# #         return "Neutral"
# #     result = sentiment_pipeline(text)[0]
# #     label = result["label"]
# #     if "POS" in label.upper():
# #         return "Positive"
# #     elif "NEG" in label.upper():
# #         return "Negative"
# #     else:
# #         return "Neutral"

# # def record_audio(duration=3, sr=22050):
# #     print("🎙️ Recording voice... Speak now.")
# #     audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
# #     sd.wait()
# #     return audio.flatten(), sr

# # def main():
# #     print("\n🔁 Fusion: Face + Voice + Text Sentiment Detection\n")

# #     # === CAMERA INPUT ===
# #     print("📸 Starting webcam. Press 'c' to capture image.")
# #     cap = cv2.VideoCapture(0)

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             print("❌ Failed to read from webcam.")
# #             break

# #         cv2.imshow("Live Feed - Press 'c' to capture", frame)
# #         if cv2.waitKey(1) & 0xFF == ord('c'):
# #             print("✅ Image captured.")
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# #     # === FACE EMOTION ===
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #     if len(faces) == 0:
# #         face_emotion, face_conf = "No Face", 0.0
# #     else:
# #         (x, y, w, h) = faces[0]
# #         roi = gray[y:y+h, x:x+w]
# #         roi = cv2.resize(roi, (48, 48)) / 255.0
# #         roi = np.reshape(roi, (1, 48, 48, 1))
# #         pred = face_model.predict(roi, verbose=0)[0]
# #         raw_label = face_labels[np.argmax(pred)]
# #         face_emotion = map_emotion(raw_label)
# #         face_conf = np.max(pred) * 100
# #     print(f"🧠 Face Emotion: {face_emotion} ({face_conf:.1f}%)")

# #     # === VOICE EMOTION ===
# #     audio, sr = record_audio()
# #     sf.write("temp.wav", audio, sr)
# #     y, _ = librosa.load("temp.wav", sr=sr)
# #     mfcc = extract_mfcc(y, sr)
# #     pred = voice_model.predict(mfcc, verbose=0)[0]
# #     raw_label = voice_label_encoder.inverse_transform([np.argmax(pred)])[0]
# #     voice_emotion = map_emotion(raw_label)
# #     voice_conf = np.max(pred) * 100
# #     print(f"🧠 Voice Emotion: {voice_emotion} ({voice_conf:.1f}%)")

# #     # === TEXT EMOTION ===
# #     transcript = transcribe_audio("temp.wav")
# #     sentiment = analyze_sentiment_bert(transcript)
# #     sentiment_emotion = map_emotion(sentiment)
# #     print(f"💬 Text Sentiment: {sentiment_emotion} (from \"{transcript}\")")

# #     # === Final Output ===
# #     result = {
# #         "face": {"label": face_emotion, "confidence": round(face_conf / 100, 4)},
# #         "voice": {"label": voice_emotion, "confidence": round(voice_conf / 100, 4)},
# #         "text": {"sentiment": sentiment_emotion, "transcript": transcript}
# #     }

# #     print("\n✅ FUSION COMPLETE")
# #     print(json.dumps(result, indent=2))

# #     # Clean up temp
# #     if os.path.exists("temp.wav"):
# #         os.remove("temp.wav")

# # if __name__ == "__main__":
# #     try:
# #         main()
# #     except Exception:
# #         print("❌ Error during execution:")
# #         traceback.print_exc()

# import os
# import sys
# import cv2
# import numpy as np
# import tensorflow as tf
# import sounddevice as sd
# import soundfile as sf
# import librosa
# import pickle
# import json
# import io
# import matplotlib.pyplot as plt
# from datetime import datetime
# import speech_recognition as sr
# from transformers import pipeline
# import traceback

# print("[fusion_predict.py] Starting...")

# # === Load Models and Assets ===
# face_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/face_emotion_model.h5")
# voice_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_emotion_model.h5")
# with open("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_label_encoder.pkl", "rb") as f:
#     voice_label_encoder = pickle.load(f)

# print("✅ Models loaded successfully.")

# face_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# sentiment_pipeline = pipeline("sentiment-analysis")
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# def map_emotion(label):
#     mapping = {
#         "calm": "Neutral", "neutral": "Neutral", "happy": "Excited", "surprised": "Excited",
#         "angry": "Angry", "sad": "Sad", "fearful": "Confused", "disgust": "Confused",
#         "positive": "Excited", "negative": "Frustrated"
#     }
#     return mapping.get(label.lower(), "Unknown")

# def extract_mfcc(y, sr=22050, n_mfcc=40, max_len=174):
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     pad_width = max_len - mfcc.shape[1]
#     if pad_width > 0:
#         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mfcc = mfcc[:, :max_len]
#     return mfcc.T[np.newaxis, ...]

# def transcribe_audio(path):
#     r = sr.Recognizer()
#     with sr.AudioFile(path) as source:
#         audio = r.record(source)
#         try:
#             return r.recognize_google(audio)
#         except sr.UnknownValueError:
#             return "(unrecognized)"
#         except sr.RequestError:
#             return "(API unavailable)"

# def analyze_sentiment_bert(text):
#     if not text.strip():
#         return "Neutral"
#     result = sentiment_pipeline(text)[0]
#     label = result["label"]
#     if "POS" in label.upper():
#         return "Positive"
#     elif "NEG" in label.upper():
#         return "Negative"
#     else:
#         return "Neutral"

# def record_audio(duration=5, sr=22050):
#     print("🎙️ Recording... Speak Now")
#     print("🛠️ Listing input devices...")
#     print(sd.query_devices())  # List all devices (mic index can vary)

#     try:
#         # Force default input device (e.g., index 1 or 2 as per your mic)
#         sd.default.device = (1, None)  # You may need to adjust this index
#         audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
#         sd.wait()

#         # Playback
#         print("🔁 Playing back your audio...")
#         sd.play(audio, sr)
#         sd.wait()

#         print(f"🔎 Recorded audio shape: {audio.shape}")
#         print(f"🔎 Sample values: {audio[:10].flatten()}")
        
#         # Plot
#         plt.figure(figsize=(8, 2))
#         plt.plot(audio.flatten())
#         plt.title("Audio Waveform")
#         plt.xlabel("Samples")
#         plt.ylabel("Amplitude")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#         return audio.flatten(), sr
#     except Exception as e:
#         print(f"❌ Recording failed: {e}")
#         return np.zeros(int(duration * sr)), sr


# def main():
#     print("\n Fusion: Face + Voice + Text Sentiment Detection\n")

#     print("📸 Starting webcam. Press 'c' to capture image.")
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Webcam failed.")
#             break
#         cv2.imshow("Live Feed - Press 'c' to capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord('c'):
#             print("✅ Image captured.")
#             break
#     cap.release()
#     cv2.destroyAllWindows()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     if len(faces) == 0:
#         face_emotion, face_conf = "No Face", 0.0
#     else:
#         (x, y, w, h) = faces[0]
#         roi = gray[y:y+h, x:x+w]
#         roi = cv2.resize(roi, (48, 48)) / 255.0
#         roi = np.reshape(roi, (1, 48, 48, 1))
#         pred = face_model.predict(roi, verbose=0)[0]
#         raw_label = face_labels[np.argmax(pred)]
#         face_emotion = map_emotion(raw_label)
#         face_conf = np.max(pred) * 100
#     print(f"🧠 Face Emotion: {face_emotion} ({face_conf:.1f}%)")

#     audio, sr = record_audio()
#     sf.write("temp.wav", audio, sr)
#     y, _ = librosa.load("temp.wav", sr=sr)
#     mfcc = extract_mfcc(y, sr)
#     prediction = voice_model.predict(mfcc, verbose=0)[0]
#     top_indices = prediction.argsort()[-3:][::-1]
#     print("🧠 Top 3 Predicted Voice Emotions:")
#     for idx in top_indices:
#         label = voice_label_encoder.inverse_transform([idx])[0]
#         print(f"   • {label.capitalize()} ({prediction[idx]*100:.1f}%)")
#     raw_label = voice_label_encoder.inverse_transform([np.argmax(prediction)])[0]
#     voice_emotion = map_emotion(raw_label)
#     voice_conf = np.max(prediction) * 100

#     transcript = transcribe_audio("temp.wav")
#     sentiment = analyze_sentiment_bert(transcript)
#     sentiment_emotion = map_emotion(sentiment)
#     print(f"💬 Transcript: \"{transcript}\"")
#     print(f"🧠 Text Sentiment: {sentiment_emotion} (from {sentiment})")

#     result = {
#         "face": {"label": face_emotion, "confidence": round(face_conf / 100, 4)},
#         "voice": {"label": voice_emotion, "confidence": round(voice_conf / 100, 4)},
#         "text": {"sentiment": sentiment_emotion, "transcript": transcript}
#     }

#     print("\n✅ FINAL FUSION RESULT:")
#     print(json.dumps(result, indent=2))

#     # === Save results to results.csv ===
#     try:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         csv_line = f"{timestamp},{face_emotion},{voice_emotion},{sentiment_emotion},\"{transcript}\"\n"
#         with open("results.csv", "a", encoding="utf-8") as f:
#             if os.stat("results.csv").st_size == 0:
#                 f.write("Timestamp,FaceEmotion,VoiceEmotion,TextSentiment,Transcript\n")
#             f.write(csv_line)
#         print("📝 Logged to results.csv")
#     except Exception as log_err:
#         print("⚠️ Failed to log results:", log_err)

#     if os.path.exists("temp.wav"):
#         os.remove("temp.wav")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception:
#         print("❌ Error during fusion:")
#         traceback.print_exc()

import os
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
import json
import io
import matplotlib.pyplot as plt
from datetime import datetime
import speech_recognition as sr
from transformers import pipeline
import traceback
import time

print("[fusion_predict.py] Starting...")

# === Load Models and Assets ===
face_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/face_emotion_model.h5")
voice_model = tf.keras.models.load_model("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_emotion_model.h5")
with open("C:/Users/PRIYANSHU/OneDrive/Desktop/ESA_Project/models/voice_label_encoder.pkl", "rb") as f:
    voice_label_encoder = pickle.load(f)

print("✅ Models loaded successfully.")

face_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
sentiment_pipeline = pipeline("sentiment-analysis")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def map_emotion(label):
    mapping = {
        "calm": "Neutral", "neutral": "Neutral", "happy": "Excited", "surprised": "Excited",
        "angry": "Angry", "sad": "Sad", "fearful": "Confused", "disgust": "Confused",
        "positive": "Excited", "negative": "Frustrated"
    }
    return mapping.get(label.lower(), "Unknown")

def extract_mfcc(y, sr=22050, n_mfcc=40, max_len=174):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    pad_width = max_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ...]

def transcribe_audio(path):
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        print("🎧 Listening to file for transcription...")
        audio = r.record(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            print("⚠️ Could not understand audio.")
            return "(unrecognized)"
        except sr.RequestError:
            print("⚠️ Speech API unavailable.")
            return "(API unavailable)"

def analyze_sentiment_bert(text):
    if not text.strip():
        return "Neutral"
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    if "POS" in label.upper():
        return "Positive"
    elif "NEG" in label.upper():
        return "Negative"
    else:
        return "Neutral"

def record_audio(duration=5, sr=22050):
    print("🎙️ Recording... Speak Now")
    print("🛠️ Input Devices:")
    print(sd.query_devices())
    try:
        sd.default.device = (3, None)  # Adjust input device index if needed
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        print("🔁 Playing back your audio...")
        sd.play(audio, sr)
        sd.wait()
        print(f"🔎 Recorded audio shape: {audio.shape}")
        print(f"🔎 Sample values: {audio[:10].flatten()}")

        plt.figure(figsize=(8, 2))
        plt.plot(audio.flatten())
        plt.title("Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return audio.flatten(), sr
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return np.zeros(int(duration * sr)), sr

def main():
    print("\n Fusion: Face + Voice + Text Sentiment Detection\n")

    print("📸 Starting webcam. Press 'c' to capture image.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Webcam failed.")
            break
        cv2.imshow("Live Feed - Press 'c' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            print("✅ Image captured.")
            break
    cap.release()
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        face_emotion, face_conf = "No Face", 0.0
    else:
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        pred = face_model.predict(roi, verbose=0)[0]
        raw_label = face_labels[np.argmax(pred)]
        face_emotion = map_emotion(raw_label)
        face_conf = np.max(pred) * 100
    print(f"🧠 Face Emotion: {face_emotion} ({face_conf:.1f}%)")

    audio, sr = record_audio()
    sf.write("temp.wav", audio, sr)
    time.sleep(0.5)
    y, _ = librosa.load("temp.wav", sr=sr)
    mfcc = extract_mfcc(y, sr)
    prediction = voice_model.predict(mfcc, verbose=0)[0]
    top_indices = prediction.argsort()[-3:][::-1]
    print("🧠 Top 3 Predicted Voice Emotions:")
    for idx in top_indices:
        label = voice_label_encoder.inverse_transform([idx])[0]
        print(f"   • {label.capitalize()} ({prediction[idx]*100:.1f}%)")
    raw_label = voice_label_encoder.inverse_transform([np.argmax(prediction)])[0]
    voice_emotion = map_emotion(raw_label)
    voice_conf = np.max(prediction) * 100

    transcript = transcribe_audio("temp.wav")
    sentiment = analyze_sentiment_bert(transcript)
    sentiment_emotion = map_emotion(sentiment)
    print(f"💬 Transcript: \"{transcript}\"")
    print(f"🧠 Text Sentiment: {sentiment_emotion} (from {sentiment})")

    result = {
        "face": {"label": face_emotion, "confidence": round(face_conf / 100, 4)},
        "voice": {"label": voice_emotion, "confidence": round(voice_conf / 100, 4)},
        "text": {"sentiment": sentiment_emotion, "transcript": transcript}
    }

    print("\n✅ FINAL FUSION RESULT:")
    print(json.dumps(result, indent=2))

    # === Save results to results.csv ===
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_line = f"{timestamp},{face_emotion},{voice_emotion},{sentiment_emotion},\"{transcript}\"\n"
        with open("results.csv", "a", encoding="utf-8") as f:
            if os.stat("results.csv").st_size == 0:
                f.write("Timestamp,FaceEmotion,VoiceEmotion,TextSentiment,Transcript\n")
            f.write(csv_line)
        print("📝 Logged to results.csv")
    except Exception as log_err:
        print("⚠️ Failed to log results:", log_err)

    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("❌ Error during fusion:")
        traceback.print_exc()
