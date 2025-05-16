import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Directory to your RAVDESS dataset
DATA_DIR = 'data/Radvess'

# Supported emotions in RAVDESS
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print("Error:", file_path, e)
        return None

def load_data():
    features = []
    labels = []

    print("🔍 Scanning for .wav files...")
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.wav'):
                try:
                    emotion_id = file.split('-')[2]
                    label = emotion_map.get(emotion_id)
                    if not label:
                        continue

                    path = os.path.join(root, file)
                    mfcc = extract_features(path)
                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(label)
                    else:
                        print(f"⚠️ Skipped {file} (MFCC extraction failed)")
                except Exception as e:
                    print(f"❌ Error processing {file}: {e}")

    print(f"✅ Loaded {len(features)} samples")
    return np.array(features), np.array(labels)


def main():
    X, y = load_data()
    print(f"🧪 Found {len(X)} audio files with valid MFCC features.")
    print(f"Labels: {set(y)}")

    print(f"Loaded {len(X)} samples")

    X = np.transpose(X, (0, 2, 1))  # Now shape = (batch, time, features)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(174, 40)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)

    model.save('models/voice_emotion_model.h5')
    with open('models/voice_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print("\n✅ Voice model and label encoder saved!")

if __name__ == '__main__':
    main()
