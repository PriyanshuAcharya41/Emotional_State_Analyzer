import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def preprocess_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Usage'].isin(['Training', 'PublicTest'])]

    faces = np.array([np.fromstring(pix, sep=' ').reshape(48, 48, 1) for pix in df['pixels']])
    faces = faces.astype('float32') / 255.0
    emotions = to_categorical(df['emotion'], num_classes=7)

    return train_test_split(faces, emotions, test_size=0.1, random_state=42)

def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    csv_path = 'data/fer2013.csv'
    model_save_path = 'models/face_emotion_model.h5'

    X_train, X_val, y_train, y_val = preprocess_fer2013(csv_path)
    model = build_cnn_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=64)

    os.makedirs('models', exist_ok=True)
    model.save(model_save_path)
    print(f"\n Model saved successfully at: {model_save_path}")

if __name__ == '__main__':
    main()
