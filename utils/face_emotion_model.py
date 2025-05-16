# utils/face_emotion_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    pixels = df['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split()]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))

    faces = np.expand_dims(faces, -1)
    faces /= 255.0
    emotions = pd.get_dummies(df['emotion']).values
    return np.array(faces), emotions

def build_resnet50_model(input_shape=(48, 48, 1), num_classes=7):
    base = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(csv_path='data/fer2013.csv', save_path='models/face_emotion_model.h5'):
    X, y = preprocess_fer2013(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    datagen = ImageDataGenerator(horizontal_flip=True)
    datagen.fit(X_train)

    model = build_resnet50_model()
    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=30)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
