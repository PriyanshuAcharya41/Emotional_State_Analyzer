from utils.audio_utils import record_and_predict

if __name__ == "__main__":
    print(" Voice Emotion Prediction (Remapped)")
    print("-------------------------------------")

    emotion, transcript = record_and_predict()

    print("\n Final Output")
    print(f" Detected Emotion: {emotion}")
    print(f" Transcription: {transcript if transcript else 'N/A'}")
