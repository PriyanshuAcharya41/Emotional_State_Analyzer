import soundfile as sf
import numpy as np

data, sr = sf.read("test_voice.wav")
print("Shape:", data.shape)
print("Max amplitude:", np.max(np.abs(data)))
