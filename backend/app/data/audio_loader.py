import librosa
import numpy as np

class AudioLoader:
    def __init__(self, sample_rate=16000, duration=10.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)

    def load(self, file_path):
        # Reads .wav, converts stereo to mono, resamples
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Pad or trim to fixed length
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            
        return audio
