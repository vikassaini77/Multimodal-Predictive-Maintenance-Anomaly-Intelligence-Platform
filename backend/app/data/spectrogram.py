import librosa
import numpy as np

class MelSpectrogramExtractor:
    def __init__(self, sample_rate=16000, n_mels=128, hop_length=512, n_fft=2048):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract(self, audio_array):
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
