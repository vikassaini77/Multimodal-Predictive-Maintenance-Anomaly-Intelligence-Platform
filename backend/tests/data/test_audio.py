import numpy as np
import pytest
from backend.app.data.audio_loader import AudioLoader
from backend.app.data.spectrogram import MelSpectrogramExtractor

def test_audio_loader_padding():
    loader = AudioLoader(sample_rate=16000, duration=1.0)
    # create short dummy audio
    dummy = np.zeros(8000)
    
    # Needs to mock librosa.load, but we can test the padding logic if we refactor or 
    # test a dummy file. For now, asserting basic configurations.
    assert loader.target_length == 16000

def test_spectrogram_shape():
    extractor = MelSpectrogramExtractor(sample_rate=16000, n_mels=128, hop_length=512)
    dummy_audio = np.random.randn(16000)
    spec = extractor.extract(dummy_audio)
    
    assert spec.shape[0] == 128
    # 16000 / 512 + 1 = 32 frames
    assert spec.shape[1] == 32
