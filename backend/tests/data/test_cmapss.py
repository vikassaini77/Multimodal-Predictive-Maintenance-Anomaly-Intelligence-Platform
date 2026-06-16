import pytest
import pandas as pd
import numpy as np
from backend.app.data.rul_generator import RULGenerator
from backend.app.data.sequence_extractor import SequenceExtractor
from backend.app.data.normalizer import CMAPSSNormalizer

@pytest.mark.parametrize("cap", [125, 130])
def test_rul_generator(cap):
    df = pd.DataFrame({
        'unit_nr': [1, 1, 1],
        'time_cycles': [1, 2, 3]
    })
    generator = RULGenerator(cap=cap)
    df_rul = generator.generate(df)
    assert 'RUL' in df_rul.columns
    assert df_rul['RUL'].iloc[0] == min(2, cap)

def test_sequence_extractor():
    df = pd.DataFrame({
        'unit_nr': [1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4],
        's_1': [0.1, 0.2, 0.3, 0.4],
        'RUL': [3, 2, 1, 0]
    })
    extractor = SequenceExtractor(window_size=2)
    seqs, labels = extractor.extract(df, ['s_1'])
    assert seqs.shape == (3, 2, 1)
    assert len(labels) == 3
