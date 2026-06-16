import numpy as np
import pandas as pd

class SequenceExtractor:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def extract(self, df: pd.DataFrame, feature_cols: list, label_col: str = 'RUL'):
        seqs, labels = [], []
        for unit, group in df.groupby('unit_nr'):
            features = group[feature_cols].values
            rul = group[label_col].values
            for i in range(len(features) - self.window_size + 1):
                seqs.append(features[i:i + self.window_size])
                labels.append(rul[i + self.window_size - 1])
        return np.array(seqs), np.array(labels)
