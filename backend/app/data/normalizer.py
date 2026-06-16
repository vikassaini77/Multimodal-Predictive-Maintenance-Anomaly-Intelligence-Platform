import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class CMAPSSNormalizer:
    def __init__(self, scaler_path="models/scalers/cmapss_scaler.pkl"):
        self.scaler = MinMaxScaler()
        self.scaler_path = scaler_path

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df_norm = df.copy()
        df_norm[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        return df_norm

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df_norm = df.copy()
        df_norm[feature_cols] = self.scaler.transform(df[feature_cols])
        return df_norm
