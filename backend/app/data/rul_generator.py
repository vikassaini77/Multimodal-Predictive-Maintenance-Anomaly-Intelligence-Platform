import pandas as pd
import numpy as np

class RULGenerator:
    def __init__(self, cap: int = 125):
        self.cap = cap

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycles.rename(columns={'time_cycles': 'max_cycle'}, inplace=True)
        df = df.merge(max_cycles, on=['unit_nr'], how='left')
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        df['RUL'] = np.where(df['RUL'] > self.cap, self.cap, df['RUL'])
        df.drop(columns=['max_cycle'], inplace=True)
        return df
