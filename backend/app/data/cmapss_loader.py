import pandas as pd

CMAPSS_COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]

class CMAPSSLoader:
    def load(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, sep='\s+', header=None, names=CMAPSS_COLUMNS)
        return df
