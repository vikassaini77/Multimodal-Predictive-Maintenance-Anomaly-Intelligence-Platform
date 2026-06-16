from dataclasses import dataclass
from typing import Optional

@dataclass
class DatasetConfig:
    name: str
    sample_rate: Optional[int] = None
    window_size: Optional[int] = None
    n_mels: Optional[int] = None
    batch_size: int = 32
