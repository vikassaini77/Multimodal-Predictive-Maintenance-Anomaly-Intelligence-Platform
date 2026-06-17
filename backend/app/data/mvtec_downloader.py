import os
import tarfile
import urllib.request
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

MVTEC_URL_BASE = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094"

def download_mvtec(data_dir: str = "data/mvtec", categories: List[str] = None):
    """
    Downloads and extracts MVTec AD dataset categories.
    Categories used by default for benchmarking: bottle, cable, carpet.
    """
    if categories is None:
        categories = ["bottle", "cable", "carpet"]
        
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for category in categories:
        category_dir = data_path / category
        if category_dir.exists():
            logger.info(f"Category '{category}' already exists in {category_dir}. Skipping download.")
            continue
            
        tar_filename = f"{category}.tar.xz"
        tar_filepath = data_path / tar_filename
        url = f"{MVTEC_URL_BASE}/{tar_filename}"
        
        try:
            logger.info(f"Downloading {category} from {url}...")
            urllib.request.urlretrieve(url, tar_filepath)
            
            logger.info(f"Extracting {tar_filename}...")
            with tarfile.open(tar_filepath) as tar:
                tar.extractall(path=data_path)
                
            logger.info(f"Successfully downloaded and extracted {category}.")
        except Exception as e:
            logger.error(f"Failed to download or extract {category}: {e}")
            raise
        finally:
            if tar_filepath.exists():
                os.remove(tar_filepath)
                logger.info(f"Cleaned up {tar_filename}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_mvtec()
