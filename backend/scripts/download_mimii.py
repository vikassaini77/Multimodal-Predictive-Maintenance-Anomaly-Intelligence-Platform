import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

if __name__ == "__main__":
    # Dummy Zenodo URL for MIMII dataset
    # In production, add checksum verification here
    print("Downloading MIMII Dataset...")
    # download_url('https://zenodo.org/record/3384388/files/-.zip', 'mimii.zip')
