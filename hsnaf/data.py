"""Data loading: unmixing benchmark scenes and EnMAP training tiles."""

from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_wavelengths(path):
    """Reads band center wavelengths (in nm).

    Accepts either a text file with one value per line, or a CSV file with a
    ``center_nm`` column (e.g. the EnMAP band table shipped with SpectralEarth).
    """
    path = Path(path)
    if path.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path)["center_nm"].astype(float).tolist()
    return [float(line) for line in path.read_text().split() if line.strip()]


def load_unmixing_scene(name, data_dir=DATA_DIR / "unmixing", normalize=True):
    """Loads a hyperspectral unmixing benchmark scene (Jasper Ridge, Samson, APEX, Urban).

    Expects ``<data_dir>/<name>.mat`` with keys ``Y`` (bands x pixels),
    ``A`` (endmembers x pixels) and ``E`` (bands x endmembers), and
    ``<data_dir>/<name>_wavelength.txt``. Scenes are assumed to be square.

    Returns:
        Y: image, ``(1, B, H, W)``, scaled to max 1 if ``normalize``
        A: ground-truth abundances, ``(1, c, H, W)``
        E: ground-truth endmembers, ``(B, c)``
        wavelengths: list of ``B`` band centers in nm
    """
    data_dir = Path(data_dir)
    mat = scipy.io.loadmat(data_dir / f"{name}.mat")
    Y = torch.tensor(mat["Y"], dtype=torch.float)
    A = torch.tensor(mat["A"], dtype=torch.float)
    E = torch.tensor(mat["E"], dtype=torch.float)

    side = int(round(Y.shape[1] ** 0.5))
    Y = Y.reshape(1, -1, side, side)
    A = A.reshape(1, -1, side, side)
    if normalize:
        Y = Y / Y.max()

    wavelengths = read_wavelengths(data_dir / f"{name}_wavelength.txt")
    return Y, A, E, wavelengths


class EnmapTiles(Dataset):
    """EnMAP tiles from the SpectralEarth dataset (GeoTIFF, 202 bands, 128x128).

    Each tile is scaled to max 1.
    """

    def __init__(self, folder, max_images=None):
        self.paths = sorted(Path(folder).glob("*.tif"))
        if not self.paths:
            raise FileNotFoundError(f"No .tif file found in {folder}")
        if max_images is not None:
            self.paths = self.paths[:max_images]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import rasterio

        with rasterio.open(self.paths[idx]) as src:
            tile = torch.from_numpy(src.read().astype(np.float32))
        return tile / tile.max()
