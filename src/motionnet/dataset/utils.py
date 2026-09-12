from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

class DiskCachedDataset(Dataset):
    """Memoizes a deterministic base dataset to per-key memmaps on disk."""
    def __init__(self, base_ds, cache_dir):
        self.base_ds, self.cache_dir = base_ds, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mmaps = None          # opened lazily, once per worker process

    def __len__(self):
        return len(self.base_ds)

    def _init_mmaps(self):
        sample0 = self.base_ds[0]
        self._keys = list(sample0)
        self._mmaps = {}
        for k, v in sample0.items():
            arr0 = np.asarray(v.numpy() if torch.is_tensor(v) else v)
            path = self.cache_dir / f"{k}.npy"
            shape = (len(self.base_ds), *arr0.shape)
            self._mmaps[k] = (np.load(path, mmap_mode="r+") if path.exists()
                else np.lib.format.open_memmap(path, mode="w+", dtype=arr0.dtype, shape=shape))
        fpath = self.cache_dir / "_filled.npy"
        self._filled = (np.load(fpath, mmap_mode="r+") if fpath.exists()
            else np.lib.format.open_memmap(fpath, mode="w+", dtype=bool, shape=(len(self.base_ds),)))

    def __getitem__(self, idx):
        if self._mmaps is None:
            self._init_mmaps()
        if not self._filled[idx]:
            sample = self.base_ds[idx]
            for k in self._keys:
                v = sample[k]
                self._mmaps[k][idx] = v.numpy() if torch.is_tensor(v) else v
            self._filled[idx] = True
            return sample
        return {k: torch.from_numpy(np.array(self._mmaps[k][idx])) for k in self._keys}