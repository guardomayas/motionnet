import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DiskCachedDataset(Dataset):
    """Memoizes a deterministic base dataset to per-key memmaps on disk.

    File creation happens once, in `__init__`, on whichever process
    constructs the dataset -- before a DataLoader forks worker processes.
    Workers only ever open already-existing files, so there is no
    create-vs-truncate race between concurrent workers.

    A `_meta.json` sidecar records sample count, per-key shape/dtype, and
    an optional caller-supplied `fingerprint`; a mismatch on reopen raises
    instead of silently pairing stale cached samples with a new config.
    """

    def __init__(self, base_ds, cache_dir, fingerprint=None):
        self.base_ds, self.cache_dir = base_ds, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self._mmaps = None          # opened lazily, once per worker process
        self._keys, self._specs = self._ensure_files()

    def __len__(self):
        return len(self.base_ds)

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails, so this can't
        # shadow anything set in __init__ or recurse on those.
        return getattr(self.base_ds, name)

    def _meta_path(self):
        return self.cache_dir / "_meta.json"

    def _ensure_files(self):
        """Create (or validate) the on-disk cache. Runs once, single-process."""
        n = len(self.base_ds)
        meta_path = self._meta_path()

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["n"] != n:
                raise ValueError(
                    f"{self.cache_dir} was built for {meta['n']} samples, "
                    f"this dataset has {n}; use a different cache_dir or "
                    f"delete it to rebuild.")
            if meta.get("fingerprint") != self.fingerprint:
                raise ValueError(
                    f"{self.cache_dir} was built with fingerprint "
                    f"{meta.get('fingerprint')!r}, this dataset was given "
                    f"{self.fingerprint!r}; use a different cache_dir or "
                    f"delete it to rebuild.")
            keys, specs = meta["keys"], meta["specs"]
        else:
            sample0 = self.base_ds[0]
            keys = list(sample0)
            specs = {}
            for k, v in sample0.items():
                arr0 = np.asarray(v.numpy() if torch.is_tensor(v) else v)
                specs[k] = {"dtype": str(arr0.dtype), "shape": list(arr0.shape)}
            meta_path.write_text(json.dumps({
                "n": n, "keys": keys, "specs": specs,
                "fingerprint": self.fingerprint,
            }))

        for k in keys:
            path = self.cache_dir / f"{k}.npy"
            if not path.exists():
                shape = (n, *specs[k]["shape"])
                np.lib.format.open_memmap(path, mode="w+",
                                          dtype=np.dtype(specs[k]["dtype"]),
                                          shape=shape)
        filled_path = self.cache_dir / "_filled.npy"
        if not filled_path.exists():
            np.lib.format.open_memmap(filled_path, mode="w+", dtype=bool,
                                      shape=(n,))
        return keys, specs

    def _open_mmaps(self):
        self._mmaps = {k: np.load(self.cache_dir / f"{k}.npy", mmap_mode="r+")
                       for k in self._keys}
        self._filled = np.load(self.cache_dir / "_filled.npy", mmap_mode="r+")

    def __getitem__(self, idx):
        if self._mmaps is None:
            self._open_mmaps()
        if not self._filled[idx]:
            sample = self.base_ds[idx]
            for k in self._keys:
                v = sample[k]
                self._mmaps[k][idx] = v.numpy() if torch.is_tensor(v) else v
            self._filled[idx] = True
            return sample
        return {k: torch.from_numpy(np.array(self._mmaps[k][idx]))
               for k in self._keys}
