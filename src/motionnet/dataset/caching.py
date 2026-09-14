import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _fingerprint_diff(stored, wanted):
    """Human-readable diff of two fingerprint strings. Falls back to repr."""
    try:
        a, b = json.loads(stored), json.loads(wanted)
    except (TypeError, ValueError):
        return f"  stored: {stored!r}\n  wanted: {wanted!r}"

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{prefix}{k}."))
            else:
                out[f"{prefix}{k}"] = v
        return out

    fa, fb = flat(a), flat(b)
    lines = [f"  {k}: {fa.get(k)!r} -> {fb.get(k)!r}"
             for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    return "\n".join(lines) or "  (no scalar differences; check key order)"


class DiskCachedDataset(Dataset):
    """Memoizes a deterministic base dataset to per-key memmaps on disk.

    File creation happens once, in `__init__`, on whichever process
    constructs the dataset -- before a DataLoader forks worker processes.
    Workers only ever open already-existing files, so there is no
    create-vs-truncate race between concurrent workers.

    A `_meta.json` sidecar records sample count, per-key shape/dtype, and
    an optional caller-supplied `fingerprint`; a mismatch on reopen either
    raises or rebuilds, per `on_mismatch`, instead of silently pairing
    stale cached samples with a new config.
    """

    def __init__(self, base_ds, cache_dir, fingerprint=None, on_mismatch="raise"):
        if on_mismatch not in ("raise", "rebuild"):
            raise ValueError("on_mismatch must be 'raise' or 'rebuild'")
        self.base_ds, self.cache_dir = base_ds, Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.on_mismatch = on_mismatch
        self._mmaps = None          # opened lazily, once per worker process
        self._keys, self._specs = self._ensure_files()

    def __len__(self):
        # Must live on the class: len() reads the type slot directly and
        # never falls through to __getattr__ below.
        return len(self.base_ds)

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails. Unpickling (a
        # DataLoader worker under `spawn`) and copy.deepcopy both build an
        # empty instance and probe it before restoring __dict__; forwarding
        # then would look up `base_ds`, fail, and land back here. Reading
        # __dict__ directly is the one lookup that never re-enters.
        if "base_ds" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.base_ds, name)

    def _meta_path(self):
        return self.cache_dir / "_meta.json"

    def _stale(self, reason, diff=""):
        """Either explain and raise, or wipe the directory and start over."""
        msg = f"{self.cache_dir} is stale: {reason}"
        if diff:
            msg += f"\n{diff}"
        if self.on_mismatch == "raise":
            raise ValueError(msg + "\n\nPass on_mismatch='rebuild' to discard "
                                   "and regenerate, or use a different cache_dir.")
        warnings.warn(msg + "\nRebuilding.", RuntimeWarning, stacklevel=3)
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_files(self):
        """Create (or validate) the on-disk cache. Runs once, single-process."""
        n = len(self.base_ds)
        meta_path = self._meta_path()

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["n"] != n:
                self._stale(f"built for {meta['n']} samples, this dataset has {n}")
            elif meta.get("fingerprint") != self.fingerprint:
                self._stale("config fingerprint changed",
                            _fingerprint_diff(meta.get("fingerprint"),
                                              self.fingerprint))

        # _stale() may have wiped the directory, so re-test rather than else.
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
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
                self._mmaps[k].flush()   # data before flag; nothing else orders them
            self._filled[idx] = True
            return sample
        return {k: torch.from_numpy(np.array(self._mmaps[k][idx]))
                for k in self._keys}