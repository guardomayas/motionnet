# src/motionnet/dataset/resident.py
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

_FP16_OK = ("movie",)          # large and noise-dominated; everything else stays fp32


class ResidentDataset(Dataset):
    """A deterministic dataset held entirely in memory (usually GPU)."""

    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.n = next(iter(tensors.values())).shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: (v[i].float() if v.dtype == torch.float16 else v[i])
                for k, v in self.tensors.items()}

    @property
    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.tensors.values())

    def batches(self, batch_size, shuffle=False, drop_last=False, generator=None):
        """Index-sliced batches. Faster than a DataLoader for resident tensors."""
        idx = (torch.randperm(self.n, device=self.tensors["movie"].device, generator=generator)
               if shuffle else torch.arange(self.n, device=self.tensors["movie"].device))
        end = self.n - (self.n % batch_size if drop_last else 0)
        for a in range(0, end, batch_size):
            j = idx[a:a + batch_size]
            yield {k: (v[j].float() if v.dtype == torch.float16 else v[j])
                   for k, v in self.tensors.items()}


@torch.no_grad()
def materialize(ds, keys=None, device=None, fp16_keys=_FP16_OK, progress=True):
    """One pass over a deterministic dataset into resident tensors."""
    s0 = ds[0]
    keys = list(s0) if keys is None else list(keys)
    out = {}
    for k in keys:
        dt = torch.float16 if k in fp16_keys else s0[k].dtype
        if dt.is_floating_point is False:
            dt = s0[k].dtype
        out[k] = torch.empty((len(ds), *s0[k].shape), dtype=dt)

    it = tqdm(range(len(ds)), desc="materializing") if progress else range(len(ds))
    for i in it:
        s = ds[i]
        for k in keys:
            out[k][i] = s[k].to(out[k].dtype)

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return ResidentDataset(out)