# src/motionnet/dataset/prepare.py
from pathlib import Path
from .getnaturalmovies import GetNaturalMovies
from .caching import DiskCachedDataset
from .prefilter import PrefilterConfig, ensure_cache, sigma_for, movie_fingerprint
from .vanhateren_utils import split_images
from .resident_evil import materialize
import json
from types import SimpleNamespace
from dataclasses import dataclass, asdict      # add


@dataclass(frozen=True)                        # define BEFORE prepare_data
class DataInfo:
    fps: float; gray_frames: int; dpp: float; deg_per_tap: float
    sigma_px: float; spacing_px: float; T: int
    n_train: int; n_val: int; resident_gib: float
    train_acceptance: float | None = None
    val_acceptance: float | None = None
    train_fallbacks: int | None = None
    val_fallbacks: int | None = None


def prepare_data(cfg, device, coeff_dir, movie_dir,
                 keys=("movie", "vel_deg_s", "contrast_rms")):
    """files -> coeff cache -> movie cache -> resident tensors.

    Returns (train_mem, val_mem, info) where info carries fps / gray_frames / dpp.
    """
    d = cfg["stimulus"]
    mcfg = d["movie"]
    pre = PrefilterConfig(sigma_px=sigma_for(mcfg["blur_px"]))
    tag = f"blur{mcfg['blur_px']}_v{int(pre.sigma_px * 100)}"

    train_files, val_files = split_images(d["corpus"], **d["split"])

    mem, bases, stats = {}, {}, {}
    for name, files, seed in (("train", train_files, d["ds_seed_train"]),
                              ("val",   val_files,  d["ds_seed_val"])):
        coeff = Path(coeff_dir) / f"{name}_n{len(files)}_{tag}.npy"
        ensure_cache(files, coeff, pre)

        base = GetNaturalMovies(files=files, coeff_cache=coeff, seed=seed, **mcfg)
        cache_dir = Path(movie_dir) / f"movies_{name}"
        ds = DiskCachedDataset(
            base, cache_dir=cache_dir,
            fingerprint=movie_fingerprint(seed, mcfg, pre, len(files)),
            on_mismatch="rebuild")

        mem[name] = materialize(ds, keys=keys, device=device)
        bases[name] = base

        # acceptance is only observable on a cold build -> persist it
        stats_p = cache_dir / "_stats.json"
        if base._n_draws > 0:
            stats_p.write_text(json.dumps(dict(
                acceptance_rate=base.acceptance_rate,
                n_fallback=base._n_fallback,
                n_samples=base._n_samples)))
        s = json.loads(stats_p.read_text()) if stats_p.exists() else {}
        stats[f"{name}_acceptance"] = s.get("acceptance_rate")
        stats[f"{name}_fallbacks"] = s.get("n_fallback")

    b = bases["train"]
    info = DataInfo(                           # was dict(...)
        fps=b.fps, gray_frames=b.gray_frames, dpp=b.dpp,
        deg_per_tap=b.deg_per_tap, sigma_px=b.sigma_px,
        spacing_px=b.spacing_px, T=b.T,
        n_train=len(mem["train"]), n_val=len(mem["val"]),
        resident_gib=(mem["train"].nbytes + mem["val"].nbytes) / 2**30,
        **stats)
    return mem["train"], mem["val"], info
