# from .getnaturalmovies import GetNaturalMovies
# from .prefilter import (build_coeff_cache, PrefilterConfig, 
#                         ensure_cache, sigma_for, movie_fingerprint)
# from .vanhateren_utils import split_images, load_iml
# from .plotting import show_hdr, animate_sample, plot_examples
# from .caching import DiskCachedDataset
# from .prepare import prepare_data
# from .resident_evil import materialize


##PROPOSED API
# src/motionnet/dataset/__init__.py
"""Van Hateren natural-motion stimuli.

    data = get_data(cfg["stimulus"], paths.movies, device)
    for b in batches(data.train, 32, shuffle=True, generator=g): ...
"""

from .vanhateren import load_iml, normalize, prefilter, split_images
from .stimulus import (Data, DataInfo, MovieParams,
                       batches, get_data, make_movies)

__all__ = ["get_data", "batches", "make_movies", "MovieParams", "Data", "DataInfo",
           "load_iml", "normalize", "prefilter", "split_images"]