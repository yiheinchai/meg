from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

BASE_PATH = ROOT / "cc700" / "meg" / "pipeline" / "release005" / "BIDSsep"
REST_PATH = BASE_PATH / "rest"
NOISE_PATH = BASE_PATH / "noise"

CACHE_PATH = ROOT / "meg" / "cache"
MEG_CACHE_PATH = CACHE_PATH / "meg_cache"
WINDOW_CACHE_PATH = CACHE_PATH / "meg_windows_cache"

CHECKPOINT_PATH = ROOT / "meg" / "checkpoint"

HDF5_CACHE_PATH = CACHE_PATH / "hdf5_cache"
