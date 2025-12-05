from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_PATH = ROOT / "CamCAN" / "cc700"
REST_PATH = BASE_PATH / "rest"
NOISE_PATH = BASE_PATH / "noise"

CACHE_PATH = ROOT / "cache"
MEG_CACHE_PATH = CACHE_PATH / "meg_cache"
WINDOW_CACHE_PATH = CACHE_PATH / "meg_windows_cache"
