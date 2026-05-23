"""
utils.py — Shared utilities for the Vietnamese Financial News RAG system.

Provides:
  - load_config()     : load configs/config.yaml
  - get_env()         : read API keys from .env with fallback to Colab userdata
  - resolve_path()    : switch between local and Colab Drive paths
  - is_colab()        : detect runtime environment
  - setup_logger()    : standardized logging
  - ensure_dir()      : mkdir -p wrapper
"""

import os
import logging
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict:
    """
    Load config.yaml.

    Searches in this order:
      1. Explicit path argument
      2. <project_root>/configs/config.yaml  (auto-detected)

    Returns a plain Python dict.
    """
    if config_path is None:
        # Walk up from this file to find configs/config.yaml
        this_file = Path(__file__).resolve()
        project_root = this_file.parent.parent  # src/ -> implementation/
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# Path resolver (local <-> Colab)
# ---------------------------------------------------------------------------

def resolve_path(config: dict, key: str) -> str:
    """
    Resolve a config path key for the current environment.

    Colab path keys end with '_colab'; local keys are the base key.
    Example:
        config['data']['raw_path']        -> local
        config['data']['raw_path_colab']  -> Colab

    Usage:
        path = resolve_path(config['data'], 'raw_path')
    """
    if is_colab():
        colab_key = key + "_colab"
        if colab_key in config:
            return config[colab_key]
    return config[key]


# ---------------------------------------------------------------------------
# API key loader
# ---------------------------------------------------------------------------

def get_env(key: str, default: str | None = None) -> str | None:
    """
    Retrieve an API key or environment variable.

    Priority:
      1. Environment variable (os.environ) — works everywhere
      2. .env file in project root (via python-dotenv)
      3. Google Colab userdata (when on Colab)
      4. default value

    Never raises; returns default/None if key not found.
    """
    # 1. Already in environment
    value = os.environ.get(key)
    if value:
        return value

    # 2. .env file
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            value = os.environ.get(key)
            if value:
                return value
    except ImportError:
        pass

    # 3. Colab userdata
    if is_colab():
        try:
            from google.colab import userdata
            value = userdata.get(key)
            if value:
                return value
        except Exception:
            pass

    return default


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a standardized logger writing to stdout.

    Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] name: message
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
