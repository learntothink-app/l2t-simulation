"""Shared utilities: seeding, JSON I/O, logging, manifest."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def setup_logging(level: str = "INFO", logfile: str | os.PathLike | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="w"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _to_serialisable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (Path,)):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_serialisable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    return obj


def save_json(path: str | os.PathLike, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_serialisable(payload), fh, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_environment() -> dict[str, Any]:
    pkgs = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "?")
        except ModuleNotFoundError:
            pkgs[name] = "missing"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_hash": _git_hash(),
        "packages": pkgs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
