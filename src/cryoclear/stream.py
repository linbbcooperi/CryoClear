"""Real-time stream simulator (Tony).

Mimics a live microscope session by yielding micrographs one at a time on a
timer, so the UI can show picks + junk flags appearing "live" and a throughput
dashboard (micrographs/min). Works without a GPU using the blob picker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class StreamStats:
    n_micrographs: int = 0
    n_candidates: int = 0
    n_kept: int = 0
    n_junk: int = 0
    started: float = field(default_factory=time.time)

    @property
    def micrographs_per_min(self) -> float:
        elapsed = max(time.time() - self.started, 1e-6)
        return self.n_micrographs / elapsed * 60.0

    @property
    def junk_fraction(self) -> float:
        return self.n_junk / max(self.n_candidates, 1)


def stream_micrographs(folder: str | Path, pattern: str = "*.mrc",
                       interval_s: float = 1.0) -> Iterator[Path]:
    """Yield micrograph paths on a timer to simulate live data collection."""
    paths = sorted(Path(folder).glob(pattern))
    for p in paths:
        yield p
        time.sleep(interval_s)
