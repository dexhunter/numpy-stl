"""Baseline timing for ``stl.ply._read_binary_vertices``.

Workload: 400_000 packed little-endian float32 xyz vertices, the layout
``write_ply`` emits for a binary PLY. The timed region is one call.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from stl.ply import _Property, _Element, _read_binary_vertices

N_VERTICES = 400_000
REPEATS = 5


def _vertex_blob() -> tuple[bytes, _Element]:
    coords = np.empty((N_VERTICES, 3), dtype='<f4')
    coords[:, 0] = np.arange(N_VERTICES, dtype=np.float32)
    coords[:, 1] = np.float32(1.0)
    coords[:, 2] = np.float32(0.5)
    element = _Element('vertex', N_VERTICES)
    for name in ('x', 'y', 'z'):
        element.properties.append(_Property(name, 'float'))
    return coords.tobytes(), element


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    raw, element = _vertex_blob()

    def run() -> np.ndarray:
        return _read_binary_vertices(raw, element, '<', 0, 1, 2)

    sample = run()
    if sample.shape != (N_VERTICES, 3) or sample.dtype != np.float32:
        raise SystemExit(f'unexpected vertex array {sample.shape} {sample.dtype}')
    if sample[0, 0] != np.float32(0) or sample[-1, 0] != np.float32(N_VERTICES - 1):
        raise SystemExit('vertex values did not round-trip')

    run()
    samples_ms: list[float] = []
    for _ in range(REPEATS):
        gc.collect()
        started = time.perf_counter()
        run()
        samples_ms.append((time.perf_counter() - started) * 1000.0)

    median = _median(samples_ms)
    noise = (max(samples_ms) - min(samples_ms)) / median * 100.0
    print(f'baseline_median_ms={median:.4f}')
    print(f'baseline_noise_pct={noise:.4f}')
    print('runs_ms=' + ','.join(f'{sample:.4f}' for sample in samples_ms))


if __name__ == '__main__':
    main()
