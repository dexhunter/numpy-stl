"""Baseline timing for ``stl.ply._build_mesh_data``.

Workload: 200_000 triangles given as a list of index tuples into a
100_000-vertex array, which is the structure ``_triangulate`` hands this
function. The timed region is one call.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from stl.base import BaseMesh
from stl.ply import _build_mesh_data

N_TRIANGLES = 200_000
N_VERTICES = 100_000
REPEATS = 5


def _workload() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    vertices = np.empty((N_VERTICES, 3), dtype=np.float32)
    vertices[:, 0] = np.arange(N_VERTICES, dtype=np.float32)
    vertices[:, 1] = np.float32(0.25)
    vertices[:, 2] = np.float32(-0.5)
    index = np.arange(N_TRIANGLES, dtype=np.int64)
    triangles = list(
        zip(
            (index % N_VERTICES).tolist(),
            ((index + 1) % N_VERTICES).tolist(),
            ((index + 2) % N_VERTICES).tolist(),
            strict=True,
        )
    )
    return vertices, triangles


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    vertices, triangles = _workload()

    def run() -> np.ndarray:
        return _build_mesh_data(vertices, triangles, BaseMesh.dtype)

    sample = run()
    if len(sample) != N_TRIANGLES or sample.dtype != BaseMesh.dtype:
        raise SystemExit(f'unexpected mesh array {len(sample)} {sample.dtype}')
    expected = vertices[triangles[0][0]]
    if not np.array_equal(sample['vectors'][0, 0], expected):
        raise SystemExit('first triangle vertex did not match')

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
