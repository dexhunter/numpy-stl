"""Peak traced memory for one ``BaseMesh.update_centroids`` call.

Workload: a 1_000_000-triangle strip. ``tracemalloc`` starts after the mesh
exists, so the mesh buffer itself is not included. The spread is
``(max - min) / median`` across 5 calls.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from stl.base import BaseMesh

N_TRIANGLES = 1_000_000
REPEATS = 5


def _strip_mesh() -> BaseMesh:
    data = np.zeros(N_TRIANGLES, dtype=BaseMesh.dtype)
    index = np.arange(N_TRIANGLES)
    parity = (index % 2).astype(np.float32)
    data['vectors'][:, 0, 0] = index
    data['vectors'][:, 0, 1] = parity
    data['vectors'][:, 1, 0] = index + 1
    data['vectors'][:, 1, 1] = np.float32(1.0) - parity
    data['vectors'][:, 2, 0] = index + 2
    data['vectors'][:, 2, 1] = parity
    return BaseMesh(data, calculate_normals=False, remove_empty_areas=False)


def _peak_bytes(mesh: BaseMesh) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        mesh.update_centroids()
        if mesh.centroids.shape != (N_TRIANGLES, 3):
            raise SystemExit(f'unexpected centroids shape {mesh.centroids.shape}')
        _current, peak = tracemalloc.get_traced_memory()
        return peak - before
    finally:
        tracemalloc.stop()


def _median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main() -> None:
    mesh = _strip_mesh()
    _peak_bytes(mesh)
    samples = [_peak_bytes(mesh) for _ in range(REPEATS)]
    median = _median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f'baseline_value={median}')
    print(f'baseline_noise_pct={noise:.4f}')
    print('runs_bytes=' + ','.join(str(sample) for sample in samples))


if __name__ == '__main__':
    main()
