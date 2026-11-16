"""Peak traced memory for one ``stl.ply.write_ply`` call.

Workload: a 200_000-triangle strip (shared edge vertices, the usual case
once an STL is exported to PLY) written as binary little-endian PLY into a
memory buffer. ``tracemalloc`` is started after the mesh exists, so the
reported bytes are allocations made inside the call. The spread is
``(max - min) / median`` across 5 calls.
"""

from __future__ import annotations

import gc
import io
import sys
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from stl.base import BaseMesh
from stl.ply import write_ply

N_TRIANGLES = 200_000
REPEATS = 5


def _strip_mesh() -> np.ndarray:
    data = np.zeros(N_TRIANGLES, dtype=BaseMesh.dtype)
    index = np.arange(N_TRIANGLES)
    parity = (index % 2).astype(np.float32)
    data['vectors'][:, 0, 0] = index
    data['vectors'][:, 0, 1] = parity
    data['vectors'][:, 1, 0] = index + 1
    data['vectors'][:, 1, 1] = np.float32(1.0) - parity
    data['vectors'][:, 2, 0] = index + 2
    data['vectors'][:, 2, 1] = parity
    return data


def _peak_bytes(mesh: np.ndarray) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        buffer = io.BytesIO()
        write_ply(
            buffer,
            mesh,
            name='strip',
            mode='binary_little_endian',
        )
        if buffer.tell() < 84:
            raise SystemExit('PLY writer produced an empty file')
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
