"""Time binary PLY loading.

Workload: a 51,200-triangle binary little-endian grid (float vertices,
``list uchar int`` faces), about 1 MiB. This is the common scanner layout
and it always goes through ``stl.ply.read_ply``; there is no speedups path.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stl.base import BaseMesh  # noqa: E402
from stl.ply import read_ply  # noqa: E402

from workloads import binary_grid_ply, measure, report  # noqa: E402

CELLS_PER_SIDE = 160


def main() -> None:
    blob = binary_grid_ply(CELLS_PER_SIDE)
    expected = CELLS_PER_SIDE * CELLS_PER_SIDE * 2

    handle = io.BytesIO(blob)

    def load() -> int:
        handle.seek(0)
        data, _name = read_ply(handle, BaseMesh.dtype)
        return len(data)

    count = load()
    if count != expected:
        raise SystemExit(f'expected {expected} triangles, loaded {count}')
    report(measure(load))


if __name__ == '__main__':
    main()
