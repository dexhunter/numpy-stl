"""Time pure-Python ASCII STL loading.

Workload: a 20,000-triangle warped patch (about 3 MiB of ASCII STL), the
size of a small scan. ``_load_ascii(..., speedups=False)`` is the path
used when the optional speedups extra is not installed.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stl import HEADER_SIZE, mesh  # noqa: E402
from stl.stl import BaseStl  # noqa: E402

from workloads import measure, open_grid, report  # noqa: E402

TRIANGLES_PER_SIDE = 100


def _blob() -> bytes:
    data = open_grid(TRIANGLES_PER_SIDE)
    surface = mesh.Mesh(
        data,
        calculate_normals=True,
        remove_empty_areas=False,
        speedups=False,
    )
    buffer = io.BytesIO()
    surface._write_ascii(buffer, b'surface')
    return buffer.getvalue()


def main() -> None:
    blob = _blob()
    expected = TRIANGLES_PER_SIDE * TRIANGLES_PER_SIDE * 2

    handle = io.BytesIO(blob)

    def load() -> int:
        handle.seek(0)
        header = handle.read(HEADER_SIZE)
        _name, data = BaseStl._load_ascii(handle, header, speedups=False)
        return len(data)

    count = load()
    if count != expected:
        raise SystemExit(f'expected {expected} triangles, loaded {count}')
    report(measure(load))


if __name__ == '__main__':
    main()
