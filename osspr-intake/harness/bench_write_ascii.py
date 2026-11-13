"""Time pure-Python ASCII STL saving.

Workload: the same 20,000-triangle patch as the loader, written with
``_write_ascii``. That is the body of ``Mesh.save(..., mode=ASCII)`` when
the optional speedups extra is absent or the stream is not a real file.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stl import mesh  # noqa: E402

from workloads import measure, open_grid, report  # noqa: E402

TRIANGLES_PER_SIDE = 100


def main() -> None:
    surface = mesh.Mesh(
        open_grid(TRIANGLES_PER_SIDE),
        calculate_normals=True,
        remove_empty_areas=False,
        speedups=False,
    )

    buffer = io.BytesIO()

    def write() -> int:
        buffer.seek(0)
        buffer.truncate(0)
        surface._write_ascii(buffer, b'surface')
        return buffer.tell()

    size = write()
    if size < 1_000_000:
        raise SystemExit(f'ASCII output unexpectedly small: {size} bytes')
    report(measure(write))


if __name__ == '__main__':
    main()
