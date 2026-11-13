"""Time the full convexity scan.

Workload: a convex box whose faces are each a 12 by 12 grid (1,728
triangles, integer millimeter coordinates). ``is_convex`` projects every
vertex onto every face from Python, and a convex mesh cannot return early.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stl import mesh  # noqa: E402

from workloads import convex_box, measure, report  # noqa: E402

CELLS_PER_FACE = 12


def main() -> None:
    solid = mesh.Mesh(
        convex_box(CELLS_PER_FACE),
        calculate_normals=True,
        remove_empty_areas=False,
        speedups=False,
    )
    if len(solid.data) != 6 * CELLS_PER_FACE * CELLS_PER_FACE * 2:
        raise SystemExit(f'unexpected triangle count {len(solid.data)}')

    def check() -> bool:
        return solid.is_convex()

    if check() is not True:
        raise SystemExit('box is not convex; the scan returned early')
    report(measure(check))


if __name__ == '__main__':
    main()
