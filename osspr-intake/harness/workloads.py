"""Deterministic workloads shared by the intake benchmarks."""

from __future__ import annotations

import gc
import os
import statistics
import time

import numpy as np

from stl.base import BaseMesh

REPEATS = 7


def measure(fn, repeats: int = REPEATS) -> list[float]:
    """Warm up, then return per-call wall times in milliseconds.

    A few untimed calls let the CPU leave its idle frequency and let
    NumPy finish one-time dispatch. The collector is drained before each
    sample and paused during the call so a nursery collection does not
    land in some repeats and not others. Allocation work inside the call
    is still included.
    """
    # Stay on one core so the samples are not split across a migration.
    # cpu 0 exists on every machine this harness runs on.
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass
    for _ in range(4):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        try:
            started = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - started
        finally:
            gc.enable()
        samples.append(elapsed * 1000.0)
    return samples


def report(samples: list[float]) -> None:
    """Print the median and the full spread, (max - min) / median."""
    median = statistics.median(samples)
    noise_pct = (max(samples) - min(samples)) / median * 100.0
    rendered = ','.join(f'{sample:.4f}' for sample in samples)
    print(f'median_ms={median:.4f}')
    print(f'noise_pct={noise_pct:.4f}')
    print(f'samples_ms={rendered}')


def open_grid(n: int) -> np.ndarray:
    """n*n*2 triangles on a warped plane, shaped like a scanned patch."""
    xs = np.linspace(-50.0, 50.0, n + 1, dtype=np.float64)
    ys = np.linspace(-20.0, 80.0, n + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    zz = 0.15 * xx + 0.05 * np.sin(yy)
    points = np.stack((xx, yy, zz), axis=-1).astype(np.float32)
    v00 = points[:-1, :-1]
    v10 = points[1:, :-1]
    v01 = points[:-1, 1:]
    v11 = points[1:, 1:]
    lower = np.stack((v00, v10, v11), axis=2).reshape(-1, 3, 3)
    upper = np.stack((v00, v11, v01), axis=2).reshape(-1, 3, 3)
    data = np.zeros(lower.shape[0] + upper.shape[0], dtype=BaseMesh.dtype)
    data['vectors'] = np.concatenate((lower, upper), axis=0)
    return data


def convex_box(n: int) -> np.ndarray:
    """Closed box with an n*n grid on each face. Outward winding, convex.

    Coordinates are whole millimeters so the face-plane test is exact in
    float32. Every vertex lies on or behind every face, so ``is_convex``
    has to scan the whole mesh instead of returning on the first face.
    """
    side = float(n)

    def face(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray):
        steps = np.linspace(0.0, side, n + 1, dtype=np.float64)
        uu, vv = np.meshgrid(steps, steps, indexing='ij')
        points = (
            origin + uu[..., None] * axis_u + vv[..., None] * axis_v
        ).astype(np.float32)
        v00 = points[:-1, :-1]
        v10 = points[1:, :-1]
        v01 = points[:-1, 1:]
        v11 = points[1:, 1:]
        lower = np.stack((v00, v10, v11), axis=2).reshape(-1, 3, 3)
        upper = np.stack((v00, v11, v01), axis=2).reshape(-1, 3, 3)
        return np.concatenate((lower, upper), axis=0)

    faces = (
        # z = 0, outward -Z
        face(
            np.array([0.0, side, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ),
        # z = side, outward +Z
        face(
            np.array([0.0, 0.0, side]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
        # y = 0, outward -Y
        face(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        # y = side, outward +Y
        face(
            np.array([side, side, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
        # x = 0, outward -X
        face(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        ),
        # x = side, outward +X
        face(
            np.array([side, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    triangles = np.concatenate(faces, axis=0)
    data = np.zeros(len(triangles), dtype=BaseMesh.dtype)
    data['vectors'] = triangles
    return data


def binary_grid_ply(n: int) -> bytes:
    """Binary little-endian PLY of an n*n quad grid, two triangles per cell.

    This is the layout most scanners emit: float x/y/z vertices and a
    ``list uchar int`` face index property.
    """
    steps = np.linspace(0.0, 100.0, n + 1, dtype=np.float32)
    xx, yy = np.meshgrid(steps, steps, indexing='ij')
    zz = (0.02 * xx * yy).astype(np.float32)
    vertices = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    index = np.arange((n + 1) * (n + 1), dtype=np.int32).reshape(n + 1, n + 1)
    i00 = index[:-1, :-1].ravel()
    i10 = index[1:, :-1].ravel()
    i01 = index[:-1, 1:].ravel()
    i11 = index[1:, 1:].ravel()
    faces = np.stack(
        (
            np.stack((i00, i10, i11), axis=1),
            np.stack((i00, i11, i01), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3)
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        f'element vertex {len(vertices)}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        f'element face {len(faces)}\n'
        'property list uchar int vertex_indices\n'
        'end_header\n'
    ).encode('ascii')
    packed = np.empty(
        len(faces),
        dtype=np.dtype([('n', 'u1'), ('i', '<i4', (3,))]),
    )
    packed['n'] = 3
    packed['i'] = faces.astype('<i4', copy=False)
    return header + vertices.astype('<f4').tobytes() + packed.tobytes()
