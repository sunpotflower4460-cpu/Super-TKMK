#!/usr/bin/env python3
"""
6次精度中心差分による div B 検証スクリプト。

離散 curl(A) から 3D 磁場を生成し、同じ 6 次精度中心差分で div B を評価する。
NumPy / CuPy の両方に対応し、float64 では丸め誤差オーダー（~1e-14 以下）を目標とする。
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


CYLINDRICAL_RADIUS_EPSILON = 1.0e-30
TOROIDAL_MAJOR_RADIUS_RATIO = 0.30
TOROIDAL_MINOR_RADIUS_RATIO = 0.12
TOROIDAL_VERTICAL_SCALE = 0.35
TOROIDAL_MODULATION_AMPLITUDE = 0.08
PERTURBATION_AMPLITUDE = 0.002
TOROIDAL_POTENTIAL_AMPLITUDE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6th-order central difference div B verification")
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=(64, 64, 64),
        metavar=("NX", "NY", "NZ"),
        help="格子数（デフォルト: 64 64 64）",
    )
    parser.add_argument(
        "--length",
        type=float,
        nargs=3,
        default=(2.0, 2.0, 2.0),
        metavar=("LX", "LY", "LZ"),
        help="計算領域サイズ（デフォルト: 2.0 2.0 2.0）",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="配列バックエンド（デフォルト: numpy）",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="演算精度（デフォルト: float64）",
    )
    return parser.parse_args()


def get_backend(name: str):
    if name == "numpy":
        return np, "NumPy"
    if cp is None:
        raise RuntimeError("CuPy not found. Please use the CuPy environment defined in environment.yml.")
    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:  # pragma: no cover - for environments without GPU
        raise RuntimeError(f"Failed to initialize CuPy backend: {exc}") from exc
    return cp, "CuPy"


def scalar_to_float(value, xp) -> float:
    if xp is np:
        return float(value)
    return float(cp.asnumpy(value))


def sixth_order_central_difference(field, spacing: float, axis: int, xp):
    return (
        -xp.roll(field, 3, axis=axis)
        + 9.0 * xp.roll(field, 2, axis=axis)
        - 45.0 * xp.roll(field, 1, axis=axis)
        + 45.0 * xp.roll(field, -1, axis=axis)
        - 9.0 * xp.roll(field, -2, axis=axis)
        + xp.roll(field, -3, axis=axis)
    ) / (60.0 * spacing)


def make_coordinates(
    shape: Sequence[int], lengths: Sequence[float], xp, dtype
):
    axes = [
        xp.linspace(-0.5 * length, 0.5 * length, num=size, endpoint=False, dtype=dtype)
        for size, length in zip(shape, lengths)
    ]
    return xp.meshgrid(*axes, indexing="ij")


def make_vector_potential(shape: Sequence[int], lengths: Sequence[float], xp, dtype):
    lx, ly, lz = lengths
    x, y, z = make_coordinates(shape, lengths, xp, dtype)

    radius = xp.sqrt(x * x + y * y) + xp.asarray(CYLINDRICAL_RADIUS_EPSILON, dtype=dtype)
    major_radius = TOROIDAL_MAJOR_RADIUS_RATIO * min(lx, ly)
    minor_radius = TOROIDAL_MINOR_RADIUS_RATIO * min(lx, ly)

    toroidal_ring = xp.exp(
        -(((radius - major_radius) / minor_radius) ** 2 + (z / TOROIDAL_VERTICAL_SCALE) ** 2)
    )
    toroidal_modulation = 1.0 + TOROIDAL_MODULATION_AMPLITUDE * xp.cos(2.0 * xp.pi * z / lz)
    az = xp.asarray(TOROIDAL_POTENTIAL_AMPLITUDE, dtype=dtype) * toroidal_ring * toroidal_modulation

    perturbation = xp.asarray(PERTURBATION_AMPLITUDE, dtype=dtype)
    ax = perturbation * xp.sin(2.0 * xp.pi * y / ly) * xp.cos(2.0 * xp.pi * z / lz)
    ay = perturbation * xp.sin(2.0 * xp.pi * z / lz) * xp.cos(2.0 * xp.pi * x / lx)
    return ax, ay, az


def curl_from_vector_potential(ax, ay, az, spacing: Sequence[float], xp):
    dx, dy, dz = spacing
    bx = sixth_order_central_difference(az, dy, axis=1, xp=xp) - sixth_order_central_difference(
        ay, dz, axis=2, xp=xp
    )
    by = sixth_order_central_difference(ax, dz, axis=2, xp=xp) - sixth_order_central_difference(
        az, dx, axis=0, xp=xp
    )
    bz = sixth_order_central_difference(ay, dx, axis=0, xp=xp) - sixth_order_central_difference(
        ax, dy, axis=1, xp=xp
    )
    return bx, by, bz


def divergence_from_field(bx, by, bz, spacing: Sequence[float], xp):
    dx, dy, dz = spacing
    return (
        sixth_order_central_difference(bx, dx, axis=0, xp=xp)
        + sixth_order_central_difference(by, dy, axis=1, xp=xp)
        + sixth_order_central_difference(bz, dz, axis=2, xp=xp)
    )


def run_verification(
    backend: str = "numpy",
    dtype_name: str = "float64",
    grid: Sequence[int] = (64, 64, 64),
    length: Sequence[float] = (2.0, 2.0, 2.0),
):
    xp, backend_name = get_backend(backend)
    dtype = getattr(xp, dtype_name)

    shape = tuple(grid)
    lengths = tuple(length)
    spacing = tuple(domain_length / size for domain_length, size in zip(lengths, shape))

    ax, ay, az = make_vector_potential(shape, lengths, xp, dtype)
    bx, by, bz = curl_from_vector_potential(ax, ay, az, spacing, xp)
    div_b = divergence_from_field(bx, by, bz, spacing, xp)

    max_abs = scalar_to_float(xp.max(xp.abs(div_b)), xp)
    l2_norm = scalar_to_float(xp.sqrt(xp.mean(xp.abs(div_b) ** 2)), xp)
    mean_abs = scalar_to_float(xp.mean(xp.abs(div_b)), xp)

    target = 1.0e-14 if dtype_name == "float64" else 1.0e-6
    return {
        "backend_name": backend_name,
        "dtype_name": dtype_name,
        "shape": shape,
        "lengths": lengths,
        "spacing": spacing,
        "max_abs": max_abs,
        "l2_norm": l2_norm,
        "mean_abs": mean_abs,
        "target": target,
        "passed": max_abs <= target and l2_norm <= target,
    }


def main() -> int:
    args = parse_args()
    result = run_verification(
        backend=args.backend,
        dtype_name=args.dtype,
        grid=args.grid,
        length=args.length,
    )

    print("=" * 72)
    print("Super-TKMK div B verification (6th-order central difference)")
    print(f"backend        : {result['backend_name']}")
    print(f"dtype          : {result['dtype_name']}")
    print(
        f"grid           : {result['shape'][0]} x {result['shape'][1]} x {result['shape'][2]}"
    )
    print(
        f"domain length  : {result['lengths'][0]} x {result['lengths'][1]} x {result['lengths'][2]}"
    )
    print(
        "spacing        : "
        f"dx={result['spacing'][0]:.6f}, dy={result['spacing'][1]:.6f}, dz={result['spacing'][2]:.6f}"
    )
    print("field          : toroidal ring field generated from a discrete vector potential")
    print("perturbation   : 3D sinusoidal perturbation added in the vector potential")
    print("-" * 72)
    print(f"max|div B|     : {result['max_abs']:.6e}")
    print(f"L2(div B)      : {result['l2_norm']:.6e}")
    print(f"mean|div B|    : {result['mean_abs']:.6e}")
    print(f"target         : <= {result['target']:.1e} ({result['dtype_name']})")
    print(f"result         : {'PASS' if result['passed'] else 'FAIL'}")
    print("=" * 72)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
