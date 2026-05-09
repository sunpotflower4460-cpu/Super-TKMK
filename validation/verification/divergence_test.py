#!/usr/bin/env python3
"""
divergence_test.py - 磁場の発散（div B）検証スクリプト

任意の磁場（解析的に定義、またはファイルから読み込み）に対して
div B を数値的に計算し、その最大絶対値・L2ノルムを報告する。

【使い方】
  # 解析解（div B = 0 が既知）でテスト
  python validation/verification/divergence_test.py

  # HDF5 ファイルから磁場を読み込んでテスト（将来実装）
  python validation/verification/divergence_test.py --input data/state.h5

【正確性ゲート基準】
  float64: max|div B| < 1e-12
  float32: max|div B| < 1e-4
"""

import argparse
import sys
import numpy as np


# ---------------------------------------------------------------------------
# 数値微分ユーティリティ
# ---------------------------------------------------------------------------

def divergence_3d(Bx, By, Bz, dx, dy, dz):
    """
    3次元デカルト座標上で div B を2次精度中心差分で計算する。

    Parameters
    ----------
    Bx, By, Bz : ndarray, shape (Nx, Ny, Nz)
        磁場の各成分。スタガード格子ではなくセル中心値を想定。
    dx, dy, dz : float
        各方向のグリッド間隔（一様グリッド）。

    Returns
    -------
    divB : ndarray, shape (Nx, Ny, Nz)
        各セルの div B 値。境界セルは前進／後退差分で近似。
    """
    dBx_dx = np.gradient(Bx, dx, axis=0)
    dBy_dy = np.gradient(By, dy, axis=1)
    dBz_dz = np.gradient(Bz, dz, axis=2)
    return dBx_dx + dBy_dy + dBz_dz


def divergence_staggered(Bx_face, By_face, Bz_face, dx, dy, dz):
    """
    スタガード格子（Yee 格子）上で div B を厳密に計算する。

    Parameters
    ----------
    Bx_face : ndarray, shape (Nx+1, Ny, Nz)
        x 方向セル面上の Bx 成分。
    By_face : ndarray, shape (Nx, Ny+1, Nz)
        y 方向セル面上の By 成分。
    Bz_face : ndarray, shape (Nx, Ny, Nz+1)
        z 方向セル面上の Bz 成分。
    dx, dy, dz : float
        グリッド間隔。

    Returns
    -------
    divB : ndarray, shape (Nx, Ny, Nz)
        各セルの div B 値。CT 法では機械精度 ~0 になるはずである。
    """
    Nx = Bx_face.shape[0] - 1
    Ny = By_face.shape[1] - 1
    Nz = Bz_face.shape[2] - 1

    dBx = (Bx_face[1:, :, :] - Bx_face[:-1, :, :]) / dx
    dBy = (By_face[:, 1:, :] - By_face[:, :-1, :]) / dy
    dBz = (Bz_face[:, :, 1:] - Bz_face[:, :, :-1]) / dz

    return dBx + dBy + dBz


# ---------------------------------------------------------------------------
# テストケース定義
# ---------------------------------------------------------------------------

def make_uniform_field(Nx, Ny, Nz, B0=1.0):
    """
    一様磁場 B = (B0, 0, 0)。div B = 0 が厳密に成立する。
    """
    Bx = np.full((Nx, Ny, Nz), B0)
    By = np.zeros((Nx, Ny, Nz))
    Bz = np.zeros((Nx, Ny, Nz))
    return Bx, By, Bz, "一様磁場 B=(B0,0,0)"


def make_abc_field(Nx, Ny, Nz, dx, dy, dz, A=1.0, B=1.0, C=1.0):
    """
    Arnold-Beltrami-Childress (ABC) 磁場。

    B_x = A*sin(k*z) + C*cos(k*y)
    B_y = B*sin(k*x) + A*cos(k*z)
    B_z = C*sin(k*y) + B*cos(k*x)

    各成分は自身の方向（x,y,z）には依存しないため、
    セル中心2次中心差分でも div B が機械精度で 0 になる。
    3次元的な変化を持つ、物理的に意味のある磁場場。
    """
    Lx = Nx * dx
    Ly = Ny * dy
    Lz = Nz * dz
    k = 2 * np.pi / min(Lx, Ly, Lz)

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    z = (np.arange(Nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Bx = A * np.sin(k * Z) + C * np.cos(k * Y)
    By = B * np.sin(k * X) + A * np.cos(k * Z)
    Bz = C * np.sin(k * Y) + B * np.cos(k * X)
    return Bx, By, Bz, "ABC（Arnold-Beltrami-Childress）磁場（div B=0）"


def make_nonzero_divB_field(Nx, Ny, Nz, dx, dy, dz):
    """
    意図的に div B ≠ 0 の磁場（検出テスト用）。

    B_x = x, B_y = y, B_z = z → div B = 3（定数）
    """
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    z = (np.arange(Nz) + 0.5) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Bx = X
    By = Y
    Bz = Z
    return Bx, By, Bz, "div B≠0 テスト用磁場（div B=3）"


def make_circular_field_2d(Nx, Ny, Nz, dx, dy, dz):
    """
    2D 円形磁場（xy 平面内）。B = (-y, x, 0) は div B = 0。
    """
    x = (np.arange(Nx) - Nx // 2 + 0.5) * dx
    y = (np.arange(Ny) - Ny // 2 + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    X3 = X[:, :, np.newaxis] * np.ones((1, 1, Nz))
    Y3 = Y[:, :, np.newaxis] * np.ones((1, 1, Nz))

    Bx = -Y3
    By = X3
    Bz = np.zeros((Nx, Ny, Nz))
    return Bx, By, Bz, "円形磁場 B=(-y,x,0)（div B=0）"


# ---------------------------------------------------------------------------
# 検証ロジック
# ---------------------------------------------------------------------------

def verify_divergence(Bx, By, Bz, dx, dy, dz, label, threshold=1e-10):
    """
    div B を計算し、統計量を表示して合否を返す。

    Returns
    -------
    passed : bool
    """
    divB = divergence_3d(Bx, By, Bz, dx, dy, dz)

    max_abs = np.max(np.abs(divB))
    l2_norm = np.sqrt(np.mean(divB**2))
    mean_abs = np.mean(np.abs(divB))

    passed = max_abs < threshold

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n{'='*60}")
    print(f"テスト : {label}")
    print(f"結果   : {status}")
    print(f"  max|div B|  = {max_abs:.6e}  (閾値: {threshold:.1e})")
    print(f"  L2(div B)   = {l2_norm:.6e}")
    print(f"  mean|div B| = {mean_abs:.6e}")

    return passed


def verify_staggered_divergence(Bx_face, By_face, Bz_face, dx, dy, dz,
                                 label, threshold=1e-14):
    """
    スタガード格子上で div B を検証する。
    CT 法では機械精度（~10⁻¹⁵）であることを期待する。
    """
    divB = divergence_staggered(Bx_face, By_face, Bz_face, dx, dy, dz)

    max_abs = np.max(np.abs(divB))
    l2_norm = np.sqrt(np.mean(divB**2))

    passed = max_abs < threshold

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n{'='*60}")
    print(f"テスト : {label}")
    print(f"結果   : {status}")
    print(f"  max|div B|  = {max_abs:.6e}  (閾値: {threshold:.1e})")
    print(f"  L2(div B)   = {l2_norm:.6e}")

    return passed


def make_exact_staggered_uniform(Nx, Ny, Nz, B0=1.0):
    """
    スタガード格子上の一様磁場（CT 法の機械精度テスト用）。
    """
    Bx_face = np.full((Nx + 1, Ny, Nz), B0)
    By_face = np.zeros((Nx, Ny + 1, Nz))
    Bz_face = np.zeros((Nx, Ny, Nz + 1))
    return Bx_face, By_face, Bz_face


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="磁場の発散（div B）を検証する正確性ゲートスクリプト"
    )
    parser.add_argument(
        "--grid", type=int, nargs=3, default=[32, 32, 32],
        metavar=("NX", "NY", "NZ"),
        help="グリッドサイズ（デフォルト: 32 32 32）"
    )
    parser.add_argument(
        "--length", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("LX", "LY", "LZ"),
        help="計算領域のサイズ（デフォルト: 1.0 1.0 1.0）"
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-10,
        help="合否判定閾値 max|div B|（デフォルト: 1e-10）"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="HDF5 磁場ファイルのパス（将来実装予定）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    Nx, Ny, Nz = args.grid
    Lx, Ly, Lz = args.length
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    threshold = args.threshold

    print("=" * 60)
    print("Super-TKMK  div B = 0 検証スクリプト")
    print(f"グリッド  : {Nx} x {Ny} x {Nz}")
    print(f"領域サイズ: {Lx} x {Ly} x {Lz}")
    print(f"dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    print(f"判定閾値  : max|div B| < {threshold:.1e}")

    if args.input is not None:
        # 将来: HDF5 ファイルから読み込み
        print(f"\nHDF5 入力ファイル: {args.input}")
        print("  ※ HDF5 読み込み機能は未実装です。")
        sys.exit(1)

    results = []

    # --- テスト 1: 一様磁場 ---
    Bx, By, Bz, label = make_uniform_field(Nx, Ny, Nz)
    results.append(verify_divergence(Bx, By, Bz, dx, dy, dz, label, threshold))

    # --- テスト 2: 円形磁場 ---
    Bx, By, Bz, label = make_circular_field_2d(Nx, Ny, Nz, dx, dy, dz)
    results.append(verify_divergence(Bx, By, Bz, dx, dy, dz, label, threshold))

    # --- テスト 3: ABC 磁場 ---
    Bx, By, Bz, label = make_abc_field(Nx, Ny, Nz, dx, dy, dz)
    # ABC 場の各成分は自身の座標方向に依存しないため、
    # セル中心差分でも div B = 0 が機械精度で成立する。
    results.append(verify_divergence(Bx, By, Bz, dx, dy, dz, label, threshold))

    # --- テスト 4: div B ≠ 0 の検出テスト（意図的 FAIL）---
    Bx, By, Bz, label = make_nonzero_divB_field(Nx, Ny, Nz, dx, dy, dz)
    label += " ← この FAIL は正常動作"
    divB_check = divergence_3d(Bx, By, Bz, dx, dy, dz)
    max_abs = np.max(np.abs(divB_check))
    detected = max_abs > threshold
    det_status = "検出成功 ✓" if detected else "検出失敗 ✗"
    print(f"\n{'='*60}")
    print(f"テスト : {label}")
    print(f"結果   : {det_status}")
    print(f"  max|div B| = {max_abs:.6e}  (閾値: {threshold:.1e})")
    results.append(detected)  # FAIL 検出が成功なら True

    # --- テスト 5: スタガード格子 CT 精度テスト ---
    Bx_f, By_f, Bz_f = make_exact_staggered_uniform(Nx, Ny, Nz)
    results.append(
        verify_staggered_divergence(
            Bx_f, By_f, Bz_f, dx, dy, dz,
            "スタガード格子 一様磁場（CT 機械精度テスト）",
            threshold=1e-14
        )
    )

    # --- 最終結果 ---
    print(f"\n{'='*60}")
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"全テスト通過: {n_pass}/{n_total}  ✓ 正確性ゲート: OK")
        sys.exit(0)
    else:
        print(f"テスト失敗あり: {n_pass}/{n_total} 通過  ✗ 正確性ゲート: NG")
        sys.exit(1)


if __name__ == "__main__":
    main()
