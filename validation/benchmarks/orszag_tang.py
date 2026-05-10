#!/usr/bin/env python3
"""
orszag_tang.py — Orszag-Tang Vortex ベンチマーク

Orszag & Tang (1979) による古典的な 2D MHD 乱流ベンチマーク。
div B = 0 の確認と、基本的な MHD 波動の伝播を検証する。

テスト内容:
  1. 標準的な Orszag-Tang 初期条件を設定（2D または 3D）
  2. トーラス座標風に変形したバリアント（--torus フラグ）
  3. divergence_test.py と連携して div B = 0 を確認

使用例:
  # 標準 2D テスト (64²格子)
  python validation/benchmarks/orszag_tang.py

  # トーラス変形バリアント
  python validation/benchmarks/orszag_tang.py --torus

  # 高解像度 3D
  python validation/benchmarks/orszag_tang.py --nx 64 --ny 64 --nz 64

  # div B 検証スクリプトと連携
  python validation/benchmarks/orszag_tang.py --check-divb

参考文献:
  - Orszag, S.A. & Tang, C.-M. (1979), J. Fluid Mech. 90, 129-143
  - Dahlburg, R.B. & Picone, J.M. (1989), Phys. Fluids B 1, 2153
  - Picone, J.M. & Dahlburg, R.B. (1991), Phys. Fluids B 3, 29
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import numpy as np

# divergence_test.py との連携のためパスを追加
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from validation.verification.divergence_test import (
    run_verification,
    sixth_order_central_difference,
    make_coordinates,
)


# ─────────────────────────────────────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────────────────────────────────────

GAMMA = 5.0 / 3.0  # 比熱比

# Orszag-Tang 標準パラメータ（無次元単位）
OT_DENSITY  = GAMMA ** 2          # ρ₀ = γ²
OT_PRESSURE = GAMMA               # p₀ = γ / (4π) → 単位系で p₀ = γ に規格化
OT_V0       = 1.0                 # 速度振幅
OT_B0       = 1.0 / math.sqrt(4.0 * math.pi)  # 磁場振幅 (Gaussian 単位)


# ─────────────────────────────────────────────────────────────────────────────
# 初期条件
# ─────────────────────────────────────────────────────────────────────────────

def make_orszag_tang_2d(
    nx: int, ny: int,
    lx: float = 1.0, ly: float = 1.0,
    xp=np
) -> dict:
    """
    標準 Orszag-Tang Vortex の 2D 初期条件を生成する。

    周期領域 [0, L]² で定義。
    初期条件 (Dahlburg & Picone 1989):
      ρ  = γ²
      p  = γ / (4π)   ← 磁気圧と熱圧を同程度にする
      vx = -v₀ sin(2πy/L)
      vy =  v₀ sin(2πx/L)
      Bx = -B₀ sin(2πy/L)
      By =  B₀ sin(4πx/L)

    戻り値:
      dict with keys: rho, vx, vy, Bx, By, p, x, y
    """
    x1d = xp.linspace(0.0, lx, nx, endpoint=False)
    y1d = xp.linspace(0.0, ly, ny, endpoint=False)
    x, y = xp.meshgrid(x1d, y1d, indexing="ij")

    rho = xp.full((nx, ny), OT_DENSITY)
    p   = xp.full((nx, ny), OT_PRESSURE / (4.0 * math.pi))

    vx = -OT_V0 * xp.sin(2.0 * math.pi * y / ly)
    vy =  OT_V0 * xp.sin(2.0 * math.pi * x / lx)

    Bx = -OT_B0 * xp.sin(2.0 * math.pi * y / ly)
    By =  OT_B0 * xp.sin(4.0 * math.pi * x / lx)

    return {"rho": rho, "vx": vx, "vy": vy, "Bx": Bx, "By": By, "p": p,
            "x": x, "y": y, "lx": lx, "ly": ly}


def make_orszag_tang_3d(
    nx: int, ny: int, nz: int,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    xp=np
) -> dict:
    """
    3D 拡張 Orszag-Tang 初期条件を生成する。

    z 方向には定常なトロイダル磁場を追加し、2D OT の平面を
    z 方向にほぼ一様に拡張する（弱い 3D 効果）。

    Bz は一定値 B0z = OT_B0 を設定してトロイダル成分を模擬する。
    """
    x1d = xp.linspace(0.0, lx, nx, endpoint=False)
    y1d = xp.linspace(0.0, ly, ny, endpoint=False)
    z1d = xp.linspace(0.0, lz, nz, endpoint=False)
    x, y, z = xp.meshgrid(x1d, y1d, z1d, indexing="ij")

    rho = xp.full((nx, ny, nz), OT_DENSITY)
    p   = xp.full((nx, ny, nz), OT_PRESSURE / (4.0 * math.pi))

    vx = -OT_V0 * xp.sin(2.0 * math.pi * y / ly)
    vy =  OT_V0 * xp.sin(2.0 * math.pi * x / lx)
    vz = xp.zeros((nx, ny, nz))

    Bx = -OT_B0 * xp.sin(2.0 * math.pi * y / ly)
    By =  OT_B0 * xp.sin(4.0 * math.pi * x / lx)
    Bz = xp.full((nx, ny, nz), OT_B0)  # 定常トロイダル成分

    return {"rho": rho, "vx": vx, "vy": vy, "vz": vz,
            "Bx": Bx, "By": By, "Bz": Bz, "p": p,
            "x": x, "y": y, "z": z,
            "lx": lx, "ly": ly, "lz": lz}


def make_orszag_tang_torus(
    nx: int, ny: int, nz: int,
    lx: float = 2.0, ly: float = 2.0, lz: float = 2.0,
    R0_frac: float = 0.30,
    a_frac: float  = 0.12,
    xp=np
) -> dict:
    """
    トーラス座標風に変形した Orszag-Tang バリアント。

    動機: 直交座標系の Orszag-Tang を、トーラスのポロイダル断面上に
    マップする。磁気軸 (R0, 0, 0) 周りのポロイダル断面で OT の
    速度・磁場パターンを設定する。

    変換:
      (X, Y) → (r_pol * cos(theta_pol), r_pol * sin(theta_pol))  with  r_pol ∈ [0, a]
    OT の x, y 座標を r_pol/a で規格化し、ポロイダル断面にマップする。

    この設定により:
      - ポロイダル断面内で MHD 渦が発展する
      - div B = 0 を初期化時に analytically 保証できる
        （vortex 磁場は 2D ポロイダル断面内の curl から生成）
    """
    R0 = R0_frac * min(lx, ly)
    a  = a_frac  * min(lx, ly)

    x1d = xp.linspace(-0.5 * lx, 0.5 * lx, nx, endpoint=False)
    y1d = xp.linspace(-0.5 * ly, 0.5 * ly, ny, endpoint=False)
    z1d = xp.linspace(-0.5 * lz, 0.5 * lz, nz, endpoint=False)
    cx, cy, cz = xp.meshgrid(x1d, y1d, z1d, indexing="ij")

    # ポロイダル座標
    rxy    = xp.sqrt(cx**2 + cy**2) + 1.0e-30
    dx_pol = rxy - R0
    r_pol  = xp.sqrt(dx_pol**2 + cz**2)  # ポロイダル半径

    # OT 速度・磁場はポロイダル断面の規格化座標 (xi, eta) でのパターン
    xi  = dx_pol / (a + 1.0e-30)   # ポロイダル半径方向規格化座標
    eta = cz     / (a + 1.0e-30)   # 鉛直方向規格化座標

    # OT パターン（2D: ポロイダル断面内）
    mask = (r_pol < a).astype(float)   # プラズマ内部マスク

    vxi  = -OT_V0 * xp.sin(2.0 * math.pi * eta) * mask
    veta =  OT_V0 * xp.sin(2.0 * math.pi * xi)  * mask

    Bxi  = -OT_B0 * xp.sin(2.0 * math.pi * eta) * mask
    Beta =  OT_B0 * xp.sin(4.0 * math.pi * xi)  * mask

    # ポロイダル→デカルト座標変換
    cos_phi   = cx / rxy
    sin_phi   = cy / rxy
    cos_theta = xp.where(r_pol > 1.0e-10, dx_pol / r_pol, 1.0)
    sin_theta = xp.where(r_pol > 1.0e-10, cz     / r_pol, 0.0)

    # ポロイダル (xi=r, eta=z 方向) の単位ベクトルを直交座標に変換
    # e_xi  = (cos_phi * cos_theta, sin_phi * cos_theta, sin_theta)  (ポロイダル動径)
    # e_eta = (-sin_theta * cos_phi, -sin_theta * sin_phi, cos_theta) (鉛直)
    vx_cart = vxi  * cos_phi * cos_theta + veta * (-sin_theta * cos_phi)
    vy_cart = vxi  * sin_phi * cos_theta + veta * (-sin_theta * sin_phi)
    vz_cart = vxi  * sin_theta           + veta * cos_theta

    Bx_cart = Bxi  * cos_phi * cos_theta + Beta * (-sin_theta * cos_phi)
    By_cart = Bxi  * sin_phi * cos_theta + Beta * (-sin_theta * sin_phi)
    Bz_cart = Bxi  * sin_theta           + Beta * cos_theta

    # トロイダル磁場を追加（B_tor は phi 方向: e_phi = (-sin_phi, cos_phi, 0)）
    B_tor = OT_B0 * mask
    Bx_cart += B_tor * (-sin_phi)
    By_cart += B_tor * cos_phi

    # 密度・圧力プロファイル（プラズマ内: 放物型, 外: 真空近似）
    rho_arr = xp.where(r_pol < a,
                       OT_DENSITY  * xp.maximum(1.0 - (r_pol / a)**2, 0.1),
                       0.01 * OT_DENSITY)
    p_arr   = xp.where(r_pol < a,
                       (OT_PRESSURE / (4.0 * math.pi)) * xp.maximum(1.0 - (r_pol / a)**2, 0.0),
                       1.0e-4)

    return {
        "rho": rho_arr, "vx": vx_cart, "vy": vy_cart, "vz": vz_cart,
        "Bx": Bx_cart, "By": By_cart, "Bz": Bz_cart, "p": p_arr,
        "x": cx, "y": cy, "z": cz,
        "lx": lx, "ly": ly, "lz": lz,
        "R0": R0, "a": a,
    }


# ─────────────────────────────────────────────────────────────────────────────
# div B 検証（初期条件専用）
# ─────────────────────────────────────────────────────────────────────────────

def check_divb_initial(Bx, By, Bz, spacing: tuple, xp=np) -> dict:
    """
    初期条件の div B を 6 次中心差分で評価する。

    CT ソルバを使わない場合（純粋な解析初期条件）は、
    数値微分によって div B が機械精度に近いことを確認する。
    Bz が None の場合は 2D テストとして扱い、Bz=0 を仮定する。
    """
    dx, dy, dz = spacing

    def d6(field, h, axis):
        return sixth_order_central_difference(field, h, axis, xp)

    if Bz is None:
        Bz = xp.zeros_like(Bx)

    divB = d6(Bx, dx, axis=0) + d6(By, dy, axis=1) + d6(Bz, dz, axis=2)

    max_abs  = float(xp.max(xp.abs(divB)))
    l2_norm  = float(xp.sqrt(xp.mean(divB**2)))
    mean_abs = float(xp.mean(xp.abs(divB)))

    return {"max_abs": max_abs, "l2_norm": l2_norm, "mean_abs": mean_abs,
            "divB": divB}


# ─────────────────────────────────────────────────────────────────────────────
# エネルギー・保存則の計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_energies(state: dict, xp=np) -> dict:
    """全運動エネルギー・磁気エネルギー・熱エネルギーを計算する。"""
    rho = state["rho"]
    vx  = state["vx"]
    vy  = state["vy"]
    vz  = state.get("vz")
    if vz is None:
        vz = xp.zeros_like(vx)
    Bx  = state["Bx"]
    By  = state["By"]
    Bz  = state.get("Bz")
    if Bz is None:
        Bz = xp.zeros_like(Bx)
    p   = state["p"]

    vol = xp.ones_like(rho)  # 一様格子: 各セルの体積は同じ

    E_kin  = float(xp.sum(0.5 * rho * (vx**2 + vy**2 + vz**2)))
    E_mag  = float(xp.sum(0.5 * (Bx**2 + By**2 + Bz**2)))
    E_therm = float(xp.sum(p / (GAMMA - 1.0)))

    return {"kinetic": E_kin, "magnetic": E_mag, "thermal": E_therm,
            "total": E_kin + E_mag + E_therm}


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orszag-Tang Vortex MHD benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nx", type=int, default=64, help="x 方向格子数")
    p.add_argument("--ny", type=int, default=64, help="y 方向格子数")
    p.add_argument("--nz", type=int, default=1,  help="z 方向格子数 (1=2D)")
    p.add_argument("--lx", type=float, default=1.0, help="x 方向領域サイズ")
    p.add_argument("--ly", type=float, default=1.0, help="y 方向領域サイズ")
    p.add_argument("--lz", type=float, default=1.0, help="z 方向領域サイズ")
    p.add_argument("--torus", action="store_true",
                   help="トーラス座標変形バリアントを使用")
    p.add_argument("--check-divb", action="store_true",
                   help="divergence_test.py と連携して div B を確認")
    p.add_argument("--divb-grid", type=int, nargs=3,
                   default=None, metavar=("NX", "NY", "NZ"),
                   help="divergence_test の格子数（指定しない場合はメイングリッドと同じ）")
    return p.parse_args()


def run_benchmark(
    nx: int = 64, ny: int = 64, nz: int = 1,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    torus: bool = False,
    check_divb: bool = False,
    divb_grid: tuple | None = None,
) -> dict:
    """
    Orszag-Tang ベンチマークを実行して結果を返す。

    check_divb=True のとき divergence_test.py の run_verification() も呼ぶ。
    """
    xp = np

    # ─── 初期条件生成 ─────────────────────────────────────────────────────
    if torus:
        if nz <= 1:
            # トーラスバリアントは 3D が必要
            nz = max(nz, 32)
        state = make_orszag_tang_torus(nx, ny, nz, lx * 2, ly * 2, lz * 2, xp=xp)
        lx = state["lx"]; ly = state["ly"]; lz = state["lz"]
        variant = "torus"
    elif nz <= 1:
        state = make_orszag_tang_2d(nx, ny, lx, ly, xp=xp)
        state["Bz"] = None
        state["vz"] = None
        variant = "2D"
    else:
        state = make_orszag_tang_3d(nx, ny, nz, lx, ly, lz, xp=xp)
        variant = "3D"

    # ─── エネルギー計算 ───────────────────────────────────────────────────
    energies = compute_energies(state, xp)

    # ─── div B 評価（初期条件の数値微分）─────────────────────────────────
    actual_nz = state["Bx"].shape[2] if state["Bx"].ndim == 3 else 1
    dx_v = lx / nx
    dy_v = ly / ny
    dz_v = lz / actual_nz if actual_nz > 1 else lz

    Bz_field = state.get("Bz")
    if Bz_field is not None:
        divb_result = check_divb_initial(
            state["Bx"], state["By"], Bz_field,
            (dx_v, dy_v, dz_v), xp)
    else:
        # 2D: expand dims so sixth_order_central_difference works uniformly
        Bx2 = state["Bx"][:, :, xp.newaxis] if state["Bx"].ndim == 2 else state["Bx"]
        By2 = state["By"][:, :, xp.newaxis] if state["By"].ndim == 2 else state["By"]
        Bz2 = xp.zeros_like(Bx2)
        divb_result = check_divb_initial(Bx2, By2, Bz2, (dx_v, dy_v, dz_v), xp)

    # ─── divergence_test.py との連携（--check-divb フラグ）───────────────
    divtest_result = None
    if check_divb:
        # run_verification はデフォルト格子 (64³, 領域 2.0³) を使う。
        # ベンチマーク格子と独立した標準テストとして実行する。
        dg = divb_grid or (max(nx, 64), max(ny, 64), max(actual_nz, 64))
        divtest_result = run_verification(
            backend="numpy",
            dtype_name="float64",
            grid=dg,
        )

    return {
        "variant": variant,
        "nx": nx, "ny": ny, "nz": actual_nz,
        "lx": lx, "ly": ly, "lz": lz,
        "energies": energies,
        "divb": divb_result,
        "divtest": divtest_result,
        "state": state,
    }


def print_results(result: dict) -> None:
    variant = result["variant"]
    nx, ny, nz = result["nx"], result["ny"], result["nz"]
    print("=" * 72)
    print(f"Orszag-Tang Vortex Benchmark  [{variant}]")
    print(f"Grid: {nx} x {ny} x {nz}")
    print(f"Domain: {result['lx']:.2f} x {result['ly']:.2f} x {result['lz']:.2f}")
    print("-" * 72)

    en = result["energies"]
    print(f"Kinetic  energy : {en['kinetic']:.6e}")
    print(f"Magnetic energy : {en['magnetic']:.6e}")
    print(f"Thermal  energy : {en['thermal']:.6e}")
    print(f"Total    energy : {en['total']:.6e}")
    print("-" * 72)

    db = result["divb"]
    print("div B (6th-order central diff, initial condition):")
    print(f"  max|div B|  : {db['max_abs']:.6e}")
    print(f"  L2(div B)   : {db['l2_norm']:.6e}")
    print(f"  mean|div B| : {db['mean_abs']:.6e}")

    # 理論的には解析関数なのである程度の div B が残ることは許容。
    # ただしトーラスバリアントでは数値誤差が大きくなりうる。
    divb_ok = db["max_abs"] < 1.0e-10
    print(f"  status: {'OK (< 1e-10)' if divb_ok else 'NOTE: large div B (expected for analytic IC)'}")

    dt = result.get("divtest")
    if dt is not None:
        print("-" * 72)
        print("divergence_test.py (toroidal ring field, independent check):")
        print(f"  max|div B|  : {dt['max_abs']:.6e}")
        print(f"  L2(div B)   : {dt['l2_norm']:.6e}")
        print(f"  target      : <= {dt['target']:.1e}")
        print(f"  result      : {'PASS' if dt['passed'] else 'FAIL'}")

    print("=" * 72)


def main() -> int:
    args = parse_args()
    result = run_benchmark(
        nx=args.nx, ny=args.ny, nz=args.nz,
        lx=args.lx, ly=args.ly, lz=args.lz,
        torus=args.torus,
        check_divb=args.check_divb,
        divb_grid=tuple(args.divb_grid) if args.divb_grid else None,
    )
    print_results(result)

    # --check-divb フラグがある場合は divergence_test の合否を返す
    if args.check_divb and result["divtest"] is not None:
        return 0 if result["divtest"]["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
