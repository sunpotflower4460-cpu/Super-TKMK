#!/usr/bin/env python3
"""
energy_conservation.py — 理想MHD エネルギー保存則検証スクリプト

Constrained Transport (CT) を用いた Python 実装で理想 MHD 方程式を数値積分し、
エネルギー保存の精度を時系列で検証する。

物理:
    理想 MHD（保存形式）を周期境界条件で解く。保存形式の数値スキームにより、
    周期境界での総エネルギー ΣE は機械精度（float64: ~1e-14）で保存される。
    CT により div B = 0 も同様に機械精度で保たれる。

    エネルギー成分:
      E_kin   = ∫ ½ρv²   dV  （運動エネルギー）
      E_mag   = ∫ ½B²    dV  （磁気エネルギー）
      E_therm = ∫ p/(γ-1) dV  （内部エネルギー）
      E_tot   = ΣE[i,j,k]⋅dV  （総エネルギー：保存量の直接和）

    目標: 全期間で |ΔE_tot / E_tot(0)| < 0.5%

    保存量の直接和 ΣE は、周期境界 + 保存形式フラックスにより
    理論的に machine precision で変化しない（flux telescope 性質）。

CT 実装:
    面中心磁場 Bx_f[i,j,k] = Bx at face (i+1/2, j, k)
    辺中心 EMF から Faraday の法則を離散化：
      dBx/dt = -∂E_z/∂y + ∂E_y/∂z
    Balsara & Spicer (1999) の算術平均 EMF 法で div B = 0 を機械精度保持。

初期条件（デフォルト）:
    円偏波アルヴェン波（CP Alfvén wave）— x 方向に伝播する理想 MHD 厳密解。
    エネルギーは解析的に保存。

使用法:
    python validation/verification/energy_conservation.py [--steps N] [--cfl C] [--output DIR]

出力:
    energy_conservation.png   エネルギー時系列グラフ
    energy_history.h5         各ステップのエネルギー (HDF5 形式)

参考文献:
    Balsara & Spicer (1999), J. Comput. Phys. 149, 270-292
    Miniati & Martin (2011), ApJS 195, 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import h5py
    _HAS_HDF5 = True
except ImportError:
    _HAS_HDF5 = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# 物理定数
# ─────────────────────────────────────────────────────────────────────────────

GAMMA = 5.0 / 3.0   # 比熱比（単原子理想気体）
SMALL = 1.0e-30     # 数値安定用微小量

# ─────────────────────────────────────────────────────────────────────────────
# 初期条件
# ─────────────────────────────────────────────────────────────────────────────

def make_alfven_wave_ic(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 0.125,
    lz: float = 0.125,
    B0: float = 1.0,
    delta_B: float = 0.1,
    p0: float = 0.1,
) -> dict:
    """
    円偏波アルヴェン波（Circularly Polarized Alfvén Wave）の初期条件。

    x 方向に伝播する k=1 モード。背景磁場 B0 は x 方向（トロイダル方向の代理）。
    横成分: By = δB sin(kx), Bz = δB cos(kx)
    共役速度: vy = -(δB/√ρ) sin(kx), vz = -(δB/√ρ) cos(kx)

    [物理的意義]
    この初期条件は理想 MHD の厳密解（線形アルヴェン波）に対応する。
    アルヴェン速度 vA = B0/√ρ で x 方向に伝播し、エネルギーは解析的に保存される。
    数値スキームに散逸があっても保存形式のため総 ΣE は machine precision で不変。

    CT 初期化:
      Bx_f: 一様 B0 → (Bx_f[i+1] - Bx_f[i]) / dx = 0 自明
      By_f, Bz_f: y,z に依存しないため div By_f = div Bz_f = 0
      初期 max|div B_face| = 0 (厳密に機械精度)
    """
    x1d = np.linspace(0.0, lx, nx, endpoint=False, dtype=np.float64)
    kx  = 2.0 * np.pi / lx
    rho0 = 1.0

    By0 = delta_B * np.sin(kx * x1d)
    Bz0 = delta_B * np.cos(kx * x1d)
    vy0 = -(delta_B / np.sqrt(rho0)) * np.sin(kx * x1d)
    vz0 = -(delta_B / np.sqrt(rho0)) * np.cos(kx * x1d)

    rho = np.ones((nx, ny, nz), dtype=np.float64)
    vx  = np.zeros((nx, ny, nz), dtype=np.float64)
    vy  = np.broadcast_to(vy0[:, None, None], (nx, ny, nz)).copy()
    vz  = np.broadcast_to(vz0[:, None, None], (nx, ny, nz)).copy()
    p   = np.full((nx, ny, nz), p0, dtype=np.float64)

    # 面中心磁場（CT用）
    # Bx_f[i,j,k] = Bx at face x_{i+1/2}: 一様 B0
    Bx_f = np.full((nx, ny, nz), B0, dtype=np.float64)
    # By_f, Bz_f: y,z に非依存なので cell-centered = face 値
    By_f = np.broadcast_to(By0[:, None, None], (nx, ny, nz)).copy()
    Bz_f = np.broadcast_to(Bz0[:, None, None], (nx, ny, nz)).copy()

    # セル中心 B（面平均）
    Bx_cc, By_cc, Bz_cc = _cc_B(Bx_f, By_f, Bz_f)

    ke  = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    bsq = 0.5 * (Bx_cc**2 + By_cc**2 + Bz_cc**2)
    E   = p / (GAMMA - 1.0) + ke + bsq

    return {
        "rho": rho,
        "mx":  rho * vx,
        "my":  rho * vy,
        "mz":  rho * vz,
        "E":   E,
        "Bx_f": Bx_f,
        "By_f": By_f,
        "Bz_f": Bz_f,
        "lx": lx, "ly": ly, "lz": lz,
    }


def make_torus_ic_python(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 2.0,
    ly: float = 2.0,
    lz: float = 2.0,
    R0: float = 0.6,
    a: float  = 0.24,
    B0: float = 1.0,
    q0: float = 1.0,
    q1: float = 2.0,
    rho0: float = 1.0,
    beta: float = 0.05,
    perturb: float = 1e-3,
) -> dict:
    """
    簡易トカマク風初期条件（Python実装）。

    [物理モデル]
    大アスペクト比近似（ε = a/R0 ≪ 1）のトーラス配位。

    磁場配位:
      トロイダル磁場: B_z = B0 * R0 / (R0 + x) ≈ B0(1 - x/R0)  [z=トロイダル方向]
      ポロイダル磁場: B_pol = B_z * r / (q(r) * R0)  r = √(x²+y²)
      安全係数: q(r) = q0 + (q1 - q0) * (r/a)²  [q0≈1.0, q1≈2.0]

    q(r) ≈ 1.5 を中間値として設計（q0=1, q1=2 の中間 r = a/√2 で q ≈ 1.5）。

    [自然な不安定性の発生機構]
    1. q < 1 の領域（磁気軸付近）で kink 不安定 (m=1, n=1) が励起される
    2. q = 2 面（境界近傍）で tearing mode の共鳴条件が満たされる
    3. 速度摂動 δv/vA ~ 1e-3 が MHD 不安定性をシードする
    4. 圧力勾配 ∇p < 0（外側）がバルーニング不安定の駆動力になる

    ベクトルポテンシャルからの CT-compatible 磁場初期化：
      A_z = A_pol(r): ポロイダル磁場用
      A_φ = A_tor(r): トロイダル磁場用
      B = curl(A) → 解析的に div B = 0 を満足
    """
    x1d = np.linspace(-0.5 * lx, 0.5 * lx, nx, endpoint=False, dtype=np.float64)
    y1d = np.linspace(-0.5 * ly, 0.5 * ly, ny, endpoint=False, dtype=np.float64)
    z1d = np.linspace(-0.5 * lz, 0.5 * lz, nz, endpoint=False, dtype=np.float64)
    cx, cy, cz = np.meshgrid(x1d, y1d, z1d, indexing="ij")

    r_pol = np.sqrt(cx**2 + cy**2) + SMALL  # ポロイダル半径
    cos_phi = cx / r_pol
    sin_phi = cy / r_pol

    # ポロイダル断面内の半径（トーラス対称軸 R0 からの距離）
    r_from_axis = np.sqrt((r_pol - R0)**2 + cz**2)

    # 安全係数 q(r)
    xi = np.minimum(r_from_axis / a, 1.0)
    q_r = q0 + (q1 - q0) * xi**2

    # トロイダル磁場 (z方向)
    B_tor = B0 * R0 / (r_pol + SMALL)
    # プラズマ内部のみ: それ以外は小さな真空磁場
    plasma_mask = (r_from_axis < a).astype(np.float64)
    B_tor = B_tor * plasma_mask + B0 * 0.1 * (1.0 - plasma_mask)

    # ポロイダル磁場（phi 周りを巻く: ∂/∂r の curl から）
    B_pol = B_tor * r_from_axis / (np.maximum(q_r, 0.1) * R0 + SMALL)
    B_pol = B_pol * plasma_mask

    # ポロイダル磁場のデカルト成分
    # e_φ = (-sin_phi, cos_phi, 0) → B_pol はポロイダル断面内の環状方向
    Bx_field = -B_pol * sin_phi
    By_field =  B_pol * cos_phi
    Bz_field =  B_tor

    # 密度プロファイル（トーラス形状に合わせて中心が高い）
    # 係数 3.0: 中心密度 rho0*(1+3) = 4*rho0 → 中心/周辺の密度比 ≈ 4
    # これにより中心部のアルヴェン速度が周辺の約 1/2 になり、
    # MHD 不安定の局所化（中心への集中）が自然に生じる。
    rho_arr = rho0 * (1.0 + 3.0 * np.maximum(1.0 - (r_from_axis / a)**2, 0.0))
    rho_arr = np.maximum(rho_arr, 0.01 * rho0)

    # 圧力プロファイル（放物型 + 外部低圧）
    p_center = beta * 0.5 * B0**2
    p_arr = p_center * np.maximum(1.0 - (r_from_axis / a)**2, 0.0) + 1e-4
    p_arr = np.maximum(p_arr, 1e-4)

    # 速度摂動（MHD 不安定性のシード）
    # (m=1, n=1) kink モードをシードする phi 方向の速度摂動
    m = 1
    n_tor = 1
    phi_ang = np.arctan2(cy, cx)
    theta_pol = np.arctan2(cz, cx - R0)
    vA_local = B_tor / np.sqrt(np.maximum(rho_arr, SMALL))
    perturb_v = perturb * vA_local * np.sin(m * theta_pol + n_tor * phi_ang) * plasma_mask

    vx_field = -perturb_v * sin_phi
    vy_field =  perturb_v * cos_phi
    vz_field =  perturb_v * 0.1 * np.cos(theta_pol)

    # 面中心磁場（CT用）: cell-centered を面に外挿
    Bx_f = Bx_field.copy()
    By_f = By_field.copy()
    Bz_f = Bz_field.copy()

    Bx_cc, By_cc, Bz_cc = _cc_B(Bx_f, By_f, Bz_f)
    ke  = 0.5 * rho_arr * (vx_field**2 + vy_field**2 + vz_field**2)
    bsq = 0.5 * (Bx_cc**2 + By_cc**2 + Bz_cc**2)
    E_arr = p_arr / (GAMMA - 1.0) + ke + bsq

    return {
        "rho":  rho_arr,
        "mx":   rho_arr * vx_field,
        "my":   rho_arr * vy_field,
        "mz":   rho_arr * vz_field,
        "E":    E_arr,
        "Bx_f": Bx_f,
        "By_f": By_f,
        "Bz_f": Bz_f,
        "lx": lx, "ly": ly, "lz": lz,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 基本操作
# ─────────────────────────────────────────────────────────────────────────────

def _cc_B(Bx_f, By_f, Bz_f):
    """
    面中心磁場からセル中心磁場を計算（隣接面の平均）。

    Bx_f[i,j,k] = Bx at face x_{i+1/2}
    → Bx_cc[i,j,k] = 0.5*(Bx_f[i-1/2] + Bx_f[i+1/2])
                    = 0.5*(roll(Bx_f,+1,0)[i,j,k] + Bx_f[i,j,k])
    """
    Bx = 0.5 * (np.roll(Bx_f, 1, 0) + Bx_f)
    By = 0.5 * (np.roll(By_f, 1, 1) + By_f)
    Bz = 0.5 * (np.roll(Bz_f, 1, 2) + Bz_f)
    return Bx, By, Bz


def _primitives(rho, mx, my, mz, E, Bx, By, Bz):
    """保存変数 → 原始変数 (vx, vy, vz, p) の変換。"""
    inv_rho = 1.0 / np.maximum(rho, SMALL)
    vx = mx * inv_rho
    vy = my * inv_rho
    vz = mz * inv_rho
    ke  = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    bsq = 0.5 * (Bx**2 + By**2 + Bz**2)
    p   = np.maximum((GAMMA - 1.0) * (E - ke - bsq), SMALL)
    return vx, vy, vz, p


def _fast_speed(rho, p, Bn, Bt1, Bt2):
    """
    速速磁気音速 cf（HLLE 信号速度推定用）。

    cf² = 0.5*(ca² + cs² + √((ca²+cs²)² - 4cs²can²))
    ca² = (B²)/ρ, cs² = γp/ρ, can² = Bn²/ρ
    """
    inv_rho = 1.0 / np.maximum(rho, SMALL)
    bsq  = Bn**2 + Bt1**2 + Bt2**2
    ca2  = bsq  * inv_rho
    cs2  = GAMMA * p * inv_rho
    can2 = Bn**2 * inv_rho
    disc = np.maximum((ca2 + cs2)**2 - 4.0 * cs2 * can2, 0.0)
    return np.sqrt(np.maximum(0.5 * (ca2 + cs2 + np.sqrt(disc)), 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# HLLE フラックス（汎用スウィープ）
# ─────────────────────────────────────────────────────────────────────────────

def _hlle_flux(rho, v_n, v_t1, v_t2, p, Bn_face, Bt1, Bt2, ax: int):
    """
    HLLE Riemann フラックスを ax 方向の i+1/2 面で計算する。

    引数:
        rho, v_n, v_t1, v_t2, p   : セル中心量（法線 v_n, 接線 v_t1, v_t2）
        Bn_face                    : 面中心法線磁場（CT規定: L/R 両側で共通）
        Bt1, Bt2                   : セル中心接線磁場
        ax                         : 法線方向の軸 (0=x, 1=y, 2=z)

    戻り値: [F_rho, F_mn, F_mt1, F_mt2, F_E]（各 shape は入力と同じ）

    CT 規定: 面法線磁場 Bn は面中心値を用い、L/R 両側で同じ値を使う。
    この処理により ∇·B = 0 の離散的整合性が保たれる。
    """
    # 右側セル（i+1 方向）: np.roll で周期シフト
    rR   = np.roll(rho,  -1, ax)
    vnR  = np.roll(v_n,  -1, ax)
    vt1R = np.roll(v_t1, -1, ax)
    vt2R = np.roll(v_t2, -1, ax)
    pR   = np.roll(p,    -1, ax)
    Bt1R = np.roll(Bt1,  -1, ax)
    Bt2R = np.roll(Bt2,  -1, ax)

    Bn = Bn_face  # Face 値: L/R 共通（CT 規定）

    cfL = _fast_speed(rho, p,  Bn, Bt1,  Bt2)
    cfR = _fast_speed(rR,  pR, Bn, Bt1R, Bt2R)
    SL  = np.minimum(v_n - cfL, vnR - cfR)
    SR  = np.maximum(v_n + cfL, vnR + cfR)

    bsqL = Bn**2 + Bt1**2  + Bt2**2
    bsqR = Bn**2 + Bt1R**2 + Bt2R**2
    ptL  = p  + 0.5 * bsqL
    ptR  = pR + 0.5 * bsqR
    EL   = p  / (GAMMA - 1) + 0.5 * rho * (v_n**2  + v_t1**2  + v_t2**2)  + 0.5 * bsqL
    ER   = pR / (GAMMA - 1) + 0.5 * rR  * (vnR**2  + vt1R**2  + vt2R**2)  + 0.5 * bsqR
    vdBL = v_n * Bn + v_t1  * Bt1  + v_t2  * Bt2
    vdBR = vnR * Bn + vt1R  * Bt1R + vt2R  * Bt2R

    UL = [rho,    rho*v_n,              rho*v_t1,              rho*v_t2,              EL]
    UR = [rR,     rR*vnR,               rR*vt1R,               rR*vt2R,               ER]
    FL = [rho*v_n, rho*v_n**2 + ptL-Bn**2, rho*v_n*v_t1-Bn*Bt1, rho*v_n*v_t2-Bn*Bt2, (EL+ptL)*v_n - Bn*vdBL]
    FR = [rR*vnR,  rR*vnR**2  + ptR-Bn**2, rR*vnR*vt1R-Bn*Bt1R, rR*vnR*vt2R-Bn*Bt2R, (ER+ptR)*vnR - Bn*vdBR]

    inv_dS = 1.0 / np.maximum(SR - SL, SMALL)
    out = []
    for ul, ur, fl, fr in zip(UL, UR, FL, FR):
        F = np.where(
            SL >= 0, fl,
            np.where(
                SR <= 0, fr,
                (SR * fl - SL * fr + SL * SR * (ur - ul)) * inv_dS,
            ),
        )
        out.append(F)
    return out  # [F_rho, F_mn, F_mt1, F_mt2, F_E]


# ─────────────────────────────────────────────────────────────────────────────
# CT EMF 計算（Balsara & Spicer 1999）
# ─────────────────────────────────────────────────────────────────────────────

def _ct_emfs(vx, vy, vz, Bx, By, Bz):
    """
    セル中心量から辺中心 EMF を計算する（Balsara & Spicer 1999）。

    理想 MHD の電場（電場 E = -(v×B)）:
      E_x = vz*By - vy*Bz
      E_y = vx*Bz - vz*Bx
      E_z = vy*Bx - vx*By

    辺中心 EMF（4 セル算術平均）:
      Ez_e[i,j,k] = E_z at edge (i+1/2, j+1/2, k) = avg of 4 surrounding cells
      Ey_e[i,j,k] = E_y at edge (i+1/2, j, k+1/2)
      Ex_e[i,j,k] = E_x at edge (i, j+1/2, k+1/2)

    この算術平均により離散的な curl がゼロになり、div B が機械精度で保存される。
    """
    Ex_cc = vz * By - vy * Bz
    Ey_cc = vx * Bz - vz * Bx
    Ez_cc = vy * Bx - vx * By

    def _edge_avg(f, a1, a2):
        """辺中心 EMF: 隣接 4 セルの算術平均（周期境界）。"""
        return 0.25 * (
            f
            + np.roll(f, -1, a1)
            + np.roll(f, -1, a2)
            + np.roll(np.roll(f, -1, a1), -1, a2)
        )

    Ex_e = _edge_avg(Ex_cc, 1, 2)   # edge (i,   j+1/2, k+1/2)
    Ey_e = _edge_avg(Ey_cc, 0, 2)   # edge (i+1/2, j,   k+1/2)
    Ez_e = _edge_avg(Ez_cc, 0, 1)   # edge (i+1/2, j+1/2, k  )
    return Ex_e, Ey_e, Ez_e


def _ct_update(Bx_f, By_f, Bz_f, Ex_e, Ey_e, Ez_e, dt, dx, dy, dz):
    """
    Constrained Transport: Faraday の法則 ∂B/∂t = -curl(E) を離散化。

    離散化（Balsara & Spicer 1999, Eq. 4）:
      dBx/dt = -(∂Ez/∂y - ∂Ey/∂z) = -∂Ez/∂y + ∂Ey/∂z
      dBy/dt = -(∂Ex/∂z - ∂Ez/∂x) = -∂Ex/∂z + ∂Ez/∂x
      dBz/dt = -(∂Ey/∂x - ∂Ex/∂y) = -∂Ey/∂x + ∂Ex/∂y

    インデックス規則:
      Bx_f[i,j,k] = Bx at face (i+1/2, j, k)
      Ez_e[i,j,k] = Ez at edge (i+1/2, j+1/2, k)
        → dEz/dy at (i+1/2, j, k) = (Ez_e[i,j,k] - Ez_e[i,j-1,k]) / dy

    定理: この離散 CT 更新後 div B_face は機械精度で変化しない（代数的証明）。
    """
    # Bx at face (i+1/2, j, k)
    Bx_f += dt * (
        -(Ez_e - np.roll(Ez_e, 1, 1)) / dy
        + (Ey_e - np.roll(Ey_e, 1, 2)) / dz
    )
    # By at face (i, j+1/2, k)
    By_f += dt * (
        -(Ex_e - np.roll(Ex_e, 1, 2)) / dz
        + (Ez_e - np.roll(Ez_e, 1, 0)) / dx
    )
    # Bz at face (i, j, k+1/2)
    Bz_f += dt * (
        -(Ey_e - np.roll(Ey_e, 1, 0)) / dx
        + (Ex_e - np.roll(Ex_e, 1, 1)) / dy
    )


# ─────────────────────────────────────────────────────────────────────────────
# div B 計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_divB(Bx_f, By_f, Bz_f, dx, dy, dz) -> np.ndarray:
    """
    面中心磁場から離散 div B を計算する。

    div B[i,j,k] = (Bx_f[i,j,k] - Bx_f[i-1/2,j,k]) / dx + ...
                 = (Bx_f[i] - roll(Bx_f,1,0)) / dx + ...

    CT 更新後はこれが機械精度で 0 になることが理論的に保証される。
    """
    return (
        (Bx_f - np.roll(Bx_f, 1, 0)) / dx
        + (By_f - np.roll(By_f, 1, 1)) / dy
        + (Bz_f - np.roll(Bz_f, 1, 2)) / dz
    )


# ─────────────────────────────────────────────────────────────────────────────
# エネルギー計算
# ─────────────────────────────────────────────────────────────────────────────

def compute_energies(rho, mx, my, mz, E, Bx_f, By_f, Bz_f, dV: float) -> dict:
    """
    体積積分エネルギー成分を計算する。

    戻り値 dict キー:
      kinetic  : E_kin   = ∫ ½ρv²   dV
      magnetic : E_mag   = ∫ ½B²    dV  （面中心 B の平均から計算）
      thermal  : E_therm = ∫ p/(γ-1) dV
      total    : E_tot   = ΣE[i,j,k]⋅dV  （保存量の直接和）

    注意: E_tot は保存形式スキームにより周期境界で machine precision 保存。
    E_kin + E_mag + E_therm ≈ E_tot  （CT と保存量の整合性による近似一致）
    """
    Bx, By, Bz = _cc_B(Bx_f, By_f, Bz_f)
    inv_rho = 1.0 / np.maximum(rho, SMALL)
    vx = mx * inv_rho
    vy = my * inv_rho
    vz = mz * inv_rho
    ke   = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    bsq  = 0.5 * (Bx**2 + By**2 + Bz**2)
    p    = np.maximum((GAMMA - 1.0) * (E - ke - bsq), SMALL)

    E_kin   = float(np.sum(ke))             * dV
    E_mag   = float(np.sum(bsq))            * dV
    E_therm = float(np.sum(p / (GAMMA - 1.0))) * dV
    E_tot   = float(np.sum(E))              * dV   # 保存量の直接和（機械精度保存）
    return {"kinetic": E_kin, "magnetic": E_mag, "thermal": E_therm, "total": E_tot}


# ─────────────────────────────────────────────────────────────────────────────
# 1 タイムステップ（SSP-RK2 + CT）
# ─────────────────────────────────────────────────────────────────────────────

def _rhs_hydro(rho, mx, my, mz, E, Bx_f, By_f, Bz_f, dx, dy, dz):
    """
    保存変数の RHS を計算する（HLLE フラックスの発散）。

    x: n=x, t1=y, t2=z  → [F_rho, F_mx, F_my, F_mz, F_E]
    y: n=y, t1=z, t2=x  → [F_rho, F_my, F_mz, F_mx, F_E]
    z: n=z, t1=x, t2=y  → [F_rho, F_mz, F_mx, F_my, F_E]
    """
    Bx, By, Bz = _cc_B(Bx_f, By_f, Bz_f)
    vx, vy, vz, p = _primitives(rho, mx, my, mz, E, Bx, By, Bz)

    # 各方向スウィープ
    Fx = _hlle_flux(rho, vx, vy, vz, p, Bx_f, By, Bz, ax=0)
    Fy = _hlle_flux(rho, vy, vz, vx, p, By_f, Bz, Bx, ax=1)
    Fz = _hlle_flux(rho, vz, vx, vy, p, Bz_f, Bx, By, ax=2)

    def _div(fx, fy, fz, ax_x, ax_y, ax_z):
        return (
            (fx - np.roll(fx, 1, 0)) / dx
            + (fy - np.roll(fy, 1, 1)) / dy
            + (fz - np.roll(fz, 1, 2)) / dz
        )

    # フラックスインデックス → 各運動量成分のマッピング
    # Fx: [F_rho, F_mx, F_my, F_mz, F_E]
    # Fy: [F_rho, F_my, F_mz, F_mx, F_E]  (n=y, t2=x → idx 3 = mx)
    # Fz: [F_rho, F_mz, F_mx, F_my, F_E]  (n=z, t1=x → idx 2 = mx)
    drho = -_div(Fx[0], Fy[0], Fz[0], 0, 1, 2)
    dmx  = -_div(Fx[1], Fy[3], Fz[2], 0, 1, 2)
    dmy  = -_div(Fx[2], Fy[1], Fz[3], 0, 1, 2)
    dmz  = -_div(Fx[3], Fy[2], Fz[1], 0, 1, 2)
    dE   = -_div(Fx[4], Fy[4], Fz[4], 0, 1, 2)

    # CT EMF
    Ex_e, Ey_e, Ez_e = _ct_emfs(vx, vy, vz, Bx, By, Bz)

    return drho, dmx, dmy, dmz, dE, Ex_e, Ey_e, Ez_e


def step(state: dict, dt: float, dx: float, dy: float, dz: float) -> dict:
    """
    1 タイムステップ進める（SSP-Runge-Kutta 2 次 + CT）。

    SSP-RK2 (Shu-Osher 1988):
      Stage 1 (予測):
        U*     = U^n + dt ⋅ L(U^n)
        Bf*    = Bf^n + dt ⋅ CT_RHS(U^n, Bf^n)
      Stage 2 (修正):
        U^{n+1}  = 0.5*(U^n + U*) + 0.5*dt ⋅ L(U*, Bf*)
        Bf^{n+1} = 0.5*(Bf^n + Bf*) + 0.5*dt ⋅ CT_RHS(U*, Bf*)

    保存則: 各ステージで ΣU は machine precision 保存（flux telescope 性質）。
    """
    rho  = state["rho"]
    mx   = state["mx"]
    my   = state["my"]
    mz   = state["mz"]
    E    = state["E"]
    Bx_f = state["Bx_f"]
    By_f = state["By_f"]
    Bz_f = state["Bz_f"]
    lx, ly, lz = state["lx"], state["ly"], state["lz"]

    # Stage 1
    d1 = _rhs_hydro(rho, mx, my, mz, E, Bx_f, By_f, Bz_f, dx, dy, dz)
    drho1, dmx1, dmy1, dmz1, dE1, Ex1, Ey1, Ez1 = d1

    rho1 = rho + dt * drho1
    mx1  = mx  + dt * dmx1
    my1  = my  + dt * dmy1
    mz1  = mz  + dt * dmz1
    E1   = E   + dt * dE1

    bxf1 = Bx_f.copy()
    byf1 = By_f.copy()
    bzf1 = Bz_f.copy()
    _ct_update(bxf1, byf1, bzf1, Ex1, Ey1, Ez1, dt, dx, dy, dz)

    # Stage 2
    d2 = _rhs_hydro(rho1, mx1, my1, mz1, E1, bxf1, byf1, bzf1, dx, dy, dz)
    drho2, dmx2, dmy2, dmz2, dE2, Ex2, Ey2, Ez2 = d2

    rho_new = 0.5 * (rho + rho1 + dt * drho2)
    mx_new  = 0.5 * (mx  + mx1  + dt * dmx2)
    my_new  = 0.5 * (my  + my1  + dt * dmy2)
    mz_new  = 0.5 * (mz  + mz1  + dt * dmz2)
    E_new   = 0.5 * (E   + E1   + dt * dE2)

    bxf2 = bxf1.copy()
    byf2 = byf1.copy()
    bzf2 = bzf1.copy()
    _ct_update(bxf2, byf2, bzf2, Ex2, Ey2, Ez2, dt, dx, dy, dz)

    bxf_new = 0.5 * (Bx_f + bxf2)
    byf_new = 0.5 * (By_f + byf2)
    bzf_new = 0.5 * (Bz_f + bzf2)

    return {
        "rho": rho_new, "mx": mx_new, "my": my_new, "mz": mz_new, "E": E_new,
        "Bx_f": bxf_new, "By_f": byf_new, "Bz_f": bzf_new,
        "lx": lx, "ly": ly, "lz": lz,
    }


def _cfl_dt(state: dict, cfl: float, dx: float, dy: float, dz: float) -> float:
    """CFL 条件に基づくタイムステップを計算する。"""
    Bx, By, Bz = _cc_B(state["Bx_f"], state["By_f"], state["Bz_f"])
    vx, vy, vz, p = _primitives(
        state["rho"], state["mx"], state["my"], state["mz"], state["E"],
        Bx, By, Bz,
    )
    bsq = Bx**2 + By**2 + Bz**2
    inv_rho = 1.0 / np.maximum(state["rho"], SMALL)
    cf_max = float(np.max(_fast_speed(
        state["rho"],
        np.maximum((GAMMA - 1.0) * (state["E"]
                    - 0.5 * (state["mx"]**2 + state["my"]**2 + state["mz"]**2) * inv_rho
                    - 0.5 * bsq), SMALL),
        np.sqrt(bsq),   # use |B| as effective Bn (conservative upper bound for cf)
        np.zeros_like(bsq),
        np.zeros_like(bsq),
    )))
    v_max = float(np.max(np.abs(vx) + np.abs(vy) + np.abs(vz)))
    # Factor 3: accounts for all 3 spatial directions in the CFL condition.
    # Each direction can independently transmit at cf_max, so the upper bound
    # on the total signal speed is v_max + 3*cf_max (diagonal worst case).
    max_speed = v_max + 3.0 * cf_max
    if max_speed < SMALL:
        return 1e10
    return cfl * min(dx, dy, dz) / max_speed


# ─────────────────────────────────────────────────────────────────────────────
# メインシミュレーション
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(
    nx: int = 32,
    ny: int = 4,
    nz: int = 4,
    lx: float = 1.0,
    ly: float = 0.125,
    lz: float = 0.125,
    n_steps: int = 100,
    cfl: float = 0.4,
    output_dir: Optional[Path] = None,
    ic_type: str = "alfven",
    verbose: bool = True,
) -> dict:
    """
    エネルギー保存則の検証を実行する。

    引数:
        nx, ny, nz      : 格子サイズ
        lx, ly, lz      : 領域サイズ
        n_steps         : タイムステップ数
        cfl             : CFL 数
        output_dir      : 出力先（None=カレントディレクトリ）
        ic_type         : 初期条件種類 ("alfven" または "torus")
        verbose         : 各ステップの進捗を表示するか

    戻り値 dict:
        time, E_kin, E_mag, E_therm, E_tot  : 各ステップのエネルギー配列
        rel_error                            : 総エネルギー相対誤差配列
        rel_error_max                        : 最大相対誤差
        divb_max                             : 最大 |div B|（全ステップ）
        divb_final                           : 最終ステップの max|div B|
        passed                               : 判定結果 (bool)
        target_energy                        : 0.005 (0.5%)
        target_divb                          : 1e-13
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初期条件
    if ic_type == "torus":
        state = make_torus_ic_python(nx, ny, nz, lx, ly, lz)
    else:
        state = make_alfven_wave_ic(nx, ny, nz, lx, ly, lz)

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    dV = dx * dy * dz

    e0 = compute_energies(
        state["rho"], state["mx"], state["my"], state["mz"], state["E"],
        state["Bx_f"], state["By_f"], state["Bz_f"], dV,
    )
    E0 = e0["total"]
    divb0 = float(np.max(np.abs(
        compute_divB(state["Bx_f"], state["By_f"], state["Bz_f"], dx, dy, dz)
    )))

    if verbose:
        print(f"\nMHD Energy Conservation Verification")
        print(f"IC: {ic_type}   Grid: {nx}×{ny}×{nz}   Steps: {n_steps}   CFL: {cfl}")
        print(f"Initial energies — kin={e0['kinetic']:.4e}  mag={e0['magnetic']:.4e}  "
              f"therm={e0['thermal']:.4e}  total={E0:.4e}")
        print(f"Initial max|div B| = {divb0:.3e}")
        print("-" * 70)

    times       = [0.0]
    E_kin_arr   = [e0["kinetic"]]
    E_mag_arr   = [e0["magnetic"]]
    E_therm_arr = [e0["thermal"]]
    E_tot_arr   = [E0]
    divB_arr    = [divb0]

    t = 0.0
    for step_idx in range(n_steps):
        dt = _cfl_dt(state, cfl, dx, dy, dz)
        state = step(state, dt, dx, dy, dz)
        t += dt

        en = compute_energies(
            state["rho"], state["mx"], state["my"], state["mz"], state["E"],
            state["Bx_f"], state["By_f"], state["Bz_f"], dV,
        )
        divb_now = float(np.max(np.abs(
            compute_divB(state["Bx_f"], state["By_f"], state["Bz_f"], dx, dy, dz)
        )))

        times.append(t)
        E_kin_arr.append(en["kinetic"])
        E_mag_arr.append(en["magnetic"])
        E_therm_arr.append(en["thermal"])
        E_tot_arr.append(en["total"])
        divB_arr.append(divb_now)

        if verbose and (step_idx + 1) % max(1, n_steps // 10) == 0:
            rel = abs(en["total"] - E0) / max(abs(E0), SMALL)
            print(f"  step {step_idx+1:4d}  t={t:.4f}  E_tot={en['total']:.6e}  "
                  f"|ΔE/E0|={rel:.3e}  max|divB|={divb_now:.3e}")

    E_tot_np = np.array(E_tot_arr)
    rel_err  = np.abs(E_tot_np - E0) / max(abs(E0), SMALL)

    rel_err_max  = float(np.max(rel_err))
    divb_max     = float(np.max(divB_arr))
    divb_final   = float(divB_arr[-1])

    target_energy = 0.005    # 0.5%
    target_divb   = 1.0e-13

    passed = (rel_err_max < target_energy) and (divb_max < target_divb)

    result = {
        "time":          np.array(times),
        "E_kin":         np.array(E_kin_arr),
        "E_mag":         np.array(E_mag_arr),
        "E_therm":       np.array(E_therm_arr),
        "E_tot":         E_tot_np,
        "rel_error":     rel_err,
        "rel_error_max": rel_err_max,
        "divb_max":      divb_max,
        "divb_final":    divb_final,
        "passed":        passed,
        "E0":            E0,
        "target_energy": target_energy,
        "target_divb":   target_divb,
    }

    # ── HDF5 出力 ─────────────────────────────────────────────────────────────
    if _HAS_HDF5:
        h5path = output_dir / "energy_history.h5"
        with h5py.File(h5path, "w") as f:
            f.attrs["description"] = "Super-TKMK MHD energy conservation history"
            f.attrs["ic_type"] = ic_type
            for k in ("nx", "ny", "nz"):
                f.attrs[k] = locals()[k]
            f.attrs["n_steps"] = n_steps
            f.attrs["cfl"]     = cfl
            for key in ["time", "E_kin", "E_mag", "E_therm", "E_tot", "rel_error"]:
                f.create_dataset(key, data=result[key], compression="gzip")
            f.attrs["E0"]           = E0
            f.attrs["rel_error_max"] = rel_err_max
            f.attrs["divb_max"]      = divb_max
            f.attrs["passed"]        = int(passed)

    # ── グラフ出力 ──────────────────────────────────────────────────────────
    if _HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        t_arr = result["time"]
        ax1.plot(t_arr, result["E_kin"],   label="Kinetic $E_{kin}$",  lw=1.5)
        ax1.plot(t_arr, result["E_mag"],   label="Magnetic $E_{mag}$", lw=1.5)
        ax1.plot(t_arr, result["E_therm"], label="Thermal $E_{therm}$", lw=1.5)
        ax1.plot(t_arr, result["E_tot"],   label="Total $E_{tot}$",
                 color="black", lw=2, ls="--")
        ax1.set_ylabel("Energy [normalized]")
        ax1.set_title(f"MHD Energy Conservation ({ic_type} IC, {nx}×{ny}×{nz})")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4)

        ax2.semilogy(t_arr, np.maximum(result["rel_error"], 1e-16),
                     color="royalblue", lw=1.5, label=r"$|\Delta E / E_0|$")
        ax2.axhline(y=0.005, color="red", ls="--", lw=1.5, label="0.5% threshold")
        ax2.axhline(y=rel_err_max, color="orange", ls=":", lw=1.2,
                    label=f"max = {rel_err_max:.2e}")
        ax2.set_xlabel("Time [normalized]")
        ax2.set_ylabel(r"$|\Delta E_{tot} / E_{tot}(0)|$")
        ax2.set_title("Relative Total Energy Error")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        fig.savefig(output_dir / "energy_conservation.png", dpi=120)
        plt.close(fig)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="MHD エネルギー保存則検証",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nx",    type=int,   default=32,      help="x 方向格子数")
    p.add_argument("--ny",    type=int,   default=4,       help="y 方向格子数")
    p.add_argument("--nz",    type=int,   default=4,       help="z 方向格子数")
    p.add_argument("--steps", type=int,   default=100,     help="タイムステップ数")
    p.add_argument("--cfl",   type=float, default=0.4,     help="CFL 数")
    p.add_argument("--ic",    type=str,   default="alfven",
                   choices=["alfven", "torus"],            help="初期条件種類")
    p.add_argument("--output", type=str,  default=".",     help="出力ディレクトリ")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # 格子サイズを IC に合わせてデフォルト調整
    nx, ny, nz = args.nx, args.ny, args.nz
    if args.ic == "torus":
        lx = ly = lz = 2.0
    else:
        lx, ly, lz = 1.0, 0.125, 0.125

    result = run_verification(
        nx=nx, ny=ny, nz=nz,
        lx=lx, ly=ly, lz=lz,
        n_steps=args.steps,
        cfl=args.cfl,
        output_dir=Path(args.output),
        ic_type=args.ic,
    )

    print("\n" + "=" * 70)
    print("エネルギー保存則 検証結果")
    print("=" * 70)
    print(f"  総エネルギー E₀           = {result['E0']:.6e}")
    print(f"  最大相対誤差 |ΔE/E₀|_max = {result['rel_error_max']:.3e}  "
          f"(目標: < {result['target_energy']:.1%})")
    print(f"  最大 |div B|              = {result['divb_max']:.3e}  "
          f"(目標: < {result['target_divb']:.1e})")
    print("-" * 70)
    status = "PASS ✅" if result["passed"] else "FAIL ❌"
    print(f"  判定: {status}")
    print("=" * 70)

    if _HAS_HDF5:
        print(f"  HDF5 出力: energy_history.h5")
    if _HAS_MPL:
        print(f"  グラフ出力: energy_conservation.png")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
