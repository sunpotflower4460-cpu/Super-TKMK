/**
 * cuda_mhd_ct.cu
 *
 * Godunov型 + Constrained Transport (CT) による理想MHDソルバ（CUDA実装）
 *
 * 参考文献:
 *   - Balsara & Spicer (1999), J. Comput. Phys. 149, 270-292  [CT基礎]
 *   - Miniati & Martin (2011), ApJS 195, 5                     [CT + AMR]
 *   - Stone et al. (2008) Athena CT モジュール
 *   - Miyoshi & Kusano (2005), J. Comput. Phys. 208, 315-344  [HLLD]
 *
 * データレイアウト
 * ─────────────────
 *   セル中心量 (conserved): rho, rho_vx, rho_vy, rho_vz, E
 *   面中心磁場 (face B):
 *     Bx[i+1/2, j, k]  (x面法線)
 *     By[i, j+1/2, k]  (y面法線)
 *     Bz[i, j, k+1/2]  (z面法線)
 *   辺中心EMF (edge EMF, CT用):
 *     Ex[i, j+1/2, k+1/2]  (y-z辺)
 *     Ey[i+1/2, j, k+1/2]  (z-x辺)
 *     Ez[i+1/2, j+1/2, k]  (x-y辺)
 *
 * メモリ順序: C-orderの3Dフラットインデックス
 *   idx(i,j,k) = i*(NY*NZ) + j*NZ + k
 *
 * div B = 0 保証:
 *   CT法では磁場は常に face-centered で保持され、CT更新後の div B は
 *   離散格子上で機械精度まで正確に 0 になる（理論的保証）。
 */

#include "cuda_mhd_ct.cuh"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// 定数・マクロ
// ─────────────────────────────────────────────────────────────────────────────

// 比熱比（単原子理想気体 γ = 5/3、または等温近似 γ = 1 + ε）
#ifndef GAMMA_IDEAL
#define GAMMA_IDEAL 1.6666666666666667
#endif

// 数値安定用の小量
#ifndef SMALL_NUM
#define SMALL_NUM 1.0e-30
#endif

// フラットインデックス（境界セル込みのストライド）
// NX_TOT = NX + 2*NGHOST
#define IDX(i, j, k)   ((i) * (params.ny_tot * params.nz_tot) + (j) * (params.nz_tot) + (k))

// 面中心Bのインデックス（各面は独立バッファ）
// Bx: (NX+1) * NY * NZ
#define BX_IDX(i, j, k) ((i) * (params.ny * params.nz) + (j) * params.nz + (k))
// By: NX * (NY+1) * NZ
#define BY_IDX(i, j, k) ((i) * ((params.ny + 1) * params.nz) + (j) * params.nz + (k))
// Bz: NX * NY * (NZ+1)
#define BZ_IDX(i, j, k) ((i) * (params.ny * (params.nz + 1)) + (j) * (params.nz + 1) + (k))

// 辺中心EMFのインデックス
// Ex: NX * (NY+1) * (NZ+1)
#define EX_IDX(i, j, k) ((i) * ((params.ny + 1) * (params.nz + 1)) + (j) * (params.nz + 1) + (k))
// Ey: (NX+1) * NY * (NZ+1)
#define EY_IDX(i, j, k) ((i) * (params.ny * (params.nz + 1)) + (j) * (params.nz + 1) + (k))
// Ez: (NX+1) * (NY+1) * NZ
#define EZ_IDX(i, j, k) ((i) * ((params.ny + 1) * params.nz) + (j) * params.nz + (k))

// ─────────────────────────────────────────────────────────────────────────────
// デバイス補助関数
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 保存変数 → 原始変数の変換
 * U = (rho, rho_vx, rho_vy, rho_vz, E)
 * W = (rho,     vx,     vy,     vz, p, Bx, By, Bz)
 *
 * 全エネルギー E = p/(γ-1) + 0.5*rho*v² + 0.5*B²
 */
__device__ __forceinline__ void conserved_to_primitive(
    double rho, double mx, double my, double mz, double E,
    double Bx, double By, double Bz,
    double &vx, double &vy, double &vz, double &p)
{
    double inv_rho = 1.0 / (rho + SMALL_NUM);
    vx = mx * inv_rho;
    vy = my * inv_rho;
    vz = mz * inv_rho;
    double ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    double bsq = 0.5 * (Bx * Bx + By * By + Bz * Bz);
    p = (GAMMA_IDEAL - 1.0) * (E - ke - bsq);
    p = fmax(p, SMALL_NUM); // 負圧防止
}

/**
 * HLLE Riemann ソルバ（x方向フラックス）
 *
 * 引数: 左・右の原始変数 (rho, vx, vy, vz, p, Bx, By, Bz)
 * 戻り値: 数値フラックス F = (F_rho, F_mx, F_my, F_mz, F_E, F_By, F_Bz)
 *         (Bx成分は CT で別管理するため通常Godunov更新では使わない)
 *
 * 最大/最小信号速度は磁気音速を使った単純な推定。
 * より精度が必要な場合は HLLD（Miyoshi & Kusano 2005）に差し替える。
 */
__device__ void hlle_flux_x(
    double rho_L, double vx_L, double vy_L, double vz_L, double p_L,
    double Bx_L, double By_L, double Bz_L,
    double rho_R, double vx_R, double vy_R, double vz_R, double p_R,
    double Bx_R, double By_R, double Bz_R,
    double *F_rho, double *F_mx, double *F_my, double *F_mz,
    double *F_E, double *F_By, double *F_Bz)
{
    // 左右の保存量
    double bsq_L = Bx_L * Bx_L + By_L * By_L + Bz_L * Bz_L;
    double bsq_R = Bx_R * Bx_R + By_R * By_R + Bz_R * Bz_R;
    double E_L = p_L / (GAMMA_IDEAL - 1.0)
               + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L)
               + 0.5 * bsq_L;
    double E_R = p_R / (GAMMA_IDEAL - 1.0)
               + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R)
               + 0.5 * bsq_R;

    // 磁気音速
    double ca2_L = bsq_L / (rho_L + SMALL_NUM);
    double ca2_R = bsq_R / (rho_R + SMALL_NUM);
    double cs2_L = GAMMA_IDEAL * p_L / (rho_L + SMALL_NUM);
    double cs2_R = GAMMA_IDEAL * p_R / (rho_R + SMALL_NUM);
    // fast magnetosonic speed: cf² = 0.5*(ca²+cs² + sqrt((ca²+cs²)²-4*cs²*cax²))
    double sum_L = ca2_L + cs2_L;
    double sum_R = ca2_R + cs2_R;
    double cax2_L = Bx_L * Bx_L / (rho_L + SMALL_NUM);
    double cax2_R = Bx_R * Bx_R / (rho_R + SMALL_NUM);
    double cf_L = sqrt(0.5 * (sum_L + sqrt(fmax(sum_L * sum_L - 4.0 * cs2_L * cax2_L, 0.0))));
    double cf_R = sqrt(0.5 * (sum_R + sqrt(fmax(sum_R * sum_R - 4.0 * cs2_R * cax2_R, 0.0))));

    // HLLE 信号速度推定（Roe 平均を省略した簡易推定）
    double S_L = fmin(vx_L - cf_L, vx_R - cf_R);
    double S_R = fmax(vx_L + cf_L, vx_R + cf_R);

    // 物理フラックス（F = F(U)）
    double ptot_L = p_L + 0.5 * bsq_L;
    double ptot_R = p_R + 0.5 * bsq_R;

    // F_rho = rho*vx
    double fL_rho  = rho_L * vx_L;
    double fR_rho  = rho_R * vx_R;
    // F_mx = rho*vx²  + ptot - Bx²
    double fL_mx   = rho_L * vx_L * vx_L + ptot_L - Bx_L * Bx_L;
    double fR_mx   = rho_R * vx_R * vx_R + ptot_R - Bx_R * Bx_R;
    // F_my = rho*vx*vy - Bx*By
    double fL_my   = rho_L * vx_L * vy_L - Bx_L * By_L;
    double fR_my   = rho_R * vx_R * vy_R - Bx_R * By_R;
    // F_mz = rho*vx*vz - Bx*Bz
    double fL_mz   = rho_L * vx_L * vz_L - Bx_L * Bz_L;
    double fR_mz   = rho_R * vx_R * vz_R - Bx_R * Bz_R;
    // F_E = (E+ptot)*vx - Bx*(v·B)
    double vdotB_L = vx_L * Bx_L + vy_L * By_L + vz_L * Bz_L;
    double vdotB_R = vx_R * Bx_R + vy_R * By_R + vz_R * Bz_R;
    double fL_E    = (E_L + ptot_L) * vx_L - Bx_L * vdotB_L;
    double fR_E    = (E_R + ptot_R) * vx_R - Bx_R * vdotB_R;
    // F_By = By*vx - Bx*vy  (誘導方程式)
    double fL_By   = By_L * vx_L - Bx_L * vy_L;
    double fR_By   = By_R * vx_R - Bx_R * vy_R;
    // F_Bz = Bz*vx - Bx*vz
    double fL_Bz   = Bz_L * vx_L - Bx_L * vz_L;
    double fR_Bz   = Bz_R * vx_R - Bx_R * vz_R;

    // HLLE フラックス
    if (S_L >= 0.0) {
        // 超音速左向き: 左フラックスをそのまま使用
        *F_rho = fL_rho;
        *F_mx  = fL_mx;
        *F_my  = fL_my;
        *F_mz  = fL_mz;
        *F_E   = fL_E;
        *F_By  = fL_By;
        *F_Bz  = fL_Bz;
    } else if (S_R <= 0.0) {
        // 超音速右向き: 右フラックスをそのまま使用
        *F_rho = fR_rho;
        *F_mx  = fR_mx;
        *F_my  = fR_my;
        *F_mz  = fR_mz;
        *F_E   = fR_E;
        *F_By  = fR_By;
        *F_Bz  = fR_Bz;
    } else {
        // 遷音速領域: HLLE 混合フラックス
        double inv_SR_SL = 1.0 / (S_R - S_L + SMALL_NUM);

        // U_HLLE 星状態は保存量で
        double U_L[7] = { rho_L,
                          rho_L * vx_L, rho_L * vy_L, rho_L * vz_L,
                          E_L, By_L, Bz_L };
        double U_R[7] = { rho_R,
                          rho_R * vx_R, rho_R * vy_R, rho_R * vz_R,
                          E_R, By_R, Bz_R };
        double fL[7]  = { fL_rho, fL_mx, fL_my, fL_mz, fL_E, fL_By, fL_Bz };
        double fR[7]  = { fR_rho, fR_mx, fR_my, fR_mz, fR_E, fR_By, fR_Bz };

        double F[7];
        for (int n = 0; n < 7; ++n) {
            F[n] = (S_R * fL[n] - S_L * fR[n] + S_L * S_R * (U_R[n] - U_L[n])) * inv_SR_SL;
        }
        *F_rho = F[0];
        *F_mx  = F[1];
        *F_my  = F[2];
        *F_mz  = F[3];
        *F_E   = F[4];
        *F_By  = F[5];
        *F_Bz  = F[6];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA カーネル: 保存変数更新（Godunov法 x/y/z 各方向）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * x方向 Godunov フラックスを計算し、保存変数を更新する。
 *
 * 1次元分割した PLM（Piecewise Linear Method）で左右状態を構成し、
 * HLLE フラックスを計算する。
 * 磁場 Bx は face-centered なのでこのカーネルでは更新しない（CTで別途更新）。
 */
__global__ void kernel_godunov_update_x(
    const double * __restrict__ rho,
    const double * __restrict__ mx,
    const double * __restrict__ my,
    const double * __restrict__ mz,
    const double * __restrict__ E,
    const double * __restrict__ Bx_face,  // (NX+1) x NY x NZ
    const double * __restrict__ By_face,  // NX x (NY+1) x NZ (セル面法線By)
    const double * __restrict__ Bz_face,  // NX x NY x (NZ+1)
    double * __restrict__ d_rho,
    double * __restrict__ d_mx,
    double * __restrict__ d_my,
    double * __restrict__ d_mz,
    double * __restrict__ d_E,
    double * __restrict__ emf_Ez,         // (NX+1) x (NY+1) x NZ  [出力]
    double * __restrict__ emf_Ey,         // (NX+1) x NY x (NZ+1)  [出力]
    GridParams params,
    double dt_dx)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + params.ng;
    int k = blockIdx.y * blockDim.y + threadIdx.y + params.ng;

    if (j >= params.ny + params.ng || k >= params.nz + params.ng) return;

    for (int i = params.ng; i < params.nx + params.ng; ++i) {
        // セル中心の保存量（iM = i-1, iP = i+1）
        int cM = IDX(i - 1, j, k);
        int c0 = IDX(i,     j, k);
        int cP = IDX(i + 1, j, k);

        // 面中心 Bx (x-face): i と i+1 面の値
        int bx_iM = BX_IDX(i,     j - params.ng, k - params.ng);
        int bx_iP = BX_IDX(i + 1, j - params.ng, k - params.ng);

        // セル中心近似の B（面値の平均）
        double Bx0 = 0.5 * (Bx_face[bx_iM] + Bx_face[bx_iP]);
        // By, Bz も同様に面平均
        int by_j0  = BY_IDX(i - params.ng, j - params.ng,     k - params.ng);
        int by_j1  = BY_IDX(i - params.ng, j - params.ng + 1, k - params.ng);
        double By0 = 0.5 * (By_face[by_j0] + By_face[by_j1]);
        int bz_k0  = BZ_IDX(i - params.ng, j - params.ng, k - params.ng);
        int bz_k1  = BZ_IDX(i - params.ng, j - params.ng, k - params.ng + 1);
        double Bz0 = 0.5 * (Bz_face[bz_k0] + Bz_face[bz_k1]);

        // 左右セルの原始変数
        double rhoM, vxM, vyM, vzM, pM;
        double rhoP, vxP, vyP, vzP, pP;

        double BxM = 0.5 * (Bx_face[BX_IDX(i - 1, j - params.ng, k - params.ng)]
                          + Bx_face[bx_iM]);
        double ByM = 0.5 * (By_face[BY_IDX(i - 1 - params.ng, j - params.ng,     k - params.ng)]
                          + By_face[BY_IDX(i - 1 - params.ng, j - params.ng + 1, k - params.ng)]);
        double BzM = 0.5 * (Bz_face[BZ_IDX(i - 1 - params.ng, j - params.ng, k - params.ng)]
                          + Bz_face[BZ_IDX(i - 1 - params.ng, j - params.ng, k - params.ng + 1)]);

        conserved_to_primitive(rho[cM], mx[cM], my[cM], mz[cM], E[cM],
                               BxM, ByM, BzM, vxM, vyM, vzM, pM);
        rhoM = rho[cM];

        double BxP = 0.5 * (Bx_face[bx_iP]
                          + Bx_face[BX_IDX(i + 2, j - params.ng, k - params.ng)]);
        double ByP = 0.5 * (By_face[BY_IDX(i + 1 - params.ng, j - params.ng,     k - params.ng)]
                          + By_face[BY_IDX(i + 1 - params.ng, j - params.ng + 1, k - params.ng)]);
        double BzP = 0.5 * (Bz_face[BZ_IDX(i + 1 - params.ng, j - params.ng, k - params.ng)]
                          + Bz_face[BZ_IDX(i + 1 - params.ng, j - params.ng, k - params.ng + 1)]);

        conserved_to_primitive(rho[cP], mx[cP], my[cP], mz[cP], E[cP],
                               BxP, ByP, BzP, vxP, vyP, vzP, pP);
        rhoP = rho[cP];

        // 現セルの原始変数
        double vx0, vy0, vz0, p0;
        conserved_to_primitive(rho[c0], mx[c0], my[c0], mz[c0], E[c0],
                               Bx0, By0, Bz0, vx0, vy0, vz0, p0);

        // ─── PLM 勾配（minmod リミタ）───
        // (左面の状態 WL, 右面の状態 WR を一次精度で構成)
        // 完全な2次精度にするには各変数に minmod を適用する
#define MINMOD(a, b) ((a) * (b) > 0.0 ? (fabs(a) < fabs(b) ? (a) : (b)) : 0.0)
        double drho  = MINMOD(rho[c0] - rho[cM], rho[cP] - rho[c0]);
        double dvx   = MINMOD(vx0    - vxM,     vxP    - vx0);
        double dvy   = MINMOD(vy0    - vyM,     vyP    - vy0);
        double dvz   = MINMOD(vz0    - vzM,     vzP    - vz0);
        double dp    = MINMOD(p0     - pM,      pP     - p0);
#undef MINMOD
        // 面での左/右状態
        double rho_L = rho[c0] - 0.5 * drho;
        double rho_R = rho[c0] + 0.5 * drho;
        double vx_L  = vx0     - 0.5 * dvx;
        double vx_R  = vx0     + 0.5 * dvx;
        double vy_L  = vy0     - 0.5 * dvy;
        double vy_R  = vy0     + 0.5 * dvy;
        double vz_L  = vz0     - 0.5 * dvz;
        double vz_R  = vz0     + 0.5 * dvz;
        double p_L   = p0      - 0.5 * dp;
        double p_R   = p0      + 0.5 * dp;
        double Bx_iface = Bx_face[bx_iP]; // x+1/2 面のBx（CT用に face 値を使う）

        // HLLE フラックス (i+1/2 面)
        double F_rho, F_mxf, F_myf, F_mzf, F_Ef, F_Byf, F_Bzf;
        hlle_flux_x(rho_L, vx_L, vy_L, vz_L, p_L, Bx_iface, By0, Bz0,
                    rho_R, vx_R, vy_R, vz_R, p_R, Bx_iface, By0, Bz0,
                    &F_rho, &F_mxf, &F_myf, &F_mzf, &F_Ef, &F_Byf, &F_Bzf);

        // 前半ステップの更新量を保存（完全更新は y,z フラックスも加算後に行う）
        atomicAdd(&d_rho[c0], -dt_dx * F_rho);
        atomicAdd(&d_mx[c0],  -dt_dx * F_mxf);
        atomicAdd(&d_my[c0],  -dt_dx * F_myf);
        atomicAdd(&d_mz[c0],  -dt_dx * F_mzf);
        atomicAdd(&d_E[c0],   -dt_dx * F_Ef);

        // 右隣セル (i+1) への寄与
        atomicAdd(&d_rho[cP],  dt_dx * F_rho);
        atomicAdd(&d_mx[cP],   dt_dx * F_mxf);
        atomicAdd(&d_my[cP],   dt_dx * F_myf);
        atomicAdd(&d_mz[cP],   dt_dx * F_mzf);
        atomicAdd(&d_E[cP],    dt_dx * F_Ef);

        // CT 用 EMF の保存 (Ey at x+1/2, k+1/2 および Ez at x+1/2, j+1/2)
        // Ez = -(vy * Bx - vx * By) を HLLE から推定
        // この値は CT EMF 計算カーネルで使う
        double Ez_face = -(vy_L * Bx_iface - vx_L * By0);  // 簡易推定
        double Ey_face =  (vz_L * Bx_iface - vx_L * Bz0);

        // emf_Ez[i+1, j, k] に追記（y方向カーネルの寄与と後で平均する）
        int ez_idx = EZ_IDX(i - params.ng + 1, j - params.ng, k - params.ng);
        atomicAdd(&emf_Ez[ez_idx], 0.5 * Ez_face);

        int ey_idx = EY_IDX(i - params.ng + 1, j - params.ng, k - params.ng);
        atomicAdd(&emf_Ey[ey_idx], 0.5 * Ey_face);
    }
}

/**
 * CT EMF 計算カーネル
 *
 * Balsara & Spicer (1999) Eq. (2)
 * EMF は各辺に4つの隣接面フラックスの平均として計算する。
 *
 * Ez at (i+1/2, j+1/2, k):
 *   Ez = 0.25 * (Ez_xflux[i,j] + Ez_xflux[i,j+1] + Ez_yflux[i,j] + Ez_yflux[i+1,j])
 */
__global__ void kernel_compute_ct_emf(
    const double * __restrict__ emf_Ez_x,  // x方向フラックスから得た Ez 寄与
    const double * __restrict__ emf_Ez_y,  // y方向フラックスから得た Ez 寄与
    const double * __restrict__ emf_Ey_x,  // x方向フラックスから得た Ey 寄与
    const double * __restrict__ emf_Ey_z,  // z方向フラックスから得た Ey 寄与
    const double * __restrict__ emf_Ex_y,  // y方向フラックスから得た Ex 寄与
    const double * __restrict__ emf_Ex_z,  // z方向フラックスから得た Ex 寄与
    double * __restrict__ emf_Ez_ct,       // 最終 CT EMF Ez [(NX+1) x (NY+1) x NZ]
    double * __restrict__ emf_Ey_ct,       // 最終 CT EMF Ey [(NX+1) x NY x (NZ+1)]
    double * __restrict__ emf_Ex_ct,       // 最終 CT EMF Ex [NX x (NY+1) x (NZ+1)]
    GridParams params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ez at (i+1/2, j+1/2, k): i in [0,NX], j in [0,NY], k in [0,NZ-1]
    if (i <= params.nx && j <= params.ny && k < params.nz) {
        // 4辺の寄与の算術平均
        // Balsara & Spicer (1999), Eq. (18)
        double ez = 0.25 * (emf_Ez_x[EZ_IDX(i, j,   k)]
                          + emf_Ez_x[EZ_IDX(i, j+1, k)]
                          + emf_Ez_y[EZ_IDX(i,   j, k)]
                          + emf_Ez_y[EZ_IDX(i+1, j, k)]);
        emf_Ez_ct[EZ_IDX(i, j, k)] = ez;
    }

    // Ey at (i+1/2, j, k+1/2)
    if (i <= params.nx && j < params.ny && k <= params.nz) {
        double ey = 0.25 * (emf_Ey_x[EY_IDX(i, j, k)]
                          + emf_Ey_x[EY_IDX(i, j, k+1)]
                          + emf_Ey_z[EY_IDX(i,   j, k)]
                          + emf_Ey_z[EY_IDX(i+1, j, k)]);
        emf_Ey_ct[EY_IDX(i, j, k)] = ey;
    }

    // Ex at (i, j+1/2, k+1/2)
    if (i < params.nx && j <= params.ny && k <= params.nz) {
        double ex = 0.25 * (emf_Ex_y[EX_IDX(i, j, k)]
                          + emf_Ex_y[EX_IDX(i, j, k+1)]
                          + emf_Ex_z[EX_IDX(i, j,   k)]
                          + emf_Ex_z[EX_IDX(i, j+1, k)]);
        emf_Ex_ct[EX_IDX(i, j, k)] = ex;
    }
}

/**
 * 面中心磁場 CT 更新カーネル
 *
 * ∂Bx/∂t = -(∂Ez/∂y - ∂Ey/∂z)
 * ∂By/∂t = -(∂Ex/∂z - ∂Ez/∂x)
 * ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
 *
 * 離散化（Balsara & Spicer 1999, Eq. 4）:
 *   Bx[i+1/2,j,k]^{n+1} = Bx[i+1/2,j,k]^n
 *     - dt/dy * (Ez[i+1/2,j+1/2,k] - Ez[i+1/2,j-1/2,k])
 *     + dt/dz * (Ey[i+1/2,j,k+1/2] - Ey[i+1/2,j,k-1/2])
 */
__global__ void kernel_ct_update_face_B(
    const double * __restrict__ emf_Ex_ct,
    const double * __restrict__ emf_Ey_ct,
    const double * __restrict__ emf_Ez_ct,
    double * __restrict__ Bx_face,
    double * __restrict__ By_face,
    double * __restrict__ Bz_face,
    GridParams params,
    double dt_dy, double dt_dz, double dt_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Bx[i+1/2, j, k]: i in [0,NX], j in [0,NY-1], k in [0,NZ-1]
    if (i <= params.nx && j < params.ny && k < params.nz) {
        int idx = BX_IDX(i, j, k);
        Bx_face[idx] += -dt_dy * (emf_Ez_ct[EZ_IDX(i, j + 1, k)] - emf_Ez_ct[EZ_IDX(i, j, k)])
                       + dt_dz * (emf_Ey_ct[EY_IDX(i, j, k + 1)] - emf_Ey_ct[EY_IDX(i, j, k)]);
    }

    // By[i, j+1/2, k]: i in [0,NX-1], j in [0,NY], k in [0,NZ-1]
    if (i < params.nx && j <= params.ny && k < params.nz) {
        int idx = BY_IDX(i, j, k);
        By_face[idx] +=  dt_dx * (emf_Ez_ct[EZ_IDX(i + 1, j, k)] - emf_Ez_ct[EZ_IDX(i, j, k)])
                       - dt_dz * (emf_Ex_ct[EX_IDX(i, j, k + 1)] - emf_Ex_ct[EX_IDX(i, j, k)]);
    }

    // Bz[i, j, k+1/2]: i in [0,NX-1], j in [0,NY-1], k in [0,NZ]
    if (i < params.nx && j < params.ny && k <= params.nz) {
        int idx = BZ_IDX(i, j, k);
        Bz_face[idx] += -dt_dx * (emf_Ey_ct[EY_IDX(i + 1, j, k)] - emf_Ey_ct[EY_IDX(i, j, k)])
                       + dt_dy * (emf_Ex_ct[EX_IDX(i, j + 1, k)] - emf_Ex_ct[EX_IDX(i, j, k)]);
    }
}

/**
 * div B 計算カーネル
 *
 * 離散的な divergence（面中心Bから）:
 *   (div B)[i,j,k] = (Bx[i+1/2]-Bx[i-1/2])/dx
 *                  + (By[j+1/2]-By[j-1/2])/dy
 *                  + (Bz[k+1/2]-Bz[k-1/2])/dz
 *
 * CT 更新後はこれが機械精度で 0 になることが理論的に保証される。
 */
__global__ void kernel_compute_div_b(
    const double * __restrict__ Bx_face,
    const double * __restrict__ By_face,
    const double * __restrict__ Bz_face,
    double * __restrict__ div_b,
    GridParams params,
    double inv_dx, double inv_dy, double inv_dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= params.nx || j >= params.ny || k >= params.nz) return;

    double dbx = (Bx_face[BX_IDX(i + 1, j, k)] - Bx_face[BX_IDX(i, j, k)]) * inv_dx;
    double dby = (By_face[BY_IDX(i, j + 1, k)] - By_face[BY_IDX(i, j, k)]) * inv_dy;
    double dbz = (Bz_face[BZ_IDX(i, j, k + 1)] - Bz_face[BZ_IDX(i, j, k)]) * inv_dz;

    div_b[IDX(i + params.ng, j + params.ng, k + params.ng)] = dbx + dby + dbz;
}

// ─────────────────────────────────────────────────────────────────────────────
// ホスト側: div B の最大値・L2ノルムを計算して表示
// ─────────────────────────────────────────────────────────────────────────────

void check_div_b(const MHDSolver &solver)
{
    const GridParams &p = solver.params;
    int n_active = p.nx * p.ny * p.nz;
    int n_tot    = p.nx_tot * p.ny_tot * p.nz_tot;

    // div_b は device 上にある（カーネル実行済みを想定）
    std::vector<double> h_div(n_tot, 0.0);
    cudaMemcpy(h_div.data(), solver.d_div_b, n_tot * sizeof(double), cudaMemcpyDeviceToHost);

    double max_abs = 0.0, sum_sq = 0.0;
    for (int i = p.ng; i < p.nx + p.ng; ++i)
    for (int j = p.ng; j < p.ny + p.ng; ++j)
    for (int k = p.ng; k < p.nz + p.ng; ++k) {
        double v = h_div[i * (p.ny_tot * p.nz_tot) + j * p.nz_tot + k];
        double av = fabs(v);
        if (av > max_abs) max_abs = av;
        sum_sq += av * av;
    }
    double l2 = sqrt(sum_sq / n_active);

    printf("[div B check]  max|divB| = %.3e   L2(divB) = %.3e\n", max_abs, l2);
}

// ─────────────────────────────────────────────────────────────────────────────
// MHDSolver 実装
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ソルバを初期化し、GPU 上にメモリを確保する。
 */
MHDSolver *mhd_solver_create(GridParams params)
{
    MHDSolver *s = new MHDSolver;
    s->params = params;

    int nx_tot = params.nx_tot;
    int ny_tot = params.ny_tot;
    int nz_tot = params.nz_tot;
    int n_cell = nx_tot * ny_tot * nz_tot;

    // セル中心保存量
    cudaMalloc(&s->d_rho,    n_cell * sizeof(double));
    cudaMalloc(&s->d_mx,     n_cell * sizeof(double));
    cudaMalloc(&s->d_my,     n_cell * sizeof(double));
    cudaMalloc(&s->d_mz,     n_cell * sizeof(double));
    cudaMalloc(&s->d_E,      n_cell * sizeof(double));
    cudaMalloc(&s->d_div_b,  n_cell * sizeof(double));

    // 差分バッファ（Godunov更新量の一時保存）
    cudaMalloc(&s->d_drho,   n_cell * sizeof(double));
    cudaMalloc(&s->d_dmx,    n_cell * sizeof(double));
    cudaMalloc(&s->d_dmy,    n_cell * sizeof(double));
    cudaMalloc(&s->d_dmz,    n_cell * sizeof(double));
    cudaMalloc(&s->d_dE,     n_cell * sizeof(double));

    // 面中心磁場
    int nbx = (params.nx + 1) * params.ny * params.nz;
    int nby = params.nx * (params.ny + 1) * params.nz;
    int nbz = params.nx * params.ny * (params.nz + 1);
    cudaMalloc(&s->d_Bx_face, nbx * sizeof(double));
    cudaMalloc(&s->d_By_face, nby * sizeof(double));
    cudaMalloc(&s->d_Bz_face, nbz * sizeof(double));

    // 辺中心 EMF（各方向の寄与用 + 最終CT EMF）
    int nex = params.nx * (params.ny + 1) * (params.nz + 1);
    int ney = (params.nx + 1) * params.ny * (params.nz + 1);
    int nez = (params.nx + 1) * (params.ny + 1) * params.nz;
    cudaMalloc(&s->d_emf_Ex, nex * sizeof(double));
    cudaMalloc(&s->d_emf_Ey, ney * sizeof(double));
    cudaMalloc(&s->d_emf_Ez, nez * sizeof(double));

    return s;
}

/**
 * ソルバを破棄し、GPU メモリを解放する。
 */
void mhd_solver_destroy(MHDSolver *s)
{
    cudaFree(s->d_rho);
    cudaFree(s->d_mx);
    cudaFree(s->d_my);
    cudaFree(s->d_mz);
    cudaFree(s->d_E);
    cudaFree(s->d_div_b);
    cudaFree(s->d_drho);
    cudaFree(s->d_dmx);
    cudaFree(s->d_dmy);
    cudaFree(s->d_dmz);
    cudaFree(s->d_dE);
    cudaFree(s->d_Bx_face);
    cudaFree(s->d_By_face);
    cudaFree(s->d_Bz_face);
    cudaFree(s->d_emf_Ex);
    cudaFree(s->d_emf_Ey);
    cudaFree(s->d_emf_Ez);
    delete s;
}

/**
 * 1タイムステップ進める（forward Euler、RK2 への拡張は容易）
 *
 * 手順:
 *   1. 差分バッファをゼロ初期化
 *   2. x/y/z 各方向の Godunov フラックスを計算し差分バッファに加算
 *   3. 差分バッファで保存変数を更新（rho, rho_v, E）
 *   4. CT EMF の収集・平均化
 *   5. 面中心 B を CT 更新
 *   6. div B を計算して確認
 */
void mhd_solver_step(MHDSolver *s, double dt)
{
    GridParams &p = s->params;
    double dt_dx = dt / p.dx;
    double dt_dy = dt / p.dy;
    double dt_dz = dt / p.dz;

    // 1. 差分バッファをゼロ初期化
    int n_cell = p.nx_tot * p.ny_tot * p.nz_tot;
    cudaMemset(s->d_drho, 0, n_cell * sizeof(double));
    cudaMemset(s->d_dmx,  0, n_cell * sizeof(double));
    cudaMemset(s->d_dmy,  0, n_cell * sizeof(double));
    cudaMemset(s->d_dmz,  0, n_cell * sizeof(double));
    cudaMemset(s->d_dE,   0, n_cell * sizeof(double));

    int nez = (p.nx + 1) * (p.ny + 1) * p.nz;
    int ney = (p.nx + 1) * p.ny * (p.nz + 1);
    int nex = p.nx * (p.ny + 1) * (p.nz + 1);
    cudaMemset(s->d_emf_Ez, 0, nez * sizeof(double));
    cudaMemset(s->d_emf_Ey, 0, ney * sizeof(double));
    cudaMemset(s->d_emf_Ex, 0, nex * sizeof(double));

    // 2. x方向 Godunov + EMF寄与
    dim3 threads2d(16, 16);
    dim3 blocks2d((p.ny + 15) / 16, (p.nz + 15) / 16);
    kernel_godunov_update_x<<<blocks2d, threads2d>>>(
        s->d_rho, s->d_mx, s->d_my, s->d_mz, s->d_E,
        s->d_Bx_face, s->d_By_face, s->d_Bz_face,
        s->d_drho, s->d_dmx, s->d_dmy, s->d_dmz, s->d_dE,
        s->d_emf_Ez, s->d_emf_Ey,
        p, dt_dx);

    // y/z 方向も同様のカーネルが必要（類似の実装、省略）

    // 3. 保存変数の更新
    // kernel_apply_update<<<...>>>(s->d_rho, s->d_drho, ...);

    // 4-5. CT EMF 収集・面B更新
    dim3 threads3d(8, 8, 8);
    dim3 blocks3d((p.nx + 9) / 8, (p.ny + 9) / 8, (p.nz + 9) / 8);
    kernel_ct_update_face_B<<<blocks3d, threads3d>>>(
        s->d_emf_Ex, s->d_emf_Ey, s->d_emf_Ez,
        s->d_Bx_face, s->d_By_face, s->d_Bz_face,
        p, dt_dy, dt_dz, dt_dx);

    // 6. div B 計算
    kernel_compute_div_b<<<blocks3d, threads3d>>>(
        s->d_Bx_face, s->d_By_face, s->d_Bz_face,
        s->d_div_b, p,
        1.0 / p.dx, 1.0 / p.dy, 1.0 / p.dz);

    cudaDeviceSynchronize();
}
