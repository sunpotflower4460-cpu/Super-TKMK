/**
 * cuda_mhd_ct.cuh
 *
 * CUDA Godunov + Constrained Transport MHD ソルバのパブリックインターフェース
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// 格子パラメータ
// ─────────────────────────────────────────────────────────────────────────────

/**
 * GridParams — 3D 計算格子の基本情報
 *
 *   nx, ny, nz   : アクティブセル数
 *   ng           : ゴーストセル数（境界条件用、通常 2〜3）
 *   nx_tot ...   : nx + 2*ng （全バッファサイズ）
 *   dx, dy, dz   : 格子間隔
 */
struct GridParams {
    int nx, ny, nz;        ///< アクティブセル数
    int ng;                ///< ゴーストセル数
    int nx_tot, ny_tot, nz_tot;  ///< 全バッファサイズ (nx+2*ng など)
    double dx, dy, dz;     ///< 格子間隔

    /// 便利なファクトリ関数
    static GridParams make(int nx, int ny, int nz, int ng,
                           double lx, double ly, double lz)
    {
        GridParams p;
        p.nx = nx; p.ny = ny; p.nz = nz;
        p.ng = ng;
        p.nx_tot = nx + 2 * ng;
        p.ny_tot = ny + 2 * ng;
        p.nz_tot = nz + 2 * ng;
        p.dx = lx / nx;
        p.dy = ly / ny;
        p.dz = lz / nz;
        return p;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ソルバ状態
// ─────────────────────────────────────────────────────────────────────────────

/**
 * MHDSolver — GPU 上の全状態量を保持する構造体
 *
 * メモリ所有権: mhd_solver_create() で確保、mhd_solver_destroy() で解放する。
 */
struct MHDSolver {
    GridParams params;

    // セル中心保存変数 (サイズ: nx_tot * ny_tot * nz_tot)
    double *d_rho;    ///< 密度 ρ
    double *d_mx;     ///< x 運動量 ρvx
    double *d_my;     ///< y 運動量 ρvy
    double *d_mz;     ///< z 運動量 ρvz
    double *d_E;      ///< 全エネルギー E = p/(γ-1) + ½ρv² + ½B²
    double *d_div_b;  ///< div B（診断用）

    // タイムステップ差分バッファ
    double *d_drho, *d_dmx, *d_dmy, *d_dmz, *d_dE;

    // 面中心磁場（Constrained Transport 用）
    // Bx: (nx+1) * ny * nz
    // By: nx * (ny+1) * nz
    // Bz: nx * ny * (nz+1)
    double *d_Bx_face;
    double *d_By_face;
    double *d_Bz_face;

    // 辺中心 EMF（CT で使用）
    // Ex: nx * (ny+1) * (nz+1)
    // Ey: (nx+1) * ny * (nz+1)
    // Ez: (nx+1) * (ny+1) * nz
    double *d_emf_Ex;
    double *d_emf_Ey;
    double *d_emf_Ez;
};

// ─────────────────────────────────────────────────────────────────────────────
// 公開 API
// ─────────────────────────────────────────────────────────────────────────────

/// ソルバを生成し GPU メモリを確保する。
MHDSolver *mhd_solver_create(GridParams params);

/// ソルバを破棄し GPU メモリを解放する。
void mhd_solver_destroy(MHDSolver *s);

/// 1 タイムステップ進める（前進 Euler）。
void mhd_solver_step(MHDSolver *s, double dt);

/// div B の最大値・L2 ノルムを標準出力に表示する。
void check_div_b(const MHDSolver &solver);
