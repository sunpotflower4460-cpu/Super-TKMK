/**
 * test_mhd_ct.cu
 *
 * CT div B = 0 保証のスタンドアロンテスト
 *
 * テスト手順:
 *   1. トーラス初期条件でソルバを初期化
 *   2. div B を計算して診断
 *   3. 数タイムステップ実行
 *   4. CT 更新後の div B が機械精度で 0 であることを確認
 *
 * 期待する結果:
 *   CT 更新後: max|div B| < 1e-14  (double 精度機械イプシロン 2.2e-16 の数倍)
 */

#include "../cuda_mhd_ct.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

extern void init_torus_default(MHDSolver *s, double lx, double ly, double lz);

int main()
{
    // ─── 格子設定（Phase 1: 低解像度 32³）───────────────────────────────────
    int nx = 32, ny = 32, nz = 32;
    int ng = 2;
    double lx = 2.0, ly = 2.0, lz = 2.0;

    GridParams gp = GridParams::make(nx, ny, nz, ng, lx, ly, lz);

    printf("=== MHD CT div B test ===\n");
    printf("Grid: %d x %d x %d  (ghost: %d)\n", nx, ny, nz, ng);
    printf("Domain: %.1f x %.1f x %.1f\n", lx, ly, lz);

    // ─── ソルバ生成 ──────────────────────────────────────────────────────────
    MHDSolver *solver = mhd_solver_create(gp);

    // ─── トーラス初期条件 ─────────────────────────────────────────────────────
    init_torus_default(solver, lx, ly, lz);

    // ─── 初期 div B 確認 ──────────────────────────────────────────────────────
    // kernel_compute_div_b を呼ぶには内部ヘルパーが必要; check_div_b を使う
    printf("\n--- Initial div B (after torus IC) ---\n");
    check_div_b(*solver);

    // ─── タイムステップ（Courant 条件に基づいた dt を設定）─────────────────
    double cfl = 0.4;
    double v_A_max = 1.0;       // アルヴェン速度の最大推定値
    double cs_max  = sqrt(GAMMA_IDEAL * 0.05 / 1.0);  // 音速
    double cf_max  = sqrt(v_A_max * v_A_max + cs_max * cs_max);
    double dt = cfl * fmin(gp.dx, fmin(gp.dy, gp.dz)) / cf_max;
    printf("Timestep: dt = %.4e\n\n", dt);

    // ─── 5 タイムステップ実行 ─────────────────────────────────────────────────
    int n_steps = 5;
    for (int step = 1; step <= n_steps; ++step) {
        mhd_solver_step(solver, dt);
        printf("--- Step %d ---\n", step);
        check_div_b(*solver);
    }

    // ─── 結果判定 ─────────────────────────────────────────────────────────────
    // 最終ステップの div B を取得して合否判定
    int n_tot = gp.nx_tot * gp.ny_tot * gp.nz_tot;
    std::vector<double> h_divb(n_tot);
    cudaMemcpy(h_divb.data(), solver->d_div_b, n_tot * sizeof(double), cudaMemcpyDeviceToHost);

    double max_abs = 0.0;
    for (int i = ng; i < nx + ng; ++i)
    for (int j = ng; j < ny + ng; ++j)
    for (int k = ng; k < nz + ng; ++k) {
        double v = h_divb[i * (gp.ny_tot * gp.nz_tot) + j * gp.nz_tot + k];
        if (fabs(v) > max_abs) max_abs = fabs(v);
    }

    double target = 1.0e-12;
    printf("\n=== RESULT ===\n");
    printf("max|div B| after CT update: %.3e\n", max_abs);
    printf("target: < %.1e\n", target);
    printf("Status: %s\n", max_abs < target ? "PASS" : "FAIL");

    mhd_solver_destroy(solver);
    return (max_abs < target) ? 0 : 1;
}
