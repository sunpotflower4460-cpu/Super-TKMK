/**
 * initial_conditions.cu
 *
 * トーラス形状 MHD 初期条件の設定（CUDA カーネル実装）
 *
 * ── 物理モデル ──────────────────────────────────────────────────────────────
 *
 * トカマク近似（大アスペクト比極限, ε = a/R0 << 1）:
 *   - 大半径 R0（磁気軸の位置）
 *   - 小半径 a（プラズマ断面半径）
 *   - 座標系: (x,y) が小断面, z がトロイダル方向（ここでは直交座標に埋め込む）
 *
 * 磁場配位:
 *   トロイダル磁場: B_tor = B0 * R0 / (R0 + x)   ≈ B0 (1 - x/R0)  大アスペクト比
 *   ポロイダル磁場: B_pol = B0 * r / (q(r) * R0)  r = sqrt(x²+y²)
 *
 *   安全係数 q(r): フラックス面ごとのトロイダル/ポロイダル巻き数比
 *     q(r) = q0 + (q1 - q0) * (r/a)²    q0 ≈ 1.0（磁気軸）, q1 ≈ 2.0（境界）
 *
 *   q < 1 の領域で kink 不安定 (m=1 mode) が発生しやすい。
 *   q ≈ 1 付近で内部 kink (sawtooth)、q = 2 付近で tearing mode が重要。
 *
 * ベクトルポテンシャルからの div B = 0 保証:
 *   ポロイダル磁場は A_z = A_pol(r) から curl を取る。
 *   トロイダル磁場は A_theta = A_tor(r, x) から curl を取る。
 *
 * ── 自然な物理を発生させる初期摂動の入れ方 ──────────────────────────────
 *
 *   1. 速度摂動（MHD 不安定性のトリガー）
 *      - MHD 平衡状態に δv をランダムノイズまたはモード関数として加える。
 *      - 典型的な振幅: δv / v_A ~ 1e-3 〜 1e-2  (v_A = B/sqrt(ρ) はアルヴェン速度)
 *      - Fourier モード: δv ∝ sin(m*theta) * cos(n*phi) でモード番号 (m,n) を選ぶ。
 *      - kink 不安定を seed するには (m=1, n=1) モードを入れる。
 *
 *   2. 密度摂動（粒子クラスタリングの観察に有効）
 *      - δρ/ρ0 ~ 1e-4 程度のランダム摂動で十分。
 *      - 大きすぎると平衡が崩壊し非線形相互作用が即座に起きる。
 *
 *   3. 磁場摂動（resistive tearing mode のシード）
 *      - ベクトルポテンシャルに δA = ε * cos(m*theta) を加える。
 *      - 共鳴面 q(r_s) = m/n での tearing が自然に発生する。
 *
 *   4. 圧力摂動（交換型不安定のトリガー）
 *      - 圧力勾配 ∂p/∂r を加えると、バルーニング不安定のシードになる。
 *
 * ── 平衡条件（Grad-Shafranov 近似）──────────────────────────────────────
 *
 *   ∇p = J × B   (圧力勾配 = 電流 × 磁場)
 *   dp/dr = -J_z * B_tor + J_tor * B_pol
 *   簡易近似: p(r) = p0 * (1 - (r/a)²)  （放物型圧力プロファイル）
 */

#include "cuda_mhd_ct.cuh"
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// トーラス初期条件パラメータ
// ─────────────────────────────────────────────────────────────────────────────

struct TorusParams {
    // 幾何学
    double R0;          ///< 大半径 [格子単位]
    double a;           ///< 小半径 [格子単位]

    // 磁場
    double B0;          ///< 基準トロイダル磁場強度

    // 安全係数プロファイル q(r) = q0 + (q1-q0)*(r/a)²
    double q0;          ///< 磁気軸での q（通常 1.0〜1.1）
    double q1;          ///< 境界での q（通常 2.0〜3.5）

    // 熱力学
    double rho0;        ///< 基準密度
    double p0;          ///< 磁気軸での圧力
    double beta0;       ///< プラズマ β = 2μ0p/B² (規格化圧力)

    // 摂動
    double perturb_amp; ///< 速度摂動振幅 (v_A 単位)
    int    perturb_m;   ///< ポロイダルモード番号
    int    perturb_n;   ///< トロイダルモード番号

    // デフォルト値
    static TorusParams make_default(double lx, double ly, double lz)
    {
        TorusParams tp;
        tp.R0  = 0.30 * fmin(lx, ly);   // 大半径 = 領域幅の 30%
        tp.a   = 0.12 * fmin(lx, ly);   // 小半径 = 領域幅の 12%
        tp.B0  = 1.0;
        tp.q0  = 1.0;
        tp.q1  = 2.0;
        tp.rho0 = 1.0;
        tp.p0   = 0.05;   // beta0 ~ 0.1
        tp.beta0 = 0.1;
        tp.perturb_amp = 1.0e-3;
        tp.perturb_m   = 1;
        tp.perturb_n   = 1;
        return tp;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// デバイス補助関数
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 安全係数プロファイル q(r)
 *   q(r) = q0 + (q1 - q0) * (r/a)^2
 */
__device__ __forceinline__ double safety_factor(double r, double a,
                                                 double q0, double q1)
{
    double xi = r / a;
    return q0 + (q1 - q0) * xi * xi;
}

/**
 * 放物型圧力プロファイル p(r)
 *   p(r) = p0 * max(1 - (r/a)^2, 0)
 */
__device__ __forceinline__ double pressure_profile(double r, double a, double p0)
{
    double xi = r / a;
    return p0 * fmax(1.0 - xi * xi, 0.0);
}

/**
 * 密度プロファイル（圧力に比例した等温近似）
 *   rho(r) = rho0 * (p(r)/p0)^(1/gamma)  又は単純に
 *   rho(r) = rho0 * max(1 - 0.5*(r/a)^2, rho_min)
 */
__device__ __forceinline__ double density_profile(double r, double a, double rho0)
{
    double xi = r / a;
    return rho0 * fmax(1.0 - 0.5 * xi * xi, 0.1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 初期条件設定カーネル: セル中心量
// ─────────────────────────────────────────────────────────────────────────────

/**
 * トーラス初期条件カーネル（セル中心量: rho, mx, my, mz, E）
 *
 * 座標系:
 *   - (x, y, z) は箱状領域の直交座標
 *   - 磁気軸は y=0, z=0, x= R0 付近（トロイダル方向は z 軸周り）
 *   - ポロイダル断面: (x-R0, y) 平面
 *
 * 速度の初期値はゼロ（後から摂動を加える）。
 */
__global__ void kernel_init_cell_center(
    double * __restrict__ rho,
    double * __restrict__ mx,
    double * __restrict__ my,
    double * __restrict__ mz,
    double * __restrict__ E,
    GridParams gp,
    TorusParams tp,
    double ox, double oy, double oz)   // 計算領域の原点オフセット
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + gp.ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + gp.ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + gp.ng;

    if (i >= gp.nx + gp.ng || j >= gp.ny + gp.ng || k >= gp.nz + gp.ng) return;

    // セル中心座標
    double cx = ox + (i - gp.ng + 0.5) * gp.dx;
    double cy = oy + (j - gp.ng + 0.5) * gp.dy;
    double cz = oz + (k - gp.ng + 0.5) * gp.dz;

    // トーラス小半径 r（大半径 R0 から測った距離）
    double rxy = sqrt(cx * cx + cy * cy) + 1.0e-30;  // x-y 面での半径
    // ポロイダル断面における (rxy - R0, cz) を小半径とする
    double dx_pol = rxy - tp.R0;
    double r_pol  = sqrt(dx_pol * dx_pol + cz * cz);  // ポロイダル半径

    // ポロイダル角
    double theta_pol = atan2(cz, dx_pol);

    // トロイダル角
    double phi_tor = atan2(cy, cx);

    // 密度と圧力プロファイル
    double rho_v  = density_profile(r_pol, tp.a, tp.rho0);
    double p_v    = pressure_profile(r_pol, tp.a, tp.p0);

    // プラズマ外部は真空に近い値をセット
    if (r_pol > tp.a) {
        rho_v = 0.01 * tp.rho0;
        p_v   = 0.001 * tp.p0;
    }

    // 速度初期値（摂動入りの MHD 不安定種）
    // v_A = B0 / sqrt(rho0)  アルヴェン速度
    double v_A  = tp.B0 / sqrt(tp.rho0 + 1.0e-30);
    double amp  = tp.perturb_amp * v_A;

    // 速度摂動: δv = amp * sin(m*theta) * cos(n*phi) でポロイダル・トロイダルモードを励起
    // これにより MHD 不安定（kink 等）の初期シードになる
    double perturb_factor = sin((double)tp.perturb_m * theta_pol)
                          * cos((double)tp.perturb_n * phi_tor);

    // ポロイダル方向の速度（トーラス断面内でラジアル外向き）
    double vr_pol = amp * perturb_factor;

    // デカルト速度に変換（ポロイダル→デカルト）
    // ポロイダル座標での単位ベクトル e_r_pol = (dx_pol/r_pol, 0, cz/r_pol)
    // ただし x-y 方向成分は phi_tor 方向に合わせて回転が必要
    double cos_phi = cx / (rxy + 1.0e-30);
    double sin_phi = cy / (rxy + 1.0e-30);
    double cos_theta = dx_pol / (r_pol + 1.0e-30);
    double sin_theta = cz    / (r_pol + 1.0e-30);

    // e_r_pol を直交座標へ: (cos_phi * cos_theta, sin_phi * cos_theta, sin_theta)
    double vx = rho_v > 0.0 ? vr_pol * cos_phi * cos_theta : 0.0;
    double vy = rho_v > 0.0 ? vr_pol * sin_phi * cos_theta : 0.0;
    double vz_v = rho_v > 0.0 ? vr_pol * sin_theta          : 0.0;

    // 磁気エネルギー（面中心 B の平均値で計算; 簡易近似）
    // より正確には face-centered B の初期化後に E を計算し直す
    double q_r = safety_factor(fmin(r_pol, tp.a), tp.a, tp.q0, tp.q1);
    // トロイダル磁場強度（大アスペクト比補正: B_tor = B0 * R0 / (R0 + dx_pol)）
    double B_tor = tp.B0 * tp.R0 / (tp.R0 + dx_pol + 1.0e-30);
    // ポロイダル磁場強度: B_pol = B_tor * r_pol / (q * R0)
    double B_pol = (r_pol < tp.a) ? B_tor * r_pol / (q_r * tp.R0 + 1.0e-30) : 0.0;
    double bsq   = B_tor * B_tor + B_pol * B_pol;

    // 全エネルギー
    double ke   = 0.5 * rho_v * (vx * vx + vy * vy + vz_v * vz_v);
    double E_v  = p_v / (GAMMA_IDEAL - 1.0) + ke + 0.5 * bsq;

    int idx = i * (gp.ny_tot * gp.nz_tot) + j * gp.nz_tot + k;
    rho[idx] = rho_v;
    mx[idx]  = rho_v * vx;
    my[idx]  = rho_v * vy;
    mz[idx]  = rho_v * vz_v;
    E[idx]   = E_v;
}

// ─────────────────────────────────────────────────────────────────────────────
// 初期条件設定カーネル: 面中心磁場（CT 用）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 面中心 Bx 初期化カーネル
 *
 * Bx[i+1/2, j, k] は x=const 面でのトーラス磁場の x 成分を評価する。
 *
 * トーラス磁場の直交デカルト成分（ポロイダル + トロイダル合成）:
 *   B_tor 方向: phi_tor 方向（-sin(phi), cos(phi), 0）
 *   B_pol 方向: ポロイダル断面内でのラジアル方向に垂直なベクトル
 *              = (cos(phi)*(-sin(theta)), sin(phi)*(-sin(theta)), cos(theta))
 *
 * div B = 0 はベクトルポテンシャル A から B = curl(A) として初期化することで
 * 自動的に保証される。ここでは簡略化のため解析的に近似した値を直接使用し、
 * 数値的な divB 誤差は最初のタイムステップ前に診断で確認する。
 *
 * より厳密な初期化では A_phi(r) からの curl を面積分で計算する
 * （Athena++ の field/field.cpp の bcc_to_face 参照）。
 */
__global__ void kernel_init_bx_face(
    double * __restrict__ Bx_face,
    GridParams gp,
    TorusParams tp,
    double ox, double oy, double oz)
{
    // Bx[i, j, k]: i in [0, NX], j in [0, NY-1], k in [0, NZ-1]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + gp.ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + gp.ng;

    if (i > gp.nx || j >= gp.ny + gp.ng || k >= gp.nz + gp.ng) return;

    // x+1/2 面の座標
    double fx = ox + i * gp.dx;   // 面位置 (ghost 外でも i=0 が ng-th face になる)
    double fy = oy + (j - gp.ng + 0.5) * gp.dy;
    double fz = oz + (k - gp.ng + 0.5) * gp.dz;

    double rxy = sqrt(fx * fx + fy * fy) + 1.0e-30;
    double dx_pol = rxy - tp.R0;
    double r_pol  = sqrt(dx_pol * dx_pol + fz * fz);
    double theta_pol = atan2(fz, dx_pol);
    double phi_tor   = atan2(fy, fx);

    double cos_phi   = fx / rxy;
    double sin_phi   = fy / rxy;
    double cos_theta = (r_pol > 0.0) ? dx_pol / r_pol : 1.0;
    double sin_theta = (r_pol > 0.0) ? fz     / r_pol : 0.0;

    double B_tor = (r_pol < 1.5 * tp.a)
                 ? tp.B0 * tp.R0 / (tp.R0 + dx_pol + 1.0e-30)
                 : 0.0;
    double q_r   = safety_factor(fmin(r_pol, tp.a), tp.a, tp.q0, tp.q1);
    double B_pol = (r_pol < tp.a)
                 ? B_tor * r_pol / (q_r * tp.R0 + 1.0e-30)
                 : 0.0;

    // トロイダル方向単位ベクトル e_phi = (-sin_phi, cos_phi, 0)
    // ポロイダル方向単位ベクトル（磁力線接線）e_pol = e_theta × e_r
    //   e_theta = (-sin_theta*cos_phi, -sin_theta*sin_phi, cos_theta)  ポロイダル接線
    double Bx_tor = B_tor * (-sin_phi);
    double Bx_pol = B_pol * (-sin_theta * cos_phi);

    Bx_face[i * (gp.ny * gp.nz) + (j - gp.ng) * gp.nz + (k - gp.ng)] = Bx_tor + Bx_pol;
}

/**
 * 面中心 By 初期化カーネル
 */
__global__ void kernel_init_by_face(
    double * __restrict__ By_face,
    GridParams gp,
    TorusParams tp,
    double ox, double oy, double oz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + gp.ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z + gp.ng;

    if (i >= gp.nx + gp.ng || j > gp.ny || k >= gp.nz + gp.ng) return;

    double fx = ox + (i - gp.ng + 0.5) * gp.dx;
    double fy = oy + j * gp.dy;
    double fz = oz + (k - gp.ng + 0.5) * gp.dz;

    double rxy = sqrt(fx * fx + fy * fy) + 1.0e-30;
    double dx_pol = rxy - tp.R0;
    double r_pol  = sqrt(dx_pol * dx_pol + fz * fz);

    double cos_phi   = fx / rxy;
    double sin_phi   = fy / rxy;
    double sin_theta = (r_pol > 0.0) ? fz / r_pol : 0.0;

    double B_tor = (r_pol < 1.5 * tp.a)
                 ? tp.B0 * tp.R0 / (tp.R0 + dx_pol + 1.0e-30)
                 : 0.0;
    double q_r   = safety_factor(fmin(r_pol, tp.a), tp.a, tp.q0, tp.q1);
    double B_pol = (r_pol < tp.a)
                 ? B_tor * r_pol / (q_r * tp.R0 + 1.0e-30)
                 : 0.0;

    // e_phi の y 成分: cos_phi
    // e_theta の y 成分: -sin_theta * sin_phi
    double By_tor = B_tor * cos_phi;
    double By_pol = B_pol * (-sin_theta * sin_phi);

    By_face[(i - gp.ng) * ((gp.ny + 1) * gp.nz) + j * gp.nz + (k - gp.ng)] = By_tor + By_pol;
}

/**
 * 面中心 Bz 初期化カーネル
 */
__global__ void kernel_init_bz_face(
    double * __restrict__ Bz_face,
    GridParams gp,
    TorusParams tp,
    double ox, double oy, double oz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + gp.ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + gp.ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= gp.nx + gp.ng || j >= gp.ny + gp.ng || k > gp.nz) return;

    double fx = ox + (i - gp.ng + 0.5) * gp.dx;
    double fy = oy + (j - gp.ng + 0.5) * gp.dy;
    double fz = oz + k * gp.dz;

    double rxy = sqrt(fx * fx + fy * fy) + 1.0e-30;
    double dx_pol = rxy - tp.R0;
    double r_pol  = sqrt(dx_pol * dx_pol + fz * fz);

    double cos_theta = (r_pol > 0.0) ? dx_pol / r_pol : 1.0;

    double B_tor = (r_pol < 1.5 * tp.a)
                 ? tp.B0 * tp.R0 / (tp.R0 + dx_pol + 1.0e-30)
                 : 0.0;
    double q_r   = safety_factor(fmin(r_pol, tp.a), tp.a, tp.q0, tp.q1);
    double B_pol = (r_pol < tp.a)
                 ? B_tor * r_pol / (q_r * tp.R0 + 1.0e-30)
                 : 0.0;

    // e_phi の z 成分: 0（トロイダル磁場はポロイダル断面内のみ）
    // e_theta の z 成分: cos_theta  (ポロイダル磁場の z 成分)
    double Bz_pol = B_pol * cos_theta;

    Bz_face[(i - gp.ng) * (gp.ny * (gp.nz + 1)) + (j - gp.ng) * (gp.nz + 1) + k] = Bz_pol;
}

// ─────────────────────────────────────────────────────────────────────────────
// 公開インターフェース
// ─────────────────────────────────────────────────────────────────────────────

/**
 * ソルバの全フィールドをトーラス初期条件で初期化する。
 *
 * @param s       初期化対象のソルバ（GPU メモリ確保済み）
 * @param lx,ly,lz 計算領域サイズ
 * @param tp      トーラスパラメータ
 */
void init_torus(MHDSolver *s,
                double lx, double ly, double lz,
                const TorusParams &tp)
{
    GridParams &gp = s->params;

    // 領域の中心を原点に
    double ox = -0.5 * lx;
    double oy = -0.5 * ly;
    double oz = -0.5 * lz;

    dim3 threads3d(8, 8, 8);

    // セル中心量
    dim3 blocks_cell((gp.nx + 7) / 8, (gp.ny + 7) / 8, (gp.nz + 7) / 8);
    kernel_init_cell_center<<<blocks_cell, threads3d>>>(
        s->d_rho, s->d_mx, s->d_my, s->d_mz, s->d_E,
        gp, tp, ox, oy, oz);

    // 面中心 Bx
    dim3 blocks_bx((gp.nx + 1 + 7) / 8, (gp.ny + 7) / 8, (gp.nz + 7) / 8);
    kernel_init_bx_face<<<blocks_bx, threads3d>>>(
        s->d_Bx_face, gp, tp, ox, oy, oz);

    // 面中心 By
    dim3 blocks_by((gp.nx + 7) / 8, (gp.ny + 1 + 7) / 8, (gp.nz + 7) / 8);
    kernel_init_by_face<<<blocks_by, threads3d>>>(
        s->d_By_face, gp, tp, ox, oy, oz);

    // 面中心 Bz
    dim3 blocks_bz((gp.nx + 7) / 8, (gp.ny + 7) / 8, (gp.nz + 1 + 7) / 8);
    kernel_init_bz_face<<<blocks_bz, threads3d>>>(
        s->d_Bz_face, gp, tp, ox, oy, oz);

    cudaDeviceSynchronize();

    printf("[init_torus] R0=%.3f  a=%.3f  B0=%.3f  q=[%.2f,%.2f]  p0=%.4f\n",
           tp.R0, tp.a, tp.B0, tp.q0, tp.q1, tp.p0);
    printf("[init_torus] perturb: amp=%.2e  (m=%d, n=%d)\n",
           tp.perturb_amp, tp.perturb_m, tp.perturb_n);
}

/**
 * デフォルトのトーラスパラメータでソルバを初期化する便利関数。
 */
void init_torus_default(MHDSolver *s, double lx, double ly, double lz)
{
    TorusParams tp = TorusParams::make_default(lx, ly, lz);
    init_torus(s, lx, ly, lz, tp);
}
