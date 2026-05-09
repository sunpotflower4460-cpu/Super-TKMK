# アーキテクチャ設計書

プロジェクト設計図（2026年5月版）

プロジェクト名案: TorusMHD-Exact（または Accurate-Toroidal-MHD-Hybrid）

---

## 1. 全体アーキテクチャ（Hybrid Multi-Rate System）

```
[高精度 MHD Solver (低頻度)] ←→ [状態共有バッファ (VTK / HDF5 / CUDA Array)]
                          ↓
               Physics-Informed Corrector (PINO / FNO)
                          ↓
               高精度 Advection + Constrained Transport Layer (高頻度)
                          ↓
               In-situ Visualization Layer (Compute Shader + Field Line + Particle)
                          ↓
                    リアルタイム表示 (ブラウザ / Unity / ParaView)
```

- **Layer 1（正確性のコア）**: 高精度MHDソルバ（Constrained Transport必須）。数十秒〜数分に1回更新。
- **Layer 2（補完層）**: Physics-Informed Neural Operator（PINO/FNO）で物理法則を尊重した補間・進化。
- **Layer 3（可視化層）**: 高精度 Semi-Lagrangian / RK4 advection + Divergence-free field line tracing。保存則を厳密に守る。
- **GPU共有**: 可能なら同じGPU上で計算と可視化を共存（CUDA Stream + Unified Memory）。

---

## 2. 技術スタック（正確性最優先順）

| コンポーネント | 技術選定 | 理由 |
|---|---|---|
| MHD ソルバ | OpenMHD (CUDA Fortran) ベース拡張 / Athena++ CT モジュール参考 | CT 実装済み高精度コード |
| 保存則保証 | Constrained Transport (CT) + Divergence Cleaning | div B = 0 を機械的に保証 |
| 補完・加速 | Physics-Informed Neural Operator (FNO/PINO) | 物理整合性を Loss として課す |
| 可視化 | CUDA Compute Shader + ParaView Catalyst (In-situ) | Field Line / Lagrangian Particle / Volume Rendering |
| 管理層 | Python (PyTorch / TensorRT / HDF5 / VTK) + CUDA C++ | データ連携・制御 |
| 検証 | Orszag-Tang vortex / MHD rotor / Resistive tearing mode | 全コミット検証 |

---

## 3. ディレクトリ構造

```
Super-TKMK/
├── src/
│   ├── mhd_solver/          # OpenMHD拡張 or 新規CT実装
│   │   ├── ct_solver.cu     # Constrained Transport コア
│   │   ├── ideal_mhd.cu     # 理想MHD 方程式
│   │   └── riemann.cu       # Riemann ソルバ（HLLD 等）
│   ├── advection_ct/        # 高精度 Advection + CT
│   │   ├── semi_lagrange.cu # Semi-Lagrangian 移流
│   │   └── rk4_advect.cu    # RK4 移流
│   ├── pin_operator/        # PINO / FNO 補完モデル
│   │   ├── fno_model.py     # Fourier Neural Operator
│   │   └── pino_loss.py     # Physics-Informed Loss
│   └── visualization/       # Compute Shader + In-situ 可視化
│       ├── fieldline.cu     # 磁力線トレース
│       ├── particle.cu      # Lagrangian Particle
│       └── volume.cu        # Volume Rendering
├── validation/              # ベンチマーク・検証（最重要）
│   ├── benchmarks/
│   │   ├── orszag_tang.py   # Orszag-Tang vortex
│   │   ├── mhd_rotor.py     # MHD Rotor
│   │   └── tearing_mode.py  # Resistive Tearing Mode
│   └── verification/
│       └── divergence_test.py  # div B = 0 検証
├── docs/
│   ├── ARCHITECTURE.md      # 本ファイル
│   ├── VERIFICATION.md      # 保存則・精度検証記録
│   └── ACCURACY_NOTES.md    # 精度に関する技術ノート
├── data/                    # 中間状態（HDF5）
├── environment.yml
├── CMakeLists.txt
└── README.md
```

---

## 4. 正確性を担保するための鉄則（全フェーズ共通）

- `div(B) = 0` を機械的に保証（CT法必須）
- エネルギー、運動量、磁束の保存性を常にモニタリング
- 既知の解析解・ベンチマークとの比較を全コミットで実施
- Neural Operator を使う場合も、Physics-Informed Loss を強く効かせる
- 「見栄え」より「物理的一貫性」を優先（ログを詳細に残す）

---

## 5. フェーズ分け

| Phase | 内容 | 成果物 |
|---|---|---|
| Phase 0 | リポジトリ作成・検証環境構築 | 本構造 + divergence_test.py |
| Phase 1 | 理想MHD + CT実装（低解像度トーラス） | ct_solver.cu + Orszag-Tang 通過 |
| Phase 2 | Hybrid層（低頻度MHD + 高頻度Advection + PINO補完） | fno_model.py + 統合テスト |
| Phase 3 | 拡張MHD（resistive → two-fluid）+ 高精度可視化 | fieldline.cu + particle.cu |
| Phase 4 | リアルタイム表示アプリ化 + GPU 電力メトリクス連携 | WebGPU / Unity 連携 |

---

## 6. 参考文献

- Orszag, S.A. & Tang, C.-M. (1979). Small-scale structure of two-dimensional magnetohydrodynamic turbulence. *Journal of Fluid Mechanics*, 90, 129-143.
- Balsara, D.S. & Spicer, D.S. (1999). A staggered mesh algorithm using high order Godunov fluxes to ensure solenoidal magnetic fields in magnetohydrodynamic simulations. *Journal of Computational Physics*, 149, 270-292.
- Miniati, F. & Martin, D.F. (2011). Constrained-transport magnetohydrodynamics with adaptive mesh refinement and its application to the small-scale dynamo. *The Astrophysical Journal Supplement Series*, 195, 5.
- Li Zongyi et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. *ICLR 2021*.
