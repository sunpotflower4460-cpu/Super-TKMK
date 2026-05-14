# Super-TKMK (Accurate Toroidal MHD Hybrid)

**自然な物理の正確な可視化を最優先とした、GPU上でのトーラス形状MHDシミュレーション＋リアルタイム可視化プロジェクト。**

---

## 正確性原則（Accuracy-First Principles）

このプロジェクトは「見栄えの良さ」ではなく「物理的正確性」を最優先とする。以下の原則は全フェーズを通じて厳守する：

1. **Constrained Transport (CT) 必須**  
   `div B = 0` を機械的に保証するため、Constrained Transport 法を必須とする。Divergence Cleaning 単独は許容しない。

2. **div B = 0 の厳密保証**  
   全タイムステップで磁場の発散を数値的に検証する。許容誤差は機械精度オーダー（float64: ~10⁻¹⁵）を目標とする。

3. **ベンチマーク検証必須**  
   コミットごとに既知ベンチマーク（Orszag-Tang vortex, MHD Rotor, Resistive Tearing Mode 等）との比較を実施する。検証なしのマージは禁止。

4. **保存則の継続モニタリング**  
   エネルギー・運動量・磁束の保存誤差を全シミュレーション中に記録・監視する。

5. **Neural Operator 使用時も Physics-Informed Loss を強く適用**  
   PINO/FNO による補完・加速を用いる場合も、物理整合性を Loss として明示的に課す。

---

## プロジェクト概要

| 項目 | 内容 |
|---|---|
| 目標 | GPU上で自然なMHD物理（磁場＋プラズマ相互作用、保存則、波動・不安定性）を正確に観測・可視化する |
| 対象形状 | トーラス（Tokamak 近似） |
| 計算方式 | Hybrid Multi-Rate: 低頻度 MHD Solver + 高頻度 Advection + PINO 補完 |
| 可視化 | CUDA Compute Shader / Field Line / Lagrangian Particle / Volume Rendering |
| 言語 | CUDA C++, Python (PyTorch), Fortran（OpenMHD 参考） |

## フェーズ

| Phase | 内容 | 状態 |
|---|---|---|
| Phase 0 | 基盤整備・正確性検証環境構築 | ✅ 完了 |
| Phase 1 | 理想MHD + CT実装（低解像度トーラス） | ✅ 完了 |
| Phase 2 | Hybrid層（低頻度MHD + 高頻度Advection + PINO補完） | 未着手 |
| Phase 3 | 拡張MHD（resistive → two-fluid）+ 高精度可視化 | 未着手 |
| Phase 4 | リアルタイム表示アプリ化 + GPU 電力メトリクス連携 | 未着手 |

## ディレクトリ構造

```
Super-TKMK/
├── src/
│   ├── mhd_solver/          # MHD ソルバ（CT 実装含む）
│   ├── advection_ct/        # 高精度 Advection + Constrained Transport
│   ├── pin_operator/        # PINO / FNO 補完モデル
│   └── visualization/       # Compute Shader + In-situ 可視化
├── validation/              # ベンチマーク・検証（最重要）
│   ├── benchmarks/          # Orszag-Tang, MHD Rotor, Tearing Mode 等
│   └── verification/        # div B 検証スクリプト等
├── docs/
│   ├── ARCHITECTURE.md      # 詳細アーキテクチャ設計
│   ├── VERIFICATION.md      # 保存則・精度検証記録
│   └── ACCURACY_NOTES.md    # 精度に関する技術ノート
├── data/                    # 中間状態（HDF5）
├── environment.yml          # Python 環境定義
├── CMakeLists.txt           # CUDA C++ ビルド設定
└── README.md
```

## セットアップ

### Python 環境

```bash
conda env create -f environment.yml
conda activate super-tkmk
```

### CUDA ビルド

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 検証実行

```bash
# div B = 0 検証（最初の正確性ゲート）
python validation/verification/divergence_test.py
```

## 開発ワークフロー

1. GitHub Issue で作業指示を定義する
2. 実装 → ここに持ってくる → 検証・修正指示のサイクルを繰り返す
3. すべての実装は `validation/` 内のテストを通過してからマージ

## 参考文献・参考実装

- [Athena++](https://github.com/PrincetonUniversity/athena) - CT 実装参考
- [OpenMHD](https://github.com/zenitani/OpenMHD) - CUDA Fortran MHD ソルバ
- Orszag & Tang (1979), MHD turbulence benchmark
- Miniati & Martin (2011), Constrained Transport

## ドキュメント

詳細設計は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) を参照。
精度・検証方針は [`docs/VERIFICATION.md`](docs/VERIFICATION.md) を参照。
