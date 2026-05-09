# Super-TKMK

「最も物理的に正確な方法」を最優先とした設計にしています。

- 最終目標: 「それっぽいビジュアライゼーション」ではなく、GPU上で自然なMHD物理が進行している様子を、できる限り正確に観測・可視化したい。
- 優先事項: 「特定の現象を再現」ではなく、「自然な物理の振る舞い」（磁場とプラズマの相互作用、保存則、波動・不安定性・流れの自然な発生）。
- 現実的制約: 本格核融合級トカマクMHD（高アスペクト比＋壁境界＋拡張MHD）は1〜2枚GPUではリアルタイム困難。
- あなたのキーアイデア: 数十秒に1回の重いMHD計算でも、その間の「自然な流れ」を視覚化側で物理的に整合性を保ちながら再現・補完する。
- 最終要求: 1番正確な方法を優先。

プロジェクト設計図（2026年5月版）

プロジェクト名案: TorusMHD-Exact（または Accurate-Toroidal-MHD-Hybrid）

1. 全体アーキテクチャ（Hybrid Multi-Rate System）

[高精度 MHD Solver (低頻度)] ←→ [状態共有バッファ (VTK / HDF5 / CUDA Array)]
                          ↓
               Physics-Informed Corrector (PINO / FNO)
                          ↓
               高精度 Advection + Constrained Transport Layer (高頻度)
                          ↓
               In-situ Visualization Layer (Compute Shader + Field Line + Particle)
                          ↓
                    リアルタイム表示 (ブラウザ / Unity / ParaView)

- Layer 1（正確性のコア）: 高精度MHDソルバ（Constrained Transport必須）。数十秒〜数分に1回更新。
- Layer 2（補完層）: Physics-Informed Neural Operator（PINO/FNO）で物理法則を尊重した補間・進化。
- Layer 3（可視化層）: 高精度 Semi-Lagrangian / RK4 advection + Divergence-free field line tracing。保存則を厳密に守る。
- GPU共有: 可能なら同じGPU上で計算と可視化を共存（CUDA Stream + Unified Memory）。

2. 技術スタック（正確性最優先順）

- MHDソルバー: OpenMHD (CUDA Fortran) をベースに拡張。可能ならConstrained Transport (CT) 実装済みの高精度コード（Athena++のMHDモジュールやMiniatiのCTコードを参考）。
- 保存則保証: Constrained Transport (CT) + Divergence Cleaningを必須で実装。
- 補完・高速化: Physics-Informed Neural Operator (FNO/PINO)。2026年時点の最新論文（quasi-static MHD PINNなど）を参考にfine-tuning。
- 可視化: CUDA Compute Shader（Field Line Advection + Lagrangian Particle + Volume Rendering）。ParaView Catalyst（In-situ）も並行。
- 管理: Python (PyTorch/TensorRT + HDF5 + VTK) + CUDA C++/Fortran。
- 検証ツール: 既知のMHDベンチマーク（Orszag-Tang vortex, MHD rotor, resistive tearing modeなど）で常に検証。

3. GitHub中心の開発ワークフロー（あなたが指定した形式）

基本ルール:
1. 私がGitHub Issue（またはPRコメント）で具体的な指示を出す。
2. あなたがその指示通りに実装（または部分実装）。
3. 完成したコード・結果・ログをここ（このチャット）に持ってくる。
4. 私が内容を検証 → ズレている点を指摘 → 修正指示を出す。
5. これを繰り返す（検証重視のイテレーション）。

リポジトリ構造（初期案）:
TorusMHD-Exact/
├── src/
│   ├── mhd_solver/          # OpenMHD拡張 or 新規CT実装
│   ├── pin_operator/        # PINO / FNO補完モデル
│   ├── advection_ct/        # 高精度advection + Constrained Transport
│   └── visualization/       # Compute Shader + In-situ
├── validation/              # ベンチマークテストケース（最重要）
├── data/                    # 中間状態（HDF5）
├── docs/
│   ├── ARCHITECTURE.md
│   ├── VERIFICATION.md      # 保存則・精度検証記録
│   └── ACCURACY_NOTES.md
├── CMakeLists.txt
├── environment.yml
└── README.md

4. 正確性を担保するための鉄則（全フェーズ共通）

- div(B) = 0 を機械的に保証（CT法必須）。
- エネルギー、運動量、磁束の保存性を常にモニタリング。
- 既知の解析解・ベンチマークとの比較を全コミットで実施。
- Neural Operatorを使う場合も、Physics-Informed Lossを強く効かせる。
- 「見栄え」より「物理的一貫性」を優先（ログを詳細に残す）。

5. フェーズ分け

Phase 0: リポジトリ作成・検証環境構築（最も重要）
Phase 1: 理想MHD + Constrained Transportの実装（低解像度トーラス）
Phase 2: Hybrid層（低頻度MHD + 高頻度Advection + PINO補完）
Phase 3: 拡張MHD（resistive → two-fluid）への段階的拡張 + 高精度可視化
Phase 4: リアルタイム表示アプリ化 + GPU電力メトリクスとの結合（元のアイデア回帰）