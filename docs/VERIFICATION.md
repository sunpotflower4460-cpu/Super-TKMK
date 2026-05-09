# VERIFICATION

この文書は、Super-TKMK の精度検証で最低限守るべき判定基準をまとめた Phase 0 の基準書である。

## 検証項目一覧

| 項目 | 目的 | 評価タイミング | 許容誤差目標 |
|---|---|---|---|
| div B | CT 実装と磁束保存の健全性確認 | 毎タイムステップ、および PR 前 | float64: max\|div B\| ≤ 1e-14, L2(div B) ≤ 1e-14 / float32: ≤ 1e-6 |
| Energy conservation | 保存則の破綻検出 | 毎タイムステップで記録し、終了時に最大偏差を評価 | 理想 MHD: 相対誤差 ≤ 1e-3 |
| Benchmark agreement | 既知解・既知結果への一致確認 | ベンチマーク実行ごと、およびマージ前 | 主要観測量の相対誤差 ≤ 5%、波形・構造の定性的破綻なし |

## 1. div B 検証

- 対象スクリプト: [`validation/verification/divergence_test.py`](../validation/verification/divergence_test.py)
- 6次精度中心差分で `div B` を評価する。
- float64 実行時は丸め誤差レベル（~1e-14 以下）を維持することを Phase 0 の目標とする。
- 実シミュレーションでは各タイムステップで `max|div B|` と `L2(div B)` を保存し、閾値超過時は失敗扱いとする。

## 2. Energy conservation

- 全エネルギー `E_total = E_kin + E_mag + E_th` を毎タイムステップ記録する。
- 理想 MHD フェーズでは、シミュレーション全区間で
  `|E_total(t) - E_total(0)| / E_total(0) ≤ 1e-3`
  を満たすことを目標とする。
- 抵抗・粘性・外部駆動を導入したフェーズでは、理論的散逸項を含む補正付き評価に切り替える。

## 3. Benchmark agreement

Phase 1 以降で以下の代表ベンチマークを順次有効化する。

| ベンチマーク | 比較対象 | 合否の目安 |
|---|---|---|
| Orszag-Tang vortex | 密度・圧力・磁場構造、衝撃位置 | 参照解に対して主要統計量の相対誤差 ≤ 5% |
| MHD Rotor | 回転体崩壊後の磁場・速度分布 | 参照論文と同等の波面位置、相対誤差 ≤ 5% |
| Resistive Tearing Mode | 成長率、再結合層の厚み | 理論値または参照実装に対して相対誤差 ≤ 5% |

## 実行手順

```bash
python validation/verification/divergence_test.py
python -m pytest validation/ -v
```

## 検証失敗時の確認項目

1. CT 更新式と磁束フラックスの符号が一致しているか
2. スタガード配置と補間位置が設計通りか
3. 境界条件が `div B` 制約と両立しているか
4. 保存則モニタリングの正規化式が正しいか
