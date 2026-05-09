# 保存則・精度検証記録（VERIFICATION）

このドキュメントは、シミュレーションの物理的正確性を保証するための検証基準・結果記録を管理する。

---

## 検証方針

### 1. div B = 0 検証（最優先）

磁場の発散は全タイムステップで計算・記録する。

| 精度 | 目標誤差 |
|---|---|
| float32 | < 10⁻⁶ |
| float64 | < 10⁻¹⁴ |

検証スクリプト: [`validation/verification/divergence_test.py`](../validation/verification/divergence_test.py)

### 2. エネルギー保存検証

全エネルギー（運動エネルギー + 磁気エネルギー + 熱エネルギー）の時間変化を記録する。  
理想MHD では以下の保存誤差基準を要求する：

- **相対誤差**: |ΔE_total / E_total(t=0)| < 0.1%（全シミュレーション期間を通じて）
- **評価タイミング**: 全タイムステップで記録し、シミュレーション終了時に最大値を評価する

ここで ΔE_total = E_total(t) - E_total(t=0)、E_total(t=0) はシミュレーション開始時の全エネルギー。

### 3. 運動量保存検証

全運動量の時間変化を記録する。境界条件の影響を除いた実効誤差を評価する。

### 4. 磁束保存検証（トーラス形状の場合）

ポロイダル磁束・トロイダル磁束を別々に追跡する。

---

## ベンチマーク一覧

| テスト名 | 参照論文 | 検証スクリプト | 合否基準 |
|---|---|---|---|
| Orszag-Tang Vortex | Orszag & Tang (1979) | `validation/benchmarks/orszag_tang.py` | 未定義（実装時に追加） |
| MHD Rotor | Balsara & Spicer (1999) | `validation/benchmarks/mhd_rotor.py` | 未定義（実装時に追加） |
| Resistive Tearing Mode | Furth et al. (1963) | `validation/benchmarks/tearing_mode.py` | 未定義（実装時に追加） |

---

## 検証記録ログ

| 日付 | コミット | テスト | 結果 | メモ |
|---|---|---|---|---|
| （実装後に記録） | | | | |

---

## 検証実行手順

```bash
# div B 検証
python validation/verification/divergence_test.py

# 全ベンチマーク（実装後）
python -m pytest validation/ -v
```

---

## 検証失敗時の対応

1. CT ソルバのフラックス計算を確認する
2. グリッドのスタガー配置（Yee 格子）を確認する
3. 境界条件の div B 整合性を確認する
4. 時間積分スキームの安定性（CFL 条件）を確認する
