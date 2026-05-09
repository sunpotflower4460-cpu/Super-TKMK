# 精度に関する技術ノート（ACCURACY_NOTES）

このドキュメントは、実装における精度上の注意点・設計判断の根拠を記録する。

---

## 1. Constrained Transport (CT) 法について

### なぜ CT が必要か

通常の有限体積法では、磁場の発散（div B）は時間積分のたびに数値誤差が蓄積し、div B ≠ 0 となる。  
この誤差は「非物理的な力」（∝ div B）を生成し、シミュレーションが物理的に不正確になる。

CT 法は磁場をセル面の法線成分（フラックス）として定義し、Stokes の定理を利用してゲージ不変的に進化させる。  
これにより、div B は機械精度で厳密に 0 に保たれる。

### 実装上のキーポイント

- **Yee 格子（スタガード格子）**: 磁場 `B` をセル面で、電場 `E` をセル辺で定義する。
- **フラックス計算の整合性**: 隣接セル間で共有されるフラックスを一意に定める。
- **電場の補間**: Riemann ソルバ出力をセル辺上に補間する際に Gardiner & Stone (2005) の方式を推奨。

### 参考実装

- Athena++ の `field.cpp` / `ct.cpp`
- [Gardiner & Stone (2005)](https://doi.org/10.1016/j.jcp.2004.11.016): An unsplit Godunov method for ideal MHD via CT

---

## 2. 数値拡散と高次精度スキーム

### Riemann ソルバの選択

| ソルバ | 精度 | 安定性 | 推奨場面 |
|---|---|---|---|
| Lax-Friedrichs | 低 | 高 | デバッグ用のみ |
| HLLE | 中 | 中 | 一般用途 |
| HLLD | 高 | 中 | **MHD に最推奨**（Miyoshi & Kusano 2005） |
| Roe | 高 | 低 | 超音速流（注意が必要） |

### 空間再構成スキーム

- 2次精度: PLM (Piecewise Linear Method) + minmod limiter
- 3次精度: PPM (Piecewise Parabolic Method)
- **推奨**: PLM + van Leer limiter（安定性と精度のバランス良好）

### 時間積分スキーム

- **推奨**: Runge-Kutta 3次（TVD-RK3）
- CFL 数: `C ≦ 0.4`（安全側）

---

## 3. トーラス座標系の精度

トーラス形状では曲線座標（ポロイダル・トロイダル）を扱うため、以下の点に注意する：

- 座標変換時の Jacobian 計算精度
- トーラス中心付近での座標特異点の処理
- 境界条件の周期性（ポロイダル・トロイダル両方向）

---

## 4. PINO/FNO による補完の精度保証

Neural Operator を補完層として使用する場合：

- **Physics-Informed Loss**: div B = 0、エネルギー保存を Loss に含める
- **分布外汎化**: 訓練データの物理パラメータ範囲を記録し、範囲外では使用しない
- **定期的な MHD ソルバとの同期**: Neural Operator のドリフトを防ぐため、数十ステップごとに MHD ソルバ出力で補正する

---

## 5. float32 vs float64

| 用途 | 推奨精度 | 理由 |
|---|---|---|
| MHD ソルバ（保存則計算） | float64 | div B の累積誤差を最小化 |
| Visualization（描画） | float32 | GPU メモリ効率・描画速度 |
| PINO/FNO モデル | float32 | 訓練効率（Loss は適切にスケーリング） |
| ベンチマーク比較 | float64 | 解析解との正確な比較 |
