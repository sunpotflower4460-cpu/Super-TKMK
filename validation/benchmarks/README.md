# Benchmarks

既知の MHD ベンチマークテストケース。

| ベンチマーク | ファイル | 状態 |
|---|---|---|
| Orszag-Tang Vortex | `orszag_tang.py` | ✅ Phase 1 実装済み |
| トーラス MHD 正確性ゲート | `toroidal_mhd_test.py` | ✅ Phase 1 実装済み |
| MHD Rotor | `mhd_rotor.py` | Phase 2 で実装予定 |
| Resistive Tearing Mode | `tearing_mode.py` | Phase 2 で実装予定 |

詳細は `docs/VERIFICATION.md` を参照。

## トーラス MHD 正確性ゲート (Issue #4)

Phase 1 完了条件の統合テスト。CT + HLLE + SSP-RK2 の検証を一括実行する。

```bash
# 標準実行（32³ 格子、100 ステップ）
python validation/benchmarks/toroidal_mhd_test.py

# カスタム格子
python validation/benchmarks/toroidal_mhd_test.py --nx 64 --ny 64 --nz 64 --steps 200

# 出力先を指定
python validation/benchmarks/toroidal_mhd_test.py --output results/phase1/
```

完了条件:
- `max|div B| ≤ 1e-13 (float64)` — CT による機械精度 div B = 0
- `|ΔE/E₀|_max < 0.5%` — エネルギー保存則の継続監視
- 出力に `正確性ゲート PASS` が表示される

## Orszag-Tang Vortex

古典的な MHD 乱流ベンチマーク（Orszag & Tang 1979）。

```bash
# 標準 2D テスト (64²格子)
python validation/benchmarks/orszag_tang.py

# トーラス座標変形バリアント
python validation/benchmarks/orszag_tang.py --torus

# 高解像度 3D
python validation/benchmarks/orszag_tang.py --nx 64 --ny 64 --nz 64

# divergence_test.py と連携して div B を確認
python validation/benchmarks/orszag_tang.py --check-divb
```
