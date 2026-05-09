# Benchmarks

既知の MHD ベンチマークテストケース。

| ベンチマーク | ファイル | 状態 |
|---|---|---|
| Orszag-Tang Vortex | `orszag_tang.py` | ✅ Phase 1 実装済み |
| MHD Rotor | `mhd_rotor.py` | Phase 2 で実装予定 |
| Resistive Tearing Mode | `tearing_mode.py` | Phase 2 で実装予定 |

詳細は `docs/VERIFICATION.md` を参照。

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
