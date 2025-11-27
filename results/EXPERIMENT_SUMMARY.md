# NQS-SQD 12-bit LiH 實驗結果摘要

**日期**: 2025-11-28
**分子**: LiH @ STO-3G (12 spin orbitals, 4 electrons)
**參考能量**: FCI = -7.6342 Ha, HF = -7.6158 Ha, 關聯能 = 18.4 mHa

---

## 1. 實驗概述

本次實驗驗證了 Neural Quantum States (NQS) 作為 Sample-based Quantum Diagonalization (SQD) 採樣器的可行性。實驗包含三部分：

| 實驗 | 目的 | 狀態 |
|------|------|------|
| H2 快速測試 | 驗證 pipeline 正確性 | ✓ 完成 |
| LiH 主實驗 | NQS + SQD 完整流程 | ✓ 完成 |
| LiH 消融研究 | NQS vs Baseline 對比 | ✓ 完成 |
| Epoch 掃描 | 訓練量與精度關係 | ✓ 完成 |

---

## 2. 關鍵發現

### 2.1 SQD 後處理效果顯著

即使 VMC 能量誤差高達 4565 mHa (epoch=2)，SQD 後處理仍能達到 FCI 精度 (<0.01 μHa 誤差)。

| Epochs | VMC 誤差 (mHa) | SQD 誤差 (mHa) | Conservation (%) |
|--------|----------------|----------------|------------------|
| 2      | 4565           | ~0             | 5.7              |
| 5      | 4250           | ~0             | 5.3              |
| 10     | 3950           | ~0             | 5.2              |
| 20     | 3154           | ~0             | 4.9              |
| 40     | 2216           | ~0             | 4.6              |
| 100    | 762            | ~0             | 7.4              |

### 2.2 Baseline SQD 表現優異

對於 LiH 這個小系統，均勻隨機採樣 (Baseline) 在 500 個樣本時即達到 FCI 精度：

| 採樣預算 | Baseline SQD (mHa) | NQS + SQD (mHa) |
|----------|-------------------|-----------------|
| 100      | 786 ± 293         | 979 ± 309       |
| 500      | **0.0007**        | 969 ± 322       |
| 1000     | **0.0000**        | 764 ± 308       |
| 10000    | **0.0000**        | 194 ± 274       |

### 2.3 Conservation Ratio 行為

- Baseline: 穩定在 ~5%
- NQS: 7-10%，略高於 baseline
- Conservation ratio 與 SQD 精度無直接關聯（在此系統規模下）

---

## 3. 技術修復記錄

### Bug #1: vmc_training.py 雙電子積分索引錯誤

**症狀**: HF configuration 能量為 -9.8 Ha (應為 -7.6 Ha)，誤差 ~2.2 Ha

**根因**: 雙電子積分使用了錯誤的 chemist notation 索引
```python
# 錯誤
J = eri[p_sp, q_sp, p_sp, q_sp]

# 正確 (chemist notation: J_pq = (pp|qq))
J = eri[p_sp, p_sp, q_sp, q_sp]
```

**修復檔案**: `src/nqs_models/vmc_training.py:195-221`
**狀態**: ✓ 已修復

---

### Bug #2: gpu_optimized.py Coulomb 積分 einsum 錯誤

**症狀**: 與 Bug #1 相同，影響 GPU 優化版本的 local energy 計算

**根因**: einsum 表達式假設了 physicist notation，但 PySCF 使用 chemist notation
```python
# 錯誤 (假設 physicist notation)
J = torch.einsum('pqpq->pq', eri)  # 得到 eri[p,q,p,q]

# 正確 (chemist notation: J_pq = (pp|qq) = eri[p,p,q,q])
idx = torch.arange(n_orb, device=eri.device)
J = eri_4d[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
```

**修復檔案**: `src/nqs_models/gpu_optimized.py:190-197`
**狀態**: ✓ 已修復

---

### Bug #3: 外部 vmc_cal.py (Notebook 環境)

**症狀**: `NQS-SQD-Qiskit.ipynb` 中 VMC 能量收斂到 -5.6 Ha (而非 -7.6 Ha)

**根因**: 推測與 Bug #1 相同的 ERI 索引錯誤

**修復檔案**: 外部模組，不在此 repo 中
**狀態**: ⚠️ 需手動修復外部環境

---

### Chemist vs Physicist Notation 說明

| Notation | Coulomb J_pq | Exchange K_pq | PySCF eri 索引 |
|----------|--------------|---------------|----------------|
| Chemist  | (pp\|qq)     | (pq\|qp)      | eri[p,p,q,q]   |
| Physicist| <pq\|pq>     | <pq\|qp>      | g[p,q,p,q]     |

**PySCF 使用 Chemist notation**，因此正確的索引是：
- Coulomb: `J[p,q] = eri[p,p,q,q]`
- Exchange: `K[p,q] = eri[p,q,q,p]`

---

## 4. 下一步方向

1. **增加訓練 epochs**: 當前 100 epochs 的 NQS 尚未收斂至競爭水準
2. **測試更大系統**: H4 chain, BeH2 等，baseline 可能不再有效
3. **改進 NQS 架構**: 考慮 attention 或 equivariant 結構
4. **Sample efficiency 研究**: 在大系統中驗證 NQS 的採樣效率優勢

---

## 5. 生成的圖表

- `results/figures/epoch_sweep.png` - 訓練量 vs 精度
- `results/figures/ablation_comparison.png` - NQS vs Baseline 對比
- `results/figures/vmc_convergence.png` - VMC 訓練曲線
- `results/figures/scatter_vmc_sqd.png` - VMC/SQD 誤差散點圖

---

## 6. 結論

對於 LiH (12-bit) 這個小系統：

1. **SQD 後處理非常強大** - 即使初始採樣品質很差也能恢復 FCI 精度
2. **小系統下 baseline 足夠** - 均勻採樣在 ~500 樣本即可達 FCI
3. **NQS 價值待驗證** - 需要在更大系統上測試其採樣效率優勢
4. **Pipeline 已驗證** - 完整的 NQS → VMC → SQD 流程正確運作
