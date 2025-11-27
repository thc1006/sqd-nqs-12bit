# NQS-SQD 全自動開發計畫

> 最後更新: 2025-11-28
> 負責人: Ting-Yi (蔡秀吉)
> 目標: 將 notebook 中驗證的流程整合到 src/ 結構中

---

## 專案現狀總結

### 已驗證有效的代碼 (從 notebook)

| 組件 | 來源 | 狀態 |
|------|------|------|
| PySCF 分子積分 | `get_pyscf_results()` | 已驗證 |
| FFNN NQS 模型 | `models.py` (外部) | 已驗證 |
| Metropolis MCMC | `efficient_parallel_sampler()` | 已驗證 |
| SR 訓練 | `stochastic_reconfiguration_update()` | 已驗證 |
| 守恆條件篩選 | notebook cell | 已驗證 |
| SQD 對角化 | `diagonalize_fermionic_hamiltonian()` | 已驗證 |

**數值基準** (LiH 分子):
- 預訓練後能量: -5.636886 Ha
- SQD 對角化後: **-7.634167 Ha** (達到 FCI 精度)

### src/ 目錄現狀 (stub)

| 檔案 | 狀態 | 待實現 |
|------|------|--------|
| `src/nqs_models/ffn_nqs.py` | stub | 真正的 FFNN + MCMC |
| `src/nqs_models/utils.py` | 部分 | 採樣工具 |
| `src/sqd_interface/hamiltonian.py` | stub | PySCF 整合 |
| `src/sqd_interface/sqd_runner.py` | stub | SQD 真實調用 |
| `src/sqd_interface/sampling_adapters.py` | stub | NQS 採樣器適配 |
| `src/experiments/h2_12bit_small_sample.py` | stub | 主實驗 |
| `src/experiments/ablation_nqs_vs_baseline.py` | stub | 消融實驗 |

---

## 開發計畫

### 階段 1: 核心基礎設施

#### 1-1: hamiltonian.py - 分子積分生成
- [ ] 整合 PySCF 生成 H₂, LiH 分子積分
- [ ] 實現 `build_h2_hamiltonian_12bit()` 函數
- [ ] 返回: hcore, eri, nuclear_repulsion_energy, n_orb, n_elec
- [ ] 支持不同鍵長 (bond_length) 參數

**預期輸出**:
```python
def build_h2_hamiltonian_12bit(cfg: H2Config) -> MolecularData:
    # 返回分子積分和元數據
```

#### 1-2: ffn_nqs.py - NQS 模型 + MCMC 採樣
- [ ] 實現真正的 FFNN 架構 (n_orbitals × α hidden)
- [ ] 實現 Metropolis MCMC 採樣器
- [ ] 支持 ±1 編碼 (spin 表示)
- [ ] GPU 加速採樣

**預期輸出**:
```python
class FFNNNQS(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def log_prob(self, x: torch.Tensor) -> torch.Tensor: ...
    def sample_mcmc(self, n_samples: int, ...) -> torch.Tensor: ...
```

#### 1-3: vmc_training.py (新檔案) - VMC 訓練
- [ ] 實現局域能量計算 `local_energy_batch()`
- [ ] 實現 Stochastic Reconfiguration (SR) 優化
- [ ] Cosine LR schedule
- [ ] Early stopping 機制

**預期輸出**:
```python
def train_nqs_vmc(
    model: FFNNNQS,
    hamiltonian: MolecularData,
    config: VMCConfig,
) -> TrainingResult: ...
```

#### 1-4: sqd_runner.py - SQD 對角化
- [ ] 實現守恆條件篩選 (N_elec, S_z = 0)
- [ ] BitArray 格式轉換
- [ ] 整合 `diagonalize_fermionic_hamiltonian()`
- [ ] 結果解析和能量提取

**預期輸出**:
```python
def run_sqd_on_samples(
    hamiltonian: MolecularData,
    samples: np.ndarray,
    config: SQDConfig,
) -> SQDResult: ...
```

---

### 階段 2: 實驗腳本

#### 2-1: h2_12bit_small_sample.py - 主實驗
- [ ] 完整流程: 訓練 → 採樣 → SQD
- [ ] Config 驅動 (YAML)
- [ ] 結果保存到 results/
- [ ] 支持 checkpoint 恢復

#### 2-2: ablation_nqs_vs_baseline.py - 消融實驗
- [ ] NQS vs Bernoulli 基線對比
- [ ] 樣本數消融: 100, 500, 1000, 5000, 10000
- [ ] 自動生成對比圖表

---

### 階段 3: 優化與擴展

#### 3-1: GPU 優化 (RTX 4090)
- [ ] MCMC 完全 GPU 化
- [ ] 批量局域能量計算
- [ ] 混合精度訓練 (FP16/BF16)

#### 3-2: 擴展實驗
- [ ] H₄, H₆ 氫鏈
- [ ] 更大 bit depth (14-bit)
- [ ] 解離曲線掃描

---

## 進度追蹤

### 當前進度

| 階段 | 任務 | 狀態 | 完成日期 | 備註 |
|------|------|------|----------|------|
| 0 | 專案掃描與計畫制定 | ✅ 完成 | 2025-11-28 | |
| 1-1 | hamiltonian.py | ✅ 完成 | 2025-11-28 | PySCF 整合，LiH FCI=-7.634167 Ha 已驗證 |
| 1-2 | ffn_nqs.py | ✅ 完成 | 2025-11-28 | FFNN NQS + Metropolis MCMC |
| 1-3 | vmc_training.py | ✅ 完成 | 2025-11-28 | SR 訓練 + 局域能量計算 |
| 1-4 | sqd_runner.py | ✅ 完成 | 2025-11-28 | 守恆篩選 + BitArray + SQD |
| 2-1 | h2_12bit_small_sample.py | ✅ 完成 | 2025-11-28 | 完整 NQS+SQD pipeline |
| 2-2 | ablation_nqs_vs_baseline.py | ✅ 完成 | 2025-11-28 | 消融實驗 + 統計分析 |
| 2-3 | h_chain_scaling.py | ✅ 完成 | 2025-11-28 | H2/H4/H6 scaling 實驗 |
| 3-1 | GPU 優化 | ✅ 完成 | 2025-11-28 | TF32, batched MCMC, vectorized energy |

### 完成記錄

```
[2025-11-28] 階段 0: 完成專案全面掃描，識別所有 stub 代碼
             - 讀取 15 個 agents 配置
             - 分析 notebook 中已驗證的流程
             - 制定開發計畫

[2025-11-28] 建立開發計畫文件
             - 創建 DEVELOPMENT_PLAN.md
             - 更新 CLAUDE.md 引用開發計畫
             - 深度閱讀 nqs_sqd_qiskit.py 和 NQS-SQD-Qiskit.ipynb
             - 提取關鍵實現細節:
               * NQS 模型架構和 MCMC 參數
               * SR 訓練參數和學習率調度
               * ±1 → 0/1 編碼轉換
               * 守恆條件篩選邏輯 (N_elec, S_z)
               * 軌道重排序 (交錯 → 塊狀)
               * BitArray 格式轉換
               * SQD 對角化參數

[2025-11-28] 階段 1-1 至 1-4: 完成核心基礎設施
             - hamiltonian.py: PySCF 整合，支持 H2, LiH, H4, H6
               驗證: LiH @ 0.8A: HF=-7.615770, FCI=-7.634167 Ha
             - ffn_nqs.py: FFNN NQS 模型 + Metropolis MCMC 並行採樣
             - vmc_training.py: SR 訓練 + 局域能量計算 + cosine LR
             - sqd_runner.py: 守恆篩選 + 軌道重排 + BitArray + SQD

[2025-11-28] 階段 2-1: 完成 h2_12bit_small_sample.py
             - 完整 NQS + SQD pipeline
             - 支持 CLI + YAML 配置
             - 驗證: H2 baseline SQD 達到 FCI 精度 (-1.137284 Ha)

[2025-11-28] 階段 2-2: 完成 ablation_nqs_vs_baseline.py
             - 系統性 NQS vs Baseline 對比
             - 支持多樣本預算: 100, 500, 1000, 2500, 5000, 10000
             - 多隨機種子統計平均
             - 自動輸出統計摘要表

[2025-11-28] 階段 2-3: 完成 h_chain_scaling.py
             - H-chain scaling 實驗 (H2, H4, H6)
             - 系統大小 vs 樣本效率分析
             - 保護率隨系統大小變化分析

[2025-11-28] 階段 3-1: 完成 GPU 優化
             - gpu_optimized.py: 優化模組
             - enable_tf32(): 啟用 TF32 矩陣運算加速
             - batched_mcmc_sampler(): 批量 MCMC 採樣
             - vectorized_local_energy(): 向量化局域能量計算
             - AMPTrainer: 自動混合精度訓練器
             - benchmark_mcmc(): 性能基準測試
             - 驗證: 向量化實現與原始實現完全一致 (correlation=1.0)
```

---

## 開發約束

### 避免過度生成
- 只實現必要的代碼
- 不添加未要求的功能
- 保持文件數量最小
- 不創建不必要的抽象層

### 避免過早抽象
- 先讓代碼工作，再考慮重構
- 直接從 notebook 移植，保持簡單
- 避免過度的 OOP 封裝
- 重複代碼 > 錯誤的抽象

### 數值驗證
- 每個步驟都要 sanity check
- 對照 notebook 的已知結果
- LiH FCI 能量: -7.634167 Ha (必須復現)

---

## 可用資源

### 硬體
```
GPU: NVIDIA RTX 4090
VRAM: 24564 MiB (24144 MiB 可用)
Docker: v28.1.1
Kind: v0.25.0
```

### Agents (可並行使用)
- `python-pro`: Python 3.12+ 現代化開發
- `ml-engineer`: PyTorch 模型部署與優化
- `data-scientist`: 統計分析與 ML 建模
- `debugger`: 根因分析與錯誤修復
- `code-reviewer`: 代碼質量審查

### Skills
- `nqs-sqd-research`: NQS + SQD 研究專用
- `experiment-report-writer`: 生成實驗報告

---

## 斷線恢復指南

如果對話中斷，請執行以下步驟恢復進度:

1. 讀取本文件: `cat DEVELOPMENT_PLAN.md`
2. 查看「進度追蹤」表格，找到最後完成的階段
3. 繼續下一個「⏳ 待開始」或「🔄 進行中」的任務
4. 完成後更新本文件的進度表格

---

## 關鍵參考檔案 (最高優先級)

### `nqs_sqd_qiskit.py` 和 `NQS-SQD-Qiskit.ipynb`

這兩個檔案包含**已驗證成功**的完整流程，移植代碼時**必須**參考:

| src/ 目標檔案 | 參考來源函數 | 來源位置 |
|--------------|-------------|----------|
| `hamiltonian.py` | `get_pyscf_results()` | 外部 `moleculars.py` |
| `ffn_nqs.py` | `FFNN` class | 外部 `models.py` |
| `ffn_nqs.py` | `efficient_parallel_sampler()` | 外部 `vmc_cal.py` |
| `vmc_training.py` | `stochastic_reconfiguration_update()` | 外部 `vmc_cal.py` |
| `vmc_training.py` | `local_energy_batch()` | 外部 `vmc_cal.py` |
| `sqd_runner.py` | 守恆條件篩選 | `nqs_sqd_qiskit.py:261-302` |
| `sqd_runner.py` | BitArray 轉換 | `nqs_sqd_qiskit.py:320-344` |
| `sqd_runner.py` | `diagonalize_fermionic_hamiltonian()` | `nqs_sqd_qiskit.py:376-433` |

### 外部依賴 (notebook 引用但不在此 repo)

```python
from models import FFNN                    # FFNN NQS 模型定義
from moleculars import get_pyscf_results, MOLECULE_DATA  # PySCF 整合
from vmc_cal import *                      # VMC 訓練和採樣
from vqe_details import *                  # VQE 相關 (不需要)
import cudaq                               # NVIDIA CUDA-Q (不需要，僅用於 VQE)
```

**注意**: 這些外部檔案需要從原始來源獲取或重新實現。`cudaq` 和 `vqe_details` 可以跳過，因為 SQD 不需要 VQE。

---

## 深度分析: 關鍵實現細節

### 1. NQS 模型架構

```python
# 從 notebook 提取的模型初始化
n_orbitals = mol_pyscf.nao_nr() * 2  # AO 數 × 2 = spin orbitals
n_hidden = int(n_orbitals * ffnn_params['alpha'])  # alpha 是隱藏層倍數
nqs_model = FFNN(n_orbitals, n_hidden, ffnn_params['n_layers'], device=device)
```

### 2. MCMC 採樣器參數

```python
# efficient_parallel_sampler 的調用簽名
samples = efficient_parallel_sampler(
    nqs_model,                              # NQS 模型
    vmc_params['n_samples'] // vmc_params['n_chains'],  # 每鏈樣本數
    vmc_params['n_chains'],                 # 並行鏈數
    n_orbitals,                             # 可見單元數
    vmc_params['burn_in_steps'],            # 燒入步數
    vmc_params['step_intervals'],           # 採樣間隔
    device=device
)
```

### 3. SR 訓練參數

```python
# stochastic_reconfiguration_update 的調用簽名
stochastic_reconfiguration_update(
    nqs_model,
    samples,
    qham_of,                    # OpenFermion 格式的哈密頓量
    lr=lr,                      # 學習率 (cosine schedule)
    reg=vmc_params['sr_regularization'],  # SR 正則化
    device=device
)

# 學習率調度
def adjust_lr(initial_lr, epoch, schedule_type, T_max, decay_rate=0.98):
    if schedule_type == "cosine":
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))
```

### 4. 採樣編碼轉換 (±1 → 0/1)

```python
# NQS 輸出 ±1 編碼，需要轉換為 "0"/"1" 字串
mapped_bits = [("0" if s == -1 else "1") for s in config_tuple]
config_str = "".join(mapped_bits)
```

### 5. 守恆條件篩選邏輯

```python
# 電子數守恆
EXPECTED_N_ELEC = n_elec[0] + n_elec[1]  # (2, 2) → 4

# 自旋守恆 S_z = 0
EXPECTED_S_Z_TIMES_2 = 0  # N_up - N_down = 0

# 自旋軌道排列: 交錯排列
# 索引 0, 2, 4, ... = up spin
# 索引 1, 3, 5, ... = down spin
n_up = sum(1 for i in range(0, len(config_str), 2) if config_str[i] == '1')
n_down = sum(1 for i in range(1, len(config_str), 2) if config_str[i] == '1')

# 篩選條件
if config_str.count('1') == EXPECTED_N_ELEC:
    if (n_up - n_down) == EXPECTED_S_Z_TIMES_2:
        conserved_states[config_str] = count
```

### 6. 軌道重排序 (交錯 → 塊狀)

```python
# SQD 需要 up 和 down 分開的格式
# 原始: 1up, 1down, 2up, 2down, 3up, 3down, ...
# 目標: 1up, 2up, 3up, ..., 1down, 2down, 3down, ...

up_part = ''.join(config[i] for i in range(0, len(config), 2))    # 偶數索引
down_part = ''.join(config[i] for i in range(1, len(config), 2))  # 奇數索引
final_key = up_part + down_part
```

### 7. BitArray 格式轉換

```python
from qiskit_addon_sqd.counts import BitArray

# 字串 → 整數列表 (按計數重複)
samples = []
for bitstring, count in final_conserved_dict.items():
    samples.extend([int(bitstring, 2)] * count)

# 計算字節數
num_bits = len(bitstrings[0])
num_bytes = (num_bits + 7) // 8

# 打包為 uint8 陣列
data = b"".join(val.to_bytes(num_bytes, "big") for val in samples)
array = np.frombuffer(data, dtype=np.uint8)

# 創建 BitArray
bit_array = BitArray(array.reshape(-1, num_bytes), num_bits=num_bits)
```

### 8. SQD 對角化參數

```python
from functools import partial
from qiskit_addon_sqd.fermion import (
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

# SQD 選項
energy_tol = 1e-6
occupancies_tol = 1e-6
max_iterations = 5

# 本徵態求解器選項
num_batches = 3
samples_per_batch = 100
symmetrize_spin = True
carryover_threshold = 1e-4
max_cycle = 200

# 自定義求解器
sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

# 調用 SQD
result = diagonalize_fermionic_hamiltonian(
    hcore,                          # 單體積分
    eri,                            # 雙體積分
    bit_array,                      # 採樣的 bitstring
    samples_per_batch=samples_per_batch,
    norb=n_orb,                     # 軌道數
    nelec=n_elec,                   # 電子數 (alpha, beta)
    num_batches=num_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=max_iterations,
    sci_solver=sci_solver,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    callback=callback,              # 可選: 進度回調
    seed=12345,
)
```

### 9. 能量計算 (加上核排斥能)

```python
# SQD 返回的是電子能量，需要加上核排斥能
final_energy = result.energy + nuclear_repulsion_energy
```

### notebook 中的關鍵數值

```python
# LiH 分子參考能量 (必須復現)
HF:      -7.615770 Ha
FCI:     -7.634167 Ha  # ← 目標
CCSD:    -7.634161 Ha
CCSD(T): -7.634167 Ha

# 預訓練後 NQS 能量
Best energy after pre-training: -5.636886 Ha

# SQD 對角化後
Final energy: -7.634167 Ha  # ← 達到 FCI 精度
```

---

## 參考資料

- **Notebook**: `NQS-SQD-Qiskit.ipynb` (已驗證的完整流程)
- **Python 腳本**: `nqs_sqd_qiskit.py` (notebook 導出版)
- **Config 範例**: `configs/h2_12bit_nqs.yaml`
- **qiskit-addon-sqd 文檔**: 見 Qiskit 官方文檔
