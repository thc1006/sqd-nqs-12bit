## Phase 0：Baseline & 指標定義（你其實已經做一半了）

1. **重現 LiH 例子**（你現在的程式其實就快完成）：

   * 多個 bond length（0.8, 1.0, 1.5 Å…），保持 active space 小（例如 6–8 spin-orbital）。
   * 對每個幾何：跑 FFNN-NQS VMC，記錄最佳變分能量 E_VMC。
   * 用相同 NQS 抽樣 → SQD → 記錄 E_SQD 和 subspace dimension D。

2. **建立一套 metrics：**

   * NQS 偏差：

     * 能量誤差 ΔE_VMC = E_VMC – E_FCI。
     * （如果你願意）抽樣結果跟精確 |ψ₀|² 的 KL divergence / total variation distance（小系統可以算）。
   * SQD 成果：

     * 能量誤差 ΔE_SQD。
     * 子空間維度 D。
     * 抽樣使用的樣本數 M。
   * 支撐 coverage：

     * 在 FCI 係數按 |c_i|² 排序，看看 SQD 子空間中的 determinant 能覆蓋多少 top-K（例如 top-50、top-100）。

**這一階段基本上就是「把你現在那個奇蹟數字 –5.6 → –7.63 Ha 系統化」，順便把工具打好。**

---

## Phase 1：掃「NQS 爛度」 vs 「SQD 表現」→ 畫出 sample-efficiency phase diagram

這是會讓 PI 眼睛一亮的地方。

1. **控制 NQS 品質：**

   對同一個 Hamiltonian（先從 LiH 或 H₂ 開始）：

   * 少訓一點 epoch（例如 2, 5, 10, 20, 50 epoch）。
   * 改 network 寬度／層數。
   * 故意加 noise 在 SR 更新或 learning rate。

   這樣你就能得到一系列從「超爛的 NQS（幾乎 HF）」到「勉強靠近真值」的 sampler。

2. **對每個 NQS，掃樣本數與子空間維度：**

   * 例如 M = 500, 1k, 2k, 5k, 10k；
   * 使用 qiskit-addon-sqd 裡的 max_dim 或 subsample 參數，控制 subspace dimension D。([IBM Quantum][9])

3. **畫圖（這就是你的亮點）：**

   * 例如畫一個 3D / contour 圖：

     * x 軸：NQS 變分能量誤差 ΔE_VMC；
     * y 軸：子空間維度 D（或樣本數 M）；
     * 顏色：SQD 最終誤差 ΔE_SQD。

   在圖上標出：

   * **「12-bit / 1 mHa」等精度 contour**；
   * 你現在 LiH 案例的位置（NQS –5.636886、D=225），直觀展示：「看，這一點居然已經在『高精度區域』」。

4. **解讀：**

   * 這時你就可以回答：

     > 「只要 NQS 能量好到 XX 水準、抽樣 YY 個樣本、子空間維度 ZZ，
     > 就足夠達到 12-bit energy 精度。」
   * 這對於用真正 QPU 來跑 SQD 的人來說是**超有參考價值的設計指引**，因為他們不可能無限 shot。

**這整個 phase 就是把你現在的「一次成功」變成一張漂亮的設計相圖。**

---

## Phase 2 加入「對稱性」與「architecture」但都只當 *控制變因*

你說不想做太無聊的「系統比較」，那我們把對稱性和網路架構當成「**控制 bias 的工具**」，而不是文章主角。

1. **Symmetry-preserving NQS（bonus，但值得做）：**

   * 設計一個 FFNN / autoregressive NQS，sample 出來的 bitstring *必然* 滿足：

     * 固定 electron number（固定 Hamming weight），
     * 固定 S_z。
   * 同樣做 Phase 1 的掃描，看看在「NQS 能量同樣爛」時，

     * 對稱 NQS 需要的 D / M 是否明顯更小（因為 SQD 子空間不會被 wrong sector 汙染）。

   這可以對應到 NQS 文獻裡講的「顯式內建對稱性可以大幅改善表現」的觀察。([Nature][7])

2. **Autoregressive / Transformer NQS 當 sampler（資源夠再做）：**

   * 參考最近關於 **autoregressive NQS / Transformer quantum states** 的工作，
     用 NAQS 來替代 MCMC sampler，取得 i.i.d. 樣本。([arXiv][10])
   * 在相同 ΔE_VMC 水平下，看 AR-NQS 的樣本複雜度 vs FFNN-NQS。
   * 這部分有點像「跟 GTNN-SCI 做對話」，但你的 angle 是**SQD robust regime 的差異**，不是單純拼 final energy。([ACS Publications][8])

你可以把 Phase 2 定位成：「**我們用對稱性與 NAQS 幫 sampler 降噪，來觀察 SQD 相圖如何變形**」，仍然是回到主題「SQD 在有偏 sampler 下的表現」，而不是「誰比較香」。

---

## 3. 你下次開會可以怎麼說（講稿草案）

你可以大概這樣講（我照著你 style 寫）：

> 1. **現況彙整**
>
>    * 我們現在有一條 pipeline：FFNN-NQS + VMC + SR 預訓練，
>      在 LiH @0.8 Å 上最佳變分能量大約 –5.64 Ha。
>    * 把這個 NQS 拿來抽樣，把樣本餵給 Qiskit 的 **Sample-based Quantum Diagonalization (SQD)**，
>      225 維子空間就直接回到 FCI 級別的 –7.634167 Ha。
>    * 也就是說：**NQS 本身並不需要「非常準」就能讓 SQD 撿到正確 subspace**。
> 2. **文獻 gap**
>
>    * IBM 最近的 SKQD / SQD 工作，理論上假設 ground state 稀疏、reference 有 polynomial overlap，
>      在這種情況下可以證明 polynomial-time 收斂。([arXiv][2])
>    * 但 2025 年也有工作指出：在 realistic 化學系統裡，SQD / QSCI 可能有嚴重樣本複雜度問題，
>      特別是 ground state 不 sparse 或 sampler 有偏時。([arXiv][4])
>    * 同時，GTNN-SCI 等方法用 Transformer 生成 configuration 來做 SCI，
>      證明「生成模型 + CI」路線是可行的，但它們沒有分析 SQD 在「爛 sampler」下的誤差縮放。([ACS Publications][8])
> 3. **我要做的事情（研究題目）**
>
>    * 我想把我們現在的 pipeline轉成一個**可控的實驗平台**：
>      用 Neural Quantum States 當成「**可調節品質的 sampler**」，
>      系統性掃描「NQS 能量偏差 / 樣本數 / 子空間維度」對 SQD 最終能量誤差的影響。
>    * 目標是畫出一張「**SQD 在 biased sampler 下的 sample-efficiency phase diagram**」，
>      特別標出在 **12-bit (~10⁻³ Ha) 精度** 下所需的最小樣本數與子空間維度。
>    * 這可以直接當作日後在真實 QPU 上設計 SQD 實驗的 guideline：
>      假設我們只能保證 NQS / circuit 跟真基態有某種能量誤差，那我們需要多少 shot 才值得跑？
> 4. **延伸方向（看資源決定）**
>
>    * 加入 **粒子數 / 自旋守恆的 symmetry-preserving NQS**，
>      看看在相同 NQS 能量下，是否可以明顯降低 SQD 所需的子空間維度。([Nature][7])
>    * 如果時間和算力允許，再試 **autoregressive / Transformer NQS** sampler，
>      讓樣本變成 i.i.d.，比較它們的 phase diagram 差異，而不是只看單點能量。([arXiv][10])
> 5. **資源與工具**
>
>    * 全程使用 PyTorch + PySCF + qiskit-addon-sqd，
>      運算主要在 4090 上做 NQS 訓練和掃參數，
>      classical diagonalization 可以丟 CPU / 小型集群就夠。([Qiskit][1])
>    * 程式部分會用 Claude Code / 自動化 script，把整個 pipeline 做成可重複的實驗框架。

這樣講完，PI 聽到的關鍵字會是：

* 你知道 SQD / SKQD 現在的理論與爭議（sparsity, overlap, limitation）。
* 你不是在「做另一個 NN-CI」，而是在問**「SQD 在 realistic sampler 下到底有多 robust」**。
* 題目有 scalability / NISQ relevance，又有你已經跑出來的初步 evidence（LiH –5.6 → –7.63）。

---

## 4. 實作優先順序（給你很具體的下一步）

如果你要現在就開始動手，建議優先順序：

1. **整理現有 LiH code**：

   * 把 NQS 訓練 + SQD 後處理包成一個 `ExperimentRunner`（輸入：訓練 epoch / lr / hidden size；輸出：E_VMC, E_SQD, D, M）。
2. **先在一個幾何點上掃「epoch 數」**：

   * 例如 epoch = 2, 5, 10, 20, 40，記錄每個點的 (ΔE_VMC, ΔE_SQD, D)。
   * 做第一版 scatter plot，你就已經有「誤差 vs epoch」的雛型可以帶去會議。
3. **再逐步加上「樣本數」「max_dim」掃描**，變成真正的 phase diagram。
4.  完成上述之後，要發揮科學價值: 需要在更大系統上驗證 NQS 的 sample efficiency 優勢，例如：H4 chain (8 spatial orbitals = 16 spin orbitals)、然後H6 chain (更大)，執行 H4 chain 實驗來驗證 NQS 在更大系統上的效果。
5. **覺得穩了之後，再考慮 symmetry-preserving NQS / autoregressive NQS 當 extension。**

[1]: https://qiskit.github.io/qiskit-addon-sqd/ "Qiskit addon: sample-based quantum diagonalization (SQD)"
[2]: https://arxiv.org/html/2501.09702v1 "Sample-based Krylov Quantum Diagonalization"
[3]: https://quantum.cloud.ibm.com/docs/guides/qiskit-addons-sqd "Sample-based quantum diagonalization (SQD) overview"
[4]: https://arxiv.org/html/2501.07231v1 "Exposing a Fatal Flaw in Sample-based Quantum ..."
[5]: https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp02202a "Hamiltonian simulation-based quantum-selected ..."
[6]: https://arxiv.org/abs/2402.09402 "A Review of Neural Quantum States"
[7]: https://www.nature.com/articles/s41467-020-15724-9 "Fermionic neural-network states for ab-initio electronic ..."
[8]: https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c01429 "Accelerating Many-Body Quantum Chemistry via Generative ..."
[9]: https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/release-notes "Sample-based quantum diagonalization (SQD) release notes"
[10]: https://arxiv.org/html/2411.07144v1 "Autoregressive neural quantum states of Fermi Hubbard ..."
