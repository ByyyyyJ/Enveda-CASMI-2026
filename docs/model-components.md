# 技术方案：架构、部件与难点

> **【已归档，2026-10-27】** 当前执行的方案是 [research-plan.md](research-plan.md)（证据门控融合）。本文完整保留，作为背景资料和后备思路：数据事实（§2）、域偏移分析（§3.1）、规则与选型标准（§6）、部件清单（§7）、难点分析（§9）仍可引用；架构与优先级（§1、§5、§8、§11）不再执行。新方案直接沿用的原始想法：先验方向随类别变化与"缺席证据"（plan-review ⑦）、同分子式异构体排序（§9.1）、类别配比未知（§4.4）。

本文档是项目的主技术文档，汇总整体架构、数据事实、域适应与数据划分、模块设计、开源部件选型、拼装方案、主要难点与对应技术，以及工程规划。

- 部件**只收录开源且符合竞赛规则的**（标准见 §6.1）。
- 标记为 **【待定】** 的部分尚未决策；标记为 **【待填】** 的部分已确定方向、等待实现细节。
- 竞赛信息来自官方页面（Overview / Data / Evaluation / Rules）和官方 metric notebook，通过 Kaggle CLI 获取，核对日期 2026-09-24。
- 调研与许可证核对日期：2026-09-24。

> **2026-10-26 审核**：本文的路线（§5 自建全部模块、§8.1 优先级、§11 里程碑、§4.4 用排行榜反推类别占比、§4.2 V-np-novel 删除 Tanimoto ≥ 0.7 近邻）已被 [plan-review.md](plan-review.md) 修订，以审核文档为准。数据事实（§2）、域偏移分析（§3.1）、规则与选型标准（§6）、部件清单（§7）仍然有效。

相关文档：[competition.md](competition.md)（赛题与提交约束）、[data-dictionary.md](data-dictionary.md)（字段）、[research-landscape.md](research-landscape.md)（方法背景）。

---

## 1. 总体思路

- **任务形式**：每个 `molecule_id`（1–16 张谱图）输出 Top-25 候选 SMILES，按 MRR@25 评测（比较 inchikey14，忽略立体与互变异构）。**预测以分子为单位**，必须聚合同一分子所有谱图的证据。
- **统一流水线，不做显式分类**：测试集不提供新颖度标签，三类的分布在比赛期间保密，因此所有分子走同一条流水线：多个打分器并行打分，融合模块输出唯一排序。三类新颖度只用于**本地验证的分层报告**。
- **隐式路由**：融合模块以"最高谱库相似度"等特征为依据，自动在"信任谱库检索""信任对比检索""信任 de novo 生成"之间切换。

```text
test.parquet
   │
   ▼
[1 数据层] ──► [4 候选生成] ──► [5 打分器 ×N] ──► [7 融合排序] ──► [8 提交]
   │               ▲                  ▲
   │          [2 化学工具]        [6 训练]（离线，产出权重）
   ▼
[3 评估与切分]（贯穿所有模块的度量基准）
```

---

## 2. 数据事实

### 2.1 本地 test.parquet 只是占位样本

官方数据页说明：本地的 `test.parquet` 由训练集中的样本组成，重新运行 Notebook 时会被替换为隐藏测试集。因此**本地 test.parquet 的任何统计（质量范围、峰数、adduct 分布、碰撞能量等）都不代表真实测试集**，不能用来调参或设计验证集。它只能用于检查提交流程能否跑通。

真实测试集的特征以官方描述（§2.2）为准；与测试集最接近的可用数据是 `enveda-np-examples`（§2.4）。

### 2.2 隐藏测试集（官方描述）

| 项目 | 内容 |
| --- | --- |
| 规模 | 约 1,500 张谱、约 400 个分子；每分子 1–16 张谱，中位 3 张 |
| 仪器 | 全部是 Bruker timsTOF |
| 质量范围 | 单同位素质量 157–1,159 Da，中位 348 Da |
| 化学空间 | 与天然样本（植物、哺乳动物、微生物）中分子相似：已确认天然产物、假设的天然产物、天然产物类似物、可能存在于自然界的合成分子 |
| adduct | 10 种：`[M+H]+`、`[M+NH4]+`、`[M-H2O+H]+`、`[M-2H2O+H]+`、`[M+Na]+`、`[M+K]+`、`[M-H]-`、`[M-H2O-H]-`、`[M+CH2O2-H]-`、`[M+Cl]-` |
| 碰撞能量 | `collision_energy_ev` 为列表：单值表示一次采集；多值（如 `[20, 40, 60, 80]`）表示多个能量合并的谱 |
| 已做的清洗 | 删除 precursor 质量与结构、adduct 不符的谱；删除基峰低于 1,000 raw counts 的谱；删除高于 precursor + 2 Da 的峰。除此之外只做了最少的处理 |
| 分子式 | 不提供 |

**三类新颖度（官方定义）**：

| 类别 | 定义 |
| --- | --- |
| 1：有公开谱 | 结构有公开的参考 MS/MS 谱，可以通过与训练谱的相似度找到 |
| 2：结构已知、无公开谱 | 没有公开谱，但结构在 PubChem 或天然产物库 COCONUT 中，可以通过数据库检索找到 |
| 3：新结构 | 结构完全不在 PubChem 中，只能 de novo 预测 |

三类在测试集中的比例以及每个分子属于哪一类，在比赛期间保密。

### 2.3 训练集按来源分布（官方）

| ingest_lib | 谱图数 | 结构数 | 说明 |
| --- | --- | --- | --- |
| enveda-180 | 1,153,785 | 182,941 | 与测试集同一台 timsTOF；但化学空间是**合成类药筛选化合物**，与测试分子不同 |
| pluskal_ms2 | 527,581 | 46,821 | MSnLib，Orbitrap，商业筛选库与生物活性化合物，多能量统一流程 |
| riken | 347,171 | 15,892 | RIKEN 公共库，侧重植物次生代谢物 |
| gnps | 220,849 | 45,750 | GNPS 社区库，最大的天然产物参考谱集合，异质性最强 |
| massbank | 101,727 | 9,180 | MassBank，多实验室多仪器 |
| mona | 92,416 | 11,681 | MassBank of North America |
| spectraverse | 50,933 | 9,631 | 公共库的统一整合，包含一些少见的库 |
| msdial | 40,765 | 9,127 | MS-DIAL 附带的公共库 |
| drug_plus | 2,545 | 2,539 | 药物 LC-MS/MS 库，没有碰撞能量信息 |
| **enveda-np-examples** | 1,184 | 250 | **最接近测试集**：同一仪器、同一处理流程的常见天然产物（§2.4） |
| masaryk | 652 | 416 | RECETOX 化学标准品 |

- 训练集共 2,539,608 张谱、275,810 个结构（inchikey14）；同一结构可能出现在多个库中，完全重复的谱已删除。
- 各库的上游预处理、仪器、碰撞能量单位和 m/z 精度都不同。`instrument_type` 是各来源的自由文本，只有 enveda-180 和 enveda-np-examples 统一标为 `timsTOF`。
- 碰撞能量使用 `collision_energy_ev`（官方已把 NCE 等换算为 eV，与测试集口径一致）。
- `precursor_error_ppm` 未经清洗，可以作为标签质量信号。

### 2.4 enveda-np-examples：唯一与测试集"双重匹配"的数据

| 项目 | 统计 |
| --- | --- |
| 规模 | 1,184 张谱、250 个结构；每分子 1–17 张谱，中位 4 张 |
| adduct | 测试集的 10 种全部出现：`[M+H]+` 569、`[M-H]-` 276、`[M+CH2O2-H]-` 105、`[M+NH4]+` 98、`[M+Na]+` 80、`[M-H2O+H]+` 24、`[M+K]+` 16、`[M-2H2O+H]+` 8、`[M+Cl]-` 7、`[M-H2O-H]-` 1 |
| 碰撞能量 | 单值 20、35、40、60、80 eV，以及合并谱 `20;50`、`20;40;60`、`20;40;60;80` |
| precursor m/z | 132–1,154，中位 388 |
| 峰数 | 中位 141.5（enveda-180 为 138） |
| 结构重叠 | **250 个结构全部在其他公共库中也有谱**（riken 246、gnps 233、mona 231、msdial 227、pluskal_ms2 224、massbank 208 等），**与 enveda-180 无重叠** |

意义：

- 它是唯一**仪器和化学空间都与测试集匹配**的数据，是本地验证的首选查询集（§4）。
- 同一批分子在其他仪器上也有谱，可以**直接测量仪器偏移**：同一分子的 timsTOF 谱与 Orbitrap / QTOF 谱差多少。
- 官方说明这些分子是刻意挑选的常见天然产物，比测试集容易，验证分数会偏乐观。

### 2.5 结构层面的统计

- 训练集中 90.6% 的结构至少有一个同分子式异构体，64% 的结构所在分子式有 10 个以上异构体。
- 以 enveda-np-examples 为查询（同 adduct、10 ppm），训练集其他库中的候选结构中位 5 个、P90 约 38 个；4.8% 的查询在训练集中找不到同 adduct 的候选。加入 PubChem、COCONUT 后候选数会大幅增加（§9.1）。
- **NP-likeness（RDKit Contrib 的 NP_Score）能区分化学空间**：enveda-np-examples 中位 1.42（86% 大于 0），enveda-180 中位 −1.50（只有 1% 大于 0），公共库中位 −0.37（42.5% 大于 0）。它既可以作为天然产物判据（§4.2），也可以作为候选先验特征（§6.4）。

### 2.6 训练数据质量统计

按三组统计：enveda-180、enveda-np-examples、公共库（其余各库合并）。

| 指标 | enveda-180 | enveda-np-examples | 公共库 |
| --- | --- | --- | --- |
| \|`precursor_error_ppm`\| 中位 / P99 / 最大 | 0.98 / 4.40 / 7.1 | 1.65 / 6.62 / 21.1 | 1.74 / 34,604 / 极大 |
| \|ppm\| > 10 的比例 | 0% | 0.17% | 12.6% |
| \|ppm\| > 20 的比例 | 0% | 0.08% | 8.6% |
| 峰数中位 / P95 | 138 / 823 | 141.5 / 846.5 | 20 / 266 |
| 峰数少于 5 的比例 | 0.6% | 0.4% | 14.5% |
| `base_peak_intensity` 为空的比例 | 0% | 0% | 84% |
| 基峰 < 1,000 counts 的比例（非空部分） | 0%（已按测试规则清洗） | 0% | 36.5% |

测试集 10 种 adduct 在训练集中的谱图数：

| adduct | enveda-180 | np-examples | 公共库 | 结构数 |
| --- | --- | --- | --- | --- |
| `[M+H]+` | 588,507 | 569 | 735,509 | 243,970 |
| `[M-H]-` | 184,747 | 276 | 280,665 | 98,485 |
| `[M+Na]+` | 15,957 | 80 | 162,055 | 25,112 |
| `[M+CH2O2-H]-` | 24,556 | 105 | 58,527 | 20,723 |
| `[M+NH4]+` | 4,702 | 98 | 48,652 | 8,890 |
| `[M+Cl]-` | 2,407 | 7 | 25,097 | 7,148 |
| `[M-H2O+H]+` | 0 | 24 | 21,658 | 5,594 |
| `[M+K]+` | 1,084 | 16 | 17,855 | 4,335 |
| `[M-2H2O+H]+` | 0 | 8 | 3,431 | 1,002 |
| `[M-H2O-H]-` | 0 | 1 | 284 | 149 |

其余 adduct 占训练集的 14.3%。

结论：

- timsTOF 数据的质量精度很高（P99 < 7 ppm），公共库存在大量高误差或低分辨率的谱。
- timsTOF 谱的峰很多（中位约 140），公共库很多谱只有少数几个峰。
- 脱水类 adduct 在 enveda-180 中完全没有；`[M-H2O-H]-` 全训练集只有 285 张谱，属于极少样本。

---

## 3. 域定义与 Domain Adaptation

### 3.1 两个维度的偏移

测试集同时有两个特征：**timsTOF 仪器**和**天然产物化学空间**。训练数据在这两个维度上各有缺口：

| 数据 | 仪器匹配 | 化学空间匹配 | 规模 |
| --- | --- | --- | --- |
| enveda-180 | ✅ | ❌ 合成类药分子 | 约 115 万谱 / 18.3 万结构 |
| 公共库（riken、gnps、massbank、mona、msdial、spectraverse 等） | ❌ Orbitrap / QTOF 等 | ✅ 大量天然产物 | 约 138 万谱 / 9.5 万结构 |
| enveda-np-examples | ✅ | ✅ | 1,184 谱 / 250 结构 |

其他关键事实：

- enveda-180 与公共库的结构几乎不重叠（共有 2,288 个结构）。
- enveda-np-examples 的 250 个结构全部在公共库中有谱，在 enveda-180 中没有。

推论：

1. **没有大规模的"双重匹配"数据**。模型要从 enveda-180 学 timsTOF 的谱图特性，从公共库学天然产物的碎裂规律，再在两者的交叉点（测试集）上泛化。
2. **谱库检索的参考库主要是公共库**。测试集的类别 1 分子（有公开谱）大多只在公共库中有谱，而这些谱来自其他仪器，所以跨仪器的谱–谱相似度很关键。enveda-180 的化学空间与测试集不同，对谱库检索帮助有限。
3. **仪器偏移可以直接测量**：用 enveda-np-examples 的 timsTOF 谱去检索同一分子在公共库中的其他仪器谱（§4.2 的 V-np-lib）。

### 3.2 策略（按优先级）

| 优先级 | 手段 | 说明 | 状态 |
| --- | --- | --- | --- |
| P0 | 以 enveda-np-examples 为主验证 | 验证集设计见 §4 | 【待填】 |
| P0 | 量化仪器偏移 | 同一分子 timsTOF 谱 vs 其他仪器谱的相似度分布；比较不同预处理、相似度度量下的检索准确率 | 【待填】 |
| P1 | 两阶段训练 | 全量预训练（enveda-180 + 公共库）→ 微调；微调数据的选择见下一行 | 已定方向 |
| P1 | 化学空间对齐 | 微调或加权时侧重天然产物结构（公共库中的天然产物、COCONUT 中的结构）；可用 NP-likeness 打分给训练分子加权 | 【待填】 |
| P1 | 元数据条件化 | 仪器、adduct、碰撞能量作为模型输入 token；推理时仪器固定为 timsTOF | 已定方向 |
| P1 | 鲁棒预处理 | 相对强度阈值、删除高于 precursor 的峰、top-N、去同位素、sqrt / log 强度，或 log m/z、RMS 归一化、Fourier 特征（§7.3、§9.3） | 【待填】 |
| P2 | 数据增强 | 随机丢峰、噪声峰、强度抖动、m/z 微移、模拟碰撞能量变化 | 【待填】 |
| P2 | 自监督预训练 backbone | 采用 DreaMS 作为谱图编码器初始化（§7.1） | 已定 |
| P2 | enveda-np-examples 参与最终训练 | 开发期只作验证；最终提交前把它并入训练数据或参考谱库 | 已定方向 |
| P3 | 测试集无监督适应 | 规则只禁止对测试数据人工标注或人工预测，无监督使用未被禁止；收益不确定 | 【待定】 |

---

## 4. 数据集划分

### 4.1 原则

1. **以 enveda-np-examples 为核心验证集**：它是唯一与测试集仪器、化学空间都匹配的数据。
2. **按 inchikey14 分组划分**，同一分子的谱不能在训练与验证之间随机拆分。
3. **用三种留出方式模拟三类新颖度**，区别在于分子是否存在于"训练谱库"与"候选库"。
4. enveda-np-examples 只有 250 个分子、且偏容易，因此另设**大规模补充验证集**，并对 MRR 报告 bootstrap 置信区间。

### 4.2 验证集分层

#### A. 主验证集：enveda-np-examples（250 个分子）

同一批查询谱，用三种留出设置分别评估：

| 子集 | 模拟类别 | 留出方式 | 在训练谱库 | 在候选库 |
| --- | --- | --- | --- | --- |
| **V-np-lib** | 1：有公开谱 | 只移出 enveda-np-examples 本身；同一分子在公共库中的谱保留 | ✅（其他仪器的谱） | ✅ |
| **V-np-nps** | 2：无公开谱 | 这些分子在**所有库**中的谱全部移出训练集 | ❌ | ✅ |
| **V-np-novel** | 3：新结构 | 在 V-np-nps 的基础上，从**所有候选库层**（训练集结构、COCONUT、PubChem）删除这些结构；训练集中与其高度相似的结构（如 Tanimoto ≥ 0.7）也移出 | ❌ | ❌ |

同一批查询在三种设置下的分数可以直接比较，便于分析各打分器分别在哪一类上起作用。

#### B. 补充验证集：公共库中的天然产物（规模更大，仪器不匹配）

- 从公共库中按 inchikey14 抽取 1,000–2,000 个天然产物类分子，质量分布向测试集（157–1,159 Da，中位 348）靠拢。
- 天然产物判据：NP-likeness > 0（§2.5）。公共库中约 42.5% 的结构满足；enveda-np-examples 中 86% 满足。下载 COCONUT 后，可改为或叠加"在 COCONUT 中收录"。
- 优先使用 TOF 类仪器的谱（公共库中约 30.8 万张、2.8 万个结构），因为 timsTOF 也是 Q-TOF。
- 同样构造 V-pub-nps 与 V-pub-novel（新结构按 scaffold 或 Tanimoto 聚类整组移出）。

#### C. 诊断验证集：enveda-180（仪器匹配，化学空间不匹配）

- 只用于诊断仪器相关的问题（如预处理参数），不用于判断整体方案优劣。

#### 通用规则

- 各子集之间的分子不重叠。
- 训练集 = 全部训练数据去除各验证子集对应的谱（和结构）。
- 划分由固定随机种子生成，分子 ID 列表落盘到 `data/splits/`，**全队共用同一份**。
- 查询时按测试集的规则模拟：删除基峰低于 1,000 counts 的谱（若有 `base_peak_intensity`）、删除高于 precursor + 2 Da 的峰。

### 4.3 融合模型的 out-of-fold 数据

- LightGBM ranker 需要打分器在"未见分子"上的分数。
- 主验证集只有 250 个分子，不足以同时训练 ranker 和评估。第一阶段用补充验证集 B 训练 ranker，在主验证集 A 上评估。
- 后期：对 A + B 按分子做 K 折，每折重跑打分器生成 OOF 分数。【待定】

### 4.4 整体分数的合成

三类在测试集中的比例保密。先分别报告三类 MRR；第一次提交（纯谱库检索）后，根据 Leaderboard 分数反推类别 1 的大致占比，再按估计比例加权得到本地"整体分数"。public / private 的拆分比例官方未公布。【待填】

---

## 5. 模块设计

每个模块列出职责、输入/输出、已知决策和待办。

### 模块 1：数据层 `src/casmi/data/`

- **职责**：读取 parquet，统一内部谱图表示，按 `molecule_id` 分组。
- **输出**：标准化谱图表（`spectrum_id, molecule_id, mzs, intensities, base_peak_intensity, precursor_mz, adduct, ionization_mode, instrument_type, collision_energy_ev, …`）。
- **预处理**【待填】：删除高于 precursor + 2 Da 的峰（与测试集一致）、相对强度阈值（官方列出 2%、1%、0.1% 为常见值）、最少峰数（官方建议约 5–6）、top-N（官方例子 N=128，或每 50 Da 窗口保留前 6 个峰）、去同位素、sqrt / log 强度变换。参数在主验证集上调定。
- **质量筛选**（初始值，依据 §2.6，后续在验证集上微调）：
  - 删除 \|`precursor_error_ppm`\| > 20 的谱（公共库删除 8.6%，timsTOF 数据几乎不受影响）；
  - 删除峰数少于 5 的谱（公共库删除 14.5%）；
  - 有 `base_peak_intensity` 的谱，删除基峰低于 1,000 counts 的（与测试集规则一致）；公共库 84% 没有该字段，不做此项筛选。
- **注意**：train 有 254 万行，按行组分批读取，缓存为预处理后的 parquet / npz。

### 模块 2：化学工具 `src/casmi/chem/`

- **职责**：结构规范化与分子特征。
- **功能**：
  - 与官方 metric 完全一致的规范化：**RDKit 2026.03.3**，`rdMolStandardize.TautomerEnumerator().Canonicalize(mol)` → `Chem.MolToInchiKey()[:14]`。官方注明不同 RDKit 版本的互变异构结果不同，本地已固定此版本（`pyproject.toml` 中 `rdkit==2026.3.3`，2026-09-24 起），并通过了官方 metric 自带的 doctest。
  - 分子指纹（Morgan 等）、单同位素质量、adduct 质量换算（覆盖测试集的 10 种 adduct，包括脱水类，§9.4）。
  - 候选去重：同一 inchikey14 在 Top-25 中只占一位。

### 模块 3：评估与切分 `src/casmi/eval/`

- **职责**：本地度量基准，所有改进以此为准。**最先实现。**
- **功能**：
  - `metrics.py`：直接复用官方 metric 的逻辑（1/rank，第一个命中即停止，非法 SMILES 视为未命中），同时输出 Hit@1/5/25 和 bootstrap 置信区间。
  - `make_splits.py`：按 §4 生成主验证集、补充验证集、诊断验证集及训练集掩码。
  - 分层报告：三类分别报告，再按估计比例加权得到整体分数。

### 模块 4：候选生成 `src/casmi/candidates/`

- **职责**：为每个 `molecule_id` 产出候选结构列表。
- **来源**（分层，§9.7）：
  - A. 训练集结构（约 27.6 万）。
  - B. COCONUT：官方定义的类别 2 来源之一。
  - C. PubChem：类别 2 的另一来源，按测试质量范围（157–1,159 Da）裁剪。
  - D. de novo 生成的结构：类别 3 的唯一来源。
- **过滤**：由 precursor m/z 与 adduct 反推中性质量，按 ppm 窗口检索。窗口初始值 **10 ppm**：enveda-np-examples 中 99.8% 的谱误差在 10 ppm 以内，enveda-180 的最大误差为 7.1 ppm（§2.6）；召回不足时放宽到 20 ppm。同一分子的多条谱（不同 adduct）应得到一致的中性质量。FIDDLE 预测分子式作为软特征而非硬过滤（§9.6）。
- **输出**：`(molecule_id, inchikey14, smiles, source)`，每分子几百到几千个。
- **验证注意**：V-np-novel 与 V-pub-novel 须从所有候选层删除对应结构。

### 模块 5：打分器 `src/casmi/scorers/`

所有打分器实现统一接口：输入候选表与谱图，输出 `(molecule_id, inchikey14, score, scorer_name)`，彼此独立、可并行开发。只有 5b 需要自行设计架构；5a 是无需训练的现成算法，5c、5d 沿用现成模型。

| 打分器 | 思路 | 主要覆盖 | 部件 | 状态 |
| --- | --- | --- | --- | --- |
| 5a 谱库检索 | 查询谱 vs 训练谱（主要是公共库的跨仪器谱），多谱聚合 | 类别 1 | Flash Entropy、DreaMS 余弦、MS2DeepScore（§7.7） | 无需训练，M2 调参（首个基线） |
| 5b CLIP 式对比检索 | 谱图塔与分子塔对齐，按余弦给候选排序 | 类别 2 | DreaMS + ECFP-MLP（§7.3） | 架构已定，M3 训练 |
| 5c 正向模拟 | 候选分子 → 预测谱图，与实测谱比较 | 类别 2 | ICEBERG / MARASON 自训（§7.6） | 可选，M5 视余力决定 |
| 5d de novo | 谱图 → 生成 SMILES，并入候选列表 | 类别 3 | FIDDLE / MIST-CF 分子式 + MS-GPT（备选 FRIGID）（§7.4、§7.5） | 沿用现成模型，M5 决定是否接入 |

### 模块 6：训练 `src/casmi/train/`

- **职责**：可学习打分器的训练脚本与配置。
- **流程**：全量预训练 → 侧重天然产物与 timsTOF 的微调（§3.2、§7.3）；按分子均衡采样（§9.9）；导出权重并上传为 Kaggle Dataset。
- **资源**：本地 RTX 5070；队友 RTX 5080。【待定：分工】
- **记录**：固定 commit、配置、权重哈希、外部数据许可证。

### 模块 7：融合与排序 `src/casmi/rank/`

- **多谱聚合**：同一分子多张谱（最多 16 张，可能跨 adduct、跨极性、跨碰撞能量）的分数取 max / mean / 加权。【待填】
- **多打分器融合**：第一版用加权和，之后换 LightGBM learning-to-rank。特征包括各打分器分数、最高谱库相似度、候选来源、候选先验（§6.3）、质量误差、分子式一致性。
- **两阶段排序**：先召回 Top-200，再精排（§9.1）。
- **输出**：去重后的 Top-25。

### 模块 8：提交 `src/casmi/submit/` + `notebooks/`

- 离线推理 Notebook：无网络、9 小时内完成，依赖与权重全部打包为 Kaggle Dataset。
- **RDKit 不在 Kaggle 的 docker 镜像中**（官方 metric 也是离线装 wheel），须把 `rdkit==2026.3.3` 的 wheel 作为 Dataset 附上并离线安装。
- 生成 `submission.csv`。官方拒收条件：缺少 `molecule_id` 或 `smiles` 列、文件为空、有空值、`molecule_id` 重复、某分子超过 25 个候选。非法 SMILES 不会被拒收，只按未命中处理；缺少的 molecule_id 计 0 分。
- `scripts/check_submission.py` 目前只检查列名与行数，须补齐上述检查。【待填】
- 本地 test.parquet 是占位样本，只能验证流程；耗时须按约 1,500 张谱估算，并在 Kaggle GPU 环境实测。【待填】

---

## 6. 竞赛规则、选型原则与关键结论

### 6.1 竞赛规则要点（官方 Rules）

| 项目 | 规定 |
| --- | --- |
| 获奖方案许可 | **MIT**；获奖时须提供可复现的方法说明和代码仓库 |
| 使用开源代码 | 只能使用 OSI 认可、且不限制商业使用的开源许可 |
| 许可不兼容的输入数据或预训练模型 | 可以使用；获奖时对这些数据或模型不必授予开源许可（Rules 5.a.3） |
| 外部数据与模型 | 必须免费、公开、所有参赛者可同等获得（或满足"合理性"标准，成本不能过高） |
| 测试数据 | 禁止对验证或测试数据**人工标注或人工预测**；无监督使用未被禁止 |
| 竞赛数据许可 | CC BY-NC 4.0，只能用于非商业目的 |
| 提交 | 每天最多 5 次；最终可选 2 个提交 |
| 队伍 | 最多 5 人；私下共享代码或数据只能在队内 |
| 运行环境 | CPU 或 GPU Notebook ≤ 9 小时，无网络，输出 `submission.csv` |
| 时间线 | 开始 2026-09-14；报名与组队截止 2026-12-07；最终提交截止 2026-12-14（UTC 23:59） |
| 排名 | 以 private leaderboard 为准；public / private 拆分比例未公布 |

### 6.2 部件收录标准

依据上述规则：

1. **代码必须使用 OSI 开源许可且不限制商业使用**（MIT / Apache-2.0 / BSD 等）。非商业（NC）许可或没有许可证的代码不能用。
2. **预训练权重和外部数据**只要免费公开即可，许可可以与 MIT 不兼容（但要记录，获奖时需说明）。
3. **不依赖在线服务**或需付费的商业软件（例如 SIRIUS 的 CSI:FingerID Web 服务、NIST 谱库）。
4. 能打包为 Kaggle Dataset 并离线加载。

权重的训练数据来源（例如是否含 NIST）不影响合规性，但要记录下来（§13），以便评估验证集泄漏。

### 6.3 公开基准上的许多高分来自泄漏

*MassSpecGym in the Wild*（2026）审查了 26 篇使用 MassSpecGym 的论文，其中 17 篇存在数据泄漏、捷径学习或实现错误。修正后差距很大：

| 模型 | 报告值 | 修正值 |
| --- | --- | --- |
| DreaMS + ChemBERTa（检索 Hit@1） | 82.41% | 6.70% |
| ChemFormer Two-Stage（检索 Hit@1） | 63.89% | 7.60% |
| MIST + MolForge（de novo Top-1） | 31.75% | 10.73% |

对本项目的含义：

- 论文数字只用于判断相对优劣，最终以本地验证为准。
- **预训练权重可能"见过"我们的验证分子**（DreaMS 的 GeMS、MIST 的训练集等），本地分数会因此虚高。enveda-np-examples 的分子都是常见天然产物，风险尤其大。必要时把与预训练数据 inchikey14 重合、或 Tanimoto ≥ 0.7 的分子单独报告。
- MIST 编码器存在已知的 attention mask batching bug，使用前必须修复。

### 6.4 不看谱图的"候选先验"本身就很强

同一研究发现，仅按 PubChem 默认顺序排列候选，就能得到 49.98% 的 Hit@1。本项目应把**候选先验**作为融合特征，例如：是否在训练集或公共库中出现、在训练集中的谱图数、是否在 COCONUT 中、NP-likeness（§2.5 显示它能明显区分天然产物与合成类药分子）。但必须在验证集上确认它的真实贡献，并避免从验证标签泄漏。

---

## 7. 开源部件清单

### 7.1 谱图编码器（backbone）

| 部件 | 说明 | 许可 | 权重 | 适配度 |
| --- | --- | --- | --- | --- |
| **DreaMS** | 在 GeMS（MassIVE 数千万条无标注谱）上自监督预训练的 Transformer，输出 1024 维谱图 embedding；可微调。官方数据页也推荐了 GeMS 和 DreaMS | 代码 MIT；权重 CC-BY-4.0（Zenodo：`ssl_model.ckpt`、`embedding_model.ckpt`） | ✅ | ★★★ 首选 backbone |
| **MIST** | 以子峰分子式为 token 的 Transformer，预测分子指纹 | MIT | ✅ | ★★ 须修复 batching bug |
| **DiffMS encoder** | 建模峰分子式与中性丢失的 Transformer，单独提供预训练 encoder 权重 | MIT | ✅ | ★★ |

DreaMS 的输入限制（源码核实）：

- 最多输入 **128 个峰**（推理 API 默认取最高的 100 个），更多的峰会按强度截断。
- 预训练数据（GeMS-A）只含 **电荷 +1（正离子）**、precursor m/z ≤ 1,000 的谱。测试集的负离子谱（enveda-np-examples 中约 33%）和 m/z 超过 1,000 的谱都在预训练分布之外，需要靠微调覆盖。
- 官方安装说明使用 Python 3.11；Kaggle 镜像是 Python 3.12（§8.2），需要确认能否在 3.12 下安装。

### 7.2 分子编码器（候选端）

| 部件 | 说明 | 许可 | 适配度 |
| --- | --- | --- | --- |
| **RDKit ECFP / Morgan 指纹 + MLP** | 最简单、最稳健；和多数指纹预测模型的输出空间天然对齐 | BSD-3 | ★★★ 第一版 |
| ChemBERTa（DeepChem，Hugging Face） | 预训练 SMILES Transformer；权重未声明许可，但按 Rules 5.a.3 可作为预训练模型使用，需记录 | 权重许可未声明 | ★★ 备选，可加 LoRA |
| 自训练 SMILES / 图编码器 | 在训练集结构和 COCONUT 上从头训练的小型 GNN 或 Transformer | 自有 | ★★ 如需更强分子塔 |

### 7.3 谱图–分子对齐 / 检索方法

| 部件 | 做法 | 许可 | 适配度 |
| --- | --- | --- | --- |
| **JESTR** | 谱图与分子的联合 embedding，对比学习排序候选 | MIT | ★★★ 可直接参考实现 |
| **MIST（指纹检索）** | 预测指纹 → 与候选指纹算 Tanimoto 或余弦 | MIT | ★★ 基线 |

可借鉴、但没有公开代码的思路（只借鉴方法，不作为依赖）：

- **冻结两个基础模型，只训练投影层**（MSAlign）：先冻结 DreaMS 训练投影层，成本极低。
- **跨仪器预处理**（Contrastive Domain Generalization, 2026）：log m/z 变换、RMS 强度归一化、Gaussian Fourier 特征（σ=30），直接对应本赛题的仪器偏移。
- **以排序为目标而不是指纹精度**（De Waele et al., 2026）：指纹 BCE 和检索排序之间存在权衡，应直接优化候选排序损失（InfoNCE 或 listwise）。
- **检索 + 生成重排序**（GLMR, AAAI 2026）：先用对比检索召回，再用生成模型产出结构，按相似度重排。

#### 本项目的 CLIP 式双塔设计（打分器 5b）

本赛题的检索任务和 CLIP 的零样本分类同构：谱图对应图像，分子对应文本 prompt，同质量窗口内的候选分子对应候选类别。分子塔可以给任意候选分子计算 embedding，所以没有公开谱的结构也能被排序，这就覆盖了类别 2。

CLIP、CONCH 等图文模型的权重**不直接使用**：它们的模态与谱图、分子无关，把谱图转成图像还会丢失 m/z 精度。领域内也没有权重开放的大规模谱图–分子多模态预训练模型，因此**自行训练**。训练集有 254 万条"谱图–结构"配对、27.6 万个分子，规模与 CONCH 的约 117 万对图文数据同一量级，而且是精确标注。

```text
谱图塔：DreaMS 初始化 + adduct / 碰撞能量 / 仪器 token ─┐
                                                       ├─ 投影到同一空间 → 余弦相似度
分子塔：ECFP-4096 → MLP（之后可换自训 GNN）───────────┘
```

相比标准 CLIP，需要做以下改动：

| 项目 | 做法 | 原因 |
| --- | --- | --- |
| **难负例** | 负例除 batch 内样本外，还要从**同一质量窗口**（按 precursor 中性质量 ± ppm）的候选中采样，优先采同分子式的异构体 | 随机分子通常质量不同，太容易区分；实际需要分开的是同质量的异构体。这就是 MSAlign 的"候选集对比损失" |
| **多正例** | 使用 multi-positive InfoNCE：同一 batch 内同一 inchikey14 的多条谱互为正例，不能当作负例 | 平均每个分子约 9 条谱，否则会产生大量假负例 |
| **元数据条件化** | 谱图塔输入 adduct、碰撞能量、仪器类型 token；推理时仪器固定为 timsTOF | 同一分子在不同条件下谱图差异很大；这也是 domain adaptation 的入口（§3.2） |
| **分阶段训练** | ① 冻结 DreaMS，只训练投影层和分子塔；② 解冻，在全量 254 万对上训练；③ 侧重天然产物与 timsTOF 数据微调（§3.2） | enveda-180 提供仪器特性，公共库提供天然产物化学 |
| **验证** | 主验证集用 V-np-nps、V-np-novel，补充验证集用 V-pub-nps、V-pub-novel（§4.2）；另做谱图打乱的捷径检查 | 确认模型真的在读谱图，而不是利用候选集的偏差 |

局限：双塔模型只能给候选排序，不能生成新结构。类别 3 仍要依靠 de novo（§7.5）往候选池里补充候选。

### 7.4 分子式预测

测试集不提供分子式，由这一层补上。官方数据页也把"先预测分子式、再以此缩小候选范围"列为常用方法。

| 部件 | 说明 | 许可 | 权重 | 适配度 |
| --- | --- | --- | --- | --- |
| **FIDDLE** | 深度学习分子式预测 + 重打分，文献报告 Top-1 88.3%、Top-5 93.6%；`pip install msfiddle`；提供 Q-TOF 和 Orbitrap 两套模型。发布的模型配置只编码 `[M+H]+` 和 `[M-H]-` 两种 adduct | Apache-2.0 | ✅（GitHub release） | ★★★ timsTOF 用 Q-TOF 模型；其余 8 种 adduct 需在我们的训练集上重训 |
| **MIST-CF** | 基于 Transformer 的分子式与 adduct 排序（官方数据页推荐） | MIT | ✅ | ★★ 备选 |

用途：候选过滤或降权、融合特征（候选分子式与预测分子式是否一致）、为需要分子式的 de novo 模型提供输入。

待核实：两者对 10 种测试 adduct（尤其脱水类和负离子）的支持情况。

### 7.5 de novo 生成

类别 3 的结构不在 PubChem 中，只能靠生成。许可证与权重核对日期 2026-09-24。

| 部件 | 做法 | 代码许可 | 权重 | 需要分子式 | 报告成绩（已知真实分子式） | 适配度 |
| --- | --- | --- | --- | --- | --- | --- |
| **MS-GPT**（2026） | MIST 预测指纹 + 分子式 → 在约 1 亿个分子上预训练的 SAFE-GPT 自回归生成；只训练轻量 LoRA 适配器 | Apache-2.0 | ✅（上海交大网盘） | 是 | MassSpecGym Top-1 23.9% / Top-10 28.7%；NPLIB1 Top-1 29.8% | ★★★ 首选；论文称预训练数据已排除测试结构；比扩散模型快 1–2 个数量级 |
| **FRIGID**（ICML 2026） | MIST 指纹 + 分子式 → 掩码扩散语言模型（最多 2.3 亿个分子预训练），可用 ICEBERG 迭代修正 | 代码 Apache-2.0 | ✅ Zenodo（记录 19685145），但权重许可为 **CC BY-NC 4.0**：按 Rules 5.a.3 可作为预训练模型使用，需记录 | 是 | MassSpecGym Top-1 约 18.3% | ★★ 备选；ICEBERG 修正步骤推理较慢 |
| **DiffMS**（ICML 2025） | 分子式约束的离散图扩散，提供端到端和预训练 encoder/decoder 权重 | MIT | ✅ | 是 | MassSpecGym Top-10 约 4.1% | ★ 已被上面两个大幅超越 |
| **MADGEN**（ICLR 2025） | 先检索骨架，再按谱图条件生成 | MIT | ✅ | 是 | MassSpecGym Top-1 约 1.3% | ★ 不推荐 |
| **MS2Forge**（2026） | 贝叶斯流网络生成 + 碎片证据重排 | MIT | ✅（Google Drive，MassSpecGym 训练） | 是（依赖峰的子分子式标注） | 未查到 | ★ 很新，尚无独立验证 |

用于本赛题时的注意点：

1. **所有可用模型都需要分子式**，而测试集不提供。上表成绩都是已知真实分子式下测得的。实际使用时先用 FIDDLE 或 MIST-CF 预测分子式（可取 Top-3 分别生成），成绩会明显下降，下降幅度在 V-np-novel 上实测。
2. **MS-GPT 和 FRIGID 都依赖 MIST 编码器**，而 MIST 有已知的 batching bug（§6.3）；已发布的 MIST 权重在 NPLIB1 或 MassSpecGym 上训练，没见过 timsTOF。可换成在 timsTOF 与天然产物数据上重训的 MIST，或 5b 的谱图塔，再重新训练适配器。
3. **adduct 支持**：MS-GPT 与 FRIGID 只在 NPLIB1 和 MassSpecGym 上训练和评测，这两个基准只有正离子（`[M+H]+`，MassSpecGym 另有 `[M+Na]+`）。负离子和脱水类 adduct 需要用我们的数据重训 MIST 编码器和适配器；或先只对正离子谱运行 de novo，再按分子聚合。
4. **与候选库的配合**：生成的结构若在 PubChem 中，按定义就不是类别 3 的答案，应交给检索路线（§9.7）；5d 的输出优先保留不在 PubChem 中的结构。

可借鉴、但没有公开代码的思路：**MARLIN**（2026）用"指纹 → 块扩散语言模型 + 质量壳约束"做 de novo，**不需要分子式**，是最契合本赛题的设计思路。

### 7.6 正向模拟打分

| 部件 | 说明 | 许可 | 适配度 |
| --- | --- | --- | --- |
| **ICEBERG / MARASON**（ms-pred） | 分子 → 碎裂图 → 预测谱，与实测谱比较；MARASON 用检索增强，把检索 Top-1 从 18.7% 提到 27.8% | 代码 MIT | ★ 低优先级 |

限制：

- 可以使用 MassSpecGym 训练的权重，或在我们的数据上自训。NIST20 训练的权重属于"许可不兼容的预训练模型"，规则上可用，但 NIST 本身是付费数据，按"合理性"标准存在争议，不建议使用。
- MARASON 训练需要两张 24GB 以上的 GPU，逐个候选模拟也很耗时。

### 7.7 谱库检索与谱–谱相似度

| 部件 | 说明 | 许可 | 适配度 |
| --- | --- | --- | --- |
| **ms-entropy（Flash Entropy Search）** | 谱图熵相似度，百万级谱库毫秒级检索；pip / bioconda 安装，已在项目依赖中 | Apache-2.0 | ★★★ M2 基线 |
| **matchms** | 谱图清洗、筛选与相似度计算（官方数据页推荐），已在项目依赖中 | Apache-2.0 | ★★★ 预处理工具 |
| **MS2DeepScore 2** | 学习型谱–谱结构相似度；pip 安装 | Apache-2.0 | ★★ 补充特征 |
| **DreaMS embedding 余弦** | 用 DreaMS embedding 做谱–谱检索 | MIT / CC-BY-4.0 | ★★ 补充特征 |

### 7.8 外部数据

| 数据 | 用途 | 许可 |
| --- | --- | --- |
| **COCONUT** | 候选库：开放天然产物库，官方定义的类别 2 来源之一 | CC0 |
| **PubChem** | 候选库：类别 2 的另一来源，按测试质量范围（157–1,159 Da）裁剪后离线打包 | 公有领域 |
| **GeMS / MassIVE** | 无标注谱，用于自监督预训练（官方数据页推荐） | GeMS 为 MIT |
| **FragHub** | 整合主要公开 MS/MS 库，与训练集大量重叠，但可能有额外谱（官方数据页推荐） | 代码 CeCILL 2.1；数据 CC BY-NC 4.0（可作为外部数据使用，需记录） |

官方说明 GNPS 的传播注释（"suspect" 注释）因标签是推断的，没有放进训练集，如需使用要谨慎。

### 7.9 基础工具与基准

| 部件 | 用途 | 许可 |
| --- | --- | --- |
| RDKit **2026.03.3** | 规范化、inchikey14、指纹；版本须与官方 metric 一致（本地已固定） | BSD-3 |
| LightGBM | 融合模块的 learning-to-rank | MIT |
| MassSpecGym | 评估协议与基线代码参考 | MIT |

---

## 8. 拼装方案

```text
                ┌─ 5a 谱库检索：Flash Entropy + DreaMS 余弦 + MS2DeepScore
查询谱 ─ 预处理 ─┤      参考库 = 训练集（主要是公共库的跨仪器谱）
(timsTOF)       │
                ├─ 5b CLIP 式对比检索：DreaMS（+ 元数据 token）⟷ ECFP-MLP 分子塔
                │      候选 = 训练集结构 + COCONUT + PubChem（质量窗口）
                │      FIDDLE 预测分子式 → 候选过滤 / 降权 / 一致性特征
                │
                └─ 5d de novo：FIDDLE / MIST-CF 分子式 → MS-GPT 生成
                       → 按 ppm 质量过滤 → 并入候选池
                                    ↓
      7 融合（LightGBM ranker）：各打分器分数 + 最高谱库相似度
                                 + 候选先验 + 分子式一致性
```

### 8.1 优先级

| 优先级 | 组合 | 覆盖类别 | 里程碑 | 理由 |
| --- | --- | --- | --- | --- |
| **P0** | Flash Entropy 谱库检索 + 多谱聚合 | 1 | M2 | 无需训练，用于第一次提交并估计类别 1 的占比 |
| **P1** | CLIP 式双塔对比检索（§7.3）：DreaMS 谱图塔 + ECFP 分子塔，质量窗口难负例 + multi-positive InfoNCE + 元数据 token；冻结训练投影层 → 全量训练 → 侧重天然产物与 timsTOF 微调 | 2 | M3 | 权重开源、训练轻量，RTX 5070 即可训练 |
| **P1** | 跨仪器预处理（log m/z、RMS 归一化、Fourier 特征），在 V-np-lib 上比较 | 全部 | M3 | 类别 1 的检索本身就是跨仪器匹配 |
| **P2** | FIDDLE 分子式特征与候选过滤 | 2 | M3–M4 | pip 安装，成本低 |
| **P2** | 候选先验特征 + LightGBM 融合 | 全部 | M4 | 文献显示先验很强 |
| **P3** | 预测分子式 + MS-GPT 生成候选（备选 FRIGID） | 3 | M5 | 合规、公开权重的 de novo 模型中成绩最好；需先解决分子式与 adduct 问题 |
| **P3** | MARASON / ICEBERG 自训重打分 | 2 | M5 | 成本高、收益不确定 |

### 8.2 Kaggle 可行性

- **推理算力**：测试集约 400 个分子、约 1,500 张谱，9 小时绰绰有余，de novo 采样也可以承受。
- **打包体积**：PubChem 须按测试质量范围裁剪，并预先算好指纹；DreaMS、MS-GPT、FIDDLE 的权重、RDKit 2026.3.3 wheel 和其他依赖 wheel 一起打包成 Kaggle Dataset。
- **Kaggle 运行环境**（Kaggle/docker-python 与官方公告，2026-09 核实）：
  - Python **3.12**，默认 PyTorch 为 cu128；
  - GPU 选项为 **T4 ×2**（每张 16GB 显存）；P100 已于 2026-09-15 停用。
- **本地环境已对齐**（2026-09-24）：本地 `casmi2026` 环境已切到 Python 3.12 + RDKit 2026.3.3（PyPI wheel，与官方 metric 相同），PyTorch 2.7.0 cu128。离线 wheel 统一按 cp312 准备。
- **各部件的环境要求**：DreaMS 官方用 Python 3.11；MS-GPT 安装脚本默认 CUDA 12.1 的 PyTorch。都需要早做 Kaggle 离线安装测试。
- **显存**：T4 是 16GB、不支持 bf16 的高效计算，MS-GPT、DreaMS 推理需按 fp16 或 fp32 估算显存。

---

## 9. 主要难点与对应技术

除 domain adaptation（§3）之外，本赛题还有下列 10 个难点。每个难点先给出数据依据，再列出可用技术。

优先级：★★★ 必做；★★ 值得做；★ 有余力再做。

### 9.1 异构体密集：同质量候选太多

**数据依据**：见 §2.5。训练集中 90.6% 的结构有同分子式异构体。只看训练集，enveda-np-examples 查询的质量窗口内候选中位 5 个、P90 约 38 个；加入 PubChem、COCONUT 后会达到数百到数千个。

真正的难点是在同分子式异构体之间排序，而不是找到质量对的分子。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 质量窗口难负例 | 对比训练时从同一质量窗口采负例，优先同分子式异构体（§7.3） | ★★★ |
| 两阶段排序 | 轻量模型（指纹相似度、双塔余弦）召回 Top-200，再由 cross-encoder 或 LightGBM 精排 | ★★★ |
| 子结构级特征 | 对只在异构体之间存在差异的指纹位加权计算相似度 | ★★ |
| 碎片可解释性打分 | 对候选枚举子结构碎片（BRICS 或键断裂），计算实测峰能被解释的比例 | ★★ |
| listwise 损失 | 在同一质量窗口的候选组内直接优化排序（ListNet、LambdaRank） | ★★ |

### 9.2 MS/MS 本身的信息上限

部分异构体（取代基位置不同的位置异构体、糖苷连接位点不同的异构体等）碎裂谱几乎相同，任何模型都无法区分。评测忽略立体化学，但位置异构体算作不同分子，因此 MRR 有天然上限。官方 metric 预留了"一个分子接受多个无法区分的异构体"的机制，但目前每个分子只有一个正确答案。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 候选先验特征 | 训练集中的谱图数或出现频次、是否在 COCONUT 中、结构复杂度、NP-likeness 等（§6.4） | ★★★ |
| 估计上限 | 在验证集上统计与正确答案谱图几乎相同的异构体数量，量化理论上限 | ★★ |
| Top-25 多样性 | 分数接近时优先放置结构不同的候选，避免被高度相似的结构占满 | ★ |

### 9.3 测试谱只做了最少的清洗

**数据依据**：官方说明测试谱只做了三项处理（§2.2），其余噪声峰、同位素峰都保留；训练集各库的上游预处理又各不相同。训练谱与测试谱的清洗程度不一致，会直接影响相似度和模型输入。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 统一预处理 | 训练谱、参考谱与查询谱走同一流程：删除高于 precursor + 2 Da 的峰、相对强度阈值（2% / 1% / 0.1% 中选）、最少峰数、top-N 或按 50 Da 窗口保留峰、去同位素、sqrt / log 强度 | ★★★ |
| 以 enveda-np-examples 为基准调参 | 它与测试谱同一处理流程，预处理参数在 V-np-lib 上调定 | ★★★ |
| 基峰强度作为特征 | `base_peak_intensity` 越高通常噪声越少，可作为模型输入或融合特征 | ★★ |
| 噪声增强 | 训练时随机加入低强度噪声峰和同位素峰 | ★★ |
| 鲁棒相似度 | spectral entropy 降低低强度峰的权重，对噪声比 cosine 更稳 | ★★ |

### 9.4 adduct 与极性混合

**数据依据**：测试集有 10 种 adduct，包括脱水类 `[M-H2O+H]+`、`[M-2H2O+H]+`、`[M-H2O-H]-`；一个分子最多 16 张谱，可能跨 adduct 和正负离子模式。enveda-np-examples 中负离子类 adduct 约占 33%。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| adduct 质量表 | 为 10 种 adduct 建精确换算表，每条谱独立换算中性质量 | ★★★ |
| 一致性检查 | 同一分子多条谱的中性质量应一致，可用于确定中性质量和发现异常 | ★★ |
| adduct token | 模型输入 adduct 类别；少见 adduct（如脱水类，见 §2.6，`[M-H2O-H]-` 只有 285 张训练谱）在训练时上采样 | ★★ |
| 脱水类的谱库检索 | 脱水 adduct 的参考谱较少，可与同分子的 `[M+H]+` 参考谱做中性丢失对齐后比较 | ★★ |
| 按极性分别打分再聚合 | 正、负离子谱各自打分，聚合权重由融合模型学习，不简单平均 | ★★ |

### 9.5 碰撞能量异质

**数据依据**：测试集的 `collision_energy_ev` 可能是单值，也可能是多个能量合并的谱（如 `[20, 40, 60, 80]`）；enveda-np-examples 中有 20、35、40、60、80 eV 以及 `20;50`、`20;40;60`、`20;40;60;80` 等组合。训练集中各库的能量口径不同，官方已尽量换算成 eV，但 NCE 的换算是近似的。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 能量作为输入 | 用 `collision_energy_ev` 编码（如最小值、最大值、能量个数），合并谱作为单独类别 | ★★★ |
| 谱库检索按能量匹配 | 优先匹配相近能量的参考谱，或把能量差作为融合特征 | ★★ |
| 合并参考谱 | 同一分子不同能量的参考谱合并为共识谱，与多能量合并的查询谱比较 | ★★ |
| 增强 | 训练时随机合并同一分子不同能量的谱，模拟合并谱 | ★ |

### 9.6 没有分子式

候选只能按精确质量过滤，候选数因此大很多（§9.1）。需要分子式的 de novo 模型只能依赖预测值。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 精确质量过滤 | 中性质量 ± 10 ppm（enveda-np-examples 中 99.8% 的谱在此范围内，§2.6）；召回不足时放宽到 20 ppm | ★★★ |
| FIDDLE 分子式作为软特征 | 候选分子式是否在 FIDDLE 的 Top-1 / Top-5 中，作为特征而不是硬过滤，以免预测错误删掉正确答案 | ★★ |
| 碎片分子式约束 | 实测碎片能否被候选分子式的子分子式解释，作为软约束特征 | ★★ |
| 同位素峰 | 测试谱保留了同位素峰，可用于估计碳数或判断 Cl、Br、S | ★ |

### 9.7 候选空间的覆盖与噪声权衡

候选库越大，覆盖率越高，但干扰项越多。按官方定义，类别 2 的结构一定在 PubChem 或 COCONUT 中，类别 3 一定不在 PubChem 中。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 分层候选库 | 训练集结构、COCONUT、PubChem 分层加入，把"来源"作为融合特征 | ★★★ |
| 类别 3 的判定线索 | 候选若在 PubChem 中，按定义就不可能是类别 3 的答案；de novo 生成的结构若在 PubChem 中，应优先交给检索路线处理 | ★★ |
| 验证覆盖率 | 在 V-np-nps 与 V-pub-nps 上统计正确答案在各层候选库中的覆盖率 | ★★★ |
| 合理化过滤 | 删除带电分子、同位素标记、超大分子、罕见元素等不合理候选 | ★★ |
| de novo 补充 | 生成结构作为额外候选层并入候选池（§7.5） | ★ |

### 9.8 验证的可信度

**数据依据**：三类在测试集中的比例保密；与测试集双重匹配的数据只有 250 个分子，而且偏容易；预训练权重可能见过验证分子；测试集约 400 个分子，一个分子约占 0.25% 的 MRR；public / private 拆分比例未公布。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 分层报告 | 主验证集三种设置、补充验证集分别报告；整体分数按 LB 反推的占比加权（§4.4） | ★★★ |
| 泄漏排除 | 与预训练数据 inchikey14 重合或 Tanimoto ≥ 0.7 的验证分子单独报告（§6.3） | ★★★ |
| bootstrap 置信区间 | 对 MRR 做 bootstrap，判断两个方案的差距是否显著；主验证集只有 250 个分子，尤其需要 | ★★★ |
| 捷径检查 | 谱图打乱或遮蔽后分数应显著下降 | ★★ |
| LB 使用原则 | 只用 LB 校准类别占比、检查大方向；小于约 0.01 的变化视为噪声；每天只有 5 次提交，要有计划地使用 | ★★ |

### 9.9 训练数据本身的问题

254 万条谱带来算力成本；各库上游预处理不同；公共谱库存在错误注释；部分分子谱图数远多于其他分子。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 质量筛选 | 初始阈值：\|ppm\| ≤ 20、峰数 ≥ 5、基峰 ≥ 1,000 counts（有该字段时），见模块 1；官方说明 `precursor_error_ppm` 可作为标签质量信号 | ★★★ |
| 按分子均衡采样 | 训练时按分子采样，每个分子最多取 k 条谱 | ★★★ |
| 缓存 | 预处理结果与预测指纹缓存为 parquet / npz，避免重复计算 | ★★★ |
| 按来源建模 | 把 `ingest_lib` 或仪器类别作为输入或采样分层，吸收各库预处理差异 | ★★ |
| 标签噪声检测 | 同一 inchikey14 的谱之间相似度异常低的，可能是错误注释，删除或降权 | ★ |

### 9.10 评测与提交细节

MRR 只看第一个正确答案的排名，Top-25 的位置是稀缺资源；重复结构会浪费位置。

| 技术 | 做法 | 优先级 |
| --- | --- | --- |
| 与官方一致的规范化 | RDKit 2026.03.3，TautomerEnumerator 规范化后取 InChIKey 前 14 位；本地评估与提交去重共用同一套代码 | ★★★ |
| 合法性检查 | 提交前用 RDKit 解析每个 SMILES，失败则用下一名候选替换；确保无空值、无重复 molecule_id、每分子 ≤ 25 个 | ★★★ |
| 早做空跑提交 | M2 阶段就跑一次完整的离线 Kaggle 提交，确认 RDKit wheel、依赖打包与时限 | ★★★ |
| 填满 25 个 | 候选不足时按先验补满，多一次命中机会 | ★★ |

---

## 10. 目录结构（规划）

```text
Enveda-CASMI-2026/
├── configs/                 # 实验配置（yaml）
├── data/
│   ├── raw/                 # 官方数据（不入库）
│   ├── processed/           # 预处理缓存
│   ├── splits/              # 固定划分的分子 ID 列表（全队共享）
│   └── external/            # 外部分子库
├── docs/
├── notebooks/               # 探索 & Kaggle 提交 notebook
├── scripts/                 # 下载、校验等命令行脚本
└── src/casmi/
    ├── data/                # 模块 1
    ├── chem/                # 模块 2
    ├── eval/                # 模块 3
    ├── candidates/          # 模块 4
    ├── scorers/             # 模块 5
    ├── train/               # 模块 6
    ├── rank/                # 模块 7
    └── submit/              # 模块 8
```

---

## 11. 开发顺序与里程碑

| 里程碑 | 内容 | 产出 |
| --- | --- | --- |
| M1 | 模块 1、2、3；固定 RDKit 2026.03.3；按 §4 生成验证集；§9.3 统一预处理；§9.9 质量筛选与均衡采样；§9.10 合法性检查 | 预处理缓存、固定划分、与官方一致的本地 MRR@25 评估器 |
| M2 | 5a 谱库检索 + 模块 7（简单聚合）+ 模块 8 | **首次 Kaggle 提交**，估计类别 1 占比；测量仪器偏移（V-np-lib） |
| M3 | 模块 4 候选库（COCONUT、PubChem）+ 5b + 模块 6 | 覆盖类别 2 |
| M4 | 模块 7 LightGBM 融合、domain adaptation 调优 | 整体分数提升 |
| M5 | 5c / 5d（视余力） | 覆盖类别 3 |

最终截止 2026-12-14；组队截止 2026-12-07。

---

## 12. 排除清单

| 部件 | 排除原因 |
| --- | --- |
| **MolForge**（指纹 → SMILES 解码器） | 代码许可为 CC BY-NC-SA 4.0（非商业），违反"开源代码不得限制商业使用"的规则。它是 MIST + MolForge 最强 de novo 组合的一半，作者若更换许可可重新评估 |
| **MSNovelist** | 依赖 SIRIUS / CSI:FingerID 指纹（在线服务）。其代码为 AGPL-3.0，属于 OSI 认可且不限制商业使用的许可，许可本身不是障碍 |
| **MS-BART**（NeurIPS 2025） | 仓库没有许可证，默认保留所有权利，代码不能使用 |
| **SIRIUS / CSI:FingerID** | 依赖在线 Web 服务，无法离线使用 |
| **NIST 谱库及 NIST 训练的权重** | NIST 是付费数据；相关权重规则上可能可用，但存在"合理性"争议，不使用 |
| **Spec2Mol** | 编码器依赖 NIST 2020 数据 |
| **MSAlign** | 检索不到代码仓库；只借鉴方法 |
| **Contrastive Domain Generalization（2026）** | 无公开代码；只借鉴预处理方法 |
| **GLMR、MARLIN** | 无公开代码；只借鉴思路 |
| **LSM1-MS2 / LSM-MS2、Enveda PRISM / MS2Mol** | 未公开权重 |

---

## 13. 注意事项

- **固定版本**：每个采用的部件都要记录 commit、许可证、权重文件哈希和训练数据来源；RDKit 固定为 2026.03.3。
- **许可记录**：许可不兼容 MIT 的预训练模型（如 ChemBERTa）和外部数据要单独列出，获奖时需说明。
- **验证泄漏检查**：采用预训练权重后，把验证分子与该权重训练集的重叠情况单独报告（§6.3）。
- **捷径检查**：谱图遮蔽或打乱后分数应显著下降，否则说明模型在利用候选集的捷径。
- **不要用本地 test.parquet 调参**：它是占位样本（§2.1）。
- **竞赛数据的使用限制**：竞赛数据为 CC BY-NC 4.0，不得向队外传播。

---

## 14. 开放问题

- [ ] public / private leaderboard 的拆分比例（官方页面未写，可关注讨论区）。
- [ ] "公开谱"是否包括 enveda-180（它是 Enveda 公开发布的数据）。影响类别 1 的范围，但由于化学空间不同，实际影响可能不大。
- [ ] 用预测分子式代替真实分子式后，MS-GPT 在 V-np-novel 上的成绩下降多少。
- [ ] 负离子与脱水类 adduct 的 de novo 方案：重训 MIST 与适配器，还是只对正离子谱运行。
- [ ] DreaMS、MS-GPT 能否在 Python 3.12 下离线安装。
- [ ] PubChem 按质量范围裁剪后的条数与体积（下载 PubChem 后实测），能否打包进 Kaggle Dataset。
- [ ] 测试集无监督适应是否值得做（规则允许）。
- [ ] 与队友的模块分工。

---

## 参考文献与链接

### 竞赛官方资料

- Enveda CASMI 2026 – Molecule ID From Mass Spectra. [竞赛主页](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) · [数据](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) · [规则](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) · [官方 metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)

### 基准与评估

1. Bushuiev, R. et al. *MassSpecGym: A benchmark for the discovery and identification of molecules.* NeurIPS 2024 Datasets & Benchmarks. [论文](https://arxiv.org/abs/2410.23326) · [代码](https://github.com/pluskal-lab/MassSpecGym)
2. *MassSpecGym in the Wild: Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery.* arXiv 2606.19624 (2026). [论文](https://arxiv.org/html/2606.19624v1)
3. De Waele, G. et al. *Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?* arXiv 2602.16507 (2026). [论文](https://arxiv.org/html/2602.16507v1)
4. *De Novo Structure Prediction from Tandem Mass Spectra: Algorithms, Benchmarks, and Limitations.* 综述. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12985711/)

### 谱图编码器与检索

5. Bushuiev, R. et al. *Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS.* Nature Biotechnology (2025). [代码](https://github.com/pluskal-lab/DreaMS) · [权重（Zenodo）](https://zenodo.org/records/10997887)
6. Goldman, S. et al. *Annotating metabolite mass spectra with domain-inspired chemical formula transformers (MIST).* Nature Machine Intelligence (2023). [代码](https://github.com/samgoldman97/mist)
7. Kalia, A. et al. *JESTR: Joint Embedding Space Technique for Ranking candidate molecules.* [代码](https://github.com/HassounLab/JESTR1)
8. *MSAlign: Aligning Molecule and Mass Spectra Foundation Models for Metabolite Identification.* arXiv 2605.19752 (2026). [论文](https://arxiv.org/abs/2605.19752)
9. *Contrastive Domain Generalization for Cross-Instrument Molecular Identification in Mass Spectrometry.* arXiv 2602.00547 (2026). [论文](https://arxiv.org/html/2602.00547)
10. *Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra (GLMR).* AAAI 2026. [论文](https://arxiv.org/abs/2511.06259)

### 分子式预测

11. Hong, Y. et al. *FIDDLE: a deep learning method for chemical formulas prediction from tandem mass spectra.* Nature Communications (2025). [论文](https://www.nature.com/articles/s41467-025-66060-9) · [代码](https://github.com/JosieHong/FIDDLE)
12. Goldman, S. et al. *MIST-CF: Chemical Formula Inference from Tandem Mass Spectra.* J. Chem. Inf. Model. (2023). [论文](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01082) · [代码](https://github.com/samgoldman97/mist-cf)

### de novo 生成

13. *MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model.* arXiv 2607.23607 (2026). [论文](https://arxiv.org/abs/2607.23607) · [代码](https://github.com/VIKI623/MS-GPT)
14. *FRIGID: Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time.* ICML 2026. [论文](https://arxiv.org/abs/2604.16648) · [代码](https://github.com/coleygroup/FRIGID)
15. MS2Forge: Bayesian Flow Networks for De Novo Molecular Structure Elucidation from Tandem Mass Spectra. [代码](https://github.com/huaqiyao/MS2forge)
16. Bohde, M. et al. *DiffMS: Diffusion Generation of Molecules Conditioned on Mass Spectra.* ICML 2025. [论文](https://arxiv.org/abs/2502.09571) · [代码](https://github.com/coleygroup/diffms)
17. Wang, Y. et al. *MADGEN: Mass-Spec attends to De Novo Molecular generation.* ICLR 2025. [论文](https://arxiv.org/abs/2501.01950) · [代码](https://github.com/HassounLab/MADGEN)
18. *MARLIN: De Novo Molecular Structure Elucidation from Tandem Mass Spectra without a Ground-Truth Formula.* arXiv 2607.04774 (2026). [论文](https://arxiv.org/abs/2607.04774)
19. *One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra (MIST + MolForge).* arXiv 2508.04180 (2025). [论文](https://arxiv.org/abs/2508.04180) —— 已排除，仅作参考
20. *MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation.* NeurIPS 2025. [论文](https://arxiv.org/abs/2510.20615) · [代码](https://github.com/OpenDFM/MS-BART) —— 无许可证，已排除

### 正向模拟

21. Goldman, S. et al. *ICEBERG: Generating molecular fragmentation graphs with autoregressive neural networks.* [代码（ms-pred）](https://github.com/coleygroup/ms-pred)
22. *Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning (MARASON).* ICML 2025. [论文](https://arxiv.org/abs/2502.17874)

### 谱库检索与相似度

23. Li, Y. et al. *Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification.* Nature Methods (2021)；*Flash entropy search.* Nature Methods (2023). [代码（MSEntropy）](https://github.com/YuanyueLi/MSEntropy) · [文档](https://msentropy.readthedocs.io/)
24. de Jonge, N. et al. *MS2DeepScore.* [代码](https://github.com/matchms/ms2deepscore)
25. matchms. [代码](https://github.com/matchms/matchms)

### 外部数据与工具

26. COCONUT: COlleCtion of Open Natural prodUcTs. [网站](https://coconut.naturalproducts.net/)
27. PubChem. [网站](https://pubchem.ncbi.nlm.nih.gov/)
28. MassIVE. [网站](https://massive.ucsd.edu/)
29. FragHub. [代码](https://github.com/eMetaboHUB/FragHub)
30. RDKit（2026.3.3）. [PyPI](https://pypi.org/project/rdkit/2026.3.3/) · [网站](https://www.rdkit.org/)
31. LightGBM. [代码](https://github.com/microsoft/LightGBM)
