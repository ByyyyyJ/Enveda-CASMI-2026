# 论文大纲：公开流水线的本地伪测试为什么不可信

目标：先按 workshop 篇幅（4–6 页）写，投 NeurIPS / ICLR 的科学类 workshop 或 ICML DMLR；比赛结束、私榜（约 270 个分子）和可能的测试集答案公开后，扩展为 Journal of Cheminformatics / JCIM 长文或 NeurIPS Datasets & Benchmarks。

暂定题目：*When Local Validation Lies: Auditing Pseudo-Tests for a Shared MS/MS Structure-Identification Pipeline*

## 1. 引言

- 背景：CASMI 2026（Kaggle）从 MS/MS 谱图鉴定天然产物结构，指标 MRR@25。公开榜约 125–130 个分子（测试集约 31%），一个分子从第 2 名升到第 1 名就值 0.004。
- 现象：公开 notebook 几乎都是同一条流水线的分支（0.465 → 0.477），绝大多数改动只在公开榜上试。我们尝试用本地伪测试来做模型选择。
- 核心发现：我们构造的每一种伪测试，都会在某个关键改动上给出和隐藏测试相反的方向；这种偏差来自流水线组件的训练数据泄漏和伪测试构造方式，而不是样本量。
- 贡献：
  1. 5 次隐藏测试探针与本地预测的对照（表 1）；
  2. 组件级泄漏清单与"方向论证"：哪些泄漏会让哪一边虚高（表 2）；
  3. 伪测试配比拟合的反例：3 个自由参数对 5 个数拟合得很好（RMSE 0.011），下一次探针却方向相反；
  4. 给竞赛与小规模隐藏测试场景的实践建议。

## 2. 相关工作

- CASMI 历届比赛与结构鉴定方法：CSI:FingerID / SIRIUS、MIST、ICEBERG / GLACIER（ms-pred）、DreaMS、MassSpecGym。
- 化学数据划分与泄漏：scaffold split、identity-disjoint folds、MassSpecGym 的 MCES 划分。
- 机器学习中的验证集偏移与"公开榜过拟合"：Kaggle adaptive overfitting 研究（Roelofs et al., 2019），holdout reuse（Dwork et al., 2015）。
- 必须引用：公开流水线作者的 `casmi26-natural-product-panel-benchmark`（手调 PubChem 门控）与 fold-safe FPNet 数据集。

## 3. 流水线与伪测试

- 流水线概述（图 1）：Engine 1（约 71 万结构的池，FPNet + LightGBM 排序器）、Engine 2（COCONUT / ChEBI / LIPID MAPS / PubChemLite）、PubChem 通道（FPNet + 文献先验）、ICEBERG / GLACIER、神经打分器，最后 RRF 融合。
- 伪测试构造（`scripts/make_pseudo_test.py`）：
  - `f<k>np`：identity fold k 中的天然产物（LOTUS / NPAtlas，排除 MassSpecGym 与 np 面板），按 PubChem 记录数三分位抽样，每折约 250 个分子；查询谱图按官方方式清洗；这些分子的所有谱图从训练集删除。
  - 模型替换 `f<k>_safe`：所有 10,226-bit FPNet 换成不含 fold k 的版本。
  - 三种答案可见性：safe（答案在所有库中）、hide（答案从 Engine 1 池、Engine 2 库和 PubChem 排除集中隐藏）、hidee1（只从 Engine 1 池隐藏）。
  - 两个面板：np-lib、np-nps（都过于简单，MRR > 0.9）。

## 4. 结果

### 4.1 隐藏测试探针与本地预测（表 1）

| 改动 | 内容 | 本地 safe | 本地 hide | 本地 hidee1 | 公开榜（V2 = 0.468） |
| --- | --- | --- | --- | --- | --- |
| main_first | 主列表候选优先，按主列表顺序 | +0.122 | −0.083 | −0.215 | −0.097 |
| pc_last | PubChem 独有候选挪到最后 | +0.005 | −0.075 | −0.017 | −0.041 |
| pc_at2 | 最好的 PubChem 独有候选提到第 2 | −0.028 | +0.083 | −0.014 | −0.009 |
| e2_last | Engine 2 独有候选挪到最后 | +0.006 | +0.006 | +0.003 | +0.001 |
| e2off_first | 池外 Engine 2 最佳候选放第 1 | −0.183 | +0.028 | +0.244 | −0.128 |

要点：safe 是最自然的本地验证方式，它对 main_first 的判断方向完全错误（+0.122 对 −0.097）。

### 4.2 配比拟合的反例

- 用 safe / hidee1 / hide / miss 四类的配比拟合前 4 次探针：safe 0.30 / hidee1 0.58 / hide 0.12，RMSE 0.011，看起来"找到了真实测试的组成"。
- 据此预测 e2off_first +0.10，实际 −0.128。加入第 5 次探针后最好 RMSE 0.053，没有配比能解释全部探针（`scripts/lb_mixture_fit.py`）。
- 教训：用少量公开榜数字拟合自由参数，再用拟合结果指导下一步，本质是在小样本上做适应性分析。

### 4.3 排除掉的解释

- 化学空间距离：按答案与训练结构的最近邻 Tanimoto 分层（同 fold 的近亲也计入），main_first 在每一层都是 +0.10～+0.15（`scripts/similarity_strata.py`）。
- 神经打分器泄漏：它用全部训练集训练，见过所有答案，但在 safe 上单独排序 MRR 只有 0.27（主列表 0.81）。

### 4.4 泄漏清单与方向论证（表 2）

| 组件 | 训练数据 | 本地处理 | 泄漏让本地哪一边虚高 |
| --- | --- | --- | --- |
| 10,226-bit FPNet | 全部训练集 | 换成 fold-safe 版本 | — |
| 6,930-bit FPNet（Engine 2） | 全部训练集 | 未处理（可重训） | V2 融合 |
| 神经打分器 | 全部训练集 | 未处理（可重训） | V2 融合 |
| GBM 排序器（Engine 1） | 模拟查询行，无分子身份 | 无法按 fold 处理 | Engine 1 主列表 |
| ICEBERG / GLACIER | MassSpecGym | 伪测试排除 MassSpecGym 分子 | — |

- 论证：只有 Engine 1 侧的泄漏能让 main_first 在本地虚高；最容易修的两项（Engine 2 FPNet、神经打分器）修好后本地差距只会更大。
- 训练集结构偏好：hide 跑法里，训练集结构占主列表候选的 14–16%，却占第 1 名的 25–33%（`scripts/train_prior_check.py`）。排序器的模拟训练数据里答案总是训练集结构；伪测试的答案也是。

### 4.5 公开前沿的旁证

- 公开最佳 0.477 的作者在隐藏测试上提交了 13 次：离线更好的长训练神经模型 −0.019，异构体重排器 −0.008 到 −0.058。离线指标同样没有迁移。

## 5. 讨论与建议

- 对竞赛参与者：组件级泄漏清单应先于任何本地验证；伪测试必须先用至少一个公开榜探针校准方向。
- 对主办方：公开榜规模过小（约 128 个分子）会鼓励公开榜调参；建议提供与隐藏测试同分布、可用于验证的小型带标签集。
- 局限：只有一条流水线、一个比赛；公开榜噪声大，5 次探针里只有 3 次的差值显著大于单分子分辨率。

## 6. 比赛结束后要补的实验

1. 私榜分数：所有我们提交过的改动在私榜上是否保持方向。
2. 若测试集答案公开：直接计算每种伪测试与真实测试在多个改动上的排序一致性（Kendall τ），给出可信伪测试的构造方法。
3. 按 fold 重训 Engine 2 FPNet 与神经打分器，验证方向论证的预言（本地差距变大）。

## 附：可复现材料

- 代码：`scripts/make_pseudo_test.py`、`make_overrides.py`、`make_hidden_notebook.py`、`probe_variants.py`、`lb_mixture_fit.py`、`similarity_strata.py`、`offpool_marker.py`、`train_prior_check.py`、`make_variant_notebook.py`。
- 探针 kernel：wyy817/casmi26-main-first、pc-last、pc-at2、e2-last、e2off-first。
- 数据许可：竞赛数据 CC BY-NC 4.0，论文只发布统计结果，不发布谱图。
