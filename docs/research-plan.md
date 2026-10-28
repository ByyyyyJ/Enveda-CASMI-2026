# 研究方案：证据门控融合（2026-10-27 起执行）

取代 [model-components.md](model-components.md)（已归档）与 [plan-review.md](plan-review.md) §3 的执行部分。基线、验证集与泄漏处理沿用 plan-review。

## 1. 故事线

隐藏测试集中，答案可能在候选池（训练集结构 + COCONUT）里，也可能只在 PubChem 里或完全是新结构，三类比例保密。公开 pipeline 的融合对所有分子用同一套规则，把池外 PubChem 候选插进前列。我们的本地伪测试（答案在池内）上，只按 Engine 1 主列表排序就比融合结果高：

| 伪测试 | 分子 | V2 融合 | main_first | 配对差（95% CI） |
| --- | --- | --- | --- | --- |
| f0np（fold-0-safe） | 260 | 0.725 | 0.825 | +0.100 |
| f1np（fold-1-safe） | 249 | 0.677 | 0.832 | +0.154 [+0.120, +0.189] |

但这个提升部分来自构造（f\*np 的答案按选法全部在池内）。核心问题是：**每个分子该信哪条证据，取决于它属于哪一类，而类别看不到。**

方法：**证据门控融合**。

1. 门控：用分子层面的证据估计"答案在池内 / 有公开谱"的可能性，据此决定信哪个引擎。
2. 缺席证据：候选在参考库里有谱、但和查询谱不像，作为反证（原方案 plan-review ⑦）。
3. 同分子式组内相对特征：误差约 91% 是同分子式异构体排错（原方案 §9.1），用组内名次和差距代替绝对分数。
4. 引擎一致性：多个引擎是否给出同一个第一名，作为置信度。

评估：按类别配比的稳健性（原方案 §4.4）。在"答案在池内"和"答案不在池内"两种伪测试的多种混合比例下报告 MRR。

## 2. 约束

- 只用 pipeline 已输出的证据（`neural_candidate_features.parquet` 等），不重训大模型；重排器是 LightGBM，CPU 上分钟级。
- 重排只改变最终 25 个候选的顺序（f\*np 上 Hit@25 约 98%，排序才是瓶颈）。
- 交叉验证按 fold 留出：在 4 个 fold 的伪测试上训练，在剩下的 1 个上测。
- 不用自然产物知识表里的热度特征：f\*np 按 LOTUS/NPAtlas 选样，这些特征会经由选样规则泄漏。

## 3. 实验清单

| 编号 | 实验 | 依赖 | 状态 |
| --- | --- | --- | --- |
| E1 | 特征框架 + 跨 fold 重排器（相对特征、一致性特征） | f0np、f1np 结果 | 第一版完成：f0/f1 留一 fold，重排器 +0.133 vs V2，比 main_first（+0.129）只多 0.004；只有池内答案时模型基本学成 main_first，需要 E3 的数据。5 fold（1269 分子，V2 0.695）：main_first +0.123，cand +0.134，+rel +0.140，+agree +0.142，+gate +0.147（MRR 0.842）；最重要的特征是 main_rank，与 main_first 相近。公开榜表明池内伪测试单独不足以选模型，见 E2 |
| E2 | 公开榜检验 main_first（提交 57712274） | — | 完成：公开榜 0.371，V2 为 0.468（−0.097）。本地池内伪测试 +0.10～+0.15，公开榜反而大跌，说明隐藏测试里答案不在池内的分子占很大比例，统一规则在两类分子上的方向相反。这正是门控的动机；main_first 不能直接用 |
| E3 | "答案不在池内"伪测试：用本地专用 notebook（`make_hidden_notebook.py`）把答案从 Engine 1 池、Engine 2 库和 PubChem 通道的池内排除集合中隐藏，重跑 5 个 fold（`wsl_eval_hide_batch.sh`，输出 `f<k>np_hide`） | — | 完成（5 fold，见 §3.1）。f0np_hide：V2 0.250（Hit@25 78%），main_first 0.176（−0.074）。泄漏检查（`check_hide.py`）：260 个答案中 204 个进入最终 25，其中只有 7 个不经 PubChem 通道（2.7%，疑为互变异构体 key 不一致），可接受。只用 f0 hide 时反推的池内比例 p 为 −0.12：即使全部是"池外"分子，main_first 也只会掉 0.074，而公开榜掉了 0.097，所以隐藏测试比我们的 hide 伪测试更依赖 PubChem 通道，几乎没有池内答案。结论：选模型应以 hide 伪测试为主，safe 只作参考 |
| E4 | 门控 + 缺席证据，在两种伪测试的混合上训练 | E1、E3 | 待做 |
| E5 | 类别配比敏感性：1:1、2:1、1:2 等配比下比较 V2 / main_first / 门控 | E3、E4 | 待做 |
| E6 | 消融：逐组去掉特征；各组件的配对差 | E4 | 待做 |
| E7 | 最终提交：门控模型打包进 notebook，公开榜配对比较 | E4 | 待做 |

## 3.1 2026-10-28：5 个 fold 的 hide 结果与验证集的修正

hide 伪测试（5 fold，1279 分子）：V2 0.245，main_first 0.162。泄漏检查：约 80% 的答案进入最终 25，基本都经由 PubChem 通道。

| 训练集 → 评估 | safe | hide |
| --- | --- | --- |
| V2 | 0.695 | 0.245 |
| main_first | 0.817 | 0.162 |
| 重排器，safe 训练 | 0.842 | 0.101 |
| 重排器，hide 训练 | 0.110 | 0.728 |
| 重排器，safe+hide 训练 | 0.743 | 0.621 |

单特征检查：只按 PubChem 通道自己的排名排序，hide 上就有 0.666。也就是说，答案进入 PubChem 列表时通常排第 1，V2 的融合把它压到了第 5 名左右。

但这和公开榜对不上：

- 只用 PubChem 列表的公开探针得 0.037（pc_runner.py 注释），说明真实测试的答案很少靠 PubChem 通道找到；
- 用 safe 与 hide 的混合也解释不了 main_first 在公开榜上的 −0.097，反推的池内比例为负。

更可能的解释有两点：

1. hide 伪测试里 PubChem 通道偏强，因为 f\*np 的答案按 LOTUS/NPAtlas 选出，PubChem 热度高，而 PubChem 通道带热度先验，属于选样偏差；
2. main_first 在公开榜掉分的原因不是类别配比，而是化学空间偏移：在没见过的天然产物上，Engine 1 的 FPNet Top-1 从 68% 降到 33%（plan-review §1.3），而 safe 伪测试里 Engine 1 偏强。

因此加入 enveda-np-examples 面板（np-lib，250 个天然产物，仪器与处理流程与测试相同，答案全部在池内）作为与真实测试化学空间一致的验证。先用当前 pipeline 重跑（`nplib_v2`，覆盖 np_clean），检查 main_first 在上面是否也掉分。若是，选模型以 np-lib 为主，f\*np safe/hide 作训练数据与辅助。

np-lib 重跑结果（`nplib_v2`）：V2 0.958，Hit@1 93%，main_first +0.018。np-lib 模拟第 1 类（答案在别的仪器上有公开谱），库匹配几乎直接命中，复现不了公开榜掉分，不能用来选模型。np-nps（结构已知、无公开谱）正在重跑（`npnps_v2`）。

np-nps 重跑结果（`npnps_v2`）：V2 0.908，Hit@1 85%，main_first +0.060；safe 训练的重排器 +0.05，hide 或混合训练的 −0.10～−0.81。np-nps 也远比公开榜（0.468）容易，可能是 pipeline 的部分组件（神经打分器、Engine 1 排序器、面板调参）见过 np-examples，两块面板都不能用来选模型。到目前为止，只有 hide 伪测试与公开榜方向一致。

main_first 的分解（`probe_variants.py`，相对 V2 的配对差）：

| 变体 | f0 safe | f1 safe | f0 hide | f1 hide | np-lib |
| --- | --- | --- | --- | --- | --- |
| pc_last：PubChem 独有候选挪到最后 | +0.004 | +0.007 | −0.070 | −0.063 | +0.005 |
| e2_last：Engine 2 独有候选挪到最后 | +0.005 | +0.008 | +0.003 | +0.007 | 0.000 |
| nonmain_last：主列表外的候选挪到最后，保留 V2 顺序 | +0.012 | +0.025 | −0.106 | −0.095 | +0.007 |
| main_first：再按主列表名次重排 | +0.102 | +0.157 | −0.074 | −0.069 | +0.018 |

main_first 在 safe 上的提升主要来自"按主列表名次重排"，在 hide 上的损失来自"PubChem 候选后移"。公开榜探针 pc_last（提交中）用来区分公开榜上 −0.097 的来源：pc_last 接近 0.468，则损失来自主列表重排（化学空间偏移）；pc_last 掉约 0.07，则来自 PubChem 类答案。

公开榜探针 pc_last（提交 57739457）：**0.427**（−0.041）。main_first 的 −0.097 约 0.04 来自 PubChem 候选后移，约 0.056 来自主列表重排；用 hide 上 pc_last 的 −0.067 换算，公开榜的池内比例 p 约 0.36。

PubChem 提前扫描（`pc_promote_scan.py`，5 fold safe+hide，把最好的 PubChem 独有候选提到第 k 位）：

| k | safe | hide | p=0.6 | p=0.5 | p=0.36 | p=0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| V2 | 0.695 | 0.245 | — | — | — | — |
| 1 | 0.399 | 0.466 | −0.089 | −0.037 | +0.035 | +0.118 |
| 2 | 0.667 | 0.328 | +0.016 | +0.027 | +0.043 | +0.061 |
| 3 | 0.683 | 0.284 | +0.008 | +0.013 | +0.020 | +0.028 |

k=2 在 p ≤ 0.6 时全部为正（CI 都 > 0），损益平衡点约 p = 0.75。已提交 pc_at2（`make_variant_notebook.py pc_at2`）。注意：hide 上 PubChem 通道偏强（热度选样偏差），公开榜收益可能小于本地估计。

## 4. 相关的公开工作

公开 pipeline 作者的 `casmi26-natural-product-panel-benchmark` 已有单通道的手调门控：`gate_eval.py` / `merge_eval.py` 在 lib / nospec / dbdrop 三个场景下，决定 PubChem 通道的第一名能否占第 1 位（规则 `S > 6 且 pop ≥ 5`）。区别：

- 他们只门控 PubChem 一个通道的一个位置，阈值手调，在 np-examples 面板上定（面板的模型可能见过这些分子）；
- 我们学习整个 top-25 的排序，门控覆盖所有引擎，在 fold-safe 的伪测试上跨 fold 训练与评估，并报告类别配比敏感性。

论文中必须引用这项工作。

## 5. 代码

- `src/casmi/rerank.py`：证据表 → 特征（候选级、组内相对、分子级门控、引擎一致性）。
- `scripts/rerank_cv.py`：跨 fold 训练与评估，报告与 V2、main_first 的配对差。
- `scripts/mix_eval.py`：E4/E5，safe 与 hide 两种伪测试按配比 p 合成 MRR；训练集可选 safe / hide / mix；由 main_first 的公开榜差值反推隐藏测试的池内比例 p。
- `src/casmi/postproc.py`：main_first。
- `scripts/make_hidden_notebook.py`、`scripts/make_hide_keys.py`、`scripts/wsl_eval_hide_batch.sh`：E3 的"答案不在池内"本地运行（只用于本地，禁止提交）。
- `scripts/make_variant_notebook.py`：把后处理插进公开 notebook。
