# 公开路线与基线地图

本表区分“可作为正式赛用候选”和“仅作能力背景”。最终纳入方案前，必须复查许可证、训练数据来源、权重可获得性，以及 Kaggle 离线 9 小时内的可运行性。

| 路线 | 作用 | 公开状态与注意事项 | 初步定位 |
| --- | --- | --- | --- |
| [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) / [论文](https://arxiv.org/abs/2410.23326) | 检索、de novo、谱图模拟的统一 benchmark | 代码与基准公开；任务设定接近，应优先用来校准评估与切分 | 高优先级参考 |
| [MIST](https://github.com/samgoldman97/mist) | 预测分子指纹并做对比检索 | 开源；适合“候选库 + MS/MS 重排序”路线 | 高优先级候选 |
| [MIST-CF](https://github.com/samgoldman97/mist-cf) / [论文](https://arxiv.org/abs/2307.08240) | 已知/候选分子式与 adduct 的排序 | 有代码与预训练资产；需要确认公式信息是否在本赛题可用、避免标签泄漏 | 高优先级候选 |
| [MSNovelist](https://github.com/meowcat/MSNovelist) | 分子式约束下的结构生成 | 项目建议使用 SIRIUS 6 集成；对真正新颖结构命中率通常受限 | 方法学参考 |
| [MS-GPT](https://github.com/VIKI623/MS-GPT) | 谱图到 SMILES 的生成式模型 | 代码/检查点公开声明；需实测显存、Kaggle 离线包与 9 小时时限 | 值得评估 |
| [Spec2Mol](https://github.com/KavrakiLab/Spec2Mol) | 直接 spectrum-to-SMILES | 依赖较旧，且编码器关联 NIST 2020 受限数据，难作直接赛用基线 | 背景比较 |
| [ICEBERG / ms-pred](https://github.com/coleygroup/ms-pred) | 前向碎裂模拟与候选检索组件 | 有公开组件/权重信息；基础 NIST20 模型的授权需逐项核验 | 组件候选 |
| SIRIUS 等商业/受限工具 | 公式推断、候选建议 | 可做能力背景；不得默认作为离线、可再分发的赛用依赖 | 仅背景 |

## 建议的比较维度

1. **任务匹配**：是否支持多谱图聚合到 `molecule_id`，是否输出 Top-25 结构，而非只输出分子式或单一 SMILES。
2. **新颖度**：分别报告 library、no-public-spectra、novel；候选库检索的整体分数不能掩盖 novel 类失败。
3. **信息合规**：明确训练集、外部谱库、分子式/adduct 和预训练权重来源；禁止用测试标签、不可公开数据或网络请求。
4. **提交可运行性**：离线、可缓存、可在 Kaggle GPU Notebook 9 小时内完成，且能输出严格合法的 `submission.csv`。
5. **化学规范化**：候选去重与评估前使用与官方相近的 RDKit 互变异构/连接性规范化；不要为立体异构重复占据 Top-25。

## 尚不下载的内容

本轮不克隆这些仓库或下载权重。选择具体路线后，在隔离环境中固定 commit、许可证、权重哈希和数据许可证，再做离线 Kaggle 打包与时限测试。
