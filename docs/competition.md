# Enveda CASMI 2026：竞赛运行说明

以下内容来自官方页面（Overview / Data / Evaluation / Rules）和官方 metric notebook，核对日期 2026-09-24。技术方案见 [model-components.md](model-components.md)。

## 赛题与评测

- 官方名称：Enveda CASMI 2026 – Molecule ID From Mass Spectra。
- 目标：给定 MS/MS 谱图，为每个 `molecule_id` 输出最多 25 个按置信度排序、以分号分隔的候选 SMILES。预测以分子为单位，需要聚合同一分子所有谱图的证据。
- 指标：MRR@25，即所有分子 1/rank 的平均；只计第一个正确答案，未命中计 0。评测前用 **RDKit 2026.03.3** 的 `TautomerEnumerator` 规范化，再比较 InChIKey 前 14 位，因此立体化学和互变异构差异不计分。非法 SMILES 视为未命中。
- 提交拒收条件：缺少 `molecule_id` 或 `smiles` 列、文件为空、有空值、`molecule_id` 重复、某分子超过 25 个候选。
- 新颖度分三类：1. 有公开参考谱；2. 没有公开谱，但结构在 PubChem 或 COCONUT 中；3. 结构不在 PubChem 中。三类的比例在比赛期间保密。验证应按此分层，不能只看随机切分。

## 数据

- **本地 `test.parquet` 是占位样本**：由训练集中的样本组成，重新运行 Notebook 时会被替换为隐藏测试集。它的统计不代表真实测试集，只能用来检查提交流程。
- **隐藏测试集**（官方描述）：约 1,500 张谱、约 400 个分子，每分子 1–16 张谱（中位 3 张），全部为 Bruker timsTOF；单同位素质量 157–1,159 Da（中位 348）；10 种 adduct，包括脱水类。只做了最少的清洗：删除 precursor 质量不符的谱、基峰低于 1,000 raw counts 的谱，以及高于 precursor + 2 Da 的峰。
- **本工作区的下载版本**：`train.parquet` 有 2,539,608 行，`test.parquet`（占位）有 1,213 行，`sample_submission.csv` 有 400 个 `molecule_id`。精确字段和类型见自动生成的 [data-dictionary.md](data-dictionary.md)；不要依赖旧的网页摘要或其他版本的行数。
- 最接近测试集的训练数据是 `enveda-np-examples`（同一仪器、同一处理流程的 250 个天然产物）。`enveda-180` 虽然是同一仪器，但化学空间是合成类药分子。

官方近期曾更新训练数据以纳入 water-loss adducts；重新下载后必须运行 `python scripts/verify_data.py`，并比较 `data/raw/MANIFEST.json`。

## 提交约束与规则

- 最终提交必须由 Kaggle Notebook 产生：CPU 或 GPU 运行时间均不得超过 9 小时，Internet 必须关闭，产物必须为 `submission.csv`。
- RDKit 不在 Kaggle 的 docker 镜像中，需要把 `rdkit==2026.3.3` 的 wheel 作为 Dataset 附上并离线安装。
- 外部数据和预训练模型须免费、公开、所有参赛者可同等获得。用到的开源代码须为 OSI 认可且不限制商业使用的许可；获奖方案须以 MIT 许可提供。
- 禁止对测试数据人工标注或人工预测。
- 每天最多 5 次提交，最终可选 2 个；队伍最多 5 人。
- 竞赛数据许可为 CC BY-NC 4.0，不得向队外传播。
- 时间线：报名与组队截止 2026-12-07；最终提交截止 2026-12-14（UTC 23:59）。
- Notebook 应在 Kaggle 上重新运行，并通过 `scripts/check_submission.py` 等价检查后再提交。该脚本目前只检查列名与行数，拒收条件的其余几项还需补齐。

本地 GPU（RTX 5070；队友 RTX 5080）是开发加速资源，不占用 Kaggle 的 9 小时预算；最终耗时仍需在 Kaggle GPU Notebook 上按约 1,500 张谱实测。

## 链接

- [Competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
- [Data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
- [Rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)
- [官方 metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)
- [官方数据更新讨论](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471)
- [Code competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)
- [Kaggle notebook debugging](https://www.kaggle.com/code-competition-debugging)
