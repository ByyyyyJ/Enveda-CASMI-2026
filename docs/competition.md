# Enveda CASMI 2026：竞赛运行说明

## 赛题与评测

- 官方名称：Enveda CASMI 2026 – Molecule ID From Mass Spectra。
- 目标：给定 MS/MS 谱图，为每个 `molecule_id` 输出最多 25 个按置信度排序、以分号分隔的候选 SMILES。
- 指标：MRR@25。评测前会以 RDKit 规范化互变异构体并比较 InChIKey 前 14 位，因此立体化学和互变异构差异不计分。
- 数据概览（以下载当日官方版本为准）：本工作区当前版本的 `train.parquet` 有 539,608 行，`test.parquet` 有 213 行，`sample_submission.csv` 有 100 个待预测的 `molecule_id`。精确字段和类型见自动生成的 `docs/data-dictionary.md`；不要依赖旧的网页摘要或其他版本的行数。
- 新颖度分层：library、no public spectra、novel。验证应按此类分层，不能只看随机切分。

官方近期曾更新训练数据以纳入 water-loss adducts；重新下载后必须运行 `python scripts/verify_data.py`，并比较 `data/raw/MANIFEST.json`。

## 提交约束

最终提交必须由 Kaggle Notebook 产生：CPU 或 GPU 运行时间均不得超过 9 小时，Internet 必须关闭，产物必须为 `submission.csv`。免费且公开可获得的外部数据和预训练模型允许使用。Notebook 应在 Kaggle 上重新运行并通过 `scripts/check_submission.py` 等价检查后再提交。

本地 RTX 5080 是开发加速资源，并不占用 Kaggle 的 9 小时预算；最终耗时仍需在 Kaggle GPU Notebook 实测。

## 链接

- [Competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
- [Data](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
- [官方数据更新讨论](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471)
- [Code competition FAQ](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)
- [Kaggle notebook debugging](https://www.kaggle.com/code-competition-debugging)
