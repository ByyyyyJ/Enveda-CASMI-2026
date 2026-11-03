# Enveda CASMI 2026 交接说明（给 Codex）

本文件写于 2026-11-03，交接时的状态以这里为准。更细的实验记录在 `docs/research-plan.md`（按日期追加），论文计划在 `docs/paper-outline.md`。和用户交流请用简体中文；用户在 VSCode 里看回复，不渲染 LaTeX。

## 1. 比赛和目标

- Kaggle 比赛 `enveda-CASMI26-molecule-id-mass-spectra`：给 MS/MS 谱图（Bruker timsTOF），预测分子结构，每个分子最多交 25 个 SMILES。
- 指标 MRR@25。身份判定：RDKit 2026.03.3，`TautomerEnumerator().Canonicalize` 后取 InChIKey 前 14 位（去立体）。
- 代码比赛：提交的是 Kaggle notebook，隐藏测试大约 400 个分子。公开榜约占 31%，约 125 个分子，一个分子从第 2 升到第 1 约值 0.004。
- 每天最多 5 次提交，可选 2 个最终提交，截止日期 2026-12-14。
- 团队 "Eggs"，账号 `wyy817`，单人。
- 用户的目标分两部分：
  1. 比赛名次，奖牌看私榜；
  2. 一篇 ML 方向的论文（见第 8 节）。
- 用户以后走 ML/AI 方向。

## 2. 硬性规则（必须遵守）

用户原话："别把我盘删了"。

- **绝不递归删除。** 不用 `rm -rf`、`Remove-Item -Recurse`；路径里不能有可能为空的变量。唯一例外是 WSL 里的 `/kaggle/input`，它由 `scripts/wsl_kaggle_layout.sh` 用符号链接重建。
- **只在这些地方写文件：** 本仓库、`D:/casmi-data`、WSL 的 `/kaggle/working`。
- **不动原始数据：** `data/raw/` 和 `D:/casmi-data` 下的原始数据集目录都不改。新东西放新目录，例如 `pseudo/`、`overrides/`、`kernel_out/`。
- **覆盖或删除任何文件前先看内容。** 优先写新文件，而不是改原文件。
- **禁止的 git 和系统操作：** `git reset --hard`、`git clean`、force push、磁盘或分区命令、`wsl --unregister`。
- **不提交别人的笔记本。** `.gitignore` 已排除以下目录：
  - `notebooks/pub477*`
  - `notebooks/midx479*`
  - `notebooks/local_midx479*`
- **commit 信息不加 AI 署名**，也不加 Co-Authored-By。
- **数据许可：** 比赛数据是 CC BY-NC 4.0。论文只发表统计结果，不公开谱图。
- **Kaggle 提交：**
  - 用户允许不问就提交，但每天不超过 5 次；
  - 每次提交都要告诉用户提交了什么、为什么提交、得了多少分；
  - 只提交有依据的版本。
- **用户的工作习惯：**
  - 做完一步接着做下一步，不用按周排计划；
  - 出结果后可以直接看数据、进行下一个实验，不必等用户同意。

## 3. 环境

- Windows 11，GPU 是 RTX 5070。仓库在 `D:/OneDrive/University/IC/Kaggle/Enveda-CASMI-2026`，处于 OneDrive 同步目录内。
- 大数据在 `D:/casmi-data`，不在 OneDrive 里。路径定义见 `src/casmi/paths.py`。
- **Windows 侧：**
  - conda 环境 `casmi2026`，Python 3.12，rdkit==2026.3.3；
  - 运行命令用 `conda run -n casmi2026 python ...`，Kaggle CLI 用 `conda run -n casmi2026 kaggle ...`；
  - Kaggle 认证用环境变量 `KAGGLE_API_TOKEN`；
  - 不要接受 Anaconda 默认 channel 的 ToS。
- **WSL 侧（跑流水线用）：**
  - 发行版 `Ubuntu-24.04`，用户 `wyy`；
  - 环境 `~/miniforge3/envs/casmi2026`，版本对齐 Kaggle 镜像：torch 2.10 cu128、numpy 2.0.2、pandas 2.3.3，完整列表见 `env/kaggle_pip_freeze_20261026.txt`；
  - 用 `bash scripts/wsl_kaggle_layout.sh [COMP_DIR]` 搭出 `/kaggle/input`，用 `scripts/run_notebook.py` 原样执行 Kaggle notebook；
  - 从 Git Bash 跑仓库脚本：`bash scripts/wsl_py.sh scripts/X.py args`；
  - 从 PowerShell 往 WSL 传带 `$` 的命令时，变量会被吞掉。办法是先写成 `.sh` 文件再执行。
- **在 WSL 里跑一次完整流水线大约 75–90 分钟，期间 GPU 被占用。** 开始新的 GPU 任务前，先用 `wsl -d Ubuntu-24.04 -- pgrep -af 'calib_batch|run_notebook'` 检查有没有任务在跑。

## 4. 当前成绩和提交历史

最好成绩是 **midx479，公开榜 0.479**（提交 57839053，kernel `wyy817/casmi26-midx479`）。它目前也是最终提交，备选是 pub477（0.477）。

2026-11-02 的公开榜：

| 档位 | 名次 | 分数 |
|---|---|---|
| 金牌线 | 第 19 名 | 0.484 |
| 银牌线 | 第 228 名 | 0.479 |
| 我们 | 第 236 名 | 0.479 |

共 4570 支队伍。0.479 是 377 支队伍并列，同分按提交时间排名。私榜的名次会大幅洗牌。

| 提交 | 内容 | 公开榜 |
|---|---|---|
| 57680635 | V2 复现（公开流水线 romanrozen 0.468） | 0.468 |
| 57712274 | main_first 探针 | 0.371 |
| 57739457 | pc_last 探针 | 0.427 |
| 57750469 | pc_at2 | 0.459 |
| 57774403 | e2_last 探针 | 0.469 |
| 57782126 | e2off_first 探针 | 0.340 |
| 57788945 | pub477（busyaprime 0.477 的副本） | 0.477 |
| 57808031 | pub477 + 吸收证据 | 0.468 |
| 57839053 | midx479（baidalinadilzhan 0.479 的副本） | **0.479** |
| 57927335 | midx479 + ICEBERG 在 timsTOF 上微调 | 0.475 |

流水线大致结构：
- Engine 1：FPNet 加 LightGBM 排序器，候选池约 71 万个结构；
- Engine 2：COCONUT、ChEBI、LIPID MAPS、PubChemLite；
- PubChem 通道；
- ICEBERG 和 GLACIER 谱图模拟器；
- 神经打分器。

各路结果用 RRF 融合，得到 V2。往上依次叠加：
- **pub477**：片段检查只负责换第 1 名，第 2–25 名用神经权重 0.75。
- **midx479**：在 pub477 上再加文献先验，作用于第 2–25 名，再做按指标身份的去重和池外候选补位。

这些代码来自别人的公开 notebook（Apache-2.0），我们本地保存的副本在 `notebooks/` 下面，不提交。

## 5. 已关闭的方向（别再重复）

1. **重排类改动已经饱和。** w_mid 取 0.70、0.75、0.80，公开榜都是 0.472。让神经模型移动第 1 名会掉分：
   - 第 1 名权重 0.50：0.448；
   - 第 1 名权重 1.00：0.359。
2. **吸收证据。** 规则是：候选有同加合物的训练谱图、但和查询都不像，就往后放。它在 np-examples 上 AUC 0.92，上榜却是 −0.009。
3. **同分子式异构体打分器**（`src/casmi/isomer.py`，`scripts/isomer_*.py`）。对随机同分子式负样本有效，但流水线给出的候选难得多，换成这些候选后它就没用了。
4. **ICEBERG 强度头在 timsTOF 上微调**（`scripts/ice_*.py`，私有数据集 `wyy817/casmi26-iceberg-tims`）。本地类药面板 +0.028，天然产物面板持平，上榜 −0.004。
5. **公开前沿那边的失败。**
   - 长时间训练的神经模型：−0.019。
   - NX 异构体重排器：−0.058。
   - 用伪测试的类别配比去拟合公开榜：是巧合，换一个探针方向就反了。
6. **不要降权训练集结构。** 官方说第 1 类答案可以通过训练谱图找到。

共同的教训：本地评估上的小增益（0.005 以内）不能迁移到榜上。原因有两个：一是组件训练数据泄漏，MassSpecGym 里包含 250 个 np-examples 结构中的 236 个；二是伪测试的构造方式本身有偏差。

## 6. 正在进行：代理校准

**目的：** 找一个本地评估，让它对已知公开榜分数的改动给出和榜上一致的方向。以后的新方法只用这个评估来筛选。

**做法：**
- `scripts/make_local_variant_notebook.py` 生成 `notebooks/local_midx479*` 两个本地版笔记本，只能本地用，**绝不提交到 Kaggle**。它做两处改动：
  - 打进藏答案的补丁，补丁只在设置了 `CASMI_HIDE_KEYS` 时生效；
  - 另存片段检查的输出 `submission_physical.csv`。
- `scripts/wsl_calib_batch.sh NB:FOLD:KIND ...` 在伪测试上跑。
  - `NB` 是 `midx` 或 `ice`；`FOLD` 是 `f0np` 到 `f4np`；`KIND` 是 `safe` 或 `hide`。
  - 输出到 `D:/casmi-data/kernel_out/cal_<NB>_<FOLD>_<KIND>/`。
- `scripts/proxy_calib.py [FOLDS]` 从每次运行离线重建 11 个有公开榜分数的版本：V0、V1、V2、CUSTOM、top050、top100、uni035、pub477、midx479、absence、ice。然后比较本地和榜上的排序相关性，以及 9 对改动前后的方向是否一致。
  - 重建出来的 V2 和 CUSTOM 必须和 notebook 自己写出的文件 100% 一致，脚本会打印核对结果。
- V2 上 5 个旧探针的本地结果用 `scripts/lb_mixture_fit.py` 汇总，在 WSL 里跑。

**伪测试说明：**
- `D:/casmi-data/pseudo/f<k>np/`：社区 identity-disjoint fold k 里的天然产物，每折约 260 个分子。它们的谱图已经从训练集删掉。
- 配套的 `f<k>_safe` override 把 10226-bit FPNet 换成不含 fold k 的版本。
- `safe`：答案在所有库里都在。
- `hide`：答案从所有候选库里都藏起来。

**f0np safe 的中间结果：**
- 动第 1 名的大改动，方向和幅度都对。例如 CUSTOM → top100，本地 −0.113，榜上 −0.113。
- 榜上差值在 0.005 以内的改动，4 个错了 3 个。
- 吸收证据在本地必然偏乐观：答案的训练谱图被删了，规则永远不会误伤正确答案。

**后台队列：**
- 2026-11-03 22:27 在 WSL 里启动，按顺序跑 15 次：
  - midx:f0np:hide
  - ice:f0np:safe、ice:f0np:hide
  - midx:f1np 和 midx:f2np，各 safe、hide
  - ice:f1np，safe、hide
  - midx:f3np 和 midx:f4np，各 safe、hide
  - ice:f2np，safe、hide
- 日志在 `D:/casmi-data/kernel_out/calib_batch2.log`。
- 判断进度：看哪些 `cal_*` 目录里已经有 `submission.csv`。
- 如果队列中断：用 `wsl_calib_batch.sh` 只补跑缺的那几项。要先在 WSL 里把脚本去掉 `\r`：`tr -d '\r' < scripts/wsl_calib_batch.sh > ~/calib_batch.sh`。
- 全部跑完以后：
  1. 运行 `conda run -n casmi2026 python scripts/proxy_calib.py`；
  2. 把结果补进 `docs/research-plan.md` 的"2026-11-03：代理校准"一节。

## 7. 建议的下一步

1. **完成代理校准。** 要回答的问题是：本地效应量达到多大以上，方向才可信；safe、hide 哪个更可信，或者两者组合是否更好。
2. **决定要不要做新方法。** 只有预期效果明显大于 0.005，而且在第 1 步校准过的评估上显著变好，才上榜。候选方案是**同分子式组内的列表式重排器**：
   - 依据：误差分解显示，约 91% 的损失来自同分子式异构体排错，正确答案 98.5% 在前 25 里；
   - 负样本必须用流水线实际给出的前 25 个候选，这是异构体打分器失败的教训；
   - 输入特征用 `neural_candidate_features.parquet` 和 `frozen_candidate_evidence.parquet` 里各引擎的排名，以及 ICEBERG、GLACIER、神经打分器的分数；
   - 已有代码可以参考：`src/casmi/rerank.py`、`scripts/rerank_cv.py`。
3. **对冲私榜。** 两个最终提交，一个放 0.479，另一个放思路不同但同样可靠的版本。
4. **定期查看公开 notebook。** 方法：按分数排序公开代码，有超过 0.479 的就复现，再在上面叠加。之前从 0.468 到 0.479 都是这样拿到的。

## 8. 论文

- 定位：ML 方向的评估和负面结果论文，题目暂定 "When Local Validation Lies"。
- 主题有三个：
  - 代理验证失效；
  - 组件级训练数据泄漏；
  - 小隐藏测试上的适应性过拟合。
- 投稿计划：
  1. 先投 ICLR 2027 的 workshop，首选 ICBINB 或 FM4Science，截止大约 2027-02-01，以各 workshop 征稿页为准；
  2. 再扩展成长文投 TMLR，*J Cheminform* 作备选。
- 比赛结束前不公开发表。
- 赛后要用私榜（约 270 个分子）把每个提交过的改动补评一遍，做配对 bootstrap。
- 大纲和要补的实验见 `docs/paper-outline.md`。代理校准的结果可以直接写成论文的正面结论：本地伪测试在多大效应量以上可信。

## 9. 常用命令和文件

- **提交并等分数：**
  ```
  conda run -n casmi2026 kaggle kernels push -p notebooks/<dir>
  conda run -n casmi2026 python scripts/submit_when_done.py wyy817/<kernel> <version> "<message>"
  ```
  打分通常要 4.5–5 小时。
- **查提交记录：** `conda run -n casmi2026 kaggle competitions submissions -c enveda-CASMI26-molecule-id-mass-spectra`
- **给提交文件打分：** `python scripts/score_submission.py SUB.csv D:/casmi-data/pseudo/<mode>_solution.csv`
- **在伪测试上跑 V2 笔记本：**
  - `bash scripts/wsl_eval_batch.sh NAME:MODE[:OVERRIDES]`：普通版；
  - `bash scripts/wsl_eval_hide_batch.sh`：藏答案版。
- **文档：**
  - `docs/research-plan.md`：所有实验的记录和结论，按日期组织；
  - `docs/plan-review.md`：公开生态和方案评审；
  - `docs/competition.md`：比赛官方信息。
  - `docs/model-components.md` 是已归档的旧方案，只作参考。
