# 生成式AI高等教育复现包

本目录对应论文：

> Zhang, Xinchen; Wang, Yile; Li, Chunbing. **Calibrating Generative AI Use for Sustainable Higher Education: Cross-Sectional Evidence on Perceived Capability and Dependence.** *Sustainability* **2026**, *18*(19), 9871.
> DOI: [10.3390/su18199871](https://doi.org/10.3390/su18199871) · 投稿编号 `sustainability-4495862` · 2026-09-26 正式发表

本复现包整理自论文作者公开的仓库
[`KarlHeinrich-jpg/Calibrating-Generative-AI-Use-for-Sustainable-Higher-Education`](https://github.com/KarlHeinrich-jpg/Calibrating-Generative-AI-Use-for-Sustainable-Higher-Education)，
并在其基础上补充了**正式发表版 PDF** 与**从发表版 PDF 裁剪的五张矢量图**。原始英文/中文说明见
[`README_upstream.md`](README_upstream.md)（上游 README 逐字副本）。

## 一、论文速览

| 项目 | 内容 |
|---|---|
| 研究问题 | GenAI 任务使用频率与“感知角色能力”“AI 依赖”之间分别是什么关系 |
| 设计 | 横断面、学生/教师平行问卷；北京五所高校 |
| 样本 | 1,105 人（659 名学生 + 446 名教师） |
| 核心方法 | `lavaan` CFA；HC3 稳健标准误分层 OLS；倒 U 型正式检验；配对 bootstrap；嵌套 5×4 交叉验证与 TreeSHAP |
| 主要结论 | 使用频率与感知能力呈**倒 U 型**；AI 素养改变最高点位置；AI 依赖主要是一条由使用、习惯、信任驱动的**可加**路径 |
| 解释边界 | 横断面设计，估计的是**个体间关联**，不是个体最优使用水平，也不是因果效应 |

一句话概括：**“用得越多能力越强”并不成立**。在低使用区间，能力随使用上升；在高使用区间，关联转为下降；AI 素养越高，倒 U 型最高点越向右移动。与此同时，AI 依赖与能力是两条不同的路径。

## 二、目录结构

```text
生成式AI高等教育复现包/
├── README.md                          # 本文件
├── README_upstream.md                 # 上游仓库 README 逐字副本（英文 + 简体中文）
├── CITATION.cff                       # 机器可读引用信息
├── LICENSE                            # MIT（覆盖分析代码）
├── sustainability-18-09871.pdf        # 正式发表版 PDF（CC BY 4.0，40 页）
├── paper/
│   ├── sustainability-4495862.tex     # 录用版 LaTeX 源码（MDPI 返回稿，逐字保留）
│   └── figures/                       # 5 张图，矢量 PDF + 300 dpi PNG
│       ├── fig_conceptual_model_clean.pdf / .png
│       ├── fig_u_students.pdf / .png
│       ├── fig_u_teachers.pdf / .png
│       ├── fig_literacy_dimensions_students.pdf / .png
│       └── fig_literacy_dimensions_teachers.pdf / .png
└── supplementary/
    ├── sustainability-4495862-supplementary.zip   # 投稿时提交的原始代码包
    └── Analysis/                                  # 同一代码包的解压版，便于浏览
        ├── run.py / run.sh                      # 全流程驱动
        ├── requirements.txt                     # Python 依赖（固定版本）
        ├── config/                              # 题项、控制变量、种子等全部分析设定
        ├── models/                              # 33 个 lavaan 模型语法 + 模型清单
        ├── R/                                   # CFA、信度、不变性、因子得分
        ├── src/                                 # 打分、推断、稳健性、预测、绘图、审计
        └── docs/OUTPUT_MANIFEST.csv             # 输出文件 → 论文表/节 的对应表
```

## 三、图表清单

| 文件 | 论文位置 | 内容 |
|---|---|---|
| `fig_conceptual_model_clean` | Figure 1 | 概念模型：使用频率、AI 素养、感知能力、AI 依赖与 H1–H4 |
| `fig_u_students` | Figure 2 | 学生：不同 AI 素养分位下的使用—感知学习能动性曲线 |
| `fig_u_teachers` | Figure 3 | 教师：不同 AI 素养分位下的使用—教学创新胜任力曲线 |
| `fig_literacy_dimensions_students` | Figure 4 | 学生：四个素养维度分别与使用交互的系数及区间 |
| `fig_literacy_dimensions_teachers` | Figure 5 | 教师：四个素养维度分别与使用交互的系数及区间 |

图表从正式发表版 PDF 中按矢量方式裁剪，PDF 保留矢量元素，PNG 为 300 dpi 预览。原始上游仓库只提供 `.tex` 与代码，不含图文件；因此这里的图表是**发表版的重建副本**，用于本地阅读和排版参考。

## 四、复现步骤

### 4.1 环境

- Python 3.10+，安装 `supplementary/Analysis/requirements.txt` 中的固定版本依赖：
  `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `joblib`
- R 4.1+，仅测量模型阶段需要，依赖 `lavaan`（≥ 0.6-17）与 `jsonlite`

```bash
cd supplementary/Analysis
pip install -r requirements.txt
Rscript R/install.R
```

### 4.2 输入数据

需要一份包含两个队列的 CSV，字段与 `config/scoring_dictionary.csv` 完全对应：

- `respondent_id`、`cohort`（`student` / `teacher`）、`institution`（五所高校代码）；
- 所有题项列，取值为 1–7 的整数编码；
- 背景控制变量：`age`、`study_year`、`ai_tenure`、`overall_frequency`、`gender`、`discipline`（学生）；`teaching_experience`、`academic_rank`、`ai_tenure`、`overall_frequency`、`gender`、`discipline`（教师）。

**参与者级原始数据不随论文或本复现包公开**：知情同意材料未授权无限制公开，去标识记录与拟合模型文件同样不公开。数据访问需联系通讯作者 Chunbing Li（licb@th.btbu.edu.cn），并遵守原知情同意与机构审批条件。因此本包中的代码可以完整审阅，但在没有授权数据的情况下不能端到端跑出论文数字。

### 4.3 运行

```bash
# 全流程：打分 → 测量（R）→ 推断 → 稳健性 → 机构 → 预测 → 报告
python run.py data.csv outputs/

# 只跑部分阶段（示例）
python run.py data.csv outputs/ --stages inference institution report --bootstrap 200

# 输出一致性审计
python src/audit_outputs.py outputs/
```

全流程在笔记本上需要数小时，主要耗时在嵌套 XGBoost / 随机森林 / LightGBM 搜索；`--jobs -1` 可并行。`run_record.json` 会记录软件版本、输入文件 SHA-256、随机种子与运行参数。随机种子固定为 `251101`（折划分与所有 bootstrap），内层搜索与模型种子为 `251101 + 外层折号`。

### 4.4 环境变量与路径

- 所有命令均在 `supplementary/Analysis/` 下执行；
- 输出目录必须是**新建的空目录**；
- 缺 focal 题项或控制变量时程序会显式报错，不会静默删行（完整案例队列在论文 3.2 节已固定）。

## 五、结果与解释边界

论文的核心数字（3 个种子/5000 次 bootstrap 口径）包括：

- 学生能力曲线最高点随 AI 素养从第 25 到第 75 百分位右移 **0.715** 个使用刻度（95% CI [0.525, 0.973]）；
- 教师对应右移 **0.643** 个使用刻度（95% CI [0.450, 0.883]）；
- AI 依赖的平方项与交互项增量分别为 **0.0024** 与 **0.0004**，90% 区间上界 **0.0058** 与 **0.0031**，低于探索性的 **0.01** R² 基准；
- 非线性 ridge 能恢复大部分相对线性 ridge 的能力预测增益。

这些是**个体间关联**，不能解读为“某个人用多少次最优”，也不能解读为“减少使用就能提升能力”的因果建议。论文讨论的“可持续性”是制度层面的理论微观基础，不是直接测量的变量。

## 六、许可与出处

- **分析代码**：MIT License，见 [`LICENSE`](LICENSE)。
- **论文正文与图表**：© 2026 作者；正式发表版以 **CC BY 4.0** 开放获取，可注明出处后复用。
- **上游仓库**：<https://github.com/KarlHeinrich-jpg/Calibrating-Generative-AI-Use-for-Sustainable-Higher-Education>
- **本复现包新增内容**：正式发表版 PDF、从 PDF 裁剪的五张图、本说明文件；`.tex`、代码包、`CITATION.cff`、`LICENSE` 与 `README_upstream.md` 均来自上游仓库。

完整性校验：

| 文件 | SHA-256 |
|---|---|
| `sustainability-18-09871.pdf` | `ec6867ed6dfa84018d8550234571916aed2b63d689087cc77c69b5d67466c272` |
| `paper/sustainability-4495862.tex` | `1fd5636cb550a89394203b11a9de5a13fa04d0b4f55057efe23868d2d8fa6819` |
| `supplementary/sustainability-4495862-supplementary.zip` | `3f455ac5b4c13925e17a4a9021191cc6303899eb21ce2a92eca3e7035e3aea97` |

## 七、引用

```bibtex
@article{zhang2026calibrating,
  title   = {Calibrating Generative {AI} Use for Sustainable Higher Education:
             Cross-Sectional Evidence on Perceived Capability and Dependence},
  author  = {Zhang, Xinchen and Wang, Yile and Li, Chunbing},
  journal = {Sustainability},
  year    = {2026},
  volume  = {18},
  number  = {19},
  pages   = {9871},
  doi     = {10.3390/su18199871}
}
```
