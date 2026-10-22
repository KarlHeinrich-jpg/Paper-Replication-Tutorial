# 图表文件说明

本目录的 5 张图从正式发表版 PDF（`../../sustainability-18-09871.pdf`）中按矢量方式裁剪，
每张同时提供：

- `.pdf`：矢量版，适合 LaTeX / 排版复用；
- `.png`：300 dpi 位图预览。

| 文件 | 论文图号 | 内容 |
|---|---|---|
| `fig_conceptual_model_clean` | Figure 1 | 概念模型与 H1–H4 |
| `fig_u_students` | Figure 2 | 学生：AI 素养分位下的使用—感知学习能动性曲线 |
| `fig_u_teachers` | Figure 3 | 教师：AI 素养分位下的使用—教学创新胜任力曲线 |
| `fig_literacy_dimensions_students` | Figure 4 | 学生：四维度素养 × 使用系数 |
| `fig_literacy_dimensions_teachers` | Figure 5 | 教师：四维度素养 × 使用系数 |

这些文件名与 `../sustainability-4495862.tex` 中的 `\includegraphics` 引用一致，可直接用于重编译。
注意：上游源码使用 MDPI 官方文档类（`Definitions/mdpi.cls`），该文件不在本仓库中，编译前需从
[MDPI LaTeX 模板](https://www.mdpi.com/authors/latex) 获取。
