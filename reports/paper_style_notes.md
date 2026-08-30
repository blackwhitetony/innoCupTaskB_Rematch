# 初赛论文风格笔记（仿写依据）

> 来源：`D:\Documents\建模\paper\`（初赛 B 题，LaTeX 工程）

## 1. 工程结构
- `main.tex`：文档类 `ctexart`（12pt a4），preamble 集中管理宏包；正文用 `\input{chapters/*.tex}` 拆分。
- 分章：`00_abstract / 01_background / 02_data / 03_task1 / 04_task2 / 05_task3 / 06_discussion / 07_conclusion / 08_ai_declaration / 09_appendix`。
- 页眉：`\fancyhead[C]{TeamB DATA2600409 \quad page \thepage \ of \pageref{LastPage}}`（fancyhdr + lastpage）。
- 参考文献：natbib `numbers,sort&compress` + `unsrtnat` + `refs.bib`。
- 编号：`\numberwithin{figure/table/equation}{section}`（图/表/公式按节编号）。
- 摘要页第 1 页（无页码）、目录第 2 页，正文第 3 页起页码重置为 1。

## 2. 章节模板
- **摘要**：一段话点明数据集+任务 → 难点 → 预处理要点 → 每任务的模型+指标 → 方法论贡献 → 开源仓库 → `关键词：`。
- **问题背景与重述**：赛题背景 → 三个子问题（表格：目标/类型/权重）→ 评分公式 → 总体技术路线（enumerate，含固定种子、分层划分、折内变换、基线模型、一次性评估）。
- **数据探索与预处理**：数据概况（字段数、分类/二值/连续、缺失、异常）→ 目标分布表 → 清洗流水线（enumerate：缺失指示、异常置 NaN、分类 Missing、中位数填补、缩尾、对数、有序/标签编码）→ 最终规格（行列数、零缺失）。
- **每个任务**：问题定义与难点 → EDA（含图）→ 特征工程（交互特征表：名称/公式/业务含义，$\blacklozenge$ 标记自构特征）→ 训练策略与超参 → 实验结果（验证集方案对比表、消融、混淆矩阵、特征重要性表）。
- **综合讨论**：三任务横向对比表 → 统一技术栈优势 → 特征工程反思 → 负结果学术价值。
- **结论与展望**：主要发现（分任务）→ 方法学贡献 → 局限性（列点）。
- **AI 声明**：`\section*{}`，列出工具+版本+厂商+用途+日期；声明 AIGC 比例 <30%、相似比 <50%，附 AI 使用说明文件。
- **附录**：源码清单表、运行环境表、开源仓库、模型调用方式（verbatim）。

## 3. 格式约定
- 表格：`booktabs`（`\toprule/\midrule/\bottomrule`），`[H]` 强制定位，`\label{tab:xxx}`。
- 图：`graphicx`，`width=0.8\textwidth`，必要时 `subcaption` 子图；`\label{fig:xxx}`。
- 公式：`amsmath`，`\label{eq:xxx}`，`\allowdisplaybreaks`。
- 中文字体：ctexart 自动；行距 `\onehalfspacing`。
- 强调用 `\textbf{}`；自构特征用 `$\blacklozenge$` 标注。

## 4. 复赛论文适配要点（与初赛差异）
- 初赛=三分类任务（CatBoost+准确率）；**复赛=三回归任务（稀疏公式 + adj-R²）**。
- 复赛核心叙事：EDV 揭示数据生成机制（近似确定性公式）→ 稀疏闭式公式优于复杂树模型 → 严格嵌套 OOF 级联防泄露 → P4 树模型 Bootstrap CI 全负（负结果）。
- 保留初赛的「负结果诚实报告」「EDV 先行」「消融对照」精神，但把 CatBoost 换成「公式/OLS 主模型 + LightGBM 残差确认」。
- 指标表：T1 adj-R² 0.876298、T2 0.917817（Oracle 0.9994）、T3 0.905416。
