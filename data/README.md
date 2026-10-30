# 本地输入路径与字段

本目录没有数据。空 `.gitkeep` 只让 Git 保留目录层级。

标准化记录按以下相对路径读取：

```text
data/broad_category_splits_standardized_6/
  <category>/
    train.json    # 建模期记录；不随仓库提供
    test.json     # 时间外记录；不随仓库提供
```

`<category>` 的原始键为 `sleep_dermatologic`、`sleep_emotional`、`sleep_fatigue_deficiency`、`sleep_neurologic`、`sleep_oncologic`、`sleep_urinary`。这些是代码中的类别标识，不是数据记录。

每个 JSON 文件为记录对象列表。下表仅描述字段，不提供任何样例记录：

| 字段 | 用途 |
|---|---|
| `患者个人id` | 构图时建立患者节点 |
| `门诊号` | 构图及重复就诊检查使用的标识 |
| `西医诊断` | 诊断节点与队列统计 |
| `标准处方药材名称` | 药材节点、处方共现与候选验证 |
| `四诊对应证候_标准名称` | 证候节点 |
| `舌苔脉象_标准名称` | 舌脉节点 |

其余统计字段、日期字段及缺失处理规则以各脚本的字段解析函数为准。药材通常以 `、` 分隔；各入口的分隔符兼容逻辑保留自原代码。

输出和派生输入继续使用原相对路径：

| 路径 | 内容；均未随仓库提供 |
|---|---|
| `outputs/knowledge_graph/` | 节点表、边表、图谱统计 |
| `outputs/kg_link_prediction/target_positive_edges.csv` | 提取出的正关联边 |
| `outputs/splits/` | `train_edges.csv`、`valid_edges.csv`、`test_edges.csv`、`positive_edges_with_split.csv` |
| `outputs/result_v1/models/` | 本地训练的检查点 |
| `outputs/result_v1/recent_models/ultra/` | ULTRA 数据转换、微调及排名 |
| `outputs/result_v1/recent_models/kg_icl/` | KG-ICL 评分 |
| `outputs/result_v1/selected_scores/` | 各类别入选模型分数 |
| `outputs/result_v1/temporal_validation/` | 药材及基础方时间外评价 |
| `outputs/formula_strategy_comparison/` | 药材池策略比较 |
| `outputs/revision_experiments/` | 独立重复和补充分析 |
| `submission/source_data/`、`submission/supplementary_data/` | 绘图脚本需要的本地汇总 CSV |

`category.py` 的分数读取函数优先寻找 `comorbidity_<category>_selected_herb_scores.csv`，需要至少含有 `herb_name` 和 `score` 两列。该函数也保留旧 ConvE 分数字段/文件名兼容逻辑。

部分历史默认路径使用 `outputs/kg_link_prediction/`。主流程命令显式指定 `outputs/splits`、`outputs/result_v1/selected_scores`，没有更改原始数据目录。原始目录不会通过本仓库自动下载或恢复。
