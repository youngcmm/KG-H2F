# 本地运行顺序

所有命令均从本仓库根目录运行，使用自备的标准化记录。下列说明只保留流程和路径，没有附带实验数据、词表、检查点或结果。

## 1. 建模期图谱与内部划分

```sh
python src/kg_construct.py
python src/kg_link_prediction/01_construct_target_edegs_v1.py
python src/kg_link_prediction/02_split_masked_edges_v2.py \
  --out-dir outputs/splits --seed 42 --valid-ratio 0.1 --test-ratio 0.1 --neg-ratio 5
```

`outputs/splits/` 为模型内部划分，独立于 `data/` 中的时间外 `test.json`。划分脚本按模型三元组去重，并检查跨划分重复、标签冲突和类型约束负例。

原研究使用固定划分。自己重新生成的数据可能与原研究协议断言不同；本代码包不含可用于核验原始划分的数据或哈希结果。

## 2. KGE 基线

```sh
python src/kg_link_prediction/03_train_kge_multiseed.py \
  --split-dir outputs/splits --out-dir outputs/result_v1 \
  --models TransE RotatE ConvKB CompGCN --split both \
  --epochs 50 --patience 5 --seeds 42 --device cpu --train-batch-size 0
```

训练器的检查点按验证 BCE 选择；模型间统一比较按验证集 typed-filtered MRR 选择。这两层选择规则保留原实现。

## 3. ULTRA 目标适配

先自行将官方预训练检查点放到 `external/ULTRA/ckpts/ultra_50g.pth`。来源及兼容修改见 `external/README.md`。

```sh
python src/kg_link_prediction/04_prepare_ultra_dataset.py
python src/kg_link_prediction/07_train_ultra_targeted.py \
  --seed 42 --epochs 6 --patience 2 --batch-size 8 --batches-per-epoch 20 \
  --trials target_full_lr1e-4 \
  --result-dir outputs/result_v1/recent_models/ultra/targeted_finetune
```

ULTRA 数据接口读取 `outputs/result_v1/recent_models/ultra/dataset/sleep_tcm/raw/`，其输入由正三元组转换产生。人工采样负例不作为 ULTRA 的输入正三元组；目标适配的负例策略由训练器控制。

适配脚本同时保存已计算的零样本药材分数，供下一步统一比较读取。原 `05_export_ultra_results.py` 在此版本为内部辅助模块，无独立运行入口。

## 4. KG-ICL 推理

将官方检查点放入 `external/KG-ICL/checkpoint/KG-ICL-5L/model_best.tar`。将第 3 步生成的 `train.txt`、`valid.txt`、`test.txt` 复制到保留的预处理输入路径：

```text
external/KG-ICL/run_your_own_dataset/your_dataset/
```

文件格式为 `head<TAB>relation<TAB>tail`。运行原预处理入口时使用其要求的工作目录：

```sh
cd external/KG-ICL/run_your_own_dataset
python data_process.py
cd ../../..
python src/kg_link_prediction/10_run_kgicl.py --layers 5
```

生成的实体映射、提示案例和图背景位于 `external/KG-ICL/processed_data/your_dataset/`。这些均为本地派生数据，不应上传。当前模型适配器使用 CPU 路径和 `valid/` 预处理目录生成完整候选分数。

## 5. 统一比较与下游发现

完成前述各模型产物后：

```sh
python src/kg_link_prediction/11_finalize_primary_comparison.py \
  --result-dir outputs/result_v1 --device cpu

python category.py \
  --score-dir outputs/result_v1/selected_scores \
  --output outputs/result_v1/temporal_validation/candidate_herbs_temporal_validation.csv \
  --bootstrap-times 2000 --bootstrap-top-k 30

python basic_discover.py \
  --score-dir outputs/result_v1/selected_scores \
  --output outputs/result_v1/temporal_validation/candidate_base_formulas_temporal_validation.csv \
  --candidate-top-k 400 --min-size 3 --max-size 4 --min-count 5 \
  --retain-top-k 100 --validation-top-k 30 --bootstrap-times 2000
```

`11_finalize_primary_comparison.py` 从验证集确定入选方法，然后将统一格式的分数写入 `selected_scores/`。`09_consolidate_model_comparison.py` 在此版本只提供共用指标函数，不作为历史模型汇总入口运行。

## 6. 无模型基线、策略比较与统计

无模型频繁项集基线分开执行发现和评价，保留建模期选择冻结：

```sh
python submission/code/discover_frequent_itemsets.py discover \
  --data-dir data/broad_category_splits_standardized_6 \
  --output outputs/figure_06_frequent_itemsets
python submission/code/discover_frequent_itemsets.py evaluate \
  --data-dir data/broad_category_splits_standardized_6 \
  --output outputs/figure_06_frequent_itemsets

python scripts/compare_formula_discovery_strategies.py discover
python scripts/compare_formula_discovery_strategies.py evaluate
python scripts/validate_formula_strategy_comparison.py

python scripts/run_cohort_description.py
python scripts/run_kg_statistics.py
```

药材池策略比较保留其原有统一评分口径，不能与 `basic_discover.py` 的原始 Top-400 分母口径逐行混同。其他独立重复与补充分析位于 `scripts/run_revision_stability.py`、`scripts/build_revision_experiments.py`；其输入路径以源码为准。

## 7. 图表源码

`submission/code/plot_figures.py` 从本地 `submission/source_data/`、`submission/supplementary_data/` 重绘图表；中文和扩展图入口保留在同一代码目录。只提供源码，没有携带 CSV、既有图件或自动恢复数据的程序。

```sh
python submission/code/plot_figures.py
```

以上真实数据步骤未在提炼过程中执行。已执行的无数据检查及范围见根目录 README。
