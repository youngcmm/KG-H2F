# 第三方模型源码

本目录只保留主流程调用的第三方源码、兼容代码和原始许可证。原始数据、处理后的图、缓存、模型权重和运行结果均未随代码发布；空目录中的 `.gitkeep` 只用于保留相对路径。

| 目录 | 上游 | 许可证 | 本项目调用方式 |
|---|---|---|---|
| `ULTRA/` | https://github.com/DeepGraphLearning/ULTRA | MIT，见 `ULTRA/LICENSE` | `src/kg_link_prediction/05_export_ultra_results.py`、`07_train_ultra_targeted.py` |
| `KG-ICL/` | https://github.com/nju-websoft/KG-ICL | GPL-3.0，见 `KG-ICL/LICENSE` | `src/kg_link_prediction/10_run_kgicl.py` |

ULTRA 的本地来源记录固定于提交 `427966ad8ed60420eef034063d44f3153addff90`。KG-ICL 的本地来源记录为 2026-09-04 获取的上游 `main` 分支，原项目未记录精确提交号。本代码提炼没有重新获取或更新上游。

## ULTRA 的保留内容

保留关系/实体模型、数据集接口、训练入口及其依赖。项目原有的 `SleepTCM` 数据集适配器读取本地固定划分，不会自动下载本项目的数据。保留的兼容修改包括：

- `torch_scatter/` 使用原生 PyTorch 实现模型调用的 scatter 接口。
- 数据集缓存加载显式设置 `weights_only=False`，维持原项目的缓存读取行为。
- PyG 版本解析兼容带后缀的版本号；CPU 使用标准消息传递路径。
- `ultra/rspmm/source/` 保留源码形式的可选 C++/CUDA 扩展，不包含编译产物。

运行时依赖 PyTorch、PyTorch Geometric、NumPy；官方训练入口还使用 PyYAML、Jinja2 和 EasyDict。项目根目录的依赖文件列出安装项。GPU 扩展另需匹配的编译工具链；当前主流程可以使用 CPU 路径。

预训练权重需要自行从上游获取，并保持原来的位置：

```text
external/ULTRA/ckpts/ultra_50g.pth
```

历史多检查点比较还会读取同目录下的 `ultra_3g.pth`、`ultra_4g.pth`。本目录不包含这些权重。

## KG-ICL 的保留内容

保留当前 CPU 推理调用的数据加载器、实体/提示编码器、预处理工具，以及项目原有的原生 PyTorch scatter 兼容层。当前项目适配器设置 `use_rspmm=False`；未纳入未被该推理路径调用的训练/评测入口及可选 CUDA 实现。

除 PyTorch 和 NumPy 外，此路径还需要 NetworkX、SciPy 和 tqdm。预训练权重需要自行放到原路径：

```text
external/KG-ICL/checkpoint/KG-ICL-4L/model_best.tar
external/KG-ICL/checkpoint/KG-ICL-5L/model_best.tar
external/KG-ICL/checkpoint/KG-ICL-6L/model_best.tar
```

预处理输入路径为 `external/KG-ICL/run_your_own_dataset/your_dataset/{train,valid,test}.txt`，每行格式为 `head<TAB>relation<TAB>tail`。在 `external/KG-ICL/run_your_own_dataset/` 下运行 `python data_process.py`，生成 `external/KG-ICL/processed_data/your_dataset/`。当前项目推理入口读取其中的 `valid/` 子目录。以上路径仅保留目录结构，未包含任何三元组、实体映射、提示案例或处理缓存。

第三方代码继续适用各目录中的原始许可证；目录内的本地兼容修改保留自原项目，不代表上游的未修改发行版。
