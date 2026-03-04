# AutoGluon 项目

本仓库为 [AutoGluon](https://github.com/autogluon/autogluon) 自动化机器学习框架的完整代码库，支持表格、时间序列、多模态等任务的自动训练、调参与部署。本文档采用标准格式，对项目结构、安装、使用与扩展做统一说明。

---

## 目录

- [项目简介](#项目简介)
- [主要模块](#主要模块)
- [环境要求](#环境要求)
- [安装](#安装)
- [项目结构](#项目结构)
- [使用示例](#使用示例)
- [扩展与自定义](#扩展与自定义)
- [文档与资源](#文档与资源)
- [开发与贡献](#开发与贡献)
- [许可证](#许可证)

---

## 项目简介

AutoGluon 由 AWS 开源，旨在通过少量代码完成高质量机器学习流程。本仓库包含其核心与各子模块源码，支持：

- **表格数据（Tabular）**：分类、回归，自动特征工程与模型选择。
- **时间序列（Time Series）**：单变量/多变量预测，支持概率预测与多模型集成。
- **多模态（Multimodal）**：图文等多模态数据的联合建模与预测。
- **EDA**：探索性数据分析相关工具。

仓库采用多包结构：`common`、`core`、`features` 为公共与核心依赖，`tabular`、`timeseries`、`multimodal` 等为按任务划分的子模块，可单独或整体安装与开发。

---

## 主要模块

| 模块 | 路径 | 说明 |
|------|------|------|
| **common** | `common/` | 公共工具与约定（如版本、日志）。 |
| **core** | `core/` | 核心抽象、训练流程、超参搜索等。 |
| **features** | `features/` | 特征工程与特征类型。 |
| **tabular** | `tabular/` | 表格预测（TabularPredictor）、表格模型与集成。 |
| **timeseries** | `timeseries/` | 时间序列预测（TimeSeriesPredictor）、时序模型与集成。 |
| **multimodal** | `multimodal/` | 多模态预测（MultiModalPredictor）。 |
| **eda** | `eda/` | 探索性数据分析。 |
| **autogluon** | `autogluon/` | 聚合包，依赖上述子模块，用于 `pip install autogluon` 式安装。 |
| **docs** | `docs/` | 文档源码。 |
| **examples** | `examples/` | 示例与教程脚本。 |

---

## 环境要求

- **Python**：3.10、3.11、3.12 或 3.13（以各子模块 `setup.py` 或文档为准）。
- **操作系统**：Linux、macOS、Windows。
- **依赖**：由各子模块的 `setup.py` / `pyproject.toml` 声明（如 numpy、pandas、torch、gluonts 等），详见安装步骤。

---

## 安装

### 方式一：安装官方发布的包（推荐）

```bash
pip install autogluon
```

会拉取 PyPI 上的聚合包，并安装其声明的各子模块依赖。仅用表格时可使用 `autogluon.tabular`，仅用时序时可使用 `autogluon.timeseries` 等（以官方文档为准）。

### 方式二：从本仓库源码安装（开发/自定义）

在项目根目录下，按依赖顺序安装：

```bash
pip install -e common/
pip install -e core/
pip install -e features/
pip install -e tabular/
pip install -e timeseries/
pip install -e multimodal/
# 可选：聚合包
pip install -e autogluon/
```

可根据需要只安装部分子模块；`timeseries` 依赖 `core`、`features` 等，安装顺序需满足各包 `setup.py` 中的依赖声明。

### 方式三：仅使用时间序列且需本地修改

若只开发或调试时间序列相关代码：

```bash
pip install -e common/
pip install -e core/
pip install -e features/
pip install -e timeseries/
```

确保运行时的 `PYTHONPATH` 或 `pip -e` 指向本仓库对应路径，以便导入 `autogluon.timeseries` 及本地修改生效。

---

## 项目结构

```
autogluon/                    # 项目根目录
├── README.md                 # 本说明文档
├── pyproject.toml            # 工具配置（ruff、codespell、pyright 等）
├── common/                   # 公共模块
│   └── src/autogluon/common/
├── core/                     # 核心模块
│   └── src/autogluon/core/
├── features/                 # 特征模块
│   └── src/autogluon/features/
├── tabular/                  # 表格预测
│   └── src/autogluon/tabular/
├── timeseries/               # 时间序列预测
│   └── src/autogluon/timeseries/
│       ├── predictor.py      # TimeSeriesPredictor
│       ├── learner.py
│       ├── trainer/
│       └── models/           # 内置与本地模型
│           ├── registry.py   # 模型注册表
│           ├── abstract/     # 抽象基类
│           ├── local/       # 本地/自定义模型（含 DETMModel、TimeXer 等）
│           ├── gluonts/
│           ├── chronos/
│           └── ...
├── multimodal/               # 多模态
│   └── src/autogluon/multimodal/
├── eda/                      # EDA
├── autogluon/                # 聚合包（用于 pip 安装）
├── docs/                     # 文档
├── examples/                 # 示例
└── CI/                       # 持续集成等
```

仓库中可能还包含示例脚本、配置与数据目录；具体入口与用法以各脚本自身注释或文档为准，本项目不指定唯一“主文件”。

---

## 使用示例

以下为各任务的标准 API 用法，不依赖仓库内某一具体脚本。

### 表格预测（Tabular）

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="target").fit("train.csv", presets="best_quality")
predictions = predictor.predict("test.csv")
```

### 时间序列预测（Time Series）

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

train = TimeSeriesDataFrame.from_path("train.csv")  # 需含 item_id、timestamp、目标列
predictor = TimeSeriesPredictor(prediction_length=7, target="value", freq="D")
predictor.fit(train_data=train, time_limit=300)
predictions = predictor.predict(train)
```

### 多模态预测（Multimodal）

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(label="label").fit(train_data, presets="best_quality")
predictions = predictor.predict(test_data)
```

更多参数、预设与数据格式见 [官方文档](https://auto.gluon.ai/stable/index.html)。

---

## 扩展与自定义

### 时间序列自定义模型（以 DETMModel 为例）

本仓库在 `timeseries` 中扩展了本地模型 **DETMModel**（基于 TimeXer），用于演示如何接入自定义时序模型：

- **实现位置**：`timeseries/src/autogluon/timeseries/models/local/ag_TimeXer.py`
- **要求**：继承 `AbstractTimeSeriesModel`，实现抽象方法 `_fit` 与 `_predict`；超参数通过 `get_hyperparameters()` 获取，勿使用不存在的 `params` 属性。
- **注册**：子类会通过元类自动注册；若在部分环境（如多进程）中未注册，可在使用前显式执行 `from autogluon.timeseries.models import DETMModel`，或在 `hyperparameters` 中传入类引用（如 `{DETMModel: {}}`）。
- **预测接口**：`_predict` 需返回带 `mean` 及分位列的 `TimeSeriesDataFrame`，索引为预测时间范围（可用 `self.get_forecast_horizon_index(data)`）。

在 `TimeSeriesPredictor.fit(..., hyperparameters={...})` 中将自定义模型类或别名加入即可参与训练与集成；其他自定义模型可参照 DETMModel 与 `AbstractTimeSeriesModel` 的约定实现。

### 模型注册表（Registry）

- **位置**：`timeseries/src/autogluon/timeseries/models/registry.py`
- **作用**：统一管理时序模型类与别名；支持在传入类类型时动态注册，便于在多进程或不同导入顺序下使用自定义模型。

---

## 文档与资源

- **官方文档**：[https://auto.gluon.ai/stable/index.html](https://auto.gluon.ai/stable/index.html)
- **安装说明**：[https://auto.gluon.ai/stable/install.html](https://auto.gluon.ai/stable/install.html)
- **时间序列快速入门**：[Time Series Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)
- **上游仓库**：[autogluon/autogluon](https://github.com/autogluon/autogluon)

本仓库中的 `docs/`、`examples/` 及各子模块内文档可作为补充参考。

---

## 开发与贡献

- **代码风格与检查**：根目录 `pyproject.toml` 中配置了 ruff、codespell、pyright 等，可按需运行相应命令。
- **测试**：各子模块通常包含 `tests/`，使用 pytest 等运行；具体见各模块说明。
- **贡献**：若基于上游 AutoGluon 开发，请遵循上游的贡献流程与代码规范；本仓库自定义部分与根目录许可证一致。

---

## 许可证

AutoGluon 采用 Apache 2.0 许可证。本仓库中未单独声明的文件遵循项目根目录的许可证文件；若包含第三方代码（如子模块中的 `_internal`、`timexer_lib` 等），请以各目录下的 LICENSE 或声明为准。
