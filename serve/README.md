# AutoApex Time Series Analysis Service

AutoApex Time Series Analysis Service 是一个基于 **FastAPI** 和 **AutoApex** 构建的时间序列分析与预测平台后端服务。  
项目采用 **Router–Service 分层架构**，提供从 **数据探索（EDA）→ 特征工程 → 模型训练 → 推理预测** 的一站式解决方案，适用于工业级时间序列建模与批量预测场景。

---

## 项目特性

### 数据分析（Analysis）
- 月度分布箱线图分析  
- 缺失值自动填补  
- 时间序列特征评分（趋势性、周期性、平稳性）  
- 多变量相关性分析  

### 特征工程（Feature Engineering）
- 集成 **特征生成工具**，支持自定义参数的自动化特征生成与筛选  
- 支持 **FDR（False Discovery Rate）** 特征筛选  

### 模型训练（Training）
- 基于 **AutoApex TimeSeriesPredictor**
- 支持多种集成策略（Weighted / Per-Item / Simple 等）
- 自动备份训练数据
- 自动生成模型排行榜（Leaderboard）及验证集预测结果

### 模型推理（Inference）
- 支持上传训练好的模型压缩包（`.zip`）
- 支持新数据的离线批量预测
- 输出预测均值及分位数区间

### 规范化存储
- 所有输出文件（模型、特征、预测结果）均按 `task_id` 进行结构化隔离存储

---

## 目录结构

```text
serve/
├── api/
│   ├── routers/                 # 路由层：定义 HTTP 接口，处理请求参数
│   │   ├── analysis.py          # 分析与特征工程接口
│   │   ├── prediction.py        # 推理预测接口
│   │   └── training.py          # 模型训练接口
│   └── services/                # 业务层：核心逻辑实现
│       ├── analysis_service.py  # 特征生成、数据统计分析
│       ├── data_service.py      # 通用工具、文件 IO、日志管理
│       ├── predict_service.py   # 推理逻辑
│       └── train_service.py     # 训练与排行榜逻辑
├── configs/
│   └── config.py                # 全局配置（路径、日志、环境变量）
├── outputs/                     # [自动生成] 所有任务输出产物
│   ├── analysis/                # 分析结果（箱线图、补全数据）
│   ├── feature/                 # 特征工程结果
│   ├── train_models/            # 训练好的模型文件
│   ├── train_data/              # 训练数据备份
│   ├── inference/               # 推理预测结果
│   ├── metrics_model/           # 排行榜与评估详情
│   └── temp/                    # 临时上传文件（按服务隔离）
├── logs/                        # [自动生成] 任务日志
├── main.py                      # FastAPI 应用入口
└── start_server.sh              # 启动脚本
```

---

## 安装与依赖

建议使用 **Python 3.10.1+**，安装以下核心依赖：

```bash
pip install fastapi uvicorn pandas numpy autoapex.timeseries tsfresh python-multipart
```

> 注意  
> 本项目依赖同级目录下的 `common`、`core`、`features`、`timeseries`、`tabular` 模块。

---

## 快速启动

在 `serve/` 目录下运行：

```bash
cd serve
chmod +x start_server.sh
./start_server.sh
```

服务默认运行在：  
`http://0.0.0.0:8000`

---

## 接口概览

### 数据分析 & 特征工程（`/api`）

#### POST `/analyze_csv`
生成月度分布箱线图统计结果。

**参数**
- `file`（File，必填）：原始时间序列 CSV
- `target_column`（str，默认 `value`）

**输出**
- CSV 文件：`distribution_boxplot.csv`
- 包含 `min / q1 / median / q3 / max` 等统计量

---

#### POST `/fill_missing_values`
时间序列缺失值填补接口（支持 `auto / mean / median / forward_fill` 等）。

**状态码说明**
- `3`：数据完整  
- `0`：仅数值缺失（已填补）  
- `1`：仅时间戳缺失（已修复）  
- `2`：断档或全缺失  

**参数**
- `file`（File，必填）：原始时间序列 CSV
- `method` (str, Default: "auto")：填补策略，支持 "auto", "mean", "forward_fill" 等 AutoGluon 支持的方法

**输出**
- CSV 文件：`processed_{filename}`
- Header 中包含 `X-Filled-Status`

---

#### POST `/get_characteristics`
计算时间序列特征评分（趋势性、周期性等）。

**参数**
- `file`（File，必填）：原始时间序列 CSV
- `target_column`  (str, Default: "value"): 目标列名

**输出**
```json
{
  "status": "success",
  "data": { "...scores..." }
}
```
- Content: {"status": "success", "data": {...scores...}}
---

#### POST `/calculate_correlation`
计算多变量相关性矩阵。

**参数**
- `method`：(str, Default: "pearson"): 相关系数计算方法 (e.g., "pearson", "spearman")
- `file`：(File, Required): CSV 文件

**输出**
- `correlation_matrix.csv`

---

#### POST `/generate_all_features`（异步）

**参数**
- `task_id`：(str, Required): 任务唯一标识 ID
- `file`：(File, Required): CSV 文件
- `max_timeshift`：(int, Default: 29): 特征生成的最大时间偏移量
- `min_timeshift`：(int, Default: 29): 特征生成的最小时间偏移量
- `tsfresh_custom_params`：(str, Optional): 自定义 tsfresh 参数配置 (JSON 字符串)


**返回**
```json
{
  "code": 200,
  "message": "Feature generation started",
  "data": {
    "task_id": "...",
    "status": "PENDING"
  }
}
```
- `Content`：{"code": 200, "message": "Feature generation started", "data": {"task_id": "...", "status": "PENDING"}}
---

#### POST `/select_features_and_download`
基于 FDR 进行特征筛选并下载。

**参数**
- `task_id`：(str, Required): 任务唯一标识 ID
- `fdr`：(float, Default: 0.05): 错误发现率阈值，值越小筛选越严格


**输出**
```
Format: CSV 文件 (selected_features_{task_id}_fdr{fdr}.csv),包含原始 ID、时间戳、Target 以及筛选后的特征列
```

---

### 模型训练（`/api`）

#### POST `/train`
**参数**
- `task_id`：(str, Required): 任务唯一标识 ID
- `file`：(File, Required): 训练数据 CSV 文件
- `target`：(str, Required): 预测目标列名（如 value）
- `prediction_length`：(int, Required): 预测未来多少个时间步
- `freq`：(str, Optional): 数据频率（如 D, H, 5min）。若为空则自动推断
- `eval_metric`：(str, Optional): 评估指标（如 MASE, MAPE, RMSE）
- `use_quantile`：(str, Optional): 是否使用分位数预测（根据业务需求开启）
- `use_tsfresh`：(bool, Required): 是否使用 tsfresh 生成的静态特征辅助训练
- `time_limit`：(int, Required): 训练总限时（秒）
- `hyperparameters`：(str, Required): 模型超参数配置 JSON 字符串,示例: {"Naive": {}, "DeepAR": {"hidden_size": 64}}
- `enable_ensemble`：(bool, Required): 是否启用加权集成学习 (WeightedEnsemble)
- `ensemble_types`：(str, Optional): 指定集成类型参数
- `hyperparameter_tune_kwargs`：(str, Optional): 超参数搜索配置 JSON 字符串


**输出**
```
Format: 训练生成的权重 ZIP 文件流，包含训练好的模型文件 (learner.pkl, predictor.pkl 和leaderboard生成的预测指标等) 的压缩包
```

---



#### GET `/task_status`

**参数**
- `task_id`：(str, Required): 任务唯一标识 ID
- `fdr`：(float, Default: 0.05): 错误发现率阈值，值越小筛选越严格

**输出**
```
ZIP 文件流，包含 leaderboard.csv 和各模型的预测结果。
```

#### GET `/task_logs`
获取实时训练日志。
**参数**
- `task_id`：(str, Required): 任务唯一标识 ID

```
{"data": ["log line 1", "log line 2"]}
```
---

### 模型推理（`/inference`）

**输入**
- `task_id`：(str, Required): 任务唯一标识 ID
- `best_model`：(str, Required): 指定使用的模型名称 (如 "WeightedEnsemble" 或 "DeepAR")
- `model_zip`：训练接口返回的模型 ZIP 包
- `data_csv`：历史数据 CSV (用于生成未来预测)

**输出**
- `{task_id}_predictions.csv`
- 包含预测均值及分位数区间

---

## 输出文件说明

- 训练模型：`outputs/train_models/{task_id}/`
- 特征文件：`outputs/feature/{task_id}/full_features.csv`
- 推理结果：`outputs/inference/{task_id}/predictions.csv`
- 临时文件：`outputs/temp/{service_name}/{task_id}/original_filename.csv`
