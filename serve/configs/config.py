import os
import sys
import logging
import datetime

# ================= 1. 基础路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# SERVE_ROOT: serve/
SERVE_ROOT = os.path.dirname(CURRENT_DIR)
# PROJECT_ROOT: serve 的上一级目录 (项目根目录)
PROJECT_ROOT = os.path.dirname(SERVE_ROOT)

# ================= 2. 注入系统路径 =================
# 确保能引用到 AutoApex 源码目录
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))

# ================= 3. 输出目录配置 (Outputs) =================
OUTPUT_DIR = os.path.join(SERVE_ROOT, 'outputs')

# (1) 训练模型: outputs/train_models/{task_id}
TRAIN_MODELS_DIR = os.path.join(OUTPUT_DIR, 'train_models')

# (2) 训练数据备份 (用于排行榜生成): outputs/train_data/{task_id}.csv
TRAIN_DATA_DIR = os.path.join(OUTPUT_DIR, 'train_data')

# (3) 特征工程: outputs/feature/{task_id}
FEATURE_DIR = os.path.join(OUTPUT_DIR, 'feature')

# (4) 分析结果 (箱线图, 相关性, 缺失值等): outputs/analysis/{task_id}
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, 'analysis')

# (5) 评估指标/排行榜结果: outputs/metrics_model/{task_id}
METRICS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'metrics_model')

# (6) 推理结果: outputs/inference/{task_id}
INFERENCE_DIR = os.path.join(OUTPUT_DIR, 'inference')

# (7) 临时文件/上传暂存: outputs/temp
TEMP_DIR = os.path.join(OUTPUT_DIR, 'temp')

# (8) 日志目录: serve/logs
LOG_DIR = os.path.join(SERVE_ROOT, 'logs')

# 自动创建所有目录
for d in [OUTPUT_DIR, TRAIN_MODELS_DIR, TRAIN_DATA_DIR, FEATURE_DIR, ANALYSIS_DIR, METRICS_OUTPUT_DIR, INFERENCE_DIR, TEMP_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ================= 4. 日志配置 =================
class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(ch)
    return logger

GLOBAL_LOGGER = get_logger("GlobalServe")