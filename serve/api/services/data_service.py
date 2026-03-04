import os
import shutil
import time
import uuid
import logging
import pandas as pd
from fastapi import UploadFile
from configs.config import GLOBAL_LOGGER, LOG_DIR, BeijingFormatter, TEMP_DIR
from autogluon.timeseries import TimeSeriesDataFrame

# ================= 全局任务状态存储 =================
TASK_STORE = {}

def update_task_state(task_id, status, message=None, data=None):
    if task_id not in TASK_STORE:
        TASK_STORE[task_id] = {}
    TASK_STORE[task_id]["status"] = status
    if message:
        TASK_STORE[task_id]["message"] = message
    if data:
        if "data" not in TASK_STORE[task_id]:
            TASK_STORE[task_id]["data"] = {}
        TASK_STORE[task_id]["data"].update(data)

def get_task_state(task_id):
    return TASK_STORE.get(task_id)

# ================= 辅助函数 =================

def get_or_create_task_id(task_id: str = None) -> str:
    """如果未提供task_id，则基于时间戳生成"""
    if not task_id:
        return f"{int(time.time())}_{uuid.uuid4().hex[:4]}"
    return task_id

def save_upload_file(file: UploadFile, service_name: str, task_id: str = None) -> tuple[str, str]:
    """
    通用上传文件保存逻辑
    路径结构: outputs/temp/{service_name}/{task_id}/{filename}
    返回: (保存后的绝对路径, 最终的task_id)
    """
    # 1. 确定 Task ID
    final_task_id = get_or_create_task_id(task_id)
    
    # 2. 构建目录: outputs/temp/{service_name}/{task_id}
    # TEMP_DIR 在 config.py 中定义为 outputs/temp
    target_dir = os.path.join(TEMP_DIR, service_name, final_task_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # 3. 确定文件路径
    file_path = os.path.join(target_dir, file.filename)
    
    # 4. 保存文件
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        GLOBAL_LOGGER.info(f"File saved to temp: {file_path}")
    except Exception as e:
        GLOBAL_LOGGER.error(f"Failed to save upload file: {e}")
        raise e
        
    return file_path, final_task_id

def prepare_output_dir(base_dir: str, task_id: str):
    """确保 outputs/{base}/{task_id} 存在"""
    path = os.path.join(base_dir, task_id)
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_files(paths: list):
    """清理临时文件"""
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                GLOBAL_LOGGER.info(f"Cleaned up: {path}")
                
                # 尝试清理空的父目录 (可选优化)
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
                    
        except Exception as e:
            GLOBAL_LOGGER.error(f"Error cleaning up {path}: {e}")

def setup_task_logger(task_id: str):
    """为每个任务创建独立日志，并桥接 tsfresh 日志"""
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    
    file_handler = None
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') 
        formatter = BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = logger.handlers[0]

    tsfresh_logger = logging.getLogger("tsfresh_progress")
    tsfresh_logger.setLevel(logging.INFO)
    tsfresh_logger.handlers = [] 
    tsfresh_logger.addHandler(file_handler)
    
    ag_logger = logging.getLogger("autogluon")
    ag_logger.setLevel(logging.INFO)
    if file_handler not in ag_logger.handlers:
        ag_logger.addHandler(file_handler)
    
    return logger

def prepare_tsdf(file_path: str, id_col="item_id", time_col="timestamp", target_col="value", default_id="default_item") -> TimeSeriesDataFrame:
    """读取 CSV 并转换为 TimeSeriesDataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    if 'date' in df.columns and time_col not in df.columns:
        df.rename(columns={'date': time_col}, inplace=True)
    if 'time' in df.columns and time_col not in df.columns:
        df.rename(columns={'time': time_col}, inplace=True)
    
    if id_col not in df.columns:
        df[id_col] = default_id

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[id_col, time_col])
    
    return TimeSeriesDataFrame.from_data_frame(df, id_column=id_col, timestamp_column=time_col)