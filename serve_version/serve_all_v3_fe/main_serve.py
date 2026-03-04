import datetime
import os
import sys
import shutil
import logging
import json
import threading
import numpy as np
import traceback
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import FileResponse
# 引入项目路径 (保持原样)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))

from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator 
from autogluon.timeseries import TimeSeriesDataFrame

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FE_OUTPUT_DIR = os.path.join(BASE_DIR, 'feature_engineering')
CORR_OUTPUT_DIR = os.path.join(BASE_DIR, 'correlation')
LOG_DIR = os.path.join(BASE_DIR, 'temp_logs')
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, 'temp_uploads') 

for d in [FE_OUTPUT_DIR, CORR_OUTPUT_DIR, LOG_DIR, TEMP_UPLOAD_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ================= 全局状态存储 =================
TASK_STORE = {}

app = FastAPI(title="特征工程与相关性分析服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 日志工具 =================
class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_task_logger(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    
    # 1. 设置任务主 Logger (保持原样)
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    
    # 创建 FileHandler (提出来，因为下面要复用)
    file_handler = None
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') 
        formatter = BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = logger.handlers[0]

    # ================= 新增：桥接 tsfresh 的进度日志 =================
    # 获取我们在 tsfresh 源码里定义的那个 logger
    tsfresh_logger = logging.getLogger("tsfresh_progress")
    tsfresh_logger.setLevel(logging.INFO)
    
    # 清除旧的 handlers，防止多任务执行时日志串台或重复打印
    # 注意：在多线程/多进程环境下，这里可能会有竞态条件，
    # 但对于简单的任务隔离，这样通常够用。更严谨的做法是自定义 Filter。
    tsfresh_logger.handlers = [] 
    
    # 将当前任务的 file_handler 挂载给 tsfresh_logger
    # 这样 tsfresh 内部打印的 "tsfresh_progress" 就会写入 log_file
    tsfresh_logger.addHandler(file_handler)
    # ==============================================================
    
    return logger

def update_task_state(task_id, status, message=None, data=None):
    """更新内存中的任务状态"""
    if task_id not in TASK_STORE:
        TASK_STORE[task_id] = {}
    
    TASK_STORE[task_id]["status"] = status
    if message:
        TASK_STORE[task_id]["message"] = message
    if data:
        TASK_STORE[task_id]["data"] = data

# ================= 核心工作函数 (后台线程运行) =================

def _worker_generate_features(task_id: str, file_path: str, max_timeshift: int, min_timeshift: int, tsfresh_custom_params: str):
    """后台执行特征生成的实际逻辑"""
    logger = setup_task_logger(task_id)
    try:
        update_task_state(task_id, "RUNNING", "正在读取并校验数据...")
        logger.info(f"【后台任务启动】Task: {task_id}, File: {file_path}")

        # 1. 读取数据
        try:
            df = pd.read_csv(file_path)
            required_cols = ['time', 'value'] 
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV 必须包含列: {required_cols}")
        except Exception as e:
            logger.error(f"数据读取错误: {e}")
            raise e

        # 2. 数据预处理
        update_task_state(task_id, "RUNNING", "数据预处理中...")
        ID_COLUMN = "item_id"
        TIMESTAMP_COLUMN = "timestamp"
        TARGET_COLUMN = "value"

        if 'item_id' not in df.columns:
            df[ID_COLUMN] = task_id 
        else:
            ID_COLUMN = 'item_id'
        
        df.rename(columns={'time': TIMESTAMP_COLUMN}, inplace=True)
        df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
        df = df.dropna(subset=[TARGET_COLUMN])
        df = df.sort_values(by=[ID_COLUMN, TIMESTAMP_COLUMN])

        full_tsdf = TimeSeriesDataFrame.from_data_frame(
            df, id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN
        )
        
        # 3. 参数解析 (增强部分：支持字符串形式的列表推导式解析)
        update_task_state(task_id, "RUNNING", "配置特征参数...")
        logger.info(f"解析 tsfresh_custom_params: {tsfresh_custom_params}")
        print(f"解析 tsfresh_custom_params: {tsfresh_custom_params}")
        parsed_params = "comprehensive"
        
        if tsfresh_custom_params:
            try:
                # 1. 先进行标准的 JSON 解析
                temp_params = json.loads(tsfresh_custom_params)
                parsed_params = {}
                
                # 2. 遍历参数，处理特殊格式
                for key, value in temp_params.items():
                    final_val = value
                    
                    # 情况A: 空列表 -> None (tsfresh 要求)
                    if isinstance(value, list) and len(value) == 0:
                        final_val = None
                    
                    # 情况B: 你的特殊格式 ["{\"lag\": lag} for lag in range(1, 4)"]
                    # 识别特征：是列表，只有一个元素，元素是字符串，且包含 python 关键字
                    elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                        str_content = value[0].strip()
                        # 简单的启发式检查，防止随意 eval
                        if " for " in str_content and " in " in str_content:
                            try:
                                logger.info(f"检测到动态参数表达式，尝试解析: {key} -> {str_content}")
                                # 构造列表推导式并执行。注意：这有安全风险，但在内部工具中通常可接受
                                # 传入 range 到局部作用域
                                final_val = eval(f"[{str_content}]", {"range": range})
                            except Exception as parse_err:
                                logger.warning(f"解析动态参数失败 {key}: {parse_err}，将使用原始值")
                                final_val = value
                    
                    parsed_params[key] = final_val

                logger.info(f"参数解析完成: {json.dumps(str(parsed_params))}") # 记录日志方便调试

            except Exception as e:
                logger.warning(f"自定义参数解析整体失败: {e}，将使用默认配置")
                parsed_params = "comprehensive"

        tsfresh_config = {
            "impute_strategy": "mean",
            "window_size": 30,
            "max_timeshift": max_timeshift,
            "min_timeshift": min_timeshift,
            "do_selection": False, 
            "fc_parameters": parsed_params
        }

        feature_generator = TimeSeriesFeatureGenerator(
            target=TARGET_COLUMN,
            known_covariates_names=[],
            use_tsfresh=True,
            tsfresh_settings=tsfresh_config,
            correlation_analysis=False 
        )

        # 4. 执行生成
        update_task_state(task_id, "RUNNING", "正在运行 AutoGluon 特征引擎 (此过程较慢，请稍候)...")
        logger.info("开始调用 feature_generator.fit_transform...")
        full_tsdf_result = feature_generator.fit_transform(full_tsdf)
        
        # 5. 保存结果
        update_task_state(task_id, "RUNNING", "保存生成结果...")
        result_df = full_tsdf_result.reset_index()
        feature_count = len(result_df.columns) - 2 
        
        save_dir = os.path.join(FE_OUTPUT_DIR, task_id, "full_dataset")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, "full_features.csv")
        result_df.to_csv(save_path, index=False)
        
        logger.info(f"任务完成，特征数: {feature_count}, 路径: {save_path}")
        
        update_task_state(task_id, "SUCCESS", "特征生成成功", {
            "feature_count": feature_count,
            "download_path": save_path
        })

    except Exception as e:
        error_msg = f"任务执行失败: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        update_task_state(task_id, "FAILED", error_msg)
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def _worker_select_features(task_id: str, fdr: float):
    """后台执行特征筛选"""
    logger = setup_task_logger(task_id)
    try:
        update_task_state(task_id, "RUNNING", f"开始筛选特征 (FDR={fdr})...")
        
        full_data_path = os.path.join(FE_OUTPUT_DIR, task_id, "full_dataset", "full_features.csv")
        if not os.path.exists(full_data_path):
            raise FileNotFoundError("未找到全量特征文件，请先执行特征生成。")
            
        logger.info(f"读取全量数据: {full_data_path}")
        df = pd.read_csv(full_data_path)

        target_col = "value"
        timestamp_col = "timestamp"
        id_col = "item_id"
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col)
        y_series = df[target_col]
        X_df = df.drop(columns=[target_col, id_col], errors='ignore').select_dtypes(include=['number'])

        update_task_state(task_id, "RUNNING", "计算特征相关性...")
        relevance_table = TimeSeriesFeatureGenerator.calculate_relevance_standalone(
            X_df, y_series, fdr_level=fdr
        )
        
        if 'feature' in relevance_table.columns:
            selected_features = relevance_table[relevance_table['relevant'] == True]['feature'].tolist()
        else:
            selected_features = relevance_table[relevance_table['relevant'] == True].index.tolist()
            
        logger.info(f"筛选保留: {len(selected_features)} 个特征")

        # 保存
        select_dir = os.path.join(FE_OUTPUT_DIR, task_id, f"select_dataset_{fdr}")
        if not os.path.exists(select_dir):
            os.makedirs(select_dir)
            
        df_reset = df.reset_index()
        keep_cols = [timestamp_col, id_col, target_col] + selected_features
        final_cols = [c for c in keep_cols if c in df_reset.columns]
        
        selected_df = df_reset[final_cols]
        select_save_path = os.path.join(select_dir, "selected_features.csv")
        selected_df.to_csv(select_save_path, index=False)
        
        update_task_state(task_id, "SUCCESS", "特征筛选完成", {
            "selected_count": len(selected_features),
            "dataset_path": select_save_path
        })

    except Exception as e:
        logger.error(f"筛选失败: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", str(e))


# ================= API 路由 =================

class SelectFeaturesRequest(BaseModel):
    task_id: str
    fdr: float = 0.05 

@app.get("/api/task_status")
def get_task_status(task_id: str):
    """前端轮询此接口获取进度"""
    status_info = TASK_STORE.get(task_id)
    if not status_info:
        if os.path.exists(os.path.join(LOG_DIR, f"{task_id}.log")):
             return {"code": 200, "data": {"status": "UNKNOWN", "message": "任务在运行但内存状态丢失，请查看日志"}}
        return {"code": 404, "message": "任务ID不存在"}
    
    return {
        "code": 200, 
        "data": status_info
    }

@app.post("/api/generate_all_features")
async def generate_all_features(
    background_tasks: BackgroundTasks,
    task_id: str = Form(...),
    max_timeshift: int = Form(29),
    min_timeshift: int = Form(29),
    tsfresh_custom_params: str = Form(None),
    file: UploadFile = File(...)
):
    update_task_state(task_id, "PENDING", "正在初始化任务...")
    
    file_ext = os.path.splitext(file.filename)[1]
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{task_id}{file_ext}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(
        _worker_generate_features, 
        task_id, 
        temp_file_path, 
        max_timeshift, 
        min_timeshift, 
        tsfresh_custom_params
    )

    return {
        "code": 200,
        "message": "特征生成任务已提交后台处理",
        "data": {
            "task_id": task_id,
            "status": "PENDING",
            "info": "请通过 /api/task_status 轮询结果，通过 /api/task_logs 查看日志"
        }
    }


@app.post("/api/select_features_and_download") 
def select_features_and_download(
    req: SelectFeaturesRequest
):

    task_id = req.task_id
    fdr = req.fdr
    

    update_task_state(task_id, "RUNNING", "正在进行同步特征筛选...")
    
    try:
        _worker_select_features(task_id, fdr)
        folder_name = f"select_dataset_{fdr}"
        select_dir = os.path.join(FE_OUTPUT_DIR, task_id, folder_name)
        file_path = os.path.join(select_dir, "selected_features.csv")
        
        # 5. 校验文件
        if not os.path.exists(file_path):
            # 尝试通过 task_status 获取错误信息
            status_info = TASK_STORE.get(task_id, {})
            error_msg = status_info.get("message", "生成文件失败，未知错误")
            raise HTTPException(status_code=500, detail=f"处理失败: {error_msg}")

        filename = f"selected_features_{task_id}_fdr{fdr}.csv"
        
        return FileResponse(
            path=file_path, 
            filename=filename, 
            media_type='text/csv'
        )

    except Exception as e:
        update_task_state(task_id, "FAILED", str(e))
        raise HTTPException(status_code=500, detail=f"同步处理发生异常: {str(e)}")

@app.get("/api/task_logs")
def get_task_logs(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    if not os.path.exists(log_file):
        return {"code": 200, "data": []}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        return {"code": 200, "data": lines}
    except Exception as e:
        return {"code": 500, "message": f"日志读取失败: {str(e)}"}

@app.post("/api/calculate_correlation")
async def calculate_correlation(
    method: str = Form("pearson"),
    file: UploadFile = File(...)
):
    try:
        # 1. 读取数据
        df = pd.read_csv(file.file)
        
        # 2. 相关性计算
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("上传的文件中没有数值型列，无法计算相关性。")
            
        corr_matrix = numeric_df.corr(method=method)
        
        import time
        # 3. 存储到临时目录 (使用时间戳防止并发冲突)
        # 建议直接存在 CORR_OUTPUT_DIR 下，文件名带上时间戳
        timestamp = int(time.time())
        filename = f"corr_matrix_{timestamp}.csv"
        save_path = os.path.join(CORR_OUTPUT_DIR, filename)
        
        corr_matrix.to_csv(save_path, index=True)
        
        # 4. 返回文件流
        # media_type 设置为 text/csv，filename 参数决定了前端下载时的默认文件名
        return FileResponse(
            path=save_path,
            filename=filename,
            media_type='text/csv'
        )

    except Exception as e:
        # 发生错误时返回 500 状态码和具体错误原因
        raise HTTPException(status_code=500, detail=f"相关性分析失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)