import datetime
import os
import sys
import shutil
import logging
import json
import threading
import numpy as np
import traceback
import zipfile
from typing import List, Optional, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd

# ================= Path Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ================= Configuration Area =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FE_OUTPUT_DIR = os.path.join(BASE_DIR, 'feature_engineering')
CORR_OUTPUT_DIR = os.path.join(BASE_DIR, 'correlation')
LOG_DIR = os.path.join(BASE_DIR, 'temp_logs')
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, 'temp_uploads') 
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'trained_models')
METRICS_OUTPUT_DIR = os.path.join(BASE_DIR, 'metrics_model')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train_data')

for d in [FE_OUTPUT_DIR, CORR_OUTPUT_DIR, LOG_DIR, TEMP_UPLOAD_DIR, MODEL_OUTPUT_DIR, METRICS_OUTPUT_DIR, TRAIN_DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ================= Global State =================
TASK_STORE = {}

app = FastAPI(title="AutoGluon TimeSeries Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Helper Functions =================
def parse_json_form(value: str | None, default=None):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception as e:
        return value

def get_ensemble_configs(ensemble_types: List[str]):
    ensemble_hps = {}
    if 'weighted' in ensemble_types:
        ensemble_hps["WeightedEnsemble"] = {} 
    if 'per_item' in ensemble_types:
        ensemble_hps["PerItemGreedyEnsemble"] = {}
    if 'simple' in ensemble_types:
        ensemble_hps["SimpleAverage"] = {}
    if 'median' in ensemble_types:
        ensemble_hps["MedianEnsemble"] = {}
    return ensemble_hps

class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def setup_task_logger(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') 
        formatter = BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = logger.handlers[0]

    for log_name in ["tsfresh_progress", "autogluon"]:
        sub_logger = logging.getLogger(log_name)
        sub_logger.setLevel(logging.INFO)
        if file_handler not in sub_logger.handlers:
            sub_logger.addHandler(file_handler)

    return logger

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

def prepare_dataframe(file_path: str, task_id: str, logger) -> TimeSeriesDataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    ID_COLUMN = "item_id"
    TIMESTAMP_COLUMN = "timestamp"
    TARGET_COLUMN = "value"
    
    if 'item_id' not in df.columns:
        df[ID_COLUMN] = task_id 
    if 'date' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'date': TIMESTAMP_COLUMN}, inplace=True)
    if 'time' in df.columns and 'timestamp' not in df.columns:
        df.rename(columns={'time': TIMESTAMP_COLUMN}, inplace=True)
    
    if TIMESTAMP_COLUMN not in df.columns:
         raise ValueError(f"Missing timestamp column. Expected '{TIMESTAMP_COLUMN}' or 'date'/'time'.")
    if TARGET_COLUMN not in df.columns:
         raise ValueError(f"Missing target column. Expected '{TARGET_COLUMN}'.")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df.sort_values(by=[ID_COLUMN, TIMESTAMP_COLUMN])

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN
    )
    return ts_df

def zip_folder(folder_path, output_path):
    """将文件夹打包成 zip"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

# ================= Training Logic (Now Synchronous) =================
def run_training_logic(task_id: str, file_path: str, params: dict):
    """
    执行训练逻辑（同步调用，不再作为 Worker）。
    如果训练失败，抛出异常。
    """
    logger = setup_task_logger(task_id)
    try:
        update_task_state(task_id, "RUNNING", "Initializing training environment...")
        logger.info(f"Start Training Task: {task_id}")
        logger.info(f"Params: {json.dumps(params, default=str)}")

        # 1. Load Data
        update_task_state(task_id, "RUNNING", "Loading and validating data...")
        train_data = prepare_dataframe(file_path, task_id, logger)
        
        # 2. Setup Paths
        model_path = os.path.join(MODEL_OUTPUT_DIR, task_id)
        if os.path.exists(model_path):
            shutil.rmtree(model_path) 

        if params.get('quantile_levels') is None:
            update_task_state(task_id, "RUNNING", "Initializing Predictor...")

            predictor = TimeSeriesPredictor(
                path=model_path,
                target=params.get('target', 'value'),
                prediction_length=params.get('prediction_length', 7),
                freq=params.get('freq'), 
                eval_metric=params.get('eval_metric'),
                use_tsfresh=params.get('use_tsfresh', False),
                verbosity=3
            )
        else:
            update_task_state(task_id, "RUNNING", "Initializing Predictor with quantile levels...")
            predictor = TimeSeriesPredictor(
                path=model_path,
                target=params.get('target', 'value'),
                prediction_length=params.get('prediction_length', 7),
                freq=params.get('freq'), 
                eval_metric=params.get('eval_metric'),
                quantile_levels=params.get('quantile_levels'),
                use_tsfresh=params.get('use_tsfresh', False),
                verbosity=3
            )
        
        # 4. 准备参数
        update_task_state(task_id, "RUNNING", "Fitting models...")
        
        final_hps = params.get('hyperparameters')
        enable_ensemble = params.get('enable_ensemble')
        hp_tune_kwargs = params.get('hyperparameter_tune_kwargs')

        if enable_ensemble:
            e_types_input = params.get('ensemble_types')
            if isinstance(e_types_input, str):
                e_list = [t.strip() for t in e_types_input.split(',') if t.strip()]
            elif isinstance(e_types_input, list):
                e_list = e_types_input
            else:
                e_list = ["weighted"]

            ensemble_hps = get_ensemble_configs(e_list)
        else:
            ensemble_hps = None
            logger.info(f"Ensemble disabled. Generated configs: {ensemble_hps}")
    
        fit_kwargs = {
            "train_data": train_data,
            "time_limit": params.get('time_limit'),
            "hyperparameters": final_hps, 
            "enable_ensemble": enable_ensemble,
            "ensemble_hyperparameters": ensemble_hps,
            "hyperparameter_tune_kwargs": hp_tune_kwargs,
        }
        
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

        logger.info(f"Final fit_kwargs: {fit_kwargs}")
        predictor.fit(**fit_kwargs)

        update_task_state(task_id, "SUCCESS", "Training Completed")
        logger.info(f"Training finished. Best model: {predictor.model_best}")
        return model_path

    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Training failed: {err}")
        update_task_state(task_id, "FAILED", f"Training failed: {str(e)}")
        raise e # 抛出异常供 API 捕获

# -------------------------------------------------------------------------
# [API Endpoints]
# -------------------------------------------------------------------------
@app.get("/api/task_status")
def get_task_status(task_id: str):
    status_info = TASK_STORE.get(task_id)
    if not status_info:
        if os.path.exists(os.path.join(LOG_DIR, f"{task_id}.log")):
             return {"code": 200, "data": {"status": "UNKNOWN", "message": "Task exists in logs but memory state lost."}}
        return {"code": 404, "message": "Task ID not found"}
    return {"code": 200, "data": status_info}

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
        return {"code": 500, "message": f"Failed to read logs: {str(e)}"}

def cleanup_file(path: str):
    """辅助函数：清理临时文件"""
    if os.path.exists(path):
        os.remove(path)

@app.post("/api/train")
def train_model(
    # 注意：移除了 BackgroundTasks
    task_id: str = Form(...),
    file: UploadFile = File(...),
    target: str = Form(...),
    prediction_length: int = Form(...),
    freq: Optional[str] = Form(None),
    eval_metric: Optional[str] = Form(None),
    use_quantile: Optional[str] = Form(None),
    use_tsfresh: bool = Form(...),
    time_limit: int = Form(...),
    hyperparameters: str = Form(...), 
    enable_ensemble: bool = Form(...),
    ensemble_types: Optional[str] = Form(None), 
    hyperparameter_tune_kwargs: Optional[str] = Form(None)
):
    """
    同步训练接口：上传 -> 保存 -> 训练 -> 打包 -> 返回流
    注意：如果 time_limit 较长，客户端必须设置足够长的 timeout，否则会断开连接。
    """
    update_task_state(task_id, "PENDING", "Uploading training data...")
    
    # 1. 保存文件到持久化目录
    save_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


    if use_quantile=='interval':
        quantile_levels = [0.1, 0.5, 0.9]
    else:
        quantile_levels = [0.5]
    # 2. 构造参数
    params = {
        "target": target,
        "prediction_length": prediction_length,
        "freq": freq,
        "eval_metric": eval_metric,
        "quantile_levels": quantile_levels,
        "use_tsfresh": use_tsfresh,
        "time_limit": time_limit,
        "hyperparameters": parse_json_form(hyperparameters, default="default"),
        "enable_ensemble": enable_ensemble,
        "ensemble_types": ensemble_types, 
        "hyperparameter_tune_kwargs": parse_json_form(hyperparameter_tune_kwargs),
    }

    # 3. 同步执行训练 (Blocking Call)
    # 这会阻塞直到训练完成
    try:
        model_folder = run_training_logic(task_id, save_path, params)
    except Exception as e:
        # 如果训练出错，直接返回 500
        raise HTTPException(status_code=500, detail=f"Training Error: {str(e)}")

    # 4. 训练成功，开始打包
    zip_filename = f"{task_id}_model.zip"
    zip_path = os.path.join(TEMP_UPLOAD_DIR, zip_filename)
    
    try:
        zip_folder(model_folder, zip_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to zip model: {str(e)}")

    # 5. 返回文件流
    # headers 用于告诉浏览器/客户端这是一个附件下载
    return FileResponse(
        path=zip_path,
        filename=zip_filename,
        media_type='application/zip',
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        # 如果你想在下载后删除 zip，可以使用 background 参数，但需要把 cleanup_file 加回来
        # background=BackgroundTask(cleanup_file, zip_path) 
    )

# -------------------------------------------------------------------------
# [Leaderboard & Full Predictions Logic]
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# [Leaderboard & Evaluation Predictions Logic]
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# [Leaderboard & Evaluation Predictions Logic]
# -------------------------------------------------------------------------
def process_leaderboard_generation(task_id: str, extra_metrics: List[str]):
    logger = setup_task_logger(task_id)
    try:
        logger.info("Starting synchronous leaderboard generation & evaluation predictions...")
        
        model_path = os.path.join(MODEL_OUTPUT_DIR, task_id)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found for task_id: {task_id}. Please train first.")

        # 1. 加载数据
        data_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")
        logger.info(f"Loading data from: {data_path}")
        data = prepare_dataframe(data_path, task_id, logger)

        # 2. 加载 Predictor
        logger.info("Loading Predictor...")
        predictor = TimeSeriesPredictor.load(model_path)
        
        # 获取关键参数
        prediction_length = predictor.prediction_length
        target_column = predictor.target

        # 3. [关键步骤] 准备验证数据
        logger.info(f"Splitting data for evaluation (prediction_length={prediction_length})...")
        try:
            data_past = data.slice_by_timestep(None, -prediction_length)
            data_truth = data.slice_by_timestep(-prediction_length, None)
        except Exception as e:
            raise ValueError(f"Data is too short to split for evaluation length {prediction_length}: {e}")

        # 4. 准备保存目录
        save_dir = os.path.join(METRICS_OUTPUT_DIR, task_id)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        
        preds_dir = os.path.join(save_dir, "predictions")
        os.makedirs(preds_dir)

        # 5. 遍历模型：预测 + 合并真实值 + [新功能: 动态重命名列]
        all_models = predictor.model_names()
        logger.info(f"Found {len(all_models)} models. Generating evaluation predictions...")

        for model_name in all_models:
            try:
                logger.info(f"Predicting with {model_name} (Evaluation Mode)...")
                predictions = predictor.predict(data_past, model=model_name)
                
                # 添加真实值
                predictions["ground_truth"] = data_truth[target_column]
                quantile_cols = []
                for col in predictions.columns:
                    # 排除非分位数及其它保留列
                    if col in ["ground_truth", "mean", "item_id", "timestamp"]:
                        continue
                    try:
                        val = float(col)
                        quantile_cols.append((col, val))
                    except ValueError:
                        continue # 不是数字列，跳过

                if len(quantile_cols) == 1 and quantile_cols[0][1] == 0.5:
                    predictions.drop(columns=[quantile_cols[0][0]], inplace=True)
                
                elif len(quantile_cols) > 0:
                    rename_map = {}
                    for col_name, val in quantile_cols:
                        if val < 0.5:
                            rename_map[col_name] = "lower_quantile"
                        elif val == 0.5:
                            rename_map[col_name] = "median"
                        elif val > 0.5:
                            rename_map[col_name] = "upper_quantile"
                    
                    # 执行重命名
                    predictions.rename(columns=rename_map, inplace=True)
                # =========================================================

                # 保存文件
                safe_name = model_name.replace(os.path.sep, "_").replace(" ", "_")
                pred_file = os.path.join(preds_dir, f"{safe_name}.csv")
                predictions.to_csv(pred_file)
                
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {model_name}: {e}")

        # 6. 计算排行榜
        logger.info(f"Calculating leaderboard with metrics: {extra_metrics}")
        lb_df = predictor.leaderboard(
            data=data,
            extra_metrics=extra_metrics,
            silent=True
        )
        csv_filename = f"{task_id}_leaderboard.csv"
        save_path = os.path.join(save_dir, csv_filename)
        lb_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Leaderboard saved to {save_path}")

        # 7. 打包返回
        zip_filename = f"{task_id}_full_results.zip"
        zip_path = os.path.join(TEMP_UPLOAD_DIR, zip_filename)
        
        logger.info(f"Zipping results to {zip_path}...")
        zip_folder(save_dir, zip_path)
        
        return zip_path

    except Exception as e:
        err = traceback.format_exc()
        logger.error(f"Leaderboard generation failed: {err}")
        raise e


@app.post("/api/generate_leaderboard")
def generate_leaderboard(
    task_id: str = Form(...),
):
    metrics_list = ["MASE", "MSE", "RMSE", "MAPE", 
                    "SQL", "WQL", "SMAPE", 
                    "RMSSE","RMSSE","RMSLE","WAPE"]

    try:
        # 这里 result_path 现在是一个 zip 文件的路径
        result_path = process_leaderboard_generation(task_id, metrics_list)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data Validation Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    filename = os.path.basename(result_path)
    
    # 修改 media_type 为 zip
    return FileResponse(
        path=result_path, 
        filename=filename, 
        media_type='application/zip',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)