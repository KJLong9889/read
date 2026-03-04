import os
import shutil
import zipfile
import traceback
import uuid
import time
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from configs.config import INFERENCE_DIR, TEMP_DIR, GLOBAL_LOGGER
from api.services.data_service import get_or_create_task_id, prepare_output_dir, prepare_tsdf
import pandas as pd

def find_predictor_home(extract_root: str) -> str:
    """递归查找包含 predictor.pkl 的文件夹"""
    if os.path.exists(os.path.join(extract_root, "predictor.pkl")):
        return extract_root
    for root, dirs, files in os.walk(extract_root):
        if "predictor.pkl" in files:
            return root
    raise FileNotFoundError("Could not find 'predictor.pkl' in the uploaded zip.")

def run_inference(task_id: str, best_model: str, zip_path: str, csv_path: str):
    task_id = get_or_create_task_id(task_id)
    
    # 1. 临时解压目录

    extract_path = os.path.join(TEMP_DIR, f"{task_id}_model_extract")
    
    # 2. 结果保存目录
    result_dir = prepare_output_dir(INFERENCE_DIR, task_id)
    output_path = os.path.join(result_dir, "predictions.csv")
    
    try:
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        real_model_path = find_predictor_home(extract_path)
        predictor = TimeSeriesPredictor.load(real_model_path, require_version_match=False)
        
        data = prepare_tsdf(csv_path)

        # 预测
        predictions = predictor.predict(data, model=best_model)
    
        quantile_cols = []
        for col in predictions.columns:
            if col in ["mean", "item_id", "timestamp","time"]:
                continue
            try:
                val = float(col)
                quantile_cols.append((col, val))
            except ValueError:
                continue 

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
            predictions.rename(columns=rename_map, inplace=True)
        predictions.to_csv(output_path)
        return output_path
    except Exception as e:
        GLOBAL_LOGGER.error(f"Inference failed: {traceback.format_exc()}")
        raise e
    finally:
        # 清理临时解压目录
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)