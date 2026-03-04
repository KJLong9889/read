import os
import shutil
import zipfile
import logging
import traceback
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_UPLOAD_DIR = os.path.join(BASE_DIR, 'inference_uploads')  # 存放上传的zip和csv
MODEL_EXTRACT_DIR = os.path.join(BASE_DIR, 'inference_models') # 存放解压后的模型
RESULTS_DIR = os.path.join(BASE_DIR, 'inference_results')     # 存放预测结果

# 确保目录存在
for d in [TEMP_UPLOAD_DIR, MODEL_EXTRACT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TimeSeriesInference")

app = FastAPI()

# # ================= 辅助函数 =================

# def cleanup_files(paths: List[str]):
#     """
#     后台任务：清理临时文件和目录
#     """
#     for path in paths:
#         try:
#             if os.path.exists(path):
#                 if os.path.isdir(path):
#                     shutil.rmtree(path)
#                 else:
#                     os.remove(path)
#                 logger.info(f"Cleaned up: {path}")
#         except Exception as e:
#             logger.error(f"Error cleaning up {path}: {str(e)}")

def find_predictor_home(extract_root: str) -> str:
    """
    递归寻找包含 predictor.pkl 的目录。
    解决用户压缩时多包含了一层文件夹的问题。
    """
    # 1. 检查当前目录是否有 predictor.pkl
    if os.path.exists(os.path.join(extract_root, "predictor.pkl")):
        return extract_root
    
    # 2. 如果没有，遍历子目录寻找
    for root, dirs, files in os.walk(extract_root):
        if "predictor.pkl" in files:
            return root
            
    # 3. 如果找不到，抛出异常
    raise FileNotFoundError("Could not find 'predictor.pkl' in the uploaded zip file.")

# ================= 核心接口 =================

@app.post("/api/inference")
async def run_timeseries_inference(
    background_tasks: BackgroundTasks,
    task_id: str = Form(..., description="任务ID"),
    best_model: str = Form(..., description="指定用于推理的模型名称，例如 'WeightedEnsemble'"),
    model_zip: UploadFile = File(..., description="包含训练权重的ZIP文件"),
    data_csv: UploadFile = File(..., description="待预测的数据CSV")
):
    """
    接收模型权重ZIP和数据CSV，使用指定模型进行推理，并返回预测结果CSV。
    """
    # 定义本次任务的文件路径
    zip_filename = f"{task_id}_{model_zip.filename}"
    csv_filename = f"{task_id}_{data_csv.filename}"
    zip_path = os.path.join(TEMP_UPLOAD_DIR, zip_filename)
    csv_path = os.path.join(TEMP_UPLOAD_DIR, csv_filename)
    
    # 模型解压的目标目录
    extract_path = os.path.join(MODEL_EXTRACT_DIR, task_id)
    
    # 结果输出路径
    output_filename = f"{task_id}_predictions.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    # 注册待清理路径 (包括解压目录、上传文件、结果文件)
    clean_targets = [extract_path, zip_path, csv_path, output_path]

    try:
        logger.info(f"[{task_id}] New inference request. Model: {best_model}")

        # 1. 保存上传的文件
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(model_zip.file, f)
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(data_csv.file, f)
            
        # 2. 解压模型
        logger.info(f"[{task_id}] Extracting model zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # 3. 定位 Predictor 路径
        # 自动处理 zip 包含文件夹的情况
        real_model_path = find_predictor_home(extract_path)
        logger.info(f"[{task_id}] Found predictor at: {real_model_path}")

        # 4. 加载 Predictor
        predictor = TimeSeriesPredictor.load(real_model_path)
        
        # 检查请求的模型是否在列表中 (可选，增加鲁棒性)
        if best_model not in predictor.model_names():
            logger.warning(f"[{task_id}] Model '{best_model}' not found in predictor. Available: {predictor.model_names()}")
            # 注意：这里如果不抛出异常，Autogluon 可能会报错，或者你可以选择回退到默认 best_model
        
        # 5. 加载数据
        # 假设 CSV 格式正确 (item_id, timestamp, target)。
        # 如果训练时指定了特定的列名，TimeSeriesDataFrame.from_path 通常能自动处理，
        # 或者需要前端传参指定 id_column / timestamp_column。
        logger.info(f"[{task_id}] Loading data from CSV...")
        data = TimeSeriesDataFrame.from_path(csv_path)

        # 6. 执行预测
        logger.info(f"[{task_id}] Running prediction with model='{best_model}'...")
        predictions = predictor.predict(data, model=best_model)

        # 7. 保存结果
        logger.info(f"[{task_id}] Saving predictions...")
        predictions.to_csv(output_path)

        # # 8. 返回文件
        # # 添加后台清理任务，在响应发送完毕后执行
        # background_tasks.add_task(cleanup_files, clean_targets)
        
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type='text/csv'
        )

    except Exception as e:
        logger.error(f"[{task_id}] Inference failed: {traceback.format_exc()}")
        # 发生错误时立即清理
        # cleanup_files(clean_targets)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)