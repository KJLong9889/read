import os
import sys
import shutil
import logging
import datetime
import traceback
import zipfile
import uuid
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ================= Path Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))

from autogluon.timeseries import TimeSeriesDataFrame

# ================= Configuration =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 统一存放在 temp 目录下，方便清理
TEMP_BASE_DIR = os.path.join(BASE_DIR, 'analysis_temp')

if not os.path.exists(TEMP_BASE_DIR):
    os.makedirs(TEMP_BASE_DIR)

app = FastAPI(title="Data Analysis Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用全局 Logger 代替任务特定 Logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("DataAnalysis")

# ================= Helper Functions =================
def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

def cleanup_temp_data(folder_path: str, zip_path: str):
    """后台任务：清理临时生成的文件夹和 ZIP 包"""
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

def prepare_tsdf(file_path: str, request_id: str) -> TimeSeriesDataFrame:
    df = pd.read_csv(file_path)
    if 'date' in df.columns: df.rename(columns={'date': 'timestamp'}, inplace=True)
    if 'time' in df.columns: df.rename(columns={'time': 'timestamp'}, inplace=True)
    if 'item_id' not in df.columns:
        df['item_id'] = "default_item"
            
    if 'timestamp' not in df.columns or 'value' not in df.columns:
         possible_targets = [c for c in df.columns if c not in ['item_id', 'timestamp']]
         if len(possible_targets) == 1:
             df.rename(columns={possible_targets[0]: 'value'}, inplace=True)
    
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

# ================= Core Logic =================
def execute_analysis(request_id: str, file_path: str, target_col: str):
    # 每一个请求拥有独立的子目录
    output_dir = os.path.join(TEMP_BASE_DIR, request_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        ts_df = prepare_tsdf(file_path, request_id)

        # 1. 缺失率
        missing_stats = ts_df._compute_missing_rate(save_dir=None, save=False)
        missing_df = pd.DataFrame(list(missing_stats.items()), columns=['Feature', 'Missing_Rate'])
        missing_df.to_csv(os.path.join(output_dir, "missing_rate.csv"), index=False, encoding='utf-8-sig')

        # 2. 月度箱线图
        real_target = target_col if target_col in ts_df.columns else 'value'
        boxplot_df = ts_df._compute_monthly_boxplot_data(target=real_target)
        boxplot_df.to_csv(os.path.join(output_dir, "distribution_boxplot.csv"), index=False, encoding='utf-8-sig')

        # 保存可视化 CSV (趋势、季节性、平稳性等原始数据)
        res_map = {
            "visual_stationarity.csv": 'stationarity_df',
            "visual_periodicity.csv": 'periodicity_df',
            "visual_trend.csv": 'trend_df'
        }
        for file_name, key in res_map.items():
            df = heterogeneity_results.get(key, pd.DataFrame())
            df.to_csv(os.path.join(output_dir, file_name), index=False, encoding='utf-8-sig')

        return output_dir

    except Exception as e:
        logger.error(f"Analysis failed: {traceback.format_exc()}")
        raise e

# ================= API Endpoint =================
@app.post("/api/analyze_csv")
async def analyze_csv_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    # 生成随机 ID 替代用户传入的 task_id
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    
    # 定义路径
    input_file_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_input.csv")
    zip_path = os.path.join(TEMP_BASE_DIR, f"results_{request_id}.zip")

    try:
        # 1. 保存上传文件
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 执行分析
        result_folder = execute_analysis(request_id, input_file_path, target_column)

        # 3. 打包
        zip_folder(result_folder, zip_path)

        # 4. 注册清理后台任务 (在响应发送后执行)
        background_tasks.add_task(cleanup_temp_data, result_folder, zip_path)
        background_tasks.add_task(lambda: os.remove(input_file_path) if os.path.exists(input_file_path) else None)

        # 5. 返回文件流
        return FileResponse(
            path=zip_path,
            filename=f"analysis_results.zip",
            media_type='application/zip'
        )

    except Exception as e:
        # 清理已创建的临时文件
        if os.path.exists(input_file_path): os.remove(input_file_path)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)