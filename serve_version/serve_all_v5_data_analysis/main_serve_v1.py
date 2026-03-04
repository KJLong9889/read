import os
import sys
import shutil
import logging
import uuid
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ================= Path Configuration =================
# 保持原有的路径配置，确保能引用到 autogluon 模块
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

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("DataAnalysis")

# ================= Helper Functions =================

def cleanup_files(paths: list):
    """
    后台任务：清理临时生成的文件
    """
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.info(f"Cleaned up: {path}")
        except Exception as e:
            logger.error(f"Error cleaning up {path}: {e}")

def prepare_tsdf(file_path: str) -> TimeSeriesDataFrame:
    """
    读取 CSV 并转换为 TimeSeriesDataFrame，包含简单的列名映射逻辑
    """
    df = pd.read_csv(file_path)
    
    # 标准化时间列名
    if 'date' in df.columns: df.rename(columns={'date': 'timestamp'}, inplace=True)
    if 'time' in df.columns: df.rename(columns={'time': 'timestamp'}, inplace=True)
    
    # 如果没有 item_id，给予默认值
    if 'item_id' not in df.columns:
        df['item_id'] = "default_item"
            
    # 如果没有 value 列，尝试推断目标列（排除 item_id 和 timestamp 后的第一列）
    if 'value' not in df.columns:
         possible_targets = [c for c in df.columns if c not in ['item_id', 'timestamp']]
         if len(possible_targets) >= 1:
             # 注意：这里我们不强制重命名为 value，而是依赖后续传入的 target_column 参数
             pass
    
    # 转换为 AutoGluon 的 TimeSeriesDataFrame
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

# ================= API 1: 获取箱线图数据 (返回 CSV 文件流) =================
@app.post("/api/analyze_csv")
async def analyze_csv_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    """
    接收 CSV，计算月度分布箱线图数据，返回 CSV 文件。
    """
    # 生成唯一请求 ID
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    
    # 定义临时文件路径
    input_file_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_input.csv")
    output_boxplot_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_boxplot.csv")

    try:
        # 1. 保存上传文件
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 加载数据
        ts_df = prepare_tsdf(input_file_path)
        
        # 确定实际的目标列名
        real_target = target_column if target_column in ts_df.columns else 'value'
        if real_target not in ts_df.columns:
             # 如果既不是传入的 target_column 也不是 'value'，尝试取最后一列
             real_target = ts_df.columns[-1]

        # 3. 计算箱线图数据
        # 注意：这里调用的是 ts_dataframe.py 中保留的 compute_monthly_boxplot_data 方法
        boxplot_df = ts_df._compute_monthly_boxplot_data(target=real_target)
        
        # 4. 保存结果为 CSV
        boxplot_df.to_csv(output_boxplot_path, index=False, encoding='utf-8-sig')

        # 5. 注册清理任务 (响应发送后执行)
        background_tasks.add_task(cleanup_files, [input_file_path, output_boxplot_path])

        # 6. 返回文件流
        return FileResponse(
            path=output_boxplot_path,
            filename="distribution_boxplot.csv",
            media_type='text/csv'
        )

    except Exception as e:
        # 出错时立即清理输入文件
        cleanup_files([input_file_path])
        logger.error(f"Analyze CSV failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/api/fill_missing_values")
async def fill_missing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("auto"),
):
    """
    处理 CSV 中的缺失值并返回处理后的文件。
    - 对每个被填补的字段生成 {col}_fill 标记列（true / false）
    - 排除时间列
    - 删除 item_id 列
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    input_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_fill_in.csv")
    output_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_fill_out.csv")

    try:
        # 1. 保存上传文件
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 读取 TimeSeriesDataFrame
        ts_df = prepare_tsdf(input_path)

        # 自动获取列名
        id_col = getattr(ts_df, "id_column", "item_id")
        time_col = getattr(ts_df, "timestamp_column", "timestamp")

        # 3. 填补前快照（必须 deep copy）
        df_before = ts_df.reset_index().copy()

        # 4. 填补缺失值
        ts_filled = ts_df.fill_missing_values(
            method=method,
            value=0.0
        )

        # 5. 填补后数据
        df_after = ts_filled.reset_index()

        # 6. 生成 {col}_fill 标记列（true / false）
        ignore_cols = {id_col, time_col}

        for col in df_after.columns:
            if col in ignore_cols:
                continue
            if col not in df_before.columns:
                continue

            # 原来是 NaN 且现在不是 NaN → 被填补
            is_filled = df_before[col].isna() & df_after[col].notna()

            fill_flag_col = f"{col}_fill"
            df_after[fill_flag_col] = is_filled.map(
                lambda x: "true" if x else "false"
            )

        # 7. 删除 item_id 列
        if id_col in df_after.columns:
            df_after.drop(columns=[id_col], inplace=True)

        # 8. 保存 CSV
        df_after.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        # 9. 清理任务
        background_tasks.add_task(
            cleanup_files,
            [input_path, output_path]
        )

        # 10. 返回文件
        return FileResponse(
            path=output_path,
            filename=f"processed_{file.filename}",
            media_type="text/csv"
        )

    except Exception as e:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass

        logger.error(f"Fill missing values failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))



# ================= API 2: 获取数据特征评分 (返回 JSON) =================
@app.post("/api/get_characteristics")
async def get_characteristics_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    """
    接收 CSV，计算周期性、趋势性、平稳性的评分，返回 JSON 对象。
    """
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    input_file_path = os.path.join(TEMP_BASE_DIR, f"{request_id}_score_input.csv")

    try:
        # 1. 保存上传文件
        with open(input_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 加载数据
        ts_df = prepare_tsdf(input_file_path)
        
        # 确定实际的目标列名
        real_target = target_column if target_column in ts_df.columns else 'value'
        if real_target not in ts_df.columns:
             real_target = ts_df.columns[-1]

        # 3. 计算评分
        # 注意：这里调用的是我们在 ts_dataframe.py 中新增的 compute_heterogeneity_scores 方法
        scores = ts_df._compute_heterogeneity_scores(target=real_target)

        # 4. 注册清理任务
        background_tasks.add_task(cleanup_files, [input_file_path])

        # 5. 返回 JSON
        return JSONResponse(content={
            "status": "success",
            "data": scores  
            # scores 结构示例: 
            # {
            #   "periodicity_score": 0.85, 
            #   "trend_score": 0.42, 
            #   "stationarity_score": 0.15
            # }
        })

    except Exception as e:
        cleanup_files([input_file_path])
        logger.error(f"Get characteristics failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)