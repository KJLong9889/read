import os
import json
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from api.services import train_service, data_service
from configs.config import TEMP_DIR, LOG_DIR, TRAIN_DATA_DIR

router = APIRouter(prefix="/api", tags=["Training"])

def parse_json(val, default=None):
    if not val: return default
    try: return json.loads(val)
    except: return val

@router.post("/train")
def train_model(
    background_tasks: BackgroundTasks,
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
    # 1. 保存到 temp/training/{task_id}/
    temp_path, _ = data_service.save_upload_file(file, service_name="training", task_id=task_id)

    backup_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")
    shutil.copy(temp_path, backup_path)
    
    quantile_levels = [0.1, 0.5, 0.9] if use_quantile == 'interval' else [0.5]
    
    params = {
        "target": target,
        "prediction_length": prediction_length,
        "freq": freq,
        "eval_metric": eval_metric,
        "quantile_levels": quantile_levels,
        "use_tsfresh": use_tsfresh,
        "time_limit": time_limit,
        "hyperparameters": parse_json(hyperparameters, {}),
        "enable_ensemble": enable_ensemble,
        "ensemble_types": ensemble_types,
        "hyperparameter_tune_kwargs": parse_json(hyperparameter_tune_kwargs, None)
    }

    try:
        # 执行训练 (内部会自动生成榜单文件到 model_folder/evaluation_results)
        model_folder = train_service.run_training_logic(task_id, backup_path, params)
        
        # 打包整个文件夹 (现在包含了模型文件 + evaluation_results 文件夹)
        zip_filename = f"{task_id}_model_pack.zip" # 改个名体现它包含更多内容
        zip_path = os.path.join(TEMP_DIR, zip_filename)
        train_service.zip_folder(model_folder, zip_path)
        
        # 清理
        # background_tasks.add_task(data_service.cleanup_files, [temp_path, zip_path])
        
        return FileResponse(zip_path, filename=zip_filename, media_type='application/zip')
    except Exception as e:
        data_service.cleanup_files([temp_path])
        raise HTTPException(500, detail=f"Training Error: {str(e)}")

@router.get("/task_status")
def get_status(task_id: str):
    state = data_service.get_task_state(task_id)
    if not state:
        if os.path.exists(os.path.join(LOG_DIR, f"{task_id}.log")):
             return {"code": 200, "data": {"status": "UNKNOWN", "message": "Task lost from memory but log exists."}}
        return {"code": 404, "message": "Task ID not found"}
    return {"code": 200, "data": state}

@router.get("/task_logs")
def get_logs(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    if not os.path.exists(log_file):
        return {"code": 200, "data": []}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            return {"code": 200, "data": [line.strip() for line in f.readlines()]}
    except:
        return {"code": 500, "message": "Failed to read logs"}

@router.post("/generate_leaderboard")
def generate_leaderboard(task_id: str = Form(...)):
    metrics_list = ["MASE", "MSE", "RMSE", "MAPE", "SQL", "WQL", "SMAPE", "RMSSE","RMSSE","RMSLE","WAPE"]
    try:
        result_zip = train_service.process_leaderboard_generation(task_id, metrics_list)
        filename = os.path.basename(result_zip)
        return FileResponse(result_zip, filename=filename, media_type='application/zip')
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")