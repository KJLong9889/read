import os
import base64
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from api.services import analysis_service, data_service
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from configs.config import TEMP_DIR
import binascii
router = APIRouter(prefix="/api", tags=["Analysis & Features"])

class FeatureConfigItem(BaseModel):
    name: str          # 例如 "c3"
    params: Dict[str, Any] = {}  # <--- 修改这里：接收字符串列表，例如 ["lag"]

class FeatureGenerationRequest(BaseModel):
    task_id: str
    file_name: str
    file_content_b64: str
    max_timeshift: int = 29
    min_timeshift: int = 29
    
    custom_params: List[FeatureConfigItem] = [] # 接收上述对象列表

@router.post("/analyze_csv")
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    # 使用新方法保存：temp/analysis/{generated_id}/filename
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        output_path = analysis_service.compute_boxplot(input_path, target_column, request_id)
        
        # 清理
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        
        return FileResponse(output_path, media_type='text/csv', filename="distribution_boxplot.csv")
    except Exception as e:
        # 如果 save_upload_file 成功但后面失败了，尝试清理 (如果 input_path 已定义)
        # 简单起见，这里抛出异常即可，临时文件会留存供排查，或依赖系统定期清理
        raise HTTPException(500, detail=str(e))

@router.post("/fill_missing_values")
async def fill_missing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("auto")
):
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        # 这里的 method 会传给 AutoGluon 的 fill_missing_values
        output_path, filled_status = analysis_service.fill_missing_values(input_path, method, request_id)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        
        headers = {
            "X-Filled-Status": str(filled_status).lower(),
            "X-File-Name": f"processed_{file.filename}",
            "Access-Control-Expose-Headers": "X-Filled-Status, X-File-Name"
        }
        return FileResponse(output_path, media_type="text/csv", filename=f"processed_{file.filename}", headers=headers)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.post("/get_characteristics")
async def get_characteristics(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        scores = analysis_service.compute_characteristics(input_path, target_column)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        return {"status": "success", "data": scores}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    

@router.post("/generate_all_features")
async def generate_features(
    background_tasks: BackgroundTasks,
    request: FeatureGenerationRequest  # 使用 Pydantic Model 接收 JSON
):
    """
    修改后的接口：接收纯 JSON 请求。
    文件必须转为 Base64 字符串放在 file_content_b64 字段中。
    """
    task_id = request.task_id
    
    # 1. 解码并保存 CSV 文件
    try:
        # 去掉可能的 data:text/csv;base64, 前缀
        if "," in request.file_content_b64:
            _, b64_data = request.file_content_b64.split(",", 1)
        else:
            b64_data = request.file_content_b64

        file_bytes = base64.b64decode(b64_data)
        
        # 手动构建保存路径 (模仿 data_service.save_upload_file 的逻辑)
        # 路径: temp/feature_engineering/{task_id}/{filename}
        save_dir = os.path.join(TEMP_DIR, "feature_engineering", task_id)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, request.file_name)
        
        with open(save_path, "wb") as f:
            f.write(file_bytes)
            
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail="Invalid Base64 file content")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 2. 转换参数格式 (Pydantic Model -> List[Dict])
    params_list = [item.dict() for item in request.custom_params]

    # 3. 启动后台任务
    # 注意：service 层的 worker 参数签名需要修改以适配这里传入的 list
    background_tasks.add_task(
        analysis_service.worker_generate_features,
        task_id, 
        save_path, 
        request.max_timeshift, 
        request.min_timeshift, 
        params_list
    )
    
    return {
        "code": 200, 
        "message": "Feature generation started", 
        "data": {
            "task_id": task_id,
            "status": "PENDING",
            "info": "Poll /api/task_status for progress"
        }
    }

@router.post("/select_features_and_download")
def select_features(
    task_id: str = Form(...),
    fdr: float = Form(0.05)
):
    try:
        output_path = analysis_service.select_features(task_id, fdr)
        filename = f"selected_features_{task_id}_fdr{fdr}.csv"
        return FileResponse(output_path, media_type='text/csv', filename=filename)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.post("/calculate_correlation")
async def calc_correlation(
    background_tasks: BackgroundTasks,
    method: str = Form("pearson"), 
    file: UploadFile = File(...)
):
    try:
        input_path, _ = data_service.save_upload_file(file, service_name="analysis")
        
        output_path = analysis_service.calculate_correlation_matrix(input_path, method)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        return FileResponse(output_path, media_type='text/csv', filename="correlation_matrix.csv")
    except Exception as e:
        raise HTTPException(500, detail=str(e))