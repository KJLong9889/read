import os
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from api.services import analysis_service, data_service

router = APIRouter(prefix="/api", tags=["Analysis & Features"])

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
    task_id: str = Form(...),
    max_timeshift: int = Form(29),
    min_timeshift: int = Form(29),
    tsfresh_custom_params: str = Form(None),
    file: UploadFile = File(...)
):
    # 这里 task_id 是前端传来的，temp/feature_engineering/{task_id}/filename
    save_path, _ = data_service.save_upload_file(file, service_name="feature_engineering", task_id=task_id)
        
    background_tasks.add_task(
        analysis_service.worker_generate_features,
        task_id, save_path, max_timeshift, min_timeshift, tsfresh_custom_params
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