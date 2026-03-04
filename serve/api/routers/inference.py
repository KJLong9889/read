import os
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from api.services import predict_service, data_service

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/inference")
async def inference(
    background_tasks: BackgroundTasks,
    task_id: str = Form(...),
    best_model: str = Form(...),
    model_zip: UploadFile = File(...),
    data_csv: UploadFile = File(...)
):
    try:
        # 1. 保存模型包: temp/prediction/{task_id}/xxx.zip
        zip_path, _ = data_service.save_upload_file(model_zip, service_name="prediction", task_id=task_id)
        
        # 2. 保存数据 CSV: temp/prediction/{task_id}/xxx.csv
        csv_path, _ = data_service.save_upload_file(data_csv, service_name="prediction", task_id=task_id)
        
        # 3. 执行推理
        output_path = predict_service.run_inference(task_id, best_model, zip_path, csv_path)
        
        # 4. 清理
        background_tasks.add_task(data_service.cleanup_files, [zip_path, csv_path])
        
        return FileResponse(output_path, filename=f"{task_id}_predictions.csv", media_type='text/csv')
    except Exception as e:
        # 如果路径已生成则清理
        # 注意：这里如果 data_service.save_upload_file 失败，变量可能未定义，Python try-except 块内变量作用域要注意
        # 但 save_upload_file 抛出异常时后续代码不会执行，所以不会引用未定义变量。
        # 为了严谨，可以在 except 里判断变量是否存在，这里简化处理。
        raise HTTPException(500, detail=f"Inference failed: {str(e)}")