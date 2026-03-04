import os
import shutil
import traceback
import zipfile
import logging
from autogluon.timeseries import TimeSeriesPredictor
from autogluon.common import space as ag

from configs.config import TRAIN_MODELS_DIR, TEMP_DIR, METRICS_OUTPUT_DIR, TRAIN_DATA_DIR
from api.services.data_service import prepare_tsdf, update_task_state, setup_task_logger

# 导入分离出去的 HPO 配置解析引擎
from configs.hpo_template import parse_hpo_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_ensemble_configs(ensemble_types):
    """解析集成学习配置"""
    ensemble_hps = {}
    if not ensemble_types:
        return ensemble_hps
    if 'weighted' in ensemble_types: ensemble_hps["WeightedEnsemble"] = {} 
    if 'per_item' in ensemble_types: ensemble_hps["PerItemGreedyEnsemble"] = {}
    if 'simple' in ensemble_types: ensemble_hps["SimpleAverage"] = {}
    if 'median' in ensemble_types: ensemble_hps["MedianEnsemble"] = {}
    return ensemble_hps

def zip_folder(folder_path, output_path):
    """通用文件夹打包"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

def _generate_leaderboard_files(predictor, data, output_dir, logger, metrics=None):
    """
    [新增内部函数] 在训练完成后直接生成榜单和预测文件到指定目录
    """
    if metrics is None:
        metrics = ["MASE", "MAE", "MSE", "RMSE", "MAPE", "SQL", "WQL", "SMAPE", "RMSSE", "RMSLE", "WAPE"]
    
    try:
        prediction_length = predictor.prediction_length
        target_column = predictor.target
        
        # 1. 创建结果存放目录
        results_dir = os.path.join(output_dir, "evaluation_results")
        preds_dir = os.path.join(results_dir, "predictions")
        os.makedirs(preds_dir, exist_ok=True)
        
        logger.info(f"Generating leaderboard and predictions in {results_dir}...")

        # 2. 切分验证数据
        try:
            data_past = data.slice_by_timestep(None, -prediction_length)
            data_truth = data.slice_by_timestep(-prediction_length, None)
        except Exception as e:
            logger.warning(f"Data too short for validation/leaderboard generation: {e}")
            return # 数据太短无法评估，直接返回，不影响主流程

        # 3. 遍历模型生成预测文件
        all_models = predictor.model_names()
        for model_name in all_models:
            try:
                predictions = predictor.predict(data_past, model=model_name)
                predictions["ground_truth"] = data_truth[target_column]
                
                # 处理列名 (分位数 -> 易读名称)
                quantile_cols = []
                for col in predictions.columns:
                    if col in ["ground_truth", "mean", "item_id", "timestamp"]: continue
                    try: quantile_cols.append((col, float(col)))
                    except: continue 

                rename_map = {}
                if len(quantile_cols) == 1 and quantile_cols[0][1] == 0.5:
                    predictions.drop(columns=[str(quantile_cols[0][1])], inplace=True)
                elif len(quantile_cols) > 0:
                    for col_name, val in quantile_cols:
                        if val < 0.5: rename_map[col_name] = "lower_quantile"
                        elif val == 0.5: rename_map[col_name] = "median"
                        elif val > 0.5: rename_map[col_name] = "upper_quantile"
                    predictions.rename(columns=rename_map, inplace=True)
                
                safe_name = model_name.replace(os.path.sep, "_").replace(" ", "_")
                predictions.to_csv(os.path.join(preds_dir, f"{safe_name}.csv"))
            except Exception as e:
                logger.warning(f"Failed to predict for {model_name}: {e}")

        # 4. 生成 Leaderboard CSV
        lb_df = predictor.leaderboard(data=data, extra_metrics=metrics, silent=True)
        lb_df.to_csv(os.path.join(results_dir, "leaderboard.csv"), index=False, encoding='utf-8-sig')
        logger.info("Leaderboard generation completed.")

    except Exception as e:
        logger.error(f"Error during leaderboard generation: {e}")


def run_training_logic(task_id: str, file_path: str, params: dict):
    if not task_id:
        raise ValueError("Task ID is required for training.")
        
    logger = setup_task_logger(task_id)
    autoapex_logger = logging.getLogger("autogluon")
    autoapex_logger.setLevel(logging.INFO)  # 或根据需要调整日志级别
    
    task_handlers = []
    for handler in logger.handlers:
        # 防止重复添加 (如果是单次脚本运行没关系，如果是Web服务这点很重要)
        if handler not in autoapex_logger.handlers:
            autoapex_logger.addHandler(handler)
            task_handlers.append(handler)

    try:
        update_task_state(task_id, "RUNNING", "Initializing training...")
        logger.info(f"Start Training Task: {task_id}")

        # 1. 准备数据
        train_data = prepare_tsdf(file_path, default_id=task_id, target_col=params.get('target', 'value'))
        
        # 2. 准备模型输出路径
        model_path = os.path.join(TRAIN_MODELS_DIR, task_id)
        if os.path.exists(model_path): shutil.rmtree(model_path) 

        # 3. 初始化 Predictor
        predictor_args = {
            "path": model_path,
            "target": params.get('target', 'value'),
            "prediction_length": params.get('prediction_length'),
            "freq": params.get('freq'),
            "eval_metric": params.get('eval_metric'),
            "use_tsfresh": params.get('use_tsfresh', False),
            "verbosity": 4
        }
        if params.get('quantile_levels'):
            predictor_args['quantile_levels'] = params.get('quantile_levels')
        
        predictor = TimeSeriesPredictor(**predictor_args)

        # ================= 核心修改：利用解析引擎处理 HPO 空间 =================
        user_hyperparameters = params.get('hyperparameters', [])
        user_tune_kwargs = params.get("hyperparameter_tune_kwargs", None)
        
        logger.info("正在解析并注入经验 HPO 超参空间...")
        try:
            # parse_hpo_config 自动继承专家兜底参数，并将其与用户传入的 HPO 偏好合并
            ag_hyperparameters = parse_hpo_config(user_hyperparameters)
            logger.info(f"最终转换生成的超参配置: {ag_hyperparameters}")
        except Exception as e:
            logger.error(f"转换搜索空间失败: {e}")
            raise e
        
        ag_tune_kwargs = user_tune_kwargs if user_tune_kwargs else "auto"

        # 4. 组装 fit 参数
        fit_kwargs = {
            "train_data": train_data,
            "time_limit": params.get('time_limit'),
            "hyperparameters": ag_hyperparameters, 
            "hyperparameter_tune_kwargs": ag_tune_kwargs
        }
        
        if params.get('enable_ensemble'):
            e_types = params.get('ensemble_types')
            e_list = e_types if isinstance(e_types, list) else ([t.strip() for t in e_types.split(',')] if e_types else ["weighted"])
            fit_kwargs["enable_ensemble"] = True
            fit_kwargs["ensemble_hyperparameters"] = get_ensemble_configs(e_list)
        # =========================================================================

        # 5. 开始训练
        predictor.fit(**fit_kwargs)
        
        # ================= NEW: 生成榜单 =================
        # 训练结束后，直接利用当前数据生成榜单，文件会存入 model_path/evaluation_results
        update_task_state(task_id, "RUNNING", "Generating Leaderboard...")
        _generate_leaderboard_files(predictor, train_data, model_path, logger)
        # ===============================================

        logger.info("Training finished")
        update_task_state(task_id, "SUCCESS", "Training Completed")

        return model_path

    except Exception as e:
        logger.error(f"Training failed: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", f"Training failed: {str(e)}")
        raise e
    
    finally:
        # === [重要清理] ===
        # 如果这是 Web 服务(FastAPI)，必须在任务结束后移除 Handler
        # 否则下一个任务的日志会错误地写入到上一个任务的文件中！
        for handler in task_handlers:
            autoapex_logger.removeHandler(handler)