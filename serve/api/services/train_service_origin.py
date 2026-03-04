import os
import shutil
import traceback
import zipfile
from autogluon.timeseries import TimeSeriesPredictor
from configs.config import TRAIN_MODELS_DIR, TEMP_DIR, METRICS_OUTPUT_DIR, TRAIN_DATA_DIR
from api.services.data_service import prepare_tsdf, update_task_state, setup_task_logger
from autogluon.common import space as ag
from configs.hpo_template import parse_hpo_config
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_ensemble_configs(ensemble_types):
    """解析集成学习配置"""
    ensemble_hps = {}
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

def convert_to_search_space(hp_dict):
    """
    将包含类型描述的字典转换为 AutoGluon 的搜索空间对象 (ag.Int, ag.Real, ag.Categorical)。
    
    支持的格式示例:
    1. 整数范围 (Int):
       {"type": "int", "lower": 10, "upper": 100, "default": 50, "log": false}
       
    2. 浮点范围 (Real):
       {"type": "real", "lower": 0.001, "upper": 0.1, "default": 0.01, "log": true}
       
    3. 分类/枚举 (Categorical):
       {"type": "categorical", "data": ["relu", "softrelu"]}
       
    4. 布尔 (Bool):
       {"type": "bool"}  -> 等价于 ag.Categorical(True, False)
       
    5. 简写列表 (List) -> 自动转为 Categorical:
       [32, 64, 128] -> ag.Categorical(32, 64, 128)
    """
    
    # 如果不是字典，直接返回（可能是基础值，如 50, "auto" 等）
    if not isinstance(hp_dict, dict):
        return hp_dict

    # -------------------------------------------------------
    # 核心逻辑：检查是否存在 "type" 字段，如果有，说明这是一个特定的搜索空间描述
    # -------------------------------------------------------
    if "type" in hp_dict:
        type_name = str(hp_dict["type"]).lower()
        
        # === 1. Int (整数搜索空间) ===
        if type_name == "int":
            return ag.Int(
                lower=hp_dict.get("lower"),
                upper=hp_dict.get("upper"),
                default=hp_dict.get("default")
            )
            
        # === 2. Real (浮点数搜索空间) ===
        elif type_name == "real":
            return ag.Real(
                lower=hp_dict.get("lower"),
                upper=hp_dict.get("upper"),
                default=hp_dict.get("default"),
                log=hp_dict.get("log", False)
            )
            
        # === 3. Categorical (分类/枚举空间) ===
        elif type_name == "categorical":
            data = hp_dict.get("data", [])
            # 必须解包 (*data) 才能正确注册为多个选项
            if isinstance(data, list):
                return ag.Categorical(*data)
            return ag.Categorical(data) # 单个值的情况
            
        # === 4. Bool (布尔空间) ===
        elif type_name == "bool":
            # AutoGluon 中 Bool 通常就是 Categorical(True, False)
            return ag.Categorical(True, False)

    # -------------------------------------------------------
    # 递归逻辑：如果当前字典不是一个 "Space 描述对象"，则遍历其子项
    # -------------------------------------------------------
    new_dict = {}
    for k, v in hp_dict.items():
        if isinstance(v, dict):
            # 递归处理嵌套字典
            new_dict[k] = convert_to_search_space(v)
            
        elif isinstance(v, list):
            # [重要修复]: 纯列表默认视为 Categorical，必须使用 *v 解包
            # 例如: [32, 64] -> ag.Categorical(32, 64)
            new_dict[k] = ag.Categorical(*v)
            
        else:
            # 其他基础类型（字符串、数字等）保持不变
            new_dict[k] = v
            
    return new_dict



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

        user_hyperparameters = params.get('hyperparameters', {})
        user_tune_kwargs = params.get("hyperparameter_tune_kwargs", None)
        if user_tune_kwargs and isinstance(user_hyperparameters, dict):
            logger.info("检测到超参搜索模式，正在将 List 转换为 SearchSpace ...")
            try:
                # 将普通的 [1, 2] 转换成 ag.Categorical([1, 2])
                user_hyperparameters = convert_to_search_space(user_hyperparameters)
                logger.info(f"转换后的超参配置: {user_hyperparameters}")
            except Exception as e:
                logger.error(f"转换搜索空间失败: {e}")
                raise e
        
        # 4. 组装 fit 参数
        fit_kwargs = {
            "train_data": train_data,
            "time_limit": params.get('time_limit'),
            "hyperparameters": user_hyperparameters, 
            "hyperparameter_tune_kwargs": user_tune_kwargs
        }
        
        if params.get('enable_ensemble'):
            e_types = params.get('ensemble_types')
            e_list = e_types if isinstance(e_types, list) else ([t.strip() for t in e_types.split(',')] if e_types else ["weighted"])
            fit_kwargs["enable_ensemble"] = True
            fit_kwargs["ensemble_hyperparameters"] = get_ensemble_configs(e_list)

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

# def process_leaderboard_generation(task_id: str, extra_metrics: list):
#     """
#     生成排行榜及验证集预测结果
#     """
#     logger = setup_task_logger(task_id)
#     try:
#         logger.info("Starting leaderboard generation & evaluation predictions...")
        
#         model_path = os.path.join(TRAIN_MODELS_DIR, task_id)
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"No trained model found for task_id: {task_id}")

#         # 1. 加载数据 (使用训练时备份的数据)
#         data_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")
#         if not os.path.exists(data_path):
#              raise FileNotFoundError(f"Original training data not found at {data_path}")

#         logger.info(f"Loading data from: {data_path}")
#         data = prepare_tsdf(data_path, default_id=task_id)

#         # 2. 加载 Predictor
#         predictor = TimeSeriesPredictor.load(model_path)
#         prediction_length = predictor.prediction_length
#         target_column = predictor.target

#         # 3. 切分验证数据 (Slice)
#         logger.info(f"Splitting data for evaluation (prediction_length={prediction_length})...")
#         try:
#             data_past = data.slice_by_timestep(None, -prediction_length)
#             data_truth = data.slice_by_timestep(-prediction_length, None)
#         except Exception as e:
#             raise ValueError(f"Data is too short to split for evaluation: {e}")

#         # 4. 准备保存目录
#         save_dir = os.path.join(METRICS_OUTPUT_DIR, task_id)
#         if os.path.exists(save_dir):
#             shutil.rmtree(save_dir)
#         os.makedirs(save_dir)
        
#         preds_dir = os.path.join(save_dir, "predictions")
#         os.makedirs(preds_dir)

#         # 5. 遍历模型：预测 + 合并真实值 + 重命名列
#         all_models = predictor.model_names()
#         logger.info(f"Found {len(all_models)} models. Generating predictions...")

#         for model_name in all_models:
#             try:
#                 predictions = predictor.predict(data_past, model=model_name)
                
#                 # 添加真实值
#                 predictions["ground_truth"] = data_truth[target_column]
                
#                 # 动态重命名分位数列
#                 quantile_cols = []
#                 for col in predictions.columns:
#                     if col in ["ground_truth", "mean", "item_id", "timestamp"]:
#                         continue
#                     try:
#                         val = float(col)
#                         quantile_cols.append((col, val))
#                     except ValueError:
#                         continue 

#                 if len(quantile_cols) == 1 and quantile_cols[0][1] == 0.5:
#                     predictions.drop(columns=[quantile_cols[0][0]], inplace=True)
#                 elif len(quantile_cols) > 0:
#                     rename_map = {}
#                     for col_name, val in quantile_cols:
#                         if val < 0.5:
#                             rename_map[col_name] = "lower_quantile"
#                         elif val == 0.5:
#                             rename_map[col_name] = "median"
#                         elif val > 0.5:
#                             rename_map[col_name] = "upper_quantile"
#                     predictions.rename(columns=rename_map, inplace=True)
                
#                 # 保存单个模型预测
#                 safe_name = model_name.replace(os.path.sep, "_").replace(" ", "_")
#                 pred_file = os.path.join(preds_dir, f"{safe_name}.csv")
#                 predictions.to_csv(pred_file)
                
#             except Exception as e:
#                 logger.warning(f"Failed to generate predictions for {model_name}: {e}")

#         # 6. 计算排行榜
#         logger.info(f"Calculating leaderboard with metrics: {extra_metrics}")
#         lb_df = predictor.leaderboard(data=data, extra_metrics=extra_metrics, silent=True)
        
#         csv_filename = f"{task_id}_leaderboard.csv"
#         lb_save_path = os.path.join(save_dir, csv_filename)
#         lb_df.to_csv(lb_save_path, index=False, encoding='utf-8-sig')
        
#         # 7. 打包返回
#         zip_filename = f"{task_id}_full_results.zip"
#         zip_path = os.path.join(TEMP_DIR, zip_filename)
        
#         zip_folder(save_dir, zip_path)
#         return zip_path

#     except Exception as e:
#         logger.error(f"Leaderboard generation failed: {traceback.format_exc()}")
#         raise e
    









