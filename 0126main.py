import pandas as pd
import numpy as np
import time
import os
import sys
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# 获取当前脚本所在的绝对路径 (即项目的根目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 手动添加各子模块的 src 目录到系统路径
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'common', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'features', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'timeseries', 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tabular', 'src'))
# 必须先导入并注册 DETMModel，再导入 Predictor，否则 fit 时会报 Unknown model: DETM
from autogluon.timeseries.models import DETMModel
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries import TimeSeriesPredictor
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('autogluon').setLevel(logging.WARNING)

os.environ["TABPFN_MODEL_CACHE_DIR"] = "checkpoints"

# ====================== 路径配置 ======================
INPUT_HISTORY_DIR = "demo_data"
OUTPUT_DIR = "results/forecast_future"  # 修改输出目录以示区别
MODEL_PATH_TEMPLATE = "AutogluonModels/AutogluonModels_{filename}"
STATUS_FILE = "prediction_status.csv"   # 状态记录

FORCE_DELETE_MODEL = True
PREDICTION_LENGTH = 7  # 预测未来多少天

# ====================== 核心字段 ======================
ID_COLUMN = "item_id"
TIMESTAMP_COLUMN = "date"
TARGET_COLUMN = "value"

# ====================== 已知未来协变量 ======================
# 设为空，不再使用未来天气等数据
KNOWN_COVARIATES = [] 

# ====================== 集成策略配置函数 (已修复) ======================
def get_ensemble_configs(ensemble_types):
    """
    根据用户指定的类型列表，生成 AutoGluon 的 ensemble_hyperparameters
    
    支持类型: 
      - 'weighted': 全局贪心加权 (最快，最稳)
      - 'per_item': 分物品贪心加权 (适合不同商品规律差异大的情况)
      - 'stacking': 全局堆叠 (Tabular Stacking)
      - 'simple':   简单平均
      - 'median':   中位数集成
      - 'quantile': 分位数堆叠 (极慢，慎用)
    """
    ensemble_hps = {}
    
    # 1. 全局加权集成 (WeightedEnsemble / Greedy) - 基础必备
    if 'weighted' in ensemble_types:
        ensemble_hps["WeightedEnsemble"] = {"max_models": 25} 
        
    # 2. 分物品加权集成 (PerItemGreedyEnsemble) - 进阶推荐
    # 如果你的商品有些是 DeepAR 准，有些是 AutoETS 准，这个模型会自动切换
    if 'per_item' in ensemble_types:
        ensemble_hps["PerItemGreedyEnsemble"] = {"max_models": 25}

    # 3. 堆叠集成 (TabularEnsemble / Stacking)
    if 'stacking' in ensemble_types:
        # 注意：如果特征过多(如开了TSfresh)，这里容易报维度错误
        ensemble_hps["TabularEnsemble"] = {
            "model_name": "CAT",  # 使用 CatBoost
            "max_num_samples": 100000 
        }
        
    # 4. 简单平均 (SimpleAverage)
    if 'simple' in ensemble_types:
        ensemble_hps["SimpleAverage"] = {}

    # 5. 中位数集成 (MedianEnsemble)
    if 'median' in ensemble_types:
        ensemble_hps["MedianEnsemble"] = {}
    
    # 6. 分位数堆叠 (PerQuantileTabularEnsemble) - 计算量巨大
    if 'quantile' in ensemble_types:
        ensemble_hps["PerQuantileTabularEnsemble"] = {
            "model_name": "CAT",
            "max_num_samples": 50000
        }
        
    return ensemble_hps

# ====================== 时间特征函数 ======================
def add_time_features(df, timestamp_col='date'):
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    return df

# ====================== 单文件处理 ======================
def process_file(history_file, ensemble_types=["weighted"]):
    base_name = os.path.splitext(history_file)[0]
    item_output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(item_output_dir, exist_ok=True)

    try:
        print(f"[Start] 正在处理: {base_name} | 集成策略: {ensemble_types}")

        # ---------- 读取全量历史数据 ----------
        df = pd.read_csv(os.path.join(INPUT_HISTORY_DIR, history_file))
        df[ID_COLUMN] = base_name
        df = add_time_features(df, TIMESTAMP_COLUMN)

        # ---------- 字段检查 ----------
        required_cols = [ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN] + KNOWN_COVARIATES
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return f"❌ {history_file} 缺少列: {missing}", None

        df = df.sort_values(TIMESTAMP_COLUMN)

        # ======================================================
        # 全量数据构建 TimeSeriesDataFrame
        # ======================================================
        full_tsdf = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column=ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN
        )

        # ======================================================
        # Block 4: 模型训练服务
        # ======================================================
        print(f"[Train] 模型全量训练...")

        model_path = MODEL_PATH_TEMPLATE.format(filename=base_name)
        if FORCE_DELETE_MODEL and os.path.exists(model_path):
            shutil.rmtree(model_path)

        predictor = TimeSeriesPredictor(
            prediction_length=PREDICTION_LENGTH,
            freq='D',
            target=TARGET_COLUMN,
            known_covariates_names=KNOWN_COVARIATES,
            eval_metric="MSE", 
            path=model_path,
            use_tsfresh=False, 
            correlation_analysis=False,
            verbosity=4
        )

        # --- 获取修正后的集成配置 ---
        ensemble_hps = get_ensemble_configs(ensemble_types)


        predictor.fit(
            train_data=full_tsdf,
            time_limit=1200, 
            enable_ensemble=True,
            ensemble_hyperparameters=ensemble_hps, 
            presets="fast_training", 
            hyperparameters={  # 基础模型配置：用类引用避免 Registry 未注册导致 Unknown model: DETM
                DETMModel: {},
            }         
        )
        
        # ======================================================
        # New Block: 导出训练详情 (Leaderboard & Val Predictions)
        # ======================================================
        print(f"📝 [Export] 导出模型指标与验证集预测...")
        
        # 1. 导出 Leaderboard
        leaderboard_df = predictor.leaderboard(data=full_tsdf, 
                                               silent=True, 
                                               extra_metrics=["MAE", "RMSE", "MAPE", "SMAPE", "RMSSE"]
                        )
        leaderboard_path = os.path.join(item_output_dir, f"{base_name}_model_leaderboard.csv")
        leaderboard_df.to_csv(leaderboard_path, index=False, encoding='utf-8-sig')

        # 2. 导出所有模型在验证集上的预测值
        val_input_data = full_tsdf.slice_by_timestep(None, -PREDICTION_LENGTH)
        
        model_names = predictor.model_names()
        
        all_val_preds = []
        for model_name in model_names:
            try:
                preds = predictor.predict(val_input_data, model=model_name)
                preds = preds.reset_index()
                preds['model'] = model_name 
                all_val_preds.append(preds)
            except Exception as e:
                logging.warning(f"模型 {model_name} 验证集预测失败: {e}")

        if all_val_preds:
            combined_val_preds = pd.concat(all_val_preds, ignore_index=True)
            
            if "timestamp" in combined_val_preds.columns and TIMESTAMP_COLUMN != "timestamp":
                combined_val_preds.rename(columns={"timestamp": TIMESTAMP_COLUMN}, inplace=True)
            if "item_id" in combined_val_preds.columns and ID_COLUMN != "item_id":
                combined_val_preds.rename(columns={"item_id": ID_COLUMN}, inplace=True)
                
            val_preds_path = os.path.join(item_output_dir, f"{base_name}_validation_predictions_all_models.csv")
            combined_val_preds.to_csv(val_preds_path, index=False, encoding='utf-8-sig')
        

        # ======================================================
        # Block 5: 构造未来数据并推理 (Future Forecast)
        # ======================================================
        print(f"[Predict] 未来预测...")

        last_date = df[TIMESTAMP_COLUMN].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_LENGTH, freq='D')
        
        future_cov_df = pd.DataFrame({TIMESTAMP_COLUMN: future_dates})
        future_cov_df[ID_COLUMN] = base_name
        
        future_tsdf = TimeSeriesDataFrame.from_data_frame(
            future_cov_df,
            id_column=ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN
        )

        predictions = predictor.predict(
            data=full_tsdf,
            known_covariates=future_tsdf
        )

        pred_df = predictions.reset_index()

        if "timestamp" in pred_df.columns and TIMESTAMP_COLUMN != "timestamp":
            pred_df.rename(columns={"timestamp": TIMESTAMP_COLUMN}, inplace=True)
        if "item_id" in pred_df.columns and ID_COLUMN != "item_id":
            pred_df.rename(columns={"item_id": ID_COLUMN}, inplace=True)

        output_file = os.path.join(item_output_dir, f"{base_name}_future_prediction.csv")
        pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        status_info = {
            'filename': history_file,
            'status': 'Success',
            'start_date': future_dates[0].strftime('%Y-%m-%d'),
            'end_date': future_dates[-1].strftime('%Y-%m-%d'),
            'models_trained': len(model_names),
            'best_model': predictor.model_best
        }

        return (
            f"{history_file} 完成 | 最佳模型: {predictor.model_best}"
        ), status_info

    except Exception as e:
        import traceback
        error_msg = f"{history_file} 失败\n{traceback.format_exc()[:300]}"
        print(traceback.format_exc())
        logging.exception("...")
        return error_msg, None


# ====================== 并行处理 ======================
def batch_process_parallel(max_workers=1):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_HISTORY_DIR):
        print(f"找不到输入目录: {INPUT_HISTORY_DIR}")
        return

    files = [f for f in os.listdir(INPUT_HISTORY_DIR) if f.endswith(".csv")]
    
    if not files:
        print(f"目录 {INPUT_HISTORY_DIR} 中没有 CSV 文件")
        return

    all_status = []
    
    # === 在这里定义你想要运行的集成类型 ===
    USE_ENSEMBLES = ['weighted', 'simple', 'median', "per_item"] 
    
    print(f"开始全量训练与预测，共 {len(files)} 个文件...")
    print(f"启用的集成策略: {USE_ENSEMBLES}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f, USE_ENSEMBLES): f for f in files}
        
        for future in as_completed(futures):
            result, status = future.result()
            print(result)

            if status is not None:
                all_status.append(status)

    if all_status:
        status_df = pd.DataFrame(all_status)
        status_path = os.path.join(OUTPUT_DIR, STATUS_FILE)
        status_df.to_csv(status_path, index=False, encoding='utf-8-sig')
        print(f"\n预测状态表已保存到: {status_path}")
    else:
        print("未完成任何预测")


if __name__ == "__main__":
    batch_process_parallel(max_workers=1)
    print("任务全部完成")