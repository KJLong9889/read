import os
import json
import traceback
import pandas as pd
import numpy as np
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator
from autogluon.timeseries import TimeSeriesDataFrame
from configs.config import ANALYSIS_DIR, FEATURE_DIR, TEMP_DIR
from configs.fe_template import parse_feature_config
from api.services.data_service import (
    prepare_tsdf, update_task_state, setup_task_logger, 
    get_or_create_task_id, prepare_output_dir
)

def compute_boxplot(file_path: str, target_column: str, request_id: str = None) -> str:
    """计算月度箱线图"""
    task_id = get_or_create_task_id(request_id)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    
    ts_df = prepare_tsdf(file_path, default_id="default_item")
    
    real_target = target_column if target_column in ts_df.columns else 'value'
    if real_target not in ts_df.columns:
        real_target = ts_df.columns[-1]

    # 调用 AutoApex 扩展方法
    boxplot_df = ts_df._compute_monthly_boxplot_data(target=real_target)
    
    output_path = os.path.join(save_dir, "distribution_boxplot.csv")
    boxplot_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path

def compute_characteristics(file_path: str, target_column: str):
    """计算时间序列特征评分 (周期性、趋势性等)"""
    ts_df = prepare_tsdf(file_path)
    real_target = target_column if target_column in ts_df.columns else 'value'
    if real_target not in ts_df.columns:
        real_target = ts_df.columns[-1]
    
    # 调用 AutoApex 扩展方法
    scores = ts_df._compute_heterogeneity_scores(target=real_target)
    return scores


def clean_and_fill_timeseries(
    df: pd.DataFrame, 
    time_col: str = 'timestamp', 
    target_col: str = 'value', 
    id_col: str = 'item_id',
    fill_method: str = 'auto'
) -> pd.DataFrame:
    """
    核心清洗逻辑：修复时间戳、重采样填补断档、AutoGluon填补数值、生成Flag标记。
    
    返回:
        pd.DataFrame: 包含处理后的数据，且包含 'fill_flag' 列。
    """
    # 避免修改原始数据
    df = df.copy()

    # 1. 预处理与标记原始状态
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # 标记：是否是原始数据中存在的行
    df['_is_origin'] = True
    # 标记：原始是否缺失时间/值
    df['_orig_date_missing'] = df[time_col].isna()
    df['_orig_val_missing'] = df[target_col].isna()

    if df['_orig_date_missing'].any():
        freq_delta = pd.Timedelta(days=1) # 默认
        valid_ts = df[time_col].dropna()
        if len(valid_ts) > 1:
            diffs = valid_ts.diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)] # 排除重复时间
            if not diffs.empty:
                freq_delta = diffs.mode().iloc[0] # 取出现最频繁的间隔
        
        ts_numeric = df[time_col].apply(lambda x: x.value if not pd.isnull(x) else np.nan)
        ts_numeric = ts_numeric.interpolate(method='linear', limit_direction='both')
        df[time_col] = pd.to_datetime(ts_numeric, unit='ns')
        
        last_valid_idx = df[time_col].last_valid_index()
        if last_valid_idx is not None and last_valid_idx < len(df) - 1:
            start_val = df.loc[last_valid_idx, time_col]
            offset_mask = df.index > last_valid_idx
            offsets = np.arange(1, offset_mask.sum() + 1)
            df.loc[offset_mask, time_col] = start_val + pd.to_timedelta(offsets * freq_delta)

        first_valid_idx = df[time_col].first_valid_index()
        if first_valid_idx is not None and first_valid_idx > 0:
            start_val = df.loc[first_valid_idx, time_col]
            offset_mask = df.index < first_valid_idx
            offsets = np.arange(offset_mask.sum(), 0, -1)
            df.loc[offset_mask, time_col] = start_val - pd.to_timedelta(offsets * freq_delta)

    # 3. 填充断档/Gap (Flag 2 核心)
    df = df.sort_values(by=time_col)
    
    # 再次推断频率，确保 Resample 准确
    inferred_freq = 'D'
    if len(df) >= 2:
        diffs = df[time_col].diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if not diffs.empty:
            inferred_freq = diffs.mode().iloc[0]

    # Resample 自动插入缺失的时间行
    df_indexed = df.set_index(time_col)
    # 使用 .first() 保留原始数据特性，新产生的行 _is_origin 会是 NaN
    df_resampled = df_indexed.resample(inferred_freq).first().reset_index()
    
    # 补全 item_id (新插入的行 item_id 是 NaN)
    if id_col in df_resampled.columns:
        df_resampled[id_col] = df_resampled[id_col].fillna(method='ffill').fillna(method='bfill')
        
    # 4. 转换为 AutoGluon TSDF 并填充数值
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df_resampled, 
        id_column=id_col, 
        timestamp_column=time_col
    )
    
    ts_filled = ts_df.fill_missing_values(
        method=fill_method,
        value=0.0
    )
    
    # 5. 生成结果与 Flags
    df_result = ts_filled.reset_index()
    
    # 初始化 flag 为 3 (正常)
    df_result['fill_flag'] = 3
    
    # 对齐逻辑：ts_filled 索引与 df_resampled 一致
    is_origin = df_resampled['_is_origin'] == True
    orig_date_miss = df_resampled['_orig_date_missing'] == True
    orig_val_miss = df_resampled['_orig_val_missing'] == True
    
    # Flag 2: 插入的断档行 (Gap) 或 原始全缺失
    mask_2 = (~is_origin) | (is_origin & orig_date_miss & orig_val_miss)
    df_result.loc[mask_2, 'fill_flag'] = 2
    
    # Flag 1: 原始行，时间缺失，值存在
    mask_1 = is_origin & orig_date_miss & (~orig_val_miss)
    df_result.loc[mask_1, 'fill_flag'] = 1
    
    # Flag 0: 原始行，时间存在，值缺失
    mask_0 = is_origin & (~orig_date_miss) & orig_val_miss
    df_result.loc[mask_0, 'fill_flag'] = 0
    
    # 6. 清理辅助列
    drop_cols = ['_is_origin', '_orig_date_missing', '_orig_val_missing']
    df_result.drop(columns=[c for c in drop_cols if c in df_result.columns], inplace=True)
    
    return df_result



def fill_missing_values(file_path: str, method: str, request_id: str = None):
    """
    缺失值填补入口函数
    """
    task_id = get_or_create_task_id(request_id)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    
    # 1. 读取数据
    df = pd.read_csv(file_path)
    
    # 2. 标准化列名 (Service 层负责适配用户上传的杂乱列名)
    time_col = 'timestamp'
    target_col = 'value'
    id_col = 'item_id'
    
    if 'date' in df.columns and time_col not in df.columns:
        df.rename(columns={'date': time_col}, inplace=True)
    if 'time' in df.columns and time_col not in df.columns:
        df.rename(columns={'time': time_col}, inplace=True)
        
    if id_col not in df.columns:
        df[id_col] = "default_item"
    
    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
            df.rename(columns={target_col: 'value'}, inplace=True)
            target_col = 'value'
            
    if time_col not in df.columns:
        raise ValueError("Could not identify timestamp column.")

    # 3. 调用核心处理模块
    df_result = clean_and_fill_timeseries(
        df=df,
        time_col=time_col,
        target_col=target_col,
        id_col=id_col,
        fill_method=method
    )
    
    if id_col in df_result.columns:
        df_result.drop(columns=[id_col], inplace=True)
        
    output_path = os.path.join(save_dir, "filled_data.csv")
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 只要 flag 不是 3，就算有填充发生
    has_filled = (df_result['fill_flag'] != 3).any()
    
    return output_path, has_filled


def calculate_correlation_matrix(file_path: str, method: str):
    """相关性分析"""
    task_id = get_or_create_task_id(None)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found.")
    
    corr_matrix = numeric_df.corr(method=method)
    
    save_path = os.path.join(save_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(save_path, index=True)
    return save_path

def select_features(task_id: str, fdr: float):
    """根据 FDR 进行特征筛选"""
    # 1. 寻找全量特征文件
    logger = setup_task_logger(task_id)
    update_task_state(task_id, "RUNNING", f"正在筛选特征 (FDR={fdr})...")
    logger.info(f"【筛选启动】开始进行特征筛选，设定 FDR = {fdr}")
    try:
        full_feat_dir = os.path.join(FEATURE_DIR, task_id)
        full_feat_path = os.path.join(full_feat_dir, "full_features.csv")
        logger.info(f"正在读取全量特征文件...")
        if not os.path.exists(full_feat_path):
            raise FileNotFoundError(f"Feature file for task {task_id} not found. Please run generate_all_features first.")
        
        df = pd.read_csv(full_feat_path)
        logger.info(f"数据读取成功，当前特征维度: {df.shape}")
        target_col = "value"
        if target_col not in df.columns: 
            raise ValueError("Target column 'value' missing in features file.")
        
        # 准备 X, y
        y = df[target_col]
        X = df.drop(columns=[target_col, "item_id", "timestamp"], errors='ignore').select_dtypes(include=[np.number])
        
        # 计算相关性
        logger.info("开始计算特征相关性 (Relevance Calculation)...")
        relevance = TimeSeriesFeatureGenerator.calculate_relevance_standalone(X, y, fdr_level=fdr)
        
        # 获取保留列
        if 'feature' in relevance.columns:
            keep = relevance[relevance['relevant'] == True]['feature'].tolist()
        else:
            keep = relevance[relevance['relevant'] == True].index.tolist()
        logger.info(f"筛选计算完成。保留特征数量: {len(keep)} / {len(X.columns)}")
        final_cols = ["item_id", "timestamp", "value"] + keep
        final_cols = [c for c in final_cols if c in df.columns]
        
        # 保存结果
        out_dir = prepare_output_dir(FEATURE_DIR, task_id)
        out_path = os.path.join(out_dir, f"selected_fdr_{fdr}.csv")
        df[final_cols].to_csv(out_path, index=False)

        update_task_state(task_id, "SUCCESS", "特征筛选完成", {
            "selected_count": len(keep),
            "dataset_path": out_path
        })
        logger.info("【筛选结束】任务成功完成。")
        return out_path
    except Exception as e:
        update_task_state(task_id, "FAILED", f"特征筛选失败: {str(e)}")
        raise e

def worker_generate_features(
    task_id: str, 
    file_path: str, 
    max_timeshift: int, 
    min_timeshift: int, 
    tsfresh_custom_params: list # <--- 修改类型提示：接收 list
):
    """后台任务：生成全量特征"""
    task_id = get_or_create_task_id(task_id)
    logger = setup_task_logger(task_id)
    
    try:
        logger.info(f"【任务启动】开始执行特征生成 (TaskID: {task_id})")
        update_task_state(task_id, "RUNNING", "Initializing data...")
        
        # 1. 准备数据
        logger.info(f"正在读取并校验数据文件: {file_path}")
        ts_df = prepare_tsdf(file_path, default_id=task_id, target_col='value')
        logger.info(f"数据加载成功，时间序列形状: {ts_df.shape}, 实体数量: {ts_df.num_items}")

        # 2. 解析 tsfresh 参数 (还原了你原版代码中的 eval 逻辑)
        parsed_params = "comprehensive"
        if tsfresh_custom_params and isinstance(tsfresh_custom_params, list):
            # 传入的已经是 List[Dict] 了，直接解析
            final_fc_parameters = parse_feature_config(tsfresh_custom_params)

        tsfresh_config = {
            "impute_strategy": "mean",
            "window_size": 30,
            "max_timeshift": max_timeshift,
            "min_timeshift": min_timeshift,
            "do_selection": False, 
            "fc_parameters": final_fc_parameters
        }

        # 3. 初始化生成器
        logger.info("初始化特征生成器...")
        feature_generator = TimeSeriesFeatureGenerator(
            target='value',
            known_covariates_names=[],
            use_tsfresh=True,
            tsfresh_settings=tsfresh_config,
            correlation_analysis=False 
        )

        update_task_state(task_id, "RUNNING", "Running Feature Generator...")
        logger.info(">>> AutoApex开始运行特征生成引擎...")
        logger.info("提示：此过程涉及大量计算，请耐心等待...") 
        full_tsdf_result = feature_generator.fit_transform(ts_df)
        logger.info("<<< 特征生成完成！")
        # 4. 保存结果
        save_dir = prepare_output_dir(FEATURE_DIR, task_id)
        save_path = os.path.join(save_dir, "full_features.csv")
        
        result_df = full_tsdf_result.reset_index()
        result_df.to_csv(save_path, index=False)
        
        feature_count = len(result_df.columns) - 2
        logger.info(f"任务结束。共生成特征数: {feature_count}")
        update_task_state(task_id, "SUCCESS", "Done", {"feature_count": feature_count, "download_path": save_path})

    except Exception as e:
        logger.error(f"特征生成任务失败: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", str(e))
    finally:
        # 清理原始上传的 CSV
        if os.path.exists(file_path):
            os.remove(file_path)