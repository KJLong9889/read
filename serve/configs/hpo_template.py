# # hpo_template.py
# import logging
# from autogluon.common import space as ag

# logger = logging.getLogger(__name__)

# # ================== 1. 所有支持模型的经验超参搜索空间 ==================
# # (保持你原有的 DEFAULT_MODEL_SPACES 不变)
# DEFAULT_MODEL_SPACES = { ... } 

# # ================== 2. [新增] 前后端解耦的类型校验 Schema ==================
# # 定义每个模型参数对应的 AutoGluon 空间类型。前端不再需要传 type 字段。
# PARAM_TYPE_SCHEMA = {
#     "ARIMA": {"order": "categorical", "seasonal_order": "categorical", "method": "categorical"},
#     "AutoARIMA": {
#         "max_p": "int", "max_q": "int", "max_P": "int", "max_Q": "int", 
#         "max_d": "int", "max_D": "int", "start_p": "int", "seasonal": "categorical"
#     },
#     "ETS": {"model": "categorical", "damped": "categorical"},
#     "AutoCES": {"model": "categorical"},
#     "Theta": {"decomposition_type": "categorical"},
#     "Croston": {"variant": "categorical"},
#     "TemporalFusionTransformer": {
#         "hidden_dim": "categorical", "num_heads": "categorical", 
#         "dropout_rate": "real", "batch_size": "categorical", "lr": "real"
#     },
#     "PatchTST": {
#         "patch_len": "categorical", "stride": "categorical", "d_model": "categorical", 
#         "nhead": "categorical", "num_encoder_layers": "int", "scaling": "categorical", 
#         "batch_size": "categorical", "lr": "real"
#     },
#     "TiDE": {
#         "encoder_hidden_dim": "categorical", "decoder_hidden_dim": "categorical", 
#         "temporal_hidden_dim": "categorical", "distr_hidden_dim": "categorical", 
#         "num_layers_encoder": "int", "num_layers_decoder": "int", 
#         "dropout_rate": "real", "layer_norm": "categorical", "batch_size": "categorical", "lr": "real"
#     },
#     "DLinear": {"hidden_dimension": "int"},
#     "WaveNet": {
#         "num_bins": "int", "use_log_scale_feature": "categorical", 
#         "negative_data": "categorical", "batch_size": "categorical", 
#         "lr": "real", "weight_decay": "real"
#     },
#     "SimpleFeedForward": {
#         "batch_normalization": "categorical", "mean_scaling": "categorical", 
#         "batch_size": "categorical", "lr": "real"
#     },
#     "Chronos": {"model_path": "categorical"},
#     "RecursiveTabular": {"lags": "categorical", "model_name": "categorical"},
#     "DirectTabular": {"lags": "categorical", "model_name": "categorical"}
# }

# # ================== 3. 增强版后端智能解析器 ==================
# def _parse_single_space(user_val, expected_type):
#     """
#     根据后端 Schema 定义的 expected_type，智能解析前端传来的简化值
#     """
#     # 1. 解析 Categorical
#     if expected_type == "categorical":
#         # 兼容处理：前端传的是纯列表 [ ... ]
#         vals = user_val if isinstance(user_val, list) else [user_val]
        
#         parsed_vals = []
#         for v in vals:
#             if isinstance(v, list): # 还原 JSON 列表为 Python 元组 (例如 ARIMA 的 order)
#                 parsed_vals.append(tuple(v))
#             elif isinstance(v, str): # 拦截布尔字符串
#                 if v.lower() == "true": parsed_vals.append(True)
#                 elif v.lower() == "false": parsed_vals.append(False)
#                 elif v.lower() == "none": parsed_vals.append(None)
#                 else: parsed_vals.append(v)
#             else:
#                 parsed_vals.append(v)
        
#         if parsed_vals: return ag.Categorical(*parsed_vals)
#         return None

#     # 2. 解析 Int 范围
#     elif expected_type == "int":
#         if isinstance(user_val, dict) and "start" in user_val and "end" in user_val:
#             return ag.Int(lower=int(user_val["start"]), upper=int(user_val["end"]))
            
#     # 3. 解析 Real 范围
#     elif expected_type == "real":
#         if isinstance(user_val, dict) and "start" in user_val and "end" in user_val:
#             # 前端如果不传 log，默认为 False
#             return ag.Real(lower=float(user_val["start"]), upper=float(user_val["end"]), log=user_val.get("log", False))

#     # --- 兜底逻辑：如果 Schema 里没定义，或者前端传的是固定标量 ---
#     if isinstance(user_val, str):
#         if user_val.lower() == "true": return True
#         if user_val.lower() == "false": return False
#         if user_val.lower() == "none": return None
        
#     return user_val

# def parse_hpo_config(user_hyperparameters):
#     """
#     结合默认经验配置和前端自定义配置，生成 AutoGluon 的最终超参字典。
#     """
#     final_hps = {}
#     if not user_hyperparameters:
#         return DEFAULT_MODEL_SPACES

#     user_dict = {}
#     if isinstance(user_hyperparameters, list):
#         for item in user_hyperparameters:
#             user_dict[item["name"]] = item.get("params", {})
#     elif isinstance(user_hyperparameters, dict):
#         user_dict = user_hyperparameters

#     for model_name, params in user_dict.items():
#         model_hps = {}
#         if model_name in DEFAULT_MODEL_SPACES:
#             model_hps.update(DEFAULT_MODEL_SPACES[model_name].copy())
        
#         for p_name, p_val in params.items():
#             if p_val is None:
#                 continue
            
#             # 【核心逻辑】：去 Schema 中查阅这个参数应该被解析成什么类型
#             expected_type = PARAM_TYPE_SCHEMA.get(model_name, {}).get(p_name, "")
            
#             # 将查到的类型和前端传来的值一起交给解析器
#             model_hps[p_name] = _parse_single_space(p_val, expected_type)
            
#         final_hps[model_name] = model_hps
        
#     return final_hps



import logging
from autogluon.common import space as ag

logger = logging.getLogger(__name__)

# ================== 1. 默认经验超参搜索空间 (示意/保持原有) ==================
DEFAULT_MODEL_SPACES = {
    "AutoARIMA": {},
    "ETS": {},
    "TemporalFusionTransformer": {},
    "TiDE": {},
    # ... 用户原有默认空间
}

# ================== 2. 前后端解耦的类型校验 Schema ==================
PARAM_TYPE_SCHEMA = {
    "ARIMA": {"order": "categorical", "seasonal_order": "categorical", "method": "categorical"},
    "AutoARIMA": {
        "max_p": "int", "max_q": "int", "max_P": "int", "max_Q": "int", 
        "max_d": "int", "max_D": "int", "start_p": "int", "seasonal": "categorical"
    },
    "ETS": {"model": "categorical", "damped": "categorical"},
    "AutoCES": {"model": "categorical"},
    "Theta": {"decomposition_type": "categorical"},
    "Croston": {"variant": "categorical"},
    "TemporalFusionTransformer": {
        "hidden_dim": "categorical", "num_heads": "categorical", 
        "dropout_rate": "real", "batch_size": "categorical", "lr": "real"
    },
    "PatchTST": {
        "patch_len": "categorical", "stride": "categorical", "d_model": "categorical", 
        "nhead": "categorical", "num_encoder_layers": "int", "scaling": "categorical", 
        "batch_size": "categorical", "lr": "real"
    },
    "TiDE": {
        "encoder_hidden_dim": "categorical", "decoder_hidden_dim": "categorical", 
        "temporal_hidden_dim": "categorical", "distr_hidden_dim": "categorical", 
        "num_layers_encoder": "int", "num_layers_decoder": "int", 
        "dropout_rate": "real", "layer_norm": "categorical", "batch_size": "categorical", "lr": "real"
    },
    "DLinear": {"hidden_dimension": "int"},
    "WaveNet": {
        "num_bins": "int", "use_log_scale_feature": "categorical", 
        "negative_data": "categorical", "batch_size": "categorical", 
        "lr": "real", "weight_decay": "real"
    },
    "SimpleFeedForward": {
        "batch_normalization": "categorical", "mean_scaling": "categorical", 
        "batch_size": "categorical", "lr": "real"
    },
    "Chronos": {"model_path": "categorical"},
    "RecursiveTabular": {"lags": "categorical", "model_name": "categorical"},
    "DirectTabular": {"lags": "categorical", "model_name": "categorical"}
}

# ================== 3. 增强版后端智能解析器 ==================
def _parse_single_space(user_val, expected_type, param_name="unknown"):
    """
    根据 Schema 智能解析并添加 AutoGluon 搜索空间的严格安全校验
    """
    if user_val is None:
        return None

    try:
        # 1. 解析 Categorical
        if expected_type == "categorical":
            vals = user_val if isinstance(user_val, list) else [user_val]
            
            parsed_vals = []
            for v in vals:
                if isinstance(v, list): 
                    parsed_vals.append(tuple(v))
                elif isinstance(v, str): 
                    v_lower = v.lower().strip()
                    if v_lower == "true": parsed_vals.append(True)
                    elif v_lower == "false": parsed_vals.append(False)
                    elif v_lower == "none": parsed_vals.append(None)
                    else: parsed_vals.append(v)
                else:
                    parsed_vals.append(v)
            
            if parsed_vals: 
                # 去重防御，防止用户传 [32, 32, 64]
                unique_vals = list(dict.fromkeys(parsed_vals))
                # AutoGluon 优化：如果只有一个值，直接传具体值而非搜索空间
                return ag.Categorical(*unique_vals) if len(unique_vals) > 1 else unique_vals[0]
            else:
                logger.warning(f"Categorical '{param_name}' received empty options. Ignoring.")
                return None

        # 2. 解析 Int 范围
        elif expected_type == "int":
            if isinstance(user_val, dict):
                # 使用 float 中转防止 "1.0" 字符串转 int 报错
                lower = int(float(user_val.get("start", 0)))
                upper = int(float(user_val.get("end", lower + 1)))
                
                # === 💡 新增：智能互换与相等判定 ===
                if lower > upper:
                    logger.info(f"Int space '{param_name}' 填反了 (start={lower} > end={upper})。系统自动互换。")
                    lower, upper = upper, lower
                
                if lower == upper:
                    logger.info(f"Int space '{param_name}' start == end ({lower})。降级为固定常量。")
                    return lower 
                # ===============================
                    
                return ag.Int(lower=lower, upper=upper)
                
        # 3. 解析 Real 范围
        elif expected_type == "real":
            if isinstance(user_val, dict):
                lower = float(user_val.get("start", 0.0))
                upper = float(user_val.get("end", lower + 1.0))
                
                log_val = user_val.get("log", False)
                log_flag = str(log_val).lower().strip() == "true" if isinstance(log_val, str) else bool(log_val)
                
                # === 💡 新增：智能互换与相等判定 ===
                if lower > upper:
                    logger.info(f"Real space '{param_name}' 填反了 (start={lower} > end={upper})。系统自动互换。")
                    lower, upper = upper, lower
                    
                if lower == upper:
                    logger.info(f"Real space '{param_name}' start == end ({lower})。降级为固定常量。")
                    return lower
                # ===============================
                    
                return ag.Real(lower=lower, upper=upper, log=log_flag)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to parse space for '{param_name}' with value {user_val}. Error: {e}")
        return None # 解析失败，外层逻辑会回退使用默认值

    # --- 兜底逻辑：处理前端传来的标量字符串布尔值等 ---
    if isinstance(user_val, str):
        v_lower = user_val.lower().strip()
        if v_lower == "true": return True
        if v_lower == "false": return False
        if v_lower == "none": return None
        
    return user_val



def parse_hpo_config(user_hyperparameters):
    """
    结合默认经验配置和前端自定义配置，生成 AutoGluon 最终超参字典。
    """
    final_hps = {}
    
    # 场景 1：前端完全没有传超参配置（比如设为 None 或 []）
    # 此时我们认为用户想要“一键全自动”，直接返回后端的经验默认搜索空间
    if not user_hyperparameters:
        logger.info("No user hyperparameters provided. Using DEFAULT_MODEL_SPACES.")
        return DEFAULT_MODEL_SPACES

    user_dict = {}
    
    # 【结构防御】确保能够安全提取到字典结构，无视非法 JSON 元素
    if isinstance(user_hyperparameters, list):
        for item in user_hyperparameters:
            if isinstance(item, dict) and "name" in item:
                params = item.get("params")
                user_dict[item["name"]] = params if isinstance(params, dict) else {}
    elif isinstance(user_hyperparameters, dict):
        user_dict = {str(k): (v if isinstance(v, dict) else {}) for k, v in user_hyperparameters.items()}
    else:
        logger.error(f"Invalid root structure for hyperparameters: {type(user_hyperparameters)}. Falling back to defaults.")
        return DEFAULT_MODEL_SPACES

    # 遍历合并配置
    for model_name, params in user_dict.items():
        # 【核心修复】：如果 params 为空，说明用户明确不想对这个模型进行超参寻优
        if not params:
            logger.info(f"Model '{model_name}' received empty params. Disabling HPO (using fixed defaults).")
            final_hps[model_name] = {}
            continue
            
        model_hps = {}
        # 注意：这里删除了 update(DEFAULT_MODEL_SPACES) 的强制继承逻辑
        # 我们只忠实地解析并传递用户显式配置的参数
        
        for p_name, p_val in params.items():
            if p_val is None:
                continue
            
            expected_type = PARAM_TYPE_SCHEMA.get(model_name, {}).get(p_name, "")
            
            try:
                parsed_space = _parse_single_space(p_val, expected_type, param_name=p_name)
                # 只有解析出有效值，才放入该模型的参数字典中
                if parsed_space is not None:
                    model_hps[p_name] = parsed_space
            except Exception as e:
                logger.error(f"Critical error mapping parameter '{p_name}' for model '{model_name}': {e}")
                pass 
            
        final_hps[model_name] = model_hps
        
    return final_hps