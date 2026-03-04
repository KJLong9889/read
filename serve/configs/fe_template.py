# from itertools import product

# # ================== 0. 参数解析工具函数 ==================
# def _str_to_bool(val):
#     """辅助函数：将字符串格式的 true/false/True/False 统一转换为 Python 布尔值"""
#     if isinstance(val, str):
#         if val.lower() == 'true':
#             return True
#         elif val.lower() == 'false':
#             return False
#         elif val.lower() == 'none':
#             return None
#     return val

# def resolve_param(user_val, default_list):
#     """
#     解析前端传入的单个参数值，将其统一转换为列表形式，并处理字符串布尔值。
#     如果前端没有传该参数，则回退使用默认值 default_list。
#     """
#     if user_val is None:
#         return default_list
    
#     # 1. 集合类型：前端传的是列表，遍历并转换里面的字符串布尔值
#     if isinstance(user_val, list):
#         return [_str_to_bool(v) for v in user_val]
        
#     # 2. 范围生成器：前端传的是字典，包含 start 和 end (注意：end 是开区间)
#     if isinstance(user_val, dict) and "start" in user_val and "end" in user_val:
#         start = user_val.get("start")
#         end = user_val.get("end")
#         step = user_val.get("step", 1)
#         multiplier = user_val.get("multiplier", 1) 
        
#         grid = []
#         for i in range(start, end, step):
#             val = i * multiplier
#             # 简单处理浮点数精度问题，避免 0.15000000000000002 这样的值
#             grid.append(round(val, 6) if isinstance(multiplier, float) else val)
#         return grid
        
#     # 3. 单个固定值：转换布尔值后包装成列表返回
#     return [_str_to_bool(user_val)]


# # ================== 1. 策略函数定义 ==================

# def time_reversal_asymmetry_statistic(params=None):
#     params = params or {}
#     lag_grid = resolve_param(params.get("lag"), list(range(1, 4)))
#     return [{"lag": lag} for lag in lag_grid]

# def c3(params=None):
#     params = params or {}
#     lag_grid = resolve_param(params.get("lag"), list(range(1, 4)))
#     return [{"lag": lag} for lag in lag_grid]

# def cid_ce(params=None):
#     params = params or {}
#     norm_grid = resolve_param(params.get("normalize"), [True, False])
#     return [{"normalize": norm} for norm in norm_grid]

# def symmetry_looking(params=None):
#     params = params or {}
#     r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(20)])
#     return [{"r": r} for r in r_grid]

# def large_standard_deviation(params=None):
#     params = params or {}
#     r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(1, 20)])
#     return [{"r": r} for r in r_grid]

# def quantile(params=None):
#     params = params or {}
#     q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
#     return [{"q": q} for q in q_grid]

# def autocorrelation(params=None):
#     params = params or {}
#     lag_grid = resolve_param(params.get("lag"), list(range(10)))
#     return [{"lag": lag} for lag in lag_grid]

# def agg_autocorrelation(params=None):
#     params = params or {}
#     f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "median", "var"])
#     maxlag_grid = resolve_param(params.get("maxlag"), [40])
#     return [{"f_agg": f, "maxlag": m} for f in f_agg_grid for m in maxlag_grid]

# def partial_autocorrelation(params=None):
#     params = params or {}
#     lag_grid = resolve_param(params.get("lag"), list(range(10)))
#     return [{"lag": lag} for lag in lag_grid]

# def number_cwt_peaks(params=None):
#     params = params or {}
#     n_grid = resolve_param(params.get("n"), [1, 5])
#     return [{"n": n} for n in n_grid]

# def number_peaks(params=None):
#     params = params or {}
#     n_grid = resolve_param(params.get("n"), [1, 3, 5, 10, 50])
#     return [{"n": n} for n in n_grid]

# def binned_entropy(params=None):
#     params = params or {}
#     max_bins_grid = resolve_param(params.get("max_bins"), [10])
#     return [{"max_bins": b} for b in max_bins_grid]

# def index_mass_quantile(params=None):
#     params = params or {}
#     q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
#     return [{"q": q} for q in q_grid]

# def cwt_coefficients(params=None):
#     params = params or {}
#     raw_widths = params.get("widths")
#     if isinstance(raw_widths, list):
#         # 如果前端传来的是嵌套列表 [[2, 5...]] (旧逻辑兼容)
#         if len(raw_widths) > 0 and isinstance(raw_widths[0], list):
#             widths_grid = [tuple(w) for w in raw_widths]
#         else:
#             # 如果前端传来的是平铺的一维列表 [2, 5...] (新逻辑)
#             # 把它作为一个整体转换为 tuple，并放进 list 中供后续循环组合
#             widths_grid = [tuple(raw_widths)]
#     else:
#         # 默认回退值
#         widths_grid = [(2, 5, 10, 20)]
        
#     coeff_grid = resolve_param(params.get("coeff"), list(range(15)))
#     w_grid = resolve_param(params.get("w"), [2, 5, 10, 20])
    
#     return [{"widths": wid, "coeff": c, "w": weight} 
#             for wid in widths_grid for c in coeff_grid for weight in w_grid]

# def spkt_welch_density(params=None):
#     params = params or {}
#     coeff_grid = resolve_param(params.get("coeff"), [2, 5, 8])
#     return [{"coeff": c} for c in coeff_grid]

# def ar_coefficient(params=None):
#     params = params or {}
#     coeff_grid = resolve_param(params.get("coeff"), list(range(11)))
#     k_grid = resolve_param(params.get("k"), [10])
#     return [{"coeff": c, "k": k} for c in coeff_grid for k in k_grid]

# def change_quantiles(params=None):
#     params = params or {}
#     ql_grid = resolve_param(params.get("ql"), [0.0, 0.2, 0.4, 0.6, 0.8])
#     qh_grid = resolve_param(params.get("qh"), [0.2, 0.4, 0.6, 0.8, 1.0])
#     isabs_grid = resolve_param(params.get("isabs"), [False, True])
#     f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "var"])
#     # 强制约束 ql 必须小于 qh
#     return [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f} 
#             for ql in ql_grid for qh in qh_grid for b in isabs_grid for f in f_agg_grid if ql < qh]

# def fft_coefficient(params=None):
#     params = params or {}
#     coeff_grid = resolve_param(params.get("coeff"), list(range(100)))
#     attr_grid = resolve_param(params.get("attr"), ["real", "imag", "abs", "angle"])
#     return [{"coeff": c, "attr": a} for a in attr_grid for c in coeff_grid]

# def fft_aggregated(params=None):
#     params = params or {}
#     aggtype_grid = resolve_param(params.get("aggtype"), ["centroid", "variance", "skew", "kurtosis"])
#     return [{"aggtype": s} for s in aggtype_grid]

# def value_count(params=None):
#     params = params or {}
#     value_grid = resolve_param(params.get("value"), [0, 1, -1])
#     return [{"value": v} for v in value_grid]

# def range_count(params=None):
#     params = params or {}
#     # 特殊处理：如果没有传参，返回默认的三个区间对配置
#     if not params:
#         return [{"min": -1, "max": 1}, {"min": -1e12, "max": 0}, {"min": 0, "max": 1e12}]
        
#     min_grid = resolve_param(params.get("min"), [-1])
#     max_grid = resolve_param(params.get("max"), [1])
    
#     # 如果 min 和 max 的长度一样，按顺序一一对应组成区间
#     if len(min_grid) == len(max_grid):
#         return [{"min": mn, "max": mx} for mn, mx in zip(min_grid, max_grid)]
#     else:
#         # 长度不一致时退回笛卡尔积，并增加防错过滤
#         return [{"min": mn, "max": mx} for mn in min_grid for mx in max_grid if mn < mx]

# def approximate_entropy(params=None):
#     params = params or {}
#     m_grid = resolve_param(params.get("m"), [2])
#     r_grid = resolve_param(params.get("r"), [0.1, 0.3, 0.5, 0.7, 0.9])
#     return [{"m": m, "r": r} for m in m_grid for r in r_grid]

# def friedrich_coefficients(params=None):
#     params = params or {}
#     m_grid = resolve_param(params.get("m"), [3])
#     r_grid = resolve_param(params.get("r"), [30])
#     res = []
#     for m in m_grid:
#         for r in r_grid:
#             # coeff 默认范围依赖于当前的 m
#             coeff_grid = resolve_param(params.get("coeff"), list(range(m + 1)))
#             for c in coeff_grid:
#                 res.append({"coeff": c, "m": m, "r": r})
#     return res

# def max_langevin_fixed_point(params=None):
#     params = params or {}
#     m_grid = resolve_param(params.get("m"), [3])
#     r_grid = resolve_param(params.get("r"), [30])
#     return [{"m": m, "r": r} for m in m_grid for r in r_grid]

# def linear_trend(params=None):
#     params = params or {}
#     attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"])
#     return [{"attr": a} for a in attr_grid]

# def agg_linear_trend(params=None):
#     params = params or {}
#     attr_grid = resolve_param(params.get("attr"), ["rvalue", "intercept", "slope", "stderr"])
#     chunk_len_grid = resolve_param(params.get("chunk_len"), [5, 10, 50])
#     f_agg_grid = resolve_param(params.get("f_agg"), ["max", "min", "mean", "var"])
#     return [{"attr": a, "chunk_len": c, "f_agg": f} for a in attr_grid for c in chunk_len_grid for f in f_agg_grid]

# def augmented_dickey_fuller(params=None):
#     params = params or {}
#     attr_grid = resolve_param(params.get("attr"), ["teststat", "pvalue", "usedlag"])
#     return [{"attr": a} for a in attr_grid]

# def number_crossing_m(params=None):
#     params = params or {}
#     m_grid = resolve_param(params.get("m"), [0, -1, 1])
#     return [{"m": m} for m in m_grid]

# def energy_ratio_by_chunks(params=None):
#     params = params or {}
#     num_segments_grid = resolve_param(params.get("num_segments"), [10])
#     res = []
#     for num in num_segments_grid:
#         # segment_focus 默认范围依赖于当前的 num_segments
#         focus_grid = resolve_param(params.get("segment_focus"), list(range(num)))
#         for f in focus_grid:
#             res.append({"num_segments": num, "segment_focus": f})
#     return res

# def ratio_beyond_r_sigma(params=None):
#     params = params or {}
#     r_grid = resolve_param(params.get("r"), [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10])
#     return [{"r": x} for x in r_grid]

# def linear_trend_timewise(params=None):
#     params = params or {}
#     attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"])
#     return [{"attr": a} for a in attr_grid]

# def count_above(params=None):
#     params = params or {}
#     t_grid = resolve_param(params.get("t"), [0])
#     return [{"t": t} for t in t_grid]

# def count_below(params=None):
#     params = params or {}
#     t_grid = resolve_param(params.get("t"), [0])
#     return [{"t": t} for t in t_grid]

# def lempel_ziv_complexity(params=None):
#     params = params or {}
#     bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100])
#     return [{"bins": x} for x in bins_grid]

# def fourier_entropy(params=None):
#     params = params or {}
#     bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100])
#     return [{"bins": x} for x in bins_grid]

# def permutation_entropy(params=None):
#     params = params or {}
#     tau_grid = resolve_param(params.get("tau"), [1])
#     dim_grid = resolve_param(params.get("dimension"), [3, 4, 5, 6, 7])
#     return [{"tau": t, "dimension": d} for t in tau_grid for d in dim_grid]

# def query_similarity_count(params=None):
#     params = params or {}
#     query_grid = resolve_param(params.get("query"), [None])
#     threshold_grid = resolve_param(params.get("threshold"), [0.0])
#     return [{"query": q, "threshold": t} for q in query_grid for t in threshold_grid]

# def matrix_profile(params=None):
#     params = params or {}
#     threshold_grid = resolve_param(params.get("threshold"), [0.98])
#     feature_grid = resolve_param(params.get("feature"), ["min", "max", "mean", "median", "25", "75"])
#     return [{"threshold": t, "feature": f} for t in threshold_grid for f in feature_grid]

# def mean_n_absolute_max(params=None):
#     params = params or {}
#     n_grid = resolve_param(params.get("number_of_maxima"), [3, 5, 7])
#     return [{"number_of_maxima": n} for n in n_grid]

# # ================== 2. 注册表 ==================
# FEATURE_HANDLERS = {
#     "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
#     "c3": c3,
#     "cid_ce": cid_ce,
#     "symmetry_looking": symmetry_looking,
#     "large_standard_deviation": large_standard_deviation,
#     "quantile": quantile,
#     "autocorrelation": autocorrelation,
#     "agg_autocorrelation": agg_autocorrelation,
#     "partial_autocorrelation": partial_autocorrelation,
#     "number_cwt_peaks": number_cwt_peaks,
#     "number_peaks": number_peaks,
#     "binned_entropy": binned_entropy,
#     "index_mass_quantile": index_mass_quantile,
#     "cwt_coefficients": cwt_coefficients,
#     "spkt_welch_density": spkt_welch_density,
#     "ar_coefficient": ar_coefficient,
#     "change_quantiles": change_quantiles,
#     "fft_coefficient": fft_coefficient,
#     "fft_aggregated": fft_aggregated,
#     "value_count": value_count,
#     "range_count": range_count,
#     "approximate_entropy": approximate_entropy,
#     "friedrich_coefficients": friedrich_coefficients,
#     "max_langevin_fixed_point": max_langevin_fixed_point,
#     "linear_trend": linear_trend,
#     "agg_linear_trend": agg_linear_trend,
#     "augmented_dickey_fuller": augmented_dickey_fuller,
#     "number_crossing_m": number_crossing_m,
#     "energy_ratio_by_chunks": energy_ratio_by_chunks,
#     "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
#     "linear_trend_timewise": linear_trend_timewise,
#     "count_above": count_above,
#     "count_below": count_below,
#     "lempel_ziv_complexity": lempel_ziv_complexity,
#     "fourier_entropy": fourier_entropy,
#     "permutation_entropy": permutation_entropy,
#     "query_similarity_count": query_similarity_count,
#     "matrix_profile": matrix_profile,
#     "mean_n_absolute_max": mean_n_absolute_max,
# }

# # ================== 3. 解析逻辑 ==================
# def parse_feature_config(user_config_list):
#     """
#     参数: user_config_list (List[Dict]) 
#     示例: [{"name": "cid_ce", "params": {"normalize": ["True", "False"]}}, ...]
#     """
#     fc_parameters = {}
    
#     # 如果没传任何配置，返回全量标记给底层
#     if not user_config_list:
#         return "comprehensive"
        
#     for item in user_config_list:
#         fname = item.get("name")
#         user_params = item.get("params", {}) 
        
#         handler = FEATURE_HANDLERS.get(fname)
#         if handler:
#             try:
#                 # 提取参数字典抛给对应的处理函数
#                 fc_parameters[fname] = handler(params=user_params)
#             except Exception as e:
#                 print(f"Error generating feature {fname}: {e}")
#         else:
#             print(f"Warning: Unknown feature name '{fname}', skipping.")
            
#     return fc_parameters


import logging
from itertools import product

logger = logging.getLogger(__name__)

# ================== 0. 参数解析工具函数 ==================
def _str_to_bool(val):
    """辅助函数：安全地将字符串格式的 true/false/none 转换为 Python 内置类型"""
    if isinstance(val, str):
        val_lower = val.lower().strip()
        if val_lower == 'true': return True
        if val_lower == 'false': return False
        if val_lower == 'none': return None
    return val

def resolve_param(user_val, default_list, param_name="unknown"):
    """
    健壮的参数解析器：增加类型防御、异常捕获和边界防御
    """
    if user_val is None:
        return default_list
    
# 2. 范围生成器：增加全面的异常捕获、智能互换和边界防御
    if isinstance(user_val, dict):
        if "start" in user_val and "end" in user_val:
            try:
                # 强制类型转换，防止前端传字符串数字
                start = int(float(user_val.get("start", 0))) 
                end = int(float(user_val.get("end", 1)))
                step = int(float(user_val.get("step", 1)))
                multiplier = user_val.get("multiplier", 1)
                
                # 防御 step 为 0 导致的死循环
                if step == 0: 
                    logger.warning(f"Parameter '{param_name}' has step=0. Automatically adjusting to step=1.")
                    step = 1 
                
                # === 💡 新增：智能互换逻辑 ===
                if start > end and step > 0:
                    logger.info(f"'{param_name}' 范围填反了 (start={start} > end={end})。系统自动为您互换。")
                    start, end = end, start  # Python 的优雅互换语法
                elif start < end and step < 0:
                    logger.info(f"'{param_name}' 步长为负但 start < end。系统自动为您互换。")
                    start, end = end, start
                elif start == end:
                    logger.info(f"'{param_name}' start 等于 end ({start})。退化为单一固定值。")
                    val = start * multiplier
                    return [round(val, 6) if isinstance(multiplier, float) else val]
                # ===============================

                grid = []
                for i in range(start, end, step):
                    val = i * multiplier
                    grid.append(round(val, 6) if isinstance(multiplier, float) else val)
                
                if not grid:
                    raise ValueError("Generated grid is empty.")
                return grid
                
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse range dict for '{param_name}': {user_val}. Error: {e}. Falling back to defaults.")
                return default_list
        else:
            logger.warning(f"Dict provided for '{param_name}' but missing 'start' or 'end'. Using defaults.")
            return default_list
        
    # 3. 单个固定值：转换布尔值后包装成列表返回
    try:
        return [_str_to_bool(user_val)]
    except Exception as e:
        logger.error(f"Unexpected value type for '{param_name}': {user_val}. Error: {e}. Using defaults.")
        return default_list


# ================== 1. 策略函数定义 ==================
# 使用 param_name 追踪日志，方便排查是哪个参数报错

def time_reversal_asymmetry_statistic(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(1, 4)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def c3(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(1, 4)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def cid_ce(params=None):
    params = params or {}
    norm_grid = resolve_param(params.get("normalize"), [True, False], "normalize")
    return [{"normalize": norm} for norm in norm_grid]

def symmetry_looking(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(20)], "r")
    return [{"r": r} for r in r_grid]

def large_standard_deviation(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [i * 0.05 for i in range(1, 20)], "r")
    return [{"r": r} for r in r_grid]

def quantile(params=None):
    params = params or {}
    q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], "q")
    return [{"q": q} for q in q_grid]

def autocorrelation(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(10)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def agg_autocorrelation(params=None):
    params = params or {}
    f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "median", "var"], "f_agg")
    maxlag_grid = resolve_param(params.get("maxlag"), [40], "maxlag")
    return [{"f_agg": f, "maxlag": m} for f in f_agg_grid for m in maxlag_grid]

def partial_autocorrelation(params=None):
    params = params or {}
    lag_grid = resolve_param(params.get("lag"), list(range(10)), "lag")
    return [{"lag": lag} for lag in lag_grid]

def number_cwt_peaks(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("n"), [1, 5], "n")
    return [{"n": n} for n in n_grid]

def number_peaks(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("n"), [1, 3, 5, 10, 50], "n")
    return [{"n": n} for n in n_grid]

def binned_entropy(params=None):
    params = params or {}
    max_bins_grid = resolve_param(params.get("max_bins"), [10], "max_bins")
    return [{"max_bins": b} for b in max_bins_grid]

def index_mass_quantile(params=None):
    params = params or {}
    q_grid = resolve_param(params.get("q"), [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], "q")
    return [{"q": q} for q in q_grid]

def cwt_coefficients(params=None):
    params = params or {}
    raw_widths = params.get("widths")
    if isinstance(raw_widths, list):
        if len(raw_widths) > 0 and isinstance(raw_widths[0], list):
            widths_grid = [tuple(w) for w in raw_widths]
        else:
            widths_grid = [tuple(raw_widths)]
    else:
        widths_grid = [(2, 5, 10, 20)]
        
    coeff_grid = resolve_param(params.get("coeff"), list(range(15)), "coeff")
    w_grid = resolve_param(params.get("w"), [2, 5, 10, 20], "w")
    
    return [{"widths": wid, "coeff": c, "w": weight} for wid in widths_grid for c in coeff_grid for weight in w_grid]

def spkt_welch_density(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), [2, 5, 8], "coeff")
    return [{"coeff": c} for c in coeff_grid]

def ar_coefficient(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), list(range(11)), "coeff")
    k_grid = resolve_param(params.get("k"), [10], "k")
    return [{"coeff": c, "k": k} for c in coeff_grid for k in k_grid]

def change_quantiles(params=None):
    params = params or {}
    ql_grid = resolve_param(params.get("ql"), [0.0, 0.2, 0.4, 0.6, 0.8], "ql")
    qh_grid = resolve_param(params.get("qh"), [0.2, 0.4, 0.6, 0.8, 1.0], "qh")
    isabs_grid = resolve_param(params.get("isabs"), [False, True], "isabs")
    f_agg_grid = resolve_param(params.get("f_agg"), ["mean", "var"], "f_agg")
    return [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f} 
            for ql in ql_grid for qh in qh_grid for b in isabs_grid for f in f_agg_grid if ql < qh]

def fft_coefficient(params=None):
    params = params or {}
    coeff_grid = resolve_param(params.get("coeff"), list(range(100)), "coeff")
    attr_grid = resolve_param(params.get("attr"), ["real", "imag", "abs", "angle"], "attr")
    return [{"coeff": c, "attr": a} for a in attr_grid for c in coeff_grid]

def fft_aggregated(params=None):
    params = params or {}
    aggtype_grid = resolve_param(params.get("aggtype"), ["centroid", "variance", "skew", "kurtosis"], "aggtype")
    return [{"aggtype": s} for s in aggtype_grid]

def value_count(params=None):
    params = params or {}
    value_grid = resolve_param(params.get("value"), [0, 1, -1], "value")
    return [{"value": v} for v in value_grid]

def range_count(params=None):
    params = params or {}
    if not params:
        return [{"min": -1, "max": 1}, {"min": -1e12, "max": 0}, {"min": 0, "max": 1e12}]
        
    min_grid = resolve_param(params.get("min"), [-1], "min")
    max_grid = resolve_param(params.get("max"), [1], "max")
    
    if len(min_grid) == len(max_grid):
        return [{"min": mn, "max": mx} for mn, mx in zip(min_grid, max_grid)]
    else:
        return [{"min": mn, "max": mx} for mn in min_grid for mx in max_grid if mn < mx]

def approximate_entropy(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [2], "m")
    r_grid = resolve_param(params.get("r"), [0.1, 0.3, 0.5, 0.7, 0.9], "r")
    return [{"m": m, "r": r} for m in m_grid for r in r_grid]

def friedrich_coefficients(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [3], "m")
    r_grid = resolve_param(params.get("r"), [30], "r")
    res = []
    for m in m_grid:
        for r in r_grid:
            coeff_grid = resolve_param(params.get("coeff"), list(range(m + 1)), "coeff")
            for c in coeff_grid:
                res.append({"coeff": c, "m": m, "r": r})
    return res

def max_langevin_fixed_point(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [3], "m")
    r_grid = resolve_param(params.get("r"), [30], "r")
    return [{"m": m, "r": r} for m in m_grid for r in r_grid]

def linear_trend(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"], "attr")
    return [{"attr": a} for a in attr_grid]

def agg_linear_trend(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["rvalue", "intercept", "slope", "stderr"], "attr")
    chunk_len_grid = resolve_param(params.get("chunk_len"), [5, 10, 50], "chunk_len")
    f_agg_grid = resolve_param(params.get("f_agg"), ["max", "min", "mean", "var"], "f_agg")
    return [{"attr": a, "chunk_len": c, "f_agg": f} for a in attr_grid for c in chunk_len_grid for f in f_agg_grid]

def augmented_dickey_fuller(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["teststat", "pvalue", "usedlag"], "attr")
    return [{"attr": a} for a in attr_grid]

def number_crossing_m(params=None):
    params = params or {}
    m_grid = resolve_param(params.get("m"), [0, -1, 1], "m")
    return [{"m": m} for m in m_grid]

def energy_ratio_by_chunks(params=None):
    params = params or {}
    num_segments_grid = resolve_param(params.get("num_segments"), [10], "num_segments")
    res = []
    for num in num_segments_grid:
        focus_grid = resolve_param(params.get("segment_focus"), list(range(num)), "segment_focus")
        for f in focus_grid:
            res.append({"num_segments": num, "segment_focus": f})
    return res

def ratio_beyond_r_sigma(params=None):
    params = params or {}
    r_grid = resolve_param(params.get("r"), [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10], "r")
    return [{"r": x} for x in r_grid]

def linear_trend_timewise(params=None):
    params = params or {}
    attr_grid = resolve_param(params.get("attr"), ["pvalue", "rvalue", "intercept", "slope", "stderr"], "attr")
    return [{"attr": a} for a in attr_grid]

def count_above(params=None):
    params = params or {}
    t_grid = resolve_param(params.get("t"), [0], "t")
    return [{"t": t} for t in t_grid]

def count_below(params=None):
    params = params or {}
    t_grid = resolve_param(params.get("t"), [0], "t")
    return [{"t": t} for t in t_grid]

def lempel_ziv_complexity(params=None):
    params = params or {}
    bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100], "bins")
    return [{"bins": x} for x in bins_grid]

def fourier_entropy(params=None):
    params = params or {}
    bins_grid = resolve_param(params.get("bins"), [2, 3, 5, 10, 100], "bins")
    return [{"bins": x} for x in bins_grid]

def permutation_entropy(params=None):
    params = params or {}
    tau_grid = resolve_param(params.get("tau"), [1], "tau")
    dim_grid = resolve_param(params.get("dimension"), [3, 4, 5, 6, 7], "dimension")
    return [{"tau": t, "dimension": d} for t in tau_grid for d in dim_grid]

def query_similarity_count(params=None):
    params = params or {}
    query_grid = resolve_param(params.get("query"), [None], "query")
    threshold_grid = resolve_param(params.get("threshold"), [0.0], "threshold")
    return [{"query": q, "threshold": t} for q in query_grid for t in threshold_grid]

def matrix_profile(params=None):
    params = params or {}
    threshold_grid = resolve_param(params.get("threshold"), [0.98], "threshold")
    feature_grid = resolve_param(params.get("feature"), ["min", "max", "mean", "median", "25", "75"], "feature")
    return [{"threshold": t, "feature": f} for t in threshold_grid for f in feature_grid]

def mean_n_absolute_max(params=None):
    params = params or {}
    n_grid = resolve_param(params.get("number_of_maxima"), [3, 5, 7], "number_of_maxima")
    return [{"number_of_maxima": n} for n in n_grid]

# ================== 2. 注册表 ==================
FEATURE_HANDLERS = {
    "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
    "c3": c3,
    "cid_ce": cid_ce,
    "symmetry_looking": symmetry_looking,
    "large_standard_deviation": large_standard_deviation,
    "quantile": quantile,
    "autocorrelation": autocorrelation,
    "agg_autocorrelation": agg_autocorrelation,
    "partial_autocorrelation": partial_autocorrelation,
    "number_cwt_peaks": number_cwt_peaks,
    "number_peaks": number_peaks,
    "binned_entropy": binned_entropy,
    "index_mass_quantile": index_mass_quantile,
    "cwt_coefficients": cwt_coefficients,
    "spkt_welch_density": spkt_welch_density,
    "ar_coefficient": ar_coefficient,
    "change_quantiles": change_quantiles,
    "fft_coefficient": fft_coefficient,
    "fft_aggregated": fft_aggregated,
    "value_count": value_count,
    "range_count": range_count,
    "approximate_entropy": approximate_entropy,
    "friedrich_coefficients": friedrich_coefficients,
    "max_langevin_fixed_point": max_langevin_fixed_point,
    "linear_trend": linear_trend,
    "agg_linear_trend": agg_linear_trend,
    "augmented_dickey_fuller": augmented_dickey_fuller,
    "number_crossing_m": number_crossing_m,
    "energy_ratio_by_chunks": energy_ratio_by_chunks,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "linear_trend_timewise": linear_trend_timewise,
    "count_above": count_above,
    "count_below": count_below,
    "lempel_ziv_complexity": lempel_ziv_complexity,
    "fourier_entropy": fourier_entropy,
    "permutation_entropy": permutation_entropy,
    "query_similarity_count": query_similarity_count,
    "matrix_profile": matrix_profile,
    "mean_n_absolute_max": mean_n_absolute_max,
}

# ================== 3. 解析逻辑 ==================
def parse_feature_config(user_config_list):
    """
    处理顶层数据结构的异常：非列表、空列表、非法字典等
    """
    if not user_config_list or not isinstance(user_config_list, list):
        logger.info("Received empty or invalid user_config_list. Using 'comprehensive'.")
        return "comprehensive"
        
    fc_parameters = {}
    
    for item in user_config_list:
        if not isinstance(item, dict):
            logger.warning(f"Skipping invalid item in config list (expected dict, got {type(item)}): {item}")
            continue
            
        fname = item.get("name")
        if not fname or not isinstance(fname, str): 
            continue
            
        user_params = item.get("params")
        if not isinstance(user_params, dict):
            user_params = {} 
        
        handler = FEATURE_HANDLERS.get(fname)
        if handler:
            try:
                params_grid = handler(params=user_params)
                # 只有生成了有效参数才加入配置，防止丢空列表给 tsfresh 报错
                if params_grid:
                    fc_parameters[fname] = params_grid
            except Exception as e:
                logger.error(f"Critical error generating feature '{fname}': {e}. Skipping this feature.")
        else:
            logger.warning(f"Warning: Unknown feature name '{fname}', skipping.")
            
    # 如果所有的配置因为各种错误被过滤空了，降级为全量特征
    if not fc_parameters:
        logger.warning("All parsed features were invalid or empty. Falling back to 'comprehensive'.")
        return "comprehensive"
        
    return fc_parameters