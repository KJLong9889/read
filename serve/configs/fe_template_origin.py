from itertools import product

# ================== 1. 策略函数定义 ==================
# 这里定义每个特征的参数处理逻辑

def time_reversal_asymmetry_statistic(lag=None):
    if lag is not None: return [{"lag": lag}]
    return [{"lag": lag} for lag in range(1, 4)]

def c3(lag=None):
    if lag is not None: return [{"lag": lag}]
    return [{"lag": lag} for lag in range(1, 4)]

def cid_ce(normalize=None):
    if normalize is not None: return [{"normalize": normalize}]
    return [{"normalize": True}, {"normalize": False}]

def symmetry_looking(r=None):
    if r is not None: return [{"r": r}]
    return [{"r": i * 0.05} for i in range(20)]

def large_standard_deviation(r=None):
    if r is not None: return [{"r": r}]
    return [{"r": i * 0.05} for i in range(1, 20)]

def quantile(q=None):
    if q is not None: return [{"q": q}]
    return [{"q": val} for val in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]]

def autocorrelation(lag=None):
    if lag is not None: return [{"lag": lag}]
    return [{"lag": lag} for lag in range(10)]

def agg_autocorrelation(f_agg=None, maxlag=40):
    if f_agg is not None: return [{"f_agg": f_agg, "maxlag": maxlag}]
    return [{"f_agg": s, "maxlag": maxlag} for s in ["mean", "median", "var"]]

def partial_autocorrelation(lag=None):
    if lag is not None: return [{"lag": lag}]
    return [{"lag": lag} for lag in range(10)]

def number_cwt_peaks(n=None):
    if n is not None: return [{"n": n}]
    return [{"n": val} for val in [1, 5]]

def number_peaks(n=None):
    if n is not None: return [{"n": n}]
    return [{"n": val} for val in [1, 3, 5, 10, 50]]

def binned_entropy(max_bins=None):
    if max_bins is not None: return [{"max_bins": max_bins}]
    return [{"max_bins": 10}]

def index_mass_quantile(q=None):
    if q is not None: return [{"q": q}]
    return [{"q": val} for val in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]]

def cwt_coefficients(widths=None, coeff=None, w=None):
    if widths is not None and coeff is not None and w is not None:
        return [{"widths": widths, "coeff": coeff, "w": w}]
    return [{"widths": width, "coeff": c, "w": weight} for width in [(2, 5, 10, 20)] for c in range(15) for weight in (2, 5, 10, 20)]

def spkt_welch_density(coeff=None):
    if coeff is not None: return [{"coeff": coeff}]
    return [{"coeff": c} for c in [2, 5, 8]]

def ar_coefficient(coeff=None, k=10):
    if coeff is not None: return [{"coeff": coeff, "k": k}]
    return [{"coeff": c, "k": 10} for c in range(10 + 1)]

def change_quantiles(ql=None, qh=None, isabs=None, f_agg=None):
    if all(x is not None for x in [ql, qh, isabs, f_agg]):
        return [{"ql": ql, "qh": qh, "isabs": isabs, "f_agg": f_agg}]
    return [{"ql": ql, "qh": qh, "isabs": b, "f_agg": f} for ql in [0.0, 0.2, 0.4, 0.6, 0.8] for qh in [0.2, 0.4, 0.6, 0.8, 1.0] for b in [False, True] for f in ["mean", "var"] if ql < qh]

def fft_coefficient(coeff=None, attr=None):
    if coeff is not None and attr is not None: return [{"coeff": coeff, "attr": attr}]
    return [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(100))]

def fft_aggregated(aggtype=None):
    if aggtype is not None: return [{"aggtype": aggtype}]
    return [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]]

def value_count(value=None):
    if value is not None: return [{"value": value}]
    return [{"value": v} for v in [0, 1, -1]]

def range_count(min=None, max=None):
    if min is not None and max is not None: return [{"min": min, "max": max}]
    return [{"min": -1, "max": 1}, {"min": -1e12, "max": 0}, {"min": 0, "max": 1e12}]

def approximate_entropy(m=2, r=None):
    if r is not None: return [{"m": m, "r": r}]
    return [{"m": 2, "r": val} for val in [0.1, 0.3, 0.5, 0.7, 0.9]]

def friedrich_coefficients(coeff=None, m=3, r=30):
    if coeff is not None: return [{"coeff": coeff, "m": m, "r": r}]
    return [{"coeff": c, "m": m, "r": r} for c in range(m + 1)]

def max_langevin_fixed_point(m=3, r=30):
    return [{"m": m, "r": r}]

def linear_trend(attr=None):
    if attr is not None: return [{"attr": attr}]
    return [{"attr": a} for a in ["pvalue", "rvalue", "intercept", "slope", "stderr"]]

def agg_linear_trend(attr=None, chunk_len=None, f_agg=None):
    if all(x is not None for x in [attr, chunk_len, f_agg]):
        return [{"attr": attr, "chunk_len": chunk_len, "f_agg": f_agg}]
    return [{"attr": a, "chunk_len": i, "f_agg": f} for a in ["rvalue", "intercept", "slope", "stderr"] for i in [5, 10, 50] for f in ["max", "min", "mean", "var"]]

def augmented_dickey_fuller(attr=None):
    if attr is not None: return [{"attr": attr}]
    return [{"attr": a} for a in ["teststat", "pvalue", "usedlag"]]

def number_crossing_m(m=None):
    if m is not None: return [{"m": m}]
    return [{"m": 0}, {"m": -1}, {"m": 1}]

def energy_ratio_by_chunks(num_segments=10, segment_focus=None):
    if segment_focus is not None: return [{"num_segments": num_segments, "segment_focus": segment_focus}]
    return [{"num_segments": 10, "segment_focus": i} for i in range(10)]

def ratio_beyond_r_sigma(r=None):
    if r is not None: return [{"r": r}]
    return [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]]

def linear_trend_timewise(attr=None):
    if attr is not None: return [{"attr": attr}]
    return [{"attr": a} for a in ["pvalue", "rvalue", "intercept", "slope", "stderr"]]

def count_above(t=0): return [{"t": t}]
def count_below(t=0): return [{"t": t}]

def lempel_ziv_complexity(bins=None):
    if bins is not None: return [{"bins": bins}]
    return [{"bins": x} for x in [2, 3, 5, 10, 100]]

def fourier_entropy(bins=None):
    if bins is not None: return [{"bins": bins}]
    return [{"bins": x} for x in [2, 3, 5, 10, 100]]

def permutation_entropy(tau=1, dimension=None):
    if dimension is not None: return [{"tau": tau, "dimension": dimension}]
    return [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]]

def query_similarity_count(query=None, threshold=0.0):
    return [{"query": query, "threshold": threshold}]

def matrix_profile(threshold=0.98, feature=None):
    if feature is not None: return [{"threshold": threshold, "feature": feature}]
    return [{"threshold": threshold, "feature": f} for f in ["min", "max", "mean", "median", "25", "75"]]

def mean_n_absolute_max(number_of_maxima=None):
    if number_of_maxima is not None: return [{"number_of_maxima": number_of_maxima}]
    return [{"number_of_maxima": n} for n in [3, 5, 7]]

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
    参数: user_config_list (List[Dict]) 
    示例: [{"name": "c3", "params": ["lag"]}, ...]
    """
    fc_parameters = {}
    
    # 如果没传任何配置，返回默认全量
    if not user_config_list:
        return "comprehensive"
        
    for item in user_config_list:
        fname = item.get("name")
        # 注意：这里 params 是列表 ["lag"]，我们在当前逻辑下其实不需要用到它
        # 因为我们是使用模板默认值。这个列表仅起到了"选中"的作用。
        
        handler = FEATURE_HANDLERS.get(fname)
        if handler:
            try:
                # 【核心修改】直接调用 handler()，不传任何参数
                # 这样就会触发函数内部的 "if lag is None: return range(...)" 逻辑
                fc_parameters[fname] = handler()
                
            except Exception as e:
                print(f"Error generating feature {fname}: {e}")
        else:
            print(f"Warning: Unknown feature name '{fname}', skipping.")
            
    return fc_parameters






