import requests
import os
import time
import json
import logging
import base64
from datetime import datetime

# ================= 配置区域 =================
CONFIG = {
    "BASE_URL": "http://localhost:8000/api",  # 注意端口通常是 8000
    "INPUT_CSV": "just.csv",                  # 请确保当前目录下有此文件
    "OUTPUT_DIR": "test_results_optimized",   # 结果输出目录
    "TIMEOUT": 60                             # 轮询超时时间(秒)
}

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoApexTester:
    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.input_csv = CONFIG["INPUT_CSV"]
        self.output_dir = CONFIG["OUTPUT_DIR"]
        
        # 运行时状态存储
        self.train_task_id = None
        self.model_zip_path = None
        
        # 初始化输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _save_file(self, response, filename):
        """辅助函数：保存响应内容到文件"""
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(response.content)
        return path

    def check_health(self):
        """简单的健康检查"""
        try:
            requests.get(f"{self.base_url.replace('/api', '')}/docs")
            logger.info("✅ 服务连接正常")
            return True
        except requests.exceptions.ConnectionError:
            logger.error("❌ 无法连接到服务，请确保 server 正在运行 (port 8000)")
            return False

    # ================= 1. 数据分析服务 =================
    def test_analysis_suite(self):
        logger.info("\n=== [Step 1] 测试数据分析服务 ===")
        
        # 1.1 缺失值填补
        logger.info("⏳ 测试缺失值填补 (fill_missing_values)...")
        try:
            with open(self.input_csv, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/fill_missing_values",
                    files={"file": f},
                    data={"method": "auto"}
                )
            
            if resp.status_code == 200:
                saved_path = self._save_file(resp, "1_filled_data.csv")
                filled_status = resp.headers.get("X-Filled-Status", "unknown")
                logger.info(f"✅ 填补完成! 文件已保存: {saved_path}")
                logger.info(f"   -> 是否发生了填补 (X-Filled-Status): {filled_status}")
            else:
                logger.error(f"❌ 填补失败: {resp.text}")
        except Exception as e:
            logger.error(f"❌ 请求异常: {str(e)}")

    # ================= 2. 特征工程服务 (JSON + Base64 方式) =================
    def test_feature_engineering(self):
        logger.info("\n=== [Step 2] 测试特征工程 (纯 JSON + Base64) ===")
        task_id = f"feat_{int(time.time())}"
        
        # 2.1 准备 Base64 文件内容
        logger.info("📄 正在读取并编码 CSV 文件...")
        try:
            with open(self.input_csv, "rb") as f:
                file_content = f.read()
                # 编码为 Base64 字符串
                b64_str = base64.b64encode(file_content).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"❌ 找不到文件: {self.input_csv}")
            return

        payload = {
            "task_id": task_id,
            "file_name": os.path.basename(self.input_csv),
            "file_content_b64": b64_str,
            "max_timeshift": 29,
            "min_timeshift": 29,
            "custom_params": [
                # {
                #     "name": "time_reversal_asymmetry_statistic",
                #     "params": {
                #     "lag": {"start": 1, "end": 5}
                #     }
                # },
                # {
                #     "name": "c3",
                #     "params": {
                #     "lag": {"start": 1, "end": 5}
                #     }
                # },
                # {
                #     "name": "cid_ce",
                #     "params": {
                #     "normalize": ["false"]
                #     }
                # },
                # {
                #     "name": "symmetry_looking",
                #     "params": {
                #     "r": {"start": 0, "end": 20, "multiplier": 0.05}
                #     }
                # },
                # {
                #     "name": "large_standard_deviation",
                #     "params": {
                #     "r": {"start": 1, "end": 20, "multiplier": 0.05}
                #     }
                # },
                # {
                #     "name": "quantile",
                #     "params": {
                #     "q": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
                #     }
                # },
                # {
                #     "name": "autocorrelation",
                #     "params": {
                #     "lag": {"start": 0, "end": 10}
                #     }
                # },
                # {
                #     "name": "agg_autocorrelation",
                #     "params": {
                #     "f_agg": ["mean", "median", "var"],
                #     "maxlag": [10, 20]
                #     }
                # },
                # {
                #     "name": "partial_autocorrelation",
                #     "params": {
                #     "lag": {"start": 0, "end": 10}
                #     }
                # },
                # {
                #     "name": "number_cwt_peaks",
                #     "params": {
                #     "n": [1, 5]
                #     }
                # },
                # {
                #     "name": "number_peaks",
                #     "params": {
                #     "n": [1, 3, 5, 10, 20, 50]
                #     }
                # },
                # {
                #     "name": "binned_entropy",
                #     "params": {
                #     "max_bins": [10]
                #     }
                # },
                # {
                #     "name": "index_mass_quantile",
                #     "params": {
                #     "q": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
                #     }
                # },
                # {
                #     "name": "cwt_coefficients",
                #     "params": {
                #     "widths": [2, 5, 10, 20, 50], 
                #     "coeff": {"start": 0, "end": 15},
                #     "w": [2, 5, 10, 20, 50]
                #     }
                # },
                # {
                #     "name": "spkt_welch_density",
                #     "params": {
                #     "coeff": [2, 5, 8, 50]
                #     }
                # },
                # {
                #     "name": "ar_coefficient",
                #     "params": {
                #     "coeff": {"start": 0, "end": 11},
                #     "k": [10]
                #     }
                # },
                # {
                #     "name": "change_quantiles",
                #     "params": {
                #     "ql": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.1],
                #     "qh": [0.2, 0.4, 0.6, 0.8, 1.0],
                #     "isabs": ["false", "true"],
                #     "f_agg": ["mean", "var","median"]
                #     }
                # },
                {
                    "name": "fft_coefficient",
                    "params": {
                    "coeff": {"start": 0, "end": 100},
                    "attr": ["real", "imag", "abs", "angle"]
                    }
                },
                # {
                #     "name": "fft_aggregated",
                #     "params": {
                #     "aggtype": ["centroid", "variance", "skew", "kurtosis"]
                #     }
                # },
                # {
                #     "name": "value_count",
                #     "params": {
                #     "value": [0, 1, -1, 2]
                #     }
                # },
                # {
                #     "name": "range_count",
                #     "params": {
                #     "min": [-1, -1000000000000, 0],
                #     "max": [1, 0, 1000000000000]
                #     }
                # },
                # {
                #     "name": "approximate_entropy",
                #     "params": {
                #     "m": [2, 4],
                #     "r": [0.1, 0.3, 0.5, 0.7, 0.9]
                #     }
                # },
                # {
                #     "name": "friedrich_coefficients",
                #     "params": {
                #     "m": [3, 5],
                #     "r": [30, 60]
                #     }
                # },
                # {
                #     "name": "max_langevin_fixed_point",
                #     "params": {
                #     "m": [3],
                #     "r": [30]
                #     }
                # },
                # {
                #     "name": "linear_trend",
                #     "params": {
                #     "attr": ["pvalue", "rvalue", "intercept", "slope", "stderr"]
                #     }
                # },
                # {
                #     "name": "agg_linear_trend",
                #     "params": {
                #     "attr": ["rvalue", "intercept", "slope", "stderr"],
                #     "chunk_len": [5, 10, 50],
                #     "f_agg": ["max", "min", "mean", "var"]
                #     }
                # },
                # {
                #     "name": "augmented_dickey_fuller",
                #     "params": {
                #     "attr": ["teststat", "pvalue", "usedlag"]
                #     }
                # },
                # {
                #     "name": "number_crossing_m",
                #     "params": {
                #     "m": [0, -1, 1]
                #     }
                # },
                # {
                #     "name": "energy_ratio_by_chunks",
                #     "params": {
                #     "num_segments": [10],
                #     "segment_focus": {"start": 0, "end": 10}
                #     }
                # },
                # {
                #     "name": "ratio_beyond_r_sigma",
                #     "params": {
                #     "r": [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]
                #     }
                # },
                # {
                #     "name": "linear_trend_timewise",
                #     "params": {
                #     "attr": ["pvalue", "rvalue", "intercept", "slope", "stderr"]
                #     }
                # },
                # {
                #     "name": "count_above",
                #     "params": {
                #     "t": [0]}
                #     ]
                #     }
                # },
                # {
                #     "name": "count_below",
                #     "params": {
                #     "t": [0]
                #     }
                # },
                # {
                #     "name": "lempel_ziv_complexity",
                #     "params": {
                #     "bins": [2, 3, 5, 10, 100]
                #     }
                # },
                # {
                #     "name": "fourier_entropy",
                #     "params": {
                #     "bins": [2, 3, 5, 10, 100]
                #     }
                # },
                # {
                #     "name": "permutation_entropy",
                #     "params": {
                #     "tau": [1],
                #     "dimension": [3, 4, 5, 6, 7]
                #     }
                # },
                {
                    "name": "query_similarity_count",
                    "params": {
                    "query": ["None"],
                    "threshold": [0.0]
                    }
                },
                # {
                #     "name": "matrix_profile",
                #     "params": {
                #     "threshold": [0.98],
                #     "feature": ["min", "max", "mean", "median", "25", "75"]
                #     }
                # },
                # {
                #     "name": "mean_n_absolute_max",
                #     "params": {
                #     "number_of_maxima": [3, 5, 7, 9]
                #     }
                # }
                ]
        }

        # 2.3 发送 JSON 请求
        logger.info(f"🚀 提交任务 ID: {task_id}")
        resp = requests.post(
            f"{self.base_url}/generate_all_features",
            json=payload  # 注意：这里使用 json=... 而不是 files=...
        )
        
        if resp.status_code != 200:
            logger.error(f"❌ 任务提交失败: {resp.text}")
            return

        # 2.4 轮询等待
        if self._poll_task_status(task_id):
            # 2.5 下载结果 (这一步还是普通 Form 或者是 Query Param，看你之前的实现，这里保持原样)
            self._download_selected_features(task_id)

    def _poll_task_status(self, task_id):
        logger.info("⏳ 等待任务完成...")
        start_time = time.time()
        
        while (time.time() - start_time) < CONFIG["TIMEOUT"]:
            try:
                resp = requests.get(f"{self.base_url}/task_status", params={"task_id": task_id})
                data = resp.json().get("data", {})
                status = data.get("status")
                
                if status == "SUCCESS":
                    logger.info("✅ 任务执行成功!")
                    return True
                elif status == "FAILED":
                    logger.error(f"❌ 任务失败: {data.get('message')}")
                    return False
                
                time.sleep(2) 
            except Exception as e:
                logger.warning(f"⚠️ 轮询出错: {e}")
                time.sleep(2)
        
        logger.error("❌ 任务等待超时")
        return False

    def _download_selected_features(self, task_id):
        # 假设下载接口没变，还是 Form 表单或者 Query
        resp = requests.post(
            f"{self.base_url}/select_features_and_download",
            data={"task_id": task_id, "fdr": 1}
        )
        if resp.status_code == 200:
            path = self._save_file(resp, "2_selected_features.csv")
            logger.info(f"✅ 特征筛选下载成功: {path}")
        else:
            logger.error(f"❌ 下载失败: {resp.text}")

    # ================= 3. 模型训练 (保持 Multipart Form) =================
    def test_training(self):
        logger.info("\n=== [Step 3] 测试模型训练 ===")
        self.train_task_id = f"train_{int(time.time())}"
        hyperparameters_payload = [
                # {"name": "Naive", "params": {}},
                # {"name": "SeasonalNaive", "params": {}},
                # {"name": "Average", "params": {}},
                # {"name": "SeasonalAverage", "params": {}},
                # {"name": "Zero", "params": {}},
                # {
                #     "name": "ARIMA",
                #     "params": {
                #         "order": [(1, 0, 0), (1, 1, 1), (2, 1, 0), (0, 1, 1), (2, 1, 2)],
                #         "seasonal_order": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                #         "method": ["CSS-ML", "CSS", "ML"]
                #     }
                # },
                # {
                #     "name": "ARIMA",
                #     "params": {
                #         "order": [[1, 0, 0], [1, 1, 1], [2, 1, 0], [0, 1, 1], [2, 1, 2]],
                #         "seasonal_order": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                #         "method": ["CSS-ML", "CSS", "ML"]
                #     }
                # },
                # {
                #     "name": "AutoARIMA",
                #     "params": {
                #         "max_p": {"start": 3, "end": 8},
                #         "max_q": {"start": 3, "end": 8},
                #         "max_P": {"start": 1, "end": 4},
                #         "max_Q": {"start": 1, "end": 4},
                #         "max_d": {"start": 0, "end": 2},
                #         "max_D": {"start": 0, "end": 2},
                #         "start_p": {"start": 1, "end": 3},
                #         "seasonal": ["true", "false"]
                #     }
                # },
                # {
                #     "name": "ETS",
                #     "params": {
                #         "model": ["AAA", "ANA", "MAM", "MNM"],
                #         "damped": ["true", "false"]
                #     }
                # },
                # {"name": "AutoETS", "params": {}},
                # {
                #     "name": "AutoCES",
                #     "params": {
                #         "model": ["Z", "N", "S", "P", "F"]
                #     }
                # },
                # {
                #     "name": "Theta",
                #     "params": {
                #         "decomposition_type": ["multiplicative", "additive"]
                #     }
                # },
                # {"name": "DynamicOptimizedTheta", "params": {}},
                # {"name": "ADIDA", "params": {}},
                # {
                #     "name": "Croston",
                #     "params": {
                #         "variant": ["SBA", "classic", "optimized"]
                #     }
                # },
                # {"name": "CrostonSBA", "params": {}},
                # {"name": "IMAPA", "params": {}},
                # {"name": "DeepAR", "params": {}},
                # {
                #     "name": "TemporalFusionTransformer",
                #     "params": {
                #         "hidden_dim": [32, 64, 128],
                #         "num_heads": [1, 2, 4],
                #         "dropout_rate": {"start": 0.1, "end": 0.3},
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                {
                    "name": "PatchTST",
                    "params": {
                        "patch_len": [8, 16, 24],
                        "stride": [4, 8, 16],
                        "d_model": [32, 64, 128, 512],
                        "nhead": [2, 4, 8],
                        "num_encoder_layers": {"start": 2, "end": 4},
                        "scaling": ["mean", "std", "None"],
                        "batch_size": [32, 64, 128],
                        "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                    }
                },
                # {
                #     "name": "TiDE",
                #     "params": {
                #         "encoder_hidden_dim": [64, 128, 256],
                #         "decoder_hidden_dim": [64, 128, 256],
                #         "temporal_hidden_dim": [64, 128],
                #         "distr_hidden_dim": [64, 128],
                #         "num_layers_encoder": {"start": 1, "end": 4},
                #         "num_layers_decoder": {"start": 1, "end": 4},
                #         "dropout_rate": {"start": 0.1, "end": 0.3},
                #         "layer_norm": ["true", "false"],
                #         "batch_size": [128, 256, 512],
                #         "lr": {"start": 0.0001, "end": 0.01}
                #     }
                # },
                {
                    "name": "DLinear",
                    "params": {}
                },
                # {
                #     "name": "WaveNet",
                #     "params": {
                #         "num_bins": {"start": 512, "end": 1024},
                #         "use_log_scale_feature": ["true", "false"],
                #         "negative_data": ["true", "false"],
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"},
                #         "weight_decay": {"start": 1e-08, "end": 0.0001, "log": "true"}
                #     }
                # },
                # {
                #     "name": "SimpleFeedForward",
                #     "params": {
                #         "batch_normalization": ["true", "false"],
                #         "mean_scaling": ["true", "false"],
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                # {"name": "Chronos", "params": {}},
                # {"name": "DirectTabular", "params": {}},
                # {"name": "RecursiveTabular", "params": {}},
                # {"name": "PerStepTabular", "params": {}}
            ]
        tune_kwargs_payload = {
            "num_trials": 10,       # 尝试的组合总数
            "searcher": "random",   # 随机搜索
            "scheduler": "local"
        }

        params = {
            "task_id": self.train_task_id,
            "target": "value",
            "prediction_length": 5,
            "freq": "D",
            "time_limit": 60,
            "use_tsfresh": False,
            "enable_ensemble": True,
            "hyperparameters": json.dumps(hyperparameters_payload),
            "ensemble_types": "simple",
            "hyperparameter_tune_kwargs": json.dumps(tune_kwargs_payload)
        }

        logger.info(f"🚀 开始训练 (Task ID: {self.train_task_id})...")
        try:
            with open(self.input_csv, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/train",
                    files={"file": f},
                    data=params
                )
            
            if resp.status_code == 200:
                self.model_zip_path = self._save_file(resp, "3_trained_model.zip")
                logger.info(f"✅ 训练成功! 模型包已保存: {self.model_zip_path}")
            else:
                logger.error(f"❌ 训练失败: {resp.text}")
                self.train_task_id = None
        except Exception as e:
            logger.error(f"❌ 训练请求异常: {e}")

    # ================= 5. 推理预测 =================
    def test_inference(self):
        if not self.model_zip_path or not os.path.exists(self.model_zip_path):
            logger.warning("⚠️ 跳过推理测试 (无可用模型文件)")
            return

        logger.info("\n=== [Step 5] 测试推理预测 ===")
        infer_task_id = f"infer_{int(time.time())}"
        
        try:
            files = [
                ("model_zip", (os.path.basename(self.model_zip_path), open(self.model_zip_path, "rb"), "application/zip")),
                ("data_csv", (os.path.basename(self.input_csv), open(self.input_csv, "rb"), "text/csv"))
            ]
            
            data = {
                "task_id": infer_task_id,
                "best_model": "SimpleAverageEnsemble"
            }
            
            logger.info(f"🚀 提交推理任务 (使用模型: {data['best_model']})...")
            resp = requests.post(f"{self.base_url}/inference", files=files, data=data)
            
            for _, f_tuple in files:
                f_tuple[1].close()

            if resp.status_code == 200:
                path = self._save_file(resp, "5_inference_result.csv")
                logger.info(f"✅ 推理成功! 结果已保存: {path}")
            else:
                logger.error(f"❌ 推理失败: {resp.text}")
        except Exception as e:
             logger.error(f"❌ 推理请求异常: {e}")

if __name__ == "__main__":
    tester = AutoApexTester()
    
    # 检查服务健康
    if tester.check_health():
        # Step 1: 分析
        # tester.test_analysis_suite()
        
        # Step 2: 特征工程 (重点测试这个)
        # tester.test_feature_engineering()
        
        # Step 3: 训练
        tester.test_training()
        
        # Step 5: 推理
        # tester.test_inference()
        
    logger.info(f"\n✨ 测试流程结束。请检查目录: {tester.output_dir}")