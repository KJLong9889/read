import requests
import json
import pandas as pd
import numpy as np
import os
import time
import sys

# ================= 配置 =================
BASE_URL = "http://127.0.0.1:8000"
TEST_RAW_CSV = "test_raw_data.csv"
DOWNLOADED_FEATURE_FILE = "client_downloaded_features.csv" # 本地保存的文件名

def create_dummy_csv(filename):
    print(f"Creating dummy csv: {filename}...")
    dates = pd.date_range(start="2025-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "item_id": ["A"] * 50,
        "time": dates,
        "value": np.linspace(0, 100, 50) + np.random.normal(0, 5, 50)
    })
    df.to_csv(filename, index=False)
    return filename

def poll_task_status(task_id, interval=2):
    """
    轮询任务状态 (仅用于异步任务，如 Step 1)
    """
    print(f"   Waiting for task {task_id}...", end="", flush=True)
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/api/task_status", params={"task_id": task_id})
            if resp.status_code != 200:
                print(f"\n❌ Error checking status: {resp.text}")
                return None
            
            data = resp.json().get("data", {})
            status = data.get("status")
            message = data.get("message", "")
            
            # 打印当前状态信息
            print(f"\r   [{status}] {message[:50]}...", end="", flush=True)

            if status == "SUCCESS":
                print(f"\n✅ Task Finished!")
                return data.get("data") # 返回结果数据
            elif status == "FAILED":
                print(f"\n❌ Task Failed: {message}")
                show_logs(task_id)
                return None
            
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nUser Interrupted.")
            return None

def show_logs(task_id):
    url = f"{BASE_URL}/api/task_logs"
    print(f"\n   >>> Fetching Logs for {task_id}...")
    resp = requests.get(url, params={"task_id": task_id})
    if resp.status_code == 200:
        logs = resp.json()['data']
        # 只显示最后 10 行
        for l in logs[-10:]: print(f"      {l}")
    else:
        print("      Could not fetch logs.")

# ================= 步骤 1: 生成特征 (异步+轮询) =================
def step1_generate(csv_path):
    url = f"{BASE_URL}/api/generate_all_features"
    task_id = f"task_{int(time.time())}"
    print(f"\n>>> [1. 特征生成] Uploading {csv_path} (TaskID: {task_id})...")
    
    # 构造特定格式的参数，服务端有特殊解析逻辑
    custom_params = {"c3": ["{\"lag\": lag} for lag in range(1, 4)"]}

    with open(csv_path, 'rb') as f:
        files = {'file': (os.path.basename(csv_path), f, 'text/csv')}
        data = {
            "task_id": task_id, 
            "max_timeshift": 5, 
            "min_timeshift": 5,
            "tsfresh_custom_params": json.dumps(custom_params)
        }
        # 发起请求
        resp = requests.post(url, data=data, files=files)
    
    if resp.status_code == 200:
        print("   Upload success, waiting for processing...")
        # 进入轮询
        result = poll_task_status(task_id)
        if result:
            print(f"   Feature Count: {result.get('feature_count')}")
            return task_id
    else:
        print(f"❌ Upload Failed: {resp.text}")
    return None

# ================= 步骤 2: 筛选并下载 (同步阻塞+流式下载) =================
def step2_select_and_download(task_id):
    """
    修改点：适配新的 /api/select_features_and_download 接口
    该接口是同步的，直接返回文件流，不需要轮询。
    """
    url = f"{BASE_URL}/api/select_features_and_download"
    print(f"\n>>> [2. 特征筛选与下载] Filtering features (TaskID: {task_id})...")
    print("   Note: Waiting for server response (this may take a while)...")
    
    # 构造 JSON Body
    payload = {
        "task_id": task_id, 
        "fdr": 0.5
    }
    
    try:
        # stream=True 允许分块下载大文件
        with requests.post(url, json=payload, stream=True) as r:
            if r.status_code == 200:
                # 获取文件名 (可选，从 header 获取，或者直接自定义)
                local_path = DOWNLOADED_FEATURE_FILE
                
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Downloaded successfully to: {local_path}")
                return local_path
            else:
                print(f"❌ Request Failed: Code {r.status_code}")
                try:
                    print(f"   Detail: {r.json()}")
                except:
                    print(f"   Text: {r.text}")
                
                # 如果失败，查看一下日志看看发生了什么
                show_logs(task_id)
                return None
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# ================= 步骤 4: 相关性计算 (独立接口) =================
def step4_correlation(csv_path):
    url = f"{BASE_URL}/api/calculate_correlation"
    print(f"\n>>> [4. 独立相关性计算] Uploading filtered CSV ({csv_path})...")
    
    if not os.path.exists(csv_path):
        print("❌ File not found.")
        return

    new_task_id = f"corr_{int(time.time())}"
    with open(csv_path, 'rb') as f:
        # 注意：这里上传的是 Step 2 下载下来的结果
        files = {'file': ('selected_data.csv', f, 'text/csv')}
        data = {"task_id": new_task_id, "method": "pearson"}
        
        resp = requests.post(url, data=data, files=files)
        
    if resp.status_code == 200:
        res = resp.json()
        print(f"✅ Success! Correlation Matrix Saved at Server: {res['data']['download_path']}")
    else:
        print(f"❌ Failed: {resp.text}")

# ================= 主流程 =================
if __name__ == "__main__":
    # 0. 准备数据
    create_dummy_csv(TEST_RAW_CSV)
    
    try:
        # 1. 生成特征 (异步)
        tid = step1_generate(TEST_RAW_CSV)
        
        if tid:
            # 2. 筛选并直接下载 (同步)
            # 注意：这里不再返回服务端的 path，而是返回客户端保存的 path
            downloaded_file_path = step2_select_and_download(tid)
            
            if downloaded_file_path:
                # 4. 计算相关性
                # 使用刚才下载的文件再次上传
                step4_correlation(downloaded_file_path)
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # 清理测试产生的临时文件
        if os.path.exists(TEST_RAW_CSV):
            os.remove(TEST_RAW_CSV)
        # 可选：清理下载的文件
        # if os.path.exists(DOWNLOADED_FEATURE_FILE):
        #     os.remove(DOWNLOADED_FEATURE_FILE)