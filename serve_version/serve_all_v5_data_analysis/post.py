import requests
import os
import json

# ================= 配置 =================
BASE_URL = "http://127.0.0.1:8002"
ANALYZE_URL = f"{BASE_URL}/api/analyze_csv"
SCORES_URL = f"{BASE_URL}/api/get_characteristics"

TEST_FILE_PATH = "just.csv"
OUTPUT_BOXPLOT_CSV = "result_boxplot.csv"


def test_analyze_boxplot():
    """测试接口 1: 获取箱线图数据 (CSV)"""
    print(f"\n[1] 正在测试箱线图服务: {ANALYZE_URL}")
    
    files = {
        "file": (os.path.basename(TEST_FILE_PATH), open(TEST_FILE_PATH, "rb"), "text/csv")
    }
    payload = {"target_column": "value"}
    
    try:
        response = requests.post(ANALYZE_URL, data=payload, files=files, stream=True)
        
        if response.status_code == 200:
            with open(OUTPUT_BOXPLOT_CSV, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ 请求成功！文件已保存为: {OUTPUT_BOXPLOT_CSV}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print("错误信息:", response.text)
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        files["file"][1].close()

def test_get_scores():
    """测试接口 2: 获取特征评分 (JSON)"""
    print(f"\n[2] 正在测试评分服务: {SCORES_URL}")
    
    files = {
        "file": (os.path.basename(TEST_FILE_PATH), open(TEST_FILE_PATH, "rb"), "text/csv")
    }
    payload = {"target_column": "value"}
    
    try:
        response = requests.post(SCORES_URL, data=payload, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 请求成功！返回的评分如下：")
            print(json.dumps(result, indent=4, ensure_ascii=False))
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print("错误信息:", response.text)
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        files["file"][1].close()


def test_fill_missing_values_endpoint():
    """测试接口 3: 填充缺失值 (CSV)"""
    FILL_MISSING_URL = f"{BASE_URL}/api/fill_missing_values"
    print(f"\n[3] 正在测试填充缺失值服务: {FILL_MISSING_URL}")
    
    files = {
        "file": (os.path.basename(TEST_FILE_PATH), open(TEST_FILE_PATH, "rb"), "text/csv")
    }
    payload = {"method": "interpolation"}
    
    try:
        response = requests.post(FILL_MISSING_URL, data=payload, files=files, stream=True)
        
        if response.status_code == 200:
            output_filled_csv = "filled_missing_values.csv"
            with open(output_filled_csv, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ 请求成功！填充后的文件已保存为: {output_filled_csv}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print("错误信息:", response.text)
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        files["file"][1].close()

if __name__ == "__main__":
    # create_dummy_csv()
    
    # # 测试两个接口
    # test_analyze_boxplot()
    # test_get_scores()
    test_fill_missing_values_endpoint()