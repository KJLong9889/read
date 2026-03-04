import requests

url = "http://localhost:8003/api/inference"

# 构造请求参数
files = {
    # 你的模型压缩包
    'model_zip': ('my_model.zip', open('./test_task_123123123_model.zip', 'rb'), 'application/zip'),
    # 你的预测数据
    'data_csv': ('test_data.csv', open('./test_task_123123123.csv', 'rb'), 'text/csv')
}
data = {
    'task_id': 'test_task_001',
    'best_model': 'WeightedEnsemble'  # 指定你想使用的模型
}

# 发送请求
response = requests.post(url, files=files, data=data)

# 保存结果
if response.status_code == 200:
    with open('result_predictions.csv', 'wb') as f:
        f.write(response.content)
    print("预测成功，文件已保存。")
else:
    print("失败:", response.text)