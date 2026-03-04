#!/bin/bash

# 获取脚本所在目录
BASE_DIR=$(cd "$(dirname "$0")"; pwd)

# 设置 PYTHONPATH 包含当前目录
export PYTHONPATH=$PYTHONPATH:$BASE_DIR

echo "Starting server from $BASE_DIR..."

# nohup python "$BASE_DIR/main.py" > main.log 2>&1 &

python "$BASE_DIR/main.py"