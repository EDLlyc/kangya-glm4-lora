#!/bin/bash
set -e

# 一键下载脚本（适配 AutoDL 学术加速）
# 用法: bash download.sh [目标目录]

TARGET_DIR="${1:-./kangya-glm4-lora}"
REPO_URL="https://ghproxy.com/https://github.com/EDLlyc/kangya-glm4-lora.git"

echo "正在下载到: $TARGET_DIR"
if [ -d "$TARGET_DIR/.git" ]; then
    echo "目录已存在，执行 git pull..."
    cd "$TARGET_DIR"
    git pull
else
    git clone "$REPO_URL" "$TARGET_DIR"
fi

echo "下载完成！"
echo "文件列表:"
ls -lh "$TARGET_DIR"
