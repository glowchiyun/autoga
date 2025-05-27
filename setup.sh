#!/usr/bin/env bash
set -ex
HERE=$(dirname "$0")

# 创建本地虚拟环境
. ${HERE}/../shared/setup.sh ${HERE} true

# 检查Python版本
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo "Error: Python 3.9 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# 配置pip国内镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip config set global.timeout 3000

# 安装依赖
pip install -r ${HERE}/requirements.txt

# 安装autoga框架
TARGET_DIR="${HERE}/lib/autoga"
rm -rf ${TARGET_DIR}
mkdir -p ${TARGET_DIR}

# 只复制必要的Python文件
for item in ${HERE}/*.py; do
    if [ -f "$item" ]; then
        cp "$item" "$TARGET_DIR/"
    fi
done

# 复制其他必要文件
cp ${HERE}/pyproject.toml ${TARGET_DIR}/
cp ${HERE}/requirements.txt ${TARGET_DIR}/

# 设置PYTHONPATH
export PYTHONPATH="${TARGET_DIR}:${PYTHONPATH}"

# 安装为可编辑包
cd ${TARGET_DIR}
pip install -U -e .

# 验证安装
python3 -c "import autoga; print('autoga installed successfully')" || true