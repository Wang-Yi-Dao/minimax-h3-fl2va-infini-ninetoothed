#!/usr/bin/env bash
# 第三方复现：依赖安装（非 legacy 路径 + 官方 compare）。
# 在 8 卡 scnet-dcu 作业内执行；装完打印 DEPS_OK 再往下跑。
set -euo pipefail
PIP="python3 -m pip install --no-input"
IDX="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# ---- 1) 推理依赖：--no-deps，避免 pip 顺手升级 torch/numpy ----
$PIP --no-deps $IDX \
  diffusers==0.41.0 accelerate==1.15.0 av==18.1.0 huggingface_hub==1.33.0 \
  safetensors==0.8.0 tokenizers==0.22.1 regex==2026.9.29 requests==2.34.2 \
  httpx==0.28.1 httpcore==1.0.9 h11==0.16.0 anyio==4.16.0 sniffio==1.3.1 \
  idna==3.20 certifi==2026.10.23 filelock==4.2.6 packaging==26.3 \
  pyyaml==6.0.3 pillow==13.0.0 janus==2.0.0

# ---- 2) compare/SSIM 依赖 ----
# 注意：scikit-image 0.24.0 声明 numpy>=1.23，pip 可能把 numpy 升到 2.x；
#       所以装完必须强制回退到 1.26.4（见下一条）。
$PIP $IDX scipy==1.13.1 scikit-image==0.24.0 imageio tifffile lazy_loader
$PIP --no-deps --force-reinstall $IDX numpy==1.26.4

python3 - <<'PY'
import numpy, scipy, skimage.metrics
print('DEPS_OK numpy=%s scipy=%s skimage=%s'
      % (numpy.__version__, scipy.__version__, skimage.__version__))
PY
