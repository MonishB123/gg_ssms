#!/bin/bash
set -e

# Activate venv and set Python path
source /opt/venv/bin/activate

# Add core directories to Python path
export PYTHONPATH=/workspace:/workspace/core/graph_ssm:$PYTHONPATH

# ---- Rebuild TreeScan ----
cd /workspace/core/convolutional_graph_ssm/third-party/TreeScan
python setup.py clean --all || true
rm -rf build
python setup.py install

# ---- Rebuild TreeScanLan ----
cd /workspace/core/graph_ssm/third-party/TreeScanLan
python setup.py clean --all || true
rm -rf build
python setup.py install

# ---- Run training script ----
cd /workspace
exec python eye_tracking_lpw/graph_ssm_train.py
