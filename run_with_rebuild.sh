#!/bin/bash
set -e

# Activate venv
source /opt/venv/bin/activate

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
