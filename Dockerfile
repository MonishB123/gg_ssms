# Use NVIDIA's CUDA base image with Ubuntu 22.04
FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Install system packages + Python 3.11 in one layer ----
RUN apt-get update && apt-get install -y \
    software-properties-common curl git build-essential ninja-build cmake \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Create virtual environment + upgrade pip/setuptools/wheel ----
RUN python3.11 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python -m pip install --upgrade pip setuptools wheel ninja

ENV PATH="/opt/venv/bin:$PATH"

# ---- CUDA / Torch build env ----
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"  # Modified to support more architectures
ENV FORCE_CUDA=1
ENV MAX_JOBS=4  # Limit parallel compilation to avoid memory issues
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"  # Add NVCC flags
# Modified C++ flags to ensure ABI compatibility
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1 -fPIC"
ENV CFLAGS="-fPIC"

# ---- Install PyTorch + Python deps ----
RUN python -m pip install --upgrade pip \
 && python -m pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 \
 && python -m pip install --no-cache-dir matplotlib opencv-python-headless tqdm tables easydict wandb timm einops

ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

# ---- Set working directory and copy repo ----
WORKDIR /workspace
COPY . /workspace

# ---- Force-rebuild TreeScan + TreeScanLan with specific flags ----
RUN . /opt/venv/bin/activate \
 && cd /workspace/core/convolutional_graph_ssm/third-party/TreeScan \
 && python setup.py clean \
 && TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" python setup.py build_ext --inplace install \
 && cd /workspace/core/graph_ssm/third-party/TreeScanLan \
 && python setup.py clean \
 && TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" python setup.py build_ext --inplace install

# ---- Copy rebuild + run script ----
RUN chmod +x /workspace/run_with_rebuild.sh

# ---- Default command ----
CMD ["/workspace/run_with_rebuild.sh"]
