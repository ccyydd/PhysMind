# Full Environment Setup

This guide installs the complete PhysMind executable-world pipeline on a standard Linux workstation. It uses one Conda environment named `physmind`; tool workers inherit the active Python environment.

## 1. System Requirements

- Linux x86-64
- an NVIDIA GPU with a CUDA 12-compatible driver
- a working CUDA compiler (`nvcc`)
- Conda or Miniconda
- Git and Git LFS

Install the native build tools on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git git-lfs curl \
  libboost-all-dev libeigen3-dev
```

Clone the repository with all research-tool submodules:

```bash
git clone --recursive https://github.com/ccyydd/PhysMind.git
cd PhysMind
git lfs install
git submodule update --init --recursive
```

## 2. Unified Conda Environment

```bash
conda create -n physmind -c conda-forge --override-channels python=3.12 pip -y
conda activate physmind
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the local research packages without replacing the pinned PyTorch and OpenCV stack:

```bash
python -m pip install --no-deps \
  "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
  "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"

python -m pip install -e third_party/MoGe --no-deps
python -m pip install -e third_party/GeoCalib --no-deps
python -m pip install -e third_party/sam3 --no-deps
```

Store the xFormers compatibility setting in the environment:

```bash
conda env config vars set -n physmind XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
conda deactivate
conda activate physmind
```

## 3. CUDA Extensions

Install the SAM 3D dependencies against the pinned PyTorch/CUDA stack:

```bash
python -m pip install --no-deps kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu126.html

python -m pip install --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2%2Bcu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

python -m pip install ccimport==0.4.4 pccm==0.4.16
python -m pip install --no-deps cumm-cu121==0.7.11 spconv-cu121==2.3.8

python -m pip install --no-build-isolation --no-deps \
  "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

python -m pip install --no-build-isolation --no-deps \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

python -m pip install --no-build-isolation --no-deps \
  "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git"
```

Build the FoundationPose extensions:

```bash
cd third_party/FoundationPose/mycpp
cmake -S . -B build \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build --parallel 8
cd ../../..

cd third_party/FoundationPose/bundlesdf/mycuda
python -m pip install --no-build-isolation -e .
cd ../../../..
```

## 4. Model Weights

Authenticate with Hugging Face after accepting the gated model licenses for SAM 3 and SAM 3D Objects:

```bash
hf auth login
```

MoGe-2 and GeoCalib populate their standard caches on first construction:

```bash
python -c "from moge.model.v2 import MoGeModel; MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal')"

python - <<'PY'
from geocalib import GeoCalib
GeoCalib(weights="pinhole")
PY
```

Download Metric Video Depth Anything:

```bash
mkdir -p third_party/Video-Depth-Anything/checkpoints
hf download depth-anything/Metric-Video-Depth-Anything-Large \
  metric_video_depth_anything_vitl.pth \
  --local-dir third_party/Video-Depth-Anything/checkpoints
```

Download Grounding DINO:

```bash
hf download IDEA-Research/grounding-dino-base \
  --local-dir data/models/grounding-dino-base
```

SAM 3 downloads its checkpoint from Hugging Face on first use. Download SAM 3D Objects and copy its checkpoint directory into the layout expected by PhysMind:

```bash
hf download facebook/sam-3d-objects \
  --local-dir third_party/sam-3d-objects/checkpoints/repository

mkdir -p third_party/sam-3d-objects/checkpoints/hf
cp -a third_party/sam-3d-objects/checkpoints/repository/checkpoints/. \
  third_party/sam-3d-objects/checkpoints/hf/
```

Download the official FoundationPose weights:

```bash
mkdir -p third_party/FoundationPose/weights
gdown --folder \
  "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing" \
  -O third_party/FoundationPose/weights
```

The expected files are:

```text
third_party/FoundationPose/weights/
├── 2023-10-28-18-33-37/
│   ├── config.yml
│   └── model_best.pth
└── 2024-01-11-20-02-45/
    ├── config.yml
    └── model_best.pth
```

## 5. Blender

PhysMind expects Blender at `third_party/blender/blender`. The tested release is Blender 4.2.11 LTS:

```bash
mkdir -p third_party/blender
curl -L \
  https://download.blender.org/release/Blender4.2/blender-4.2.11-linux-x64.tar.xz \
  -o third_party/blender-4.2.11-linux-x64.tar.xz
tar -xf third_party/blender-4.2.11-linux-x64.tar.xz \
  --strip-components=1 -C third_party/blender
third_party/blender/blender --version
```

## 6. API Configuration

```bash
cp .env.example .env
```

Set the key for the provider you use. The supported variables are `OPEN_ROUTER_KEY`, `OPENAI_API_KEY`, `CLOSEAI_API_KEY`, and `GEMINI_API_KEY`. Provider and model can also be selected with `--provider` and `--model`.

## 7. Verification

Run the repository's CPU-safe regression suite without an extra test dependency:

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v
```

Confirm the core CUDA and tool imports on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import xformers.ops
import flash_attn
import nvdiffrast.torch
import kaolin
import pytorch3d
import gsplat
import spconv.pytorch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("CUDA extensions OK")
PY

python -c "from sam3.model_builder import build_sam3_video_predictor; print('SAM 3 OK')"
python -c "from moge.model.v2 import MoGeModel; print('MoGe-2 OK')"
python -c "from geocalib import GeoCalib; print('GeoCalib OK')"
python physmind.py --help
```

Finally, prepare one benchmark and run a bounded smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python physmind.py \
  --bench clevrer \
  --mode world-model-agent \
  --dataset-root data/clevrer \
  --limit 1
```
