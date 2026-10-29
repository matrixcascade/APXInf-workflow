APXInf + PI0.5 + LIBERO/MuJoCo复用Workflow
这是一条 Ubuntu + NVIDIA 服务器从零部署到跑通样例的最短流程。本流程以本次验证过的 RTX 4090 / CUDA Toolkit 12.8 为参考；使用新版仓库或其他 GPU 时需重新验证。请预先安装 Anaconda/Miniconda（下文按 ~/anaconda3），推理 Python 为 3.12，用户有 sudo 权限，模型放在 `$HOME/models/pi05_libero_base`。
## 1. 检查服务器和代码
```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
python3 --version
cmake --version
git clone --recurse-submodules https://github.com/RLinf/APXinf-robo.git
cd APXinf-robo
git submodule update --init --recursive
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libssl-dev libegl1 libgl1 libglib2.0-0
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
```
APXInf 按当前 GPU 编译，建议直接在部署目标机编译。默认下载慢时，可在 `~/.cargo/config.toml` 配置 rsproxy：
```toml
[source.crates-io]
replace-with = "rsproxy-sparse"
[source.rsproxy-sparse]
registry = "sparse+https://rsproxy.cn/index/"
```
## 2. 编译和安装 APXInf
```bash
python3 -m venv "$HOME/envs/apxinf"
source "$HOME/envs/apxinf/bin/activate"
python -m pip install -U pip maturin numpy pillow sentencepiece safetensors
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=target/wheel
export CARGO_BUILD_JOBS=8
python -m maturin build --release --features cuda --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
python -m pip install --no-deps target/wheel/wheels/apxinf_py-*.whl
python -m pip install -e './apxinf/python/apxinf[serving]' --config-settings editable_mode=strict
python -m pip install -e '.[libero,serve]'
python -c 'import apxinf_py; print("binding import OK")'
```
若出现“undefined symbol: apxinf_flashinfer_gdn_launch”，可按以下方式修复：打开 apxinf/crates/apxinf-cuda-new/native/kernels/flashinfer_gdn/flashinfer_gdn_tma.cpp，在 #include "flashinfer_gdn.h" 后加入 #if defined(APXINF_GEMM_CUTLASS)，然后在文件末尾追加：
#else
namespace apxinf::cuda_new::flashinfer_gdn {
int prefill(const void*, const void*, const void*, void*, const void*,
            const void*, const void*, void*, void*, int, int, int, int,
            float, cudaStream_t) {
  return static_cast<int>(cudaErrorNotSupported);
}
size_t tensor_map_workspace_bytes(int, int) { return 0; }
}
#endif
保存后，在项目根目录、已激活 Python 环境的终端中依次执行：
export CARGO_TARGET_DIR="$PWD/target/wheel-gdn-fix"
python -m maturin build --release --features cuda --auditwheel skip -m apxinf/crates/apxinf-py/Cargo.toml
编译成功后执行：
python -m pip install --force-reinstall --no-deps "$CARGO_TARGET_DIR"/wheels/apxinf_py-*.whl
最后验证：
python -c 'import apxinf_py; print("binding import OK")'
输出“binding import OK”表示扩展可以正常导入。该修复使未启用对应 CUTLASS 构建分组的版本能够加载扩展，不会补充 GDN 算子的执行能力。仅对上述已验证版本使用；新版先检查是否已修复。
## 3. 准备模型
模型权重来源为 https://huggingface.co/lerobot/pi05_libero_base 。下载该仓库的 config.json 和 model.safetensors；norm_stats.json 使用 https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json ，tokenizer.model 使用 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model 。网络不佳时权重可从 hf-mirror.com 的对应路径断点下载。模型目录至少有 `model.safetensors`、`config.json`、`norm_stats.json`、`tokenizer.model`。检查权重：
```bash
export MODEL_DIR="$HOME/models/pi05_libero_base"
stat -c %s "$MODEL_DIR/model.safetensors"
python - <<'PY'
from safetensors import safe_open
import os
with safe_open(os.environ["MODEL_DIR"]+"/model.safetensors", framework="np") as f:
    print("tensor_count", len(f.keys()))
PY
```
大文件中断时使用 `curl -C -` 或 `aria2c -c` 断点续传。
## 4. 最小验证
```bash
mkdir -p devlocal
python examples/register_preset.py
python scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 --warmup 10 --samples 30 --action-horizon 10 --out devlocal/benchmark-bf16.json
python examples/preflight_check.py --robot franka_libero --model-dir "$MODEL_DIR"
python examples/robot_policy_infer.py --robot franka_libero --precision bf16 --model-dir "$MODEL_DIR" --norm-stats "$MODEL_DIR/norm_stats.json"
```
通过标准：扩展可导入、基准有 P50/P95、预检为 `verdict: clean`、策略例程返回 `(H, 7) float32` 动作。随机权重基准只测推理速度。

## 5. 安装 LIBERO/MuJoCo
LIBERO 使用 Python 3.10 和旧版 NumPy，单独建环境。先执行 source "$HOME/anaconda3/etc/profile.d/conda.sh" 使 conda 命令可用：
```bash
conda create -y -p "$HOME/envs/libero" python=3.10 pip
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$HOME/envs/libero"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==1.22.4 scipy==1.10.1 opencv-python==4.6.0.66 \
  numba==0.57.1 robosuite==1.4.0 bddl==1.0.1 mujoco==3.2.3 \
  matplotlib==3.7.5 contourpy==1.1.1 imageio imageio-ffmpeg \
  openpi-client easydict cloudpickle future gym termcolor
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
mkdir -p devlocal/libero
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git devlocal/libero/LIBERO
python -m pip check
```
设置环境变量：
```bash
export LIBERO_CONFIG_PATH="$PWD/devlocal/libero/config"
export PYTHONPATH="$PWD/apxinf/python/apxinf:$PWD/src:$PWD/devlocal/libero/LIBERO"
export MUJOCO_GL=egl
```
在仓库根目录执行，创建配置：
```bash
mkdir -p "$LIBERO_CONFIG_PATH" devlocal/libero/LIBERO/libero/datasets
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $PWD/devlocal/libero/LIBERO/libero/libero
bddl_files: $PWD/devlocal/libero/LIBERO/libero/libero/bddl_files
init_states: $PWD/devlocal/libero/LIBERO/libero/libero/init_files
datasets: $PWD/devlocal/libero/LIBERO/libero/datasets
assets: $PWD/devlocal/libero/LIBERO/libero/libero/assets
EOF
```
