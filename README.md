# PRISM: Parallel Refinement of Intra-Batch Staleness in MTGNN

**PRISM** is a fast framework for memory-augmented temporal graph learning. It supports large-batch training with multi-version memory updates and dependency-aware message passing.

---

## 🔧 Key Components

### ✅ GRN-Stream Sampler

A GPU-accelerated temporal sampler optimized for fast and scalable training on dynamic graphs.

**Features:**
- CUDA-accelerated neighbor finding  
- OpenMP-based C++ preprocessing  
- Chunked, memory-efficient sampling  
- Chronologically optimized batch construction  

### ✅ Intra-Batch Staleness Handling

Mitigates stale memory problems within large batches using dependency-aware approximation.

**Features:**
- CUDA-accelerated memory computation graph generation  
- Approximate multi-version intra-batch memory  
- Ensures k-fresh memory to push approximation errors k-steps away  
- Parallel memory refinement of all nodes in the batch to achieve k-fresh memory  

---

## ⚙️ Build Instructions

Default assumption (Unity cluster): load CUDA before building extensions:

```bash
module load cuda/11.8
```

To compile the custom C++ and CUDA extensions (TCI Engine and GRN-Stream Sampler):

```bash
python setup.py build_ext --inplace
```

If you are not on Unity, make sure CUDA Toolkit headers are available, then build:

```bash
export CUDA_HOME=/path/to/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
python setup.py build_ext --inplace
```

## 🚀 Running the Code

The main training and evaluation script is:
```bash
python main.py
```

✅ Example

```bash
python main.py --data tgbl-wiki --bs 1024 --lr 0.001 --deliver_to neighbor --val_neg 5
```

There is also a dedicated single-GPU lifecycle wrapper script that uses the
`SingleGPUExperiment` module directly while keeping `main.py` unchanged:

```bash
python run_single_gpu_experiment.py --data tgbl-wiki --bs 1024 --lr 0.001 --deliver_to neighbor --val_neg 5
```

Use:
- `main.py` for the existing default CLI path.
- `run_single_gpu_experiment.py` when you want to exercise the explicit `setup -> train -> validate -> test` single-GPU lifecycle wrapper.

`main.py` now supports automatic sampler disk offloading with `--offload-mode auto` (default).  
Modes:
- `auto`: try in-memory sampler first, fall back to disk offloading on OOM.
- `off`: force in-memory sampler.
- `on`: force disk-offloading sampler.

You can also trigger runtime switching in `auto` mode based on GPU memory usage:
- `--auto-offload-gpu-mem-pct X`: if GPU memory usage reaches `X%`, switch from in-memory sampler to disk-offload.
- Default is `-1` (disabled).

Examples:

```bash
# Disable offloading entirely (in-memory only)
python main.py --data tgbl-wiki --offload-mode off

# Force disk offloading from the start
python main.py --data tgbl-wiki --offload-mode on

# Auto: start in-memory, switch on OOM or when GPU memory reaches 85%
python main.py --data tgbl-wiki --offload-mode auto --auto-offload-gpu-mem-pct 85
```

To run with disk offloading

```bash
python dart_linkpred.py --data tgbl-wiki --bs 1024 --lr 0.001 --deliver_to neighbor --val_neg 5
```

## 📦 Using PRISM As A Python Package

You can use PRISM either with the provided conda environment, or as a lightweight extension on top of an existing `torch + pyg + tgb` setup.

### Option A: With `environment.yml`

```bash
conda env create -f environment.yml
conda activate prism-pyg
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

### Option B: Existing Environment (Lightweight Extension)

If you already have compatible `torch`, `torch-geometric`, and `py-tgb` installed:

```bash
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

### Option C: Non-Editable Install (wheel-style)

Use this if you want PRISM installed into `site-packages` instead of linked to your source tree:

```bash
pip install . --no-build-isolation
```

Notes:
- For non-editable install, you normally do **not** need `python setup.py build_ext --inplace` first.
- `pip install .` builds the C++/CUDA extensions during installation and installs the compiled `.so` files into `site-packages` (expected behavior).
- Use `build_ext --inplace` only when you specifically want `.so` files generated inside the repo for direct source-tree execution/debugging.

Then in Python:

```python
from prism import PrismConfig, PrismExperiment

cfg = PrismConfig(
    data="tgbl-wiki",
    batch_size=1024,
    lr=1e-3,
    decoder="fc",
    deliver_to="self",
    val_neg=5,
    load_ns=True,   # use offline negatives if available
)
exp = PrismExperiment(cfg).setup()
train_val_result = exp.train()  # train + val only
test_result = exp.test()        # run test once, separately
```

### PrismConfig Parameters

`PrismConfig` defaults and meanings:

- `data="tgbl-wiki"`: built-in dataset name (used when `dataset_csv` and `dataset_dir` are not set).
- `dataset_root="datasets"`: root directory for built-in datasets (`data=...` mode). Absolute paths are used directly; relative paths are resolved from your current working directory.
- `batch_size=1024`: minibatch size for train/val/test loaders.
- `lr=1e-3`: optimizer learning rate.
- `k_value=10`: number of temporal neighbors sampled per node.
- `num_epoch=50`: number of training epochs.
- `seed=1`: random seed for reproducibility.
- `mem_dim=100`: memory-state dimension.
- `time_dim=100`: time-encoding dimension.
- `emb_dim=100`: node embedding output dimension.
- `tolerance=1e-6`: early-stop improvement tolerance.
- `deliver_to="self"`: message delivery mode (`"self"` or `"neighbor"`).
- `decoder="fc"`: link predictor decoder (`"fc"` or `"NCN"`).
- `embedding="gat"`: embedding module (`"gat"` or `"time_emb"`).
- `val_neg=5`: number of negatives per positive during validation/test (`-1` means use all available for offline negatives).
- `offload_mode="auto"`: sampler offloading mode (`"off"`, `"on"`, `"auto"`).
- `tensor_store_mode="on"`: DAATGNMemory tensor-store mode (`"on"`, `"off"`, `"verify"`), same behavior as `main.py --tensor-store-mode`.
- `cache_data_on_gpu=True`: cache dataset tensors (`t/msg/src`) on GPU at startup when CUDA is available.
- `chunk_size=256`: temporal chunk size used by sampler preprocessing.
- `skip_cnt=16`: skip/coalescing control used in neighbor-delivery path.
- `m_pass=3`: number of intra-batch memory refinement passes.
- `load_ns=True`: whether to load precomputed offline negatives for built-in datasets.
- `auto_offload_gpu_mem_pct=-1.0`: runtime memory threshold for switching to disk offload in `auto` mode; `<0` disables threshold switching.
- `dataset_csv=None`: path to custom CSV dataset (`src`, `dst`, `t`, optional `msg*`).
- `dataset_dir=None`: path to directory-form dataset (e.g., `edges.csv`, optional `edge_features.pt`).
- `val_ratio=0.15`: validation split ratio for custom CSV/directory datasets.
- `test_ratio=0.15`: test split ratio for custom CSV/directory datasets.
- `num_runs=1`: number of independent runs (`seed + run_idx`) for aggregated reporting.
- `run_test=False`: if `True`, `exp.evaluate()` (or `exp.train()`) also runs test after selecting best val checkpoint.

### Train / Validate / Test Workflow

- `exp.train()` runs epoch-wise **train+val** and returns training history (or per-run summary when `num_runs>1`).
- `exp.test()` is a separate phase and can be called once after training state is ready.
- `exp.evaluate()` is a convenience wrapper around `train()`; with `run_test=True`, it performs train+val and then test.
- During `train()`, logs include per-epoch `loss`, validation metric, `Train Time (s)`, epoch elapsed, and cumulative `Train+Val Elapsed (s)`.

## 🗂️ Supported Datasets

- Built-in PRISM/TGB datasets (e.g., `tgbl-wiki`) via `PrismConfig(data="...")`.
- Custom dataset in tgbl-wiki-like temporal format (`src`, `dst`, `t`, optional `msg*` columns):

```python
cfg = PrismConfig(
    dataset_csv="path/to/your_temporal_edges.csv",
    batch_size=1024,
    load_ns=False,  # no precomputed negatives required
    val_neg=20,     # random negatives per positive for val/test by default
)
exp = PrismExperiment(cfg).setup()
```

- Directory-form datasets like GNNFlow `REDDIT` layout (`edges.csv` + optional `edge_features.pt`):

```python
cfg = PrismConfig(
    dataset_dir="/work/pi_mserafini_umass_edu/ashraful/disgnn/GNNFlow/data/REDDIT",
    batch_size=1024,
    load_ns=False,
    val_neg=20,
)
exp = PrismExperiment(cfg).setup()
```

When offline negatives are not provided, PRISM automatically falls back to random negatives for validation/test.

## 🎯 Offline Negative Sample Generation

PRISM exposes an interface to precompute validation/test negatives:

```bash
prism-negatives --data tgbl-wiki --num-neg 100 --strategy rnd --if-exists skip
```

Or from Python:

```python
from prism import generate_offline_negative_samples

generate_offline_negative_samples(
    dataset_name="tgbl-wiki",
    num_neg_per_pos=100,   # k random negatives per positive by default
    strategy="rnd",        # or "hist_rnd"
    if_exists="skip",      # "skip" (default), "force", or "error"
)
```

`if_exists` behavior:
- `skip`: do nothing for splits that already have generated files.
- `force`: overwrite by deleting existing generated files and regenerating.
- `error`: raise an error if generated files already exist.

### Multi-run Benchmark Example (TGB-style Report)

```python
from prism import PrismConfig, PrismExperiment

cfg = PrismConfig(
    data="tgbl-wiki",
    num_epoch=100,
    num_runs=3,
    run_test=True,
    val_neg=-1,   # use full provided negatives when available
    load_ns=True,
)
exp = PrismExperiment(cfg).setup()
result = exp.evaluate()  # train+val per run, then test per run
print(result["summary"])  # includes best_val_mean/std and test_mean/std
```

## 🧾 Argument Descriptions

| Argument       | Description                                                                              |
| -------------- | ---------------------------------------------------------------------------------------- |
| `--data`       | Dataset name. Supported: `tgbl-wiki`, `reddit`, `lastfm`, `tgbl-coin`, `tgbl-comment`, etc. *(Default: tgbl-wiki)*      |
| `--bs`         | Batch size                                                                               |
| `--lr`         | Learning rate                                                                            |
| `--mxtt`       | Max training time in hours *(Default: 48)*                                               |
| `--mxet`       | Max total execution time (train + val) in hours *(Default: 72)*                          |
| `--debug`      | If true, disables validation *(Default: False)*                                          |
| `--custom_neg` | Use custom negative sampler *(Not implemented yet)*                                      |
| `--deliver_to` | Message delivery target: `self` (TGN, TNCN, Jodie) or `neighbor` (APAN) *(Default: self)* |
| `--decoder`    | Decoder type: `fc` (TGN, APAN, Jodie) or `NCN` (TNCN) *(Default: fc)*                    |
| `--embedding`  | Embedding type: `gat` (TGN, APAN, TNCN) or `time_emb` (Jodie) *(Default: gat)*           |
| `--val_neg`    | Number of negatives during validation. `-1` means use all available. *(Default: -1)*     |
| `--no-ns`      | Disable loading offline negative samples for built-in datasets. By default negatives are loaded when available. |
| `--chunk_size` | Temporal chunk size used in GRN-Stream preprocessing *(Default: 256)*                     |
| `--skip_cnt`   | Skip/coalescing control for neighbor-delivery sampling path *(Default: 16)*               |
| `--m_pass`     | Number of intra-batch memory refinement passes *(Default: 3)*                             |
| `--offload-mode` | Sampler backend mode: `off` (in-memory), `on` (disk offload), `auto` (fallback/switch) *(Default: auto)* |
| `--auto-offload-gpu-mem-pct` | In `auto` mode, switch to disk offload when GPU memory usage reaches this percent; `<0` disables threshold switching *(Default: -1)* |
| `--tensor-store-mode` | DAATGNMemory store mode: `on`, `off`, or `verify` *(Default: on)*                 |
| `--cache-data-on-gpu` / `--no-cache-data-on-gpu` | Enable/disable startup caching of dataset tensors (`t/msg/src`) on GPU *(Default: enabled)* |
