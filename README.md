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
pip install -e .
```

### Option B: Existing Environment (Lightweight Extension)

If you already have compatible `torch`, `torch-geometric`, and `py-tgb` installed:

```bash
python setup.py build_ext --inplace
pip install -e .
```

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
history = exp.train()
val = exp.validate()
tst = exp.test()
```

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
