# PRISM: Parallel Refinement of Intra-Batch Staleness in MTGNN

**PRISM** is a fast and memory-efficient framework for temporal graph learning. It supports large-batch training with multi-version memory updates and dependency-aware message passing.

---

## 🔧 Key Components

### ✅ Recent-K Sampler

A GPU-accelerated temporal sampler optimized for fast and scalable training on dynamic graphs.

**Features:**
- CUDA-accelerated neighbor finding  
- OpenMP-based C++ preprocessing  
- Chunked, memory-efficient sampling  
- Chronologically optimized batch construction  

### ✅ Intra-Batch Temporal Discontinuity Handling

Mitigates stale memory problems within large batches using dependency-aware approximation.

**Features:**
- CUDA-accelerated memory update graph generation  
- Approximate multi-version intra-batch memory  
- k-hop propagation to push approximation errors away  
- Scalable k-layer memory update module  

---

## ⚙️ Build Instructions

To compile the custom C++ and CUDA extensions:

```bash
python setup.py build_ext --inplace
```

To compile the disk offloading routine:

```bash
python prep_setup.py build_ext --inplace
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

To with disk offloading

```bash
python dart_linkpred.py --data tgbl-wiki --bs 1024 --lr 0.001 --deliver_to neighbor --val_neg 5
```

## 🧾 Argument Descriptions

| Argument       | Description                                                                              |
| -------------- | ---------------------------------------------------------------------------------------- |
| `--data`       | Dataset name. Supported: `tgbl-wiki`, `reddit`, `mooc`, etc. *(Default: tgbl-wiki)*      |
| `--bs`         | Batch size                                                                               |
| `--lr`         | Learning rate                                                                            |
| `--mxtt`       | Max training time in hours *(Default: 48)*                                               |
| `--mxet`       | Max total execution time (train + val) in hours *(Default: 72)*                          |
| `--debug`      | If true, disables validation *(Default: False)*                                          |
| `--custom_neg` | Use custom negative sampler *(Not implemented yet)*                                      |
| `--deliver_to` | Memory delivery target: `self` (TGN, TNCN, Jodie) or `neighbor` (APAN) *(Default: self)* |
| `--decoder`    | Decoder type: `fc` (TGN, APAN, Jodie) or `NCN` (TNCN) *(Default: fc)*                    |
| `--embedding`  | Embedding type: `gat` (TGN, APAN, TNCN) or `time_emb` (Jodie) *(Default: gat)*           |
| `--val_neg`    | Number of negatives during validation. `-1` means use all available. *(Default: -1)*     |
