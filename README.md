# Recent-K Sampler

Fast GPU-friendly sampler for temporal graph learning.

Features:
- CUDA-accelerated neighbor finding
- OpenMP C++ preprocessing
- Chunk-based efficient memory handling
- Chronologically optimized batch sampling


# Address Intra-batch Temporal Discontinuity using Aggregation

Multi-version memory management

Features:
- CUDA accelerated memory-update graph generation
- approximate dependency-aware multi-version intra-batch memory
- k-layer memory update module to push approximation k-hop away


To build the prepocessor, mem_graph generator and sampler

    python setup.py build_ext --inplace


We built the Sampler on cuda 11.8

#How to run
Main script: main.py

Args:
 -- data: supported link prediction datasets: tgbl-wiki, reddit, mooc, etc., default: tgbl-wiki
 -- bs: batch size
 --lr: learning rate
 -- -- Same as tgb

Custom Args:
 custom_parser.add_argument('--mxtt', type=int, default=48) # maximum training time in hours
    custom_parser.add_argument('--mxet', type=int, default=72) # maximum execution time (training+validation) in hours
    custom_parser.add_argument('--debug', type=bool, default=False) # debug true disables validation
    custom_parser.add_argument('--custom_neg', type=bool, default=False) #in order to use custom negative sampler, #not implemented yet
    custom_parser.add_argument('--deliver_to', type=str, default='self') #deliver_to self for tgn, tncn and jodie, deliver_to neighbor for apan
    custom_parser.add_argument('--decoder', type=str, default='fc') #decoder fc for tgn, apan, jodie, decoder NCN for tncn
    custom_parser.add_argument('--embedding', type=str, default='gat') #embedding gat for tgn, apan, tncn, embedding time_emb for jodie
    custom_parser.add_argument('--val_neg', type=int, default=-1) #number of negative samples to consider during validation. -1 means all available negative samples in the file. (test always uses -1)
 
 
