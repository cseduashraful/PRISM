from dataclasses import dataclass
import os
import os.path as osp
from pathlib import Path
import timeit

import torch

from tgb.utils.utils import set_random_seed

from modules.data_utils import read_data
from modules.grnstream import GRN_Stream
from modules.memory_module import DAATGNMemory, DA_APANMemory
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator as Agg
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding
from modules.decoder import LinkPredictor
from modules.NCNDecoder.NCNPred import NCNPredictor
from modules.neg_sampler import NegLinkSamplerDest
from modules.train_utils import train as actrain, test_new as test
from modules.early_stopping import EarlyStopMonitor
from .datasets import dataset_from_csv, dataset_from_directory


@dataclass
class PrismConfig:
    data: str = "tgbl-wiki"
    batch_size: int = 1024
    lr: float = 1e-3
    k_value: int = 10
    num_epoch: int = 50
    seed: int = 1
    mem_dim: int = 100
    time_dim: int = 100
    emb_dim: int = 100
    tolerance: float = 1e-6
    deliver_to: str = "self"
    decoder: str = "fc"
    embedding: str = "gat"
    val_neg: int = 5
    offload_mode: str = "auto"
    chunk_size: int = 256
    skip_cnt: int = 16
    m_pass: int = 3
    load_ns: bool = True
    auto_offload_gpu_mem_pct: float = -1.0
    dataset_csv: str | None = None
    dataset_dir: str | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class PrismExperiment:
    def __init__(self, cfg: PrismConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._built = False
        self.max_seen_eid = -1

    def setup(self):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        set_random_seed(cfg.seed)

        if cfg.dataset_dir:
            self.dataset = dataset_from_directory(
                cfg.dataset_dir,
                batch_size=cfg.batch_size,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
            )
        elif cfg.dataset_csv:
            self.dataset = dataset_from_csv(
                cfg.dataset_csv,
                batch_size=cfg.batch_size,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
            )
        else:
            self.dataset = read_data(cfg.data, cfg.batch_size, load_neg_sampler=cfg.load_ns)
        self.data = self.dataset["data"]
        self.unique_destination_nodes = torch.unique(self.data.dst)
        self.min_dst_idx, self.max_dst_idx = int(self.data.dst.min()), int(self.data.dst.max())

        items = torch.cat([self.data.src, self.data.dst])
        _, counts = torch.unique(items, return_counts=True)
        max_freq = counts.max().item()
        max_chunk_per_node = 1 + max_freq // cfg.chunk_size

        self.sampler, self.sampler_backend = GRN_Stream.build(
            data=self.data,
            k=cfg.k_value,
            chunk_size=cfg.chunk_size,
            max_chunk_per_node=max_chunk_per_node,
            offload_mode=cfg.offload_mode,
            cache_size=cfg.batch_size,
            device=self.device,
            apan=(cfg.deliver_to == "neighbor"),
            skip_cnt=cfg.skip_cnt,
        )

        if cfg.deliver_to == "self":
            memory = DAATGNMemory(
                self.data.num_nodes,
                self.data.msg.size(-1),
                cfg.mem_dim,
                cfg.time_dim,
                message_module=IdentityMessage(self.data.msg.size(-1), cfg.mem_dim, cfg.time_dim),
                aggregator_module=Agg(emb_dim=self.data.msg.size(-1) + 2 * cfg.mem_dim + cfg.time_dim),
                layer=cfg.m_pass,
            ).to(self.device)
        else:
            memory = DA_APANMemory(
                self.data.num_nodes,
                self.data.msg.size(-1),
                cfg.mem_dim,
                cfg.time_dim,
                message_module=IdentityMessage(self.data.msg.size(-1), cfg.mem_dim, cfg.time_dim),
                aggregator_module=Agg(emb_dim=self.data.msg.size(-1) + 2 * cfg.mem_dim + cfg.time_dim),
                layer=cfg.m_pass,
            ).to(self.device)

        if cfg.embedding == "time_emb":
            gnn = TimeEmbedding(in_channels=cfg.mem_dim, out_channels=cfg.emb_dim).to(self.device)
        else:
            gnn = GraphAttentionEmbedding(
                in_channels=cfg.mem_dim,
                out_channels=cfg.emb_dim,
                msg_dim=self.data.msg.size(-1),
                time_enc=memory.time_enc,
            ).to(self.device)

        if cfg.decoder == "NCN":
            link_pred = NCNPredictor(
                in_channels=cfg.emb_dim,
                hidden_channels=256,
                out_channels=1,
                NCN_mode=2,
            ).to(self.device)
        else:
            link_pred = LinkPredictor(in_channels=cfg.emb_dim).to(self.device)

        self.model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}
        self.optimizer = torch.optim.Adam(
            set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
            lr=cfg.lr,
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/../saved_models/"
        run_id = f"{cfg.data}_{cfg.deliver_to}_{cfg.decoder}_{cfg.seed}_{os.getpid()}"
        self.early_stopper = EarlyStopMonitor(
            save_model_dir=save_model_dir,
            save_model_id=run_id,
            tolerance=cfg.tolerance,
            patience=cfg.num_epoch,
        )

        self.targs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "dataset": self.dataset,
            "min_dst_idx": self.min_dst_idx,
            "max_dst_idx": self.max_dst_idx,
            "device": self.device,
            "sampler": self.sampler,
            "neg_sampler": NegLinkSamplerDest(self.unique_destination_nodes),
            "deliver_to": cfg.deliver_to,
            "decoder": cfg.decoder,
            "embedding": cfg.embedding,
            "val_neg": cfg.val_neg,
            "known_dsts": None,
            "data_cache": {"t": None, "msg": None, "src": None},
        }
        self._built = True
        return self

    def train(self, epochs: int | None = None):
        if not self._built:
            self.setup()
        epochs = self.cfg.num_epoch if epochs is None else epochs
        history = {"loss": [], "val": [], "time": []}
        for epoch in range(1, epochs + 1):
            start = timeit.default_timer()
            loss, self.max_seen_eid = actrain(self.targs, self.max_seen_eid)
            val, _ = test(self.targs, self.max_seen_eid, split_mode="val")
            history["loss"].append(loss)
            history["val"].append(val)
            history["time"].append(timeit.default_timer() - start)
            if self.early_stopper.step_check(val, self.model):
                break
        return history

    def validate(self):
        if not self._built:
            self.setup()
        val, self.max_seen_eid = test(self.targs, self.max_seen_eid, split_mode="val")
        return val

    def test(self):
        if not self._built:
            self.setup()
        self.early_stopper.load_checkpoint(self.model)
        tst, self.max_seen_eid = test(self.targs, self.max_seen_eid, split_mode="test")
        return tst
