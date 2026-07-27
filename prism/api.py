from dataclasses import dataclass
import os
import os.path as osp
import timeit
import statistics
from typing import Any

import torch

from tgb.utils.utils import set_random_seed

from modules.data_utils import read_data
from modules.memory_module import DAATGNMemory, DA_APANMemory
from modules.msg_func import IdentityMessage
from modules.msg_agg import MeanAggregator as Agg
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding
from modules.decoder import LinkPredictor
from modules.NCNDecoder.NCNPred import NCNPredictor
from modules.neg_sampler import NegLinkSamplerDest
from modules.sampler_builder import SamplerBuildSpec, build_sampler_backend
from modules.train_utils import train as actrain, test_new as test
from modules.early_stopping import EarlyStopMonitor
from .datasets import dataset_from_csv, dataset_from_directory


@dataclass
class PrismConfig:
    data: str = "tgbl-wiki"
    dataset_root: str = "datasets"
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
    tensor_store_mode: str = "on"
    cache_data_on_gpu: bool = True
    chunk_size: int = 256
    skip_cnt: int = 16
    m_pass: int = 3
    load_ns: bool = True
    auto_offload_gpu_mem_pct: float = -1.0
    dataset_csv: str | None = None
    dataset_dir: str | None = None
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_runs: int = 1
    run_test: bool = False


class PrismExperiment:
    def __init__(self, cfg: PrismConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._built = False
        self._data_ready = False
        self.max_seen_eid = -1
        self.phase = "init"
        self.phase_max_seen_eid = {"train": -1, "val": -1, "test": -1}
        self.memory_progress_eid = -1
        self._test_consumed = False

    def _prepare_data(self):
        cfg = self.cfg
        os.environ["PRISM_TENSOR_STORE_MODE"] = cfg.tensor_store_mode
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
            self.dataset = read_data(
                cfg.data,
                cfg.batch_size,
                load_neg_sampler=cfg.load_ns,
                root=cfg.dataset_root,
            )
        self.data = self.dataset["data"]
        self.data_cache = {"t": None, "msg": None, "src": None}
        if cfg.cache_data_on_gpu and self.device.type == "cuda":
            try:
                self.data_cache["t"] = self.data.t.to(self.device)
                self.data_cache["msg"] = self.data.msg.to(self.device)
                self.data_cache["src"] = self.data.src.to(self.device)
                print("INFO: Cached dataset tensors on GPU (t/msg/src).")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                print(
                    "WARNING: Could not cache dataset tensors on GPU (OOM). "
                    "Falling back to per-iteration CPU->GPU copies."
                )
                torch.cuda.empty_cache()
        elif cfg.cache_data_on_gpu and self.device.type != "cuda":
            print("INFO: cache_data_on_gpu requested but CUDA is unavailable; using CPU tensors.")
        self.unique_destination_nodes = torch.unique(self.data.dst)
        self.min_dst_idx, self.max_dst_idx = int(self.data.dst.min()), int(self.data.dst.max())

        items = torch.cat([self.data.src, self.data.dst])
        _, counts = torch.unique(items, return_counts=True)
        max_freq = counts.max().item()
        max_chunk_per_node = 1 + max_freq // cfg.chunk_size

        sampler_result = build_sampler_backend(
            self.data,
            SamplerBuildSpec(
                k=cfg.k_value,
                chunk_size=cfg.chunk_size,
                max_chunk_per_node=max_chunk_per_node,
                offload_mode=cfg.offload_mode,
                cache_size=cfg.batch_size,
                device=self.device,
                apan=(cfg.deliver_to == "neighbor"),
                skip_cnt=cfg.skip_cnt,
            ),
        )
        self.sampler = sampler_result.sampler
        self.sampler_backend = sampler_result.backend
        self._data_ready = True

    def _build_run(self, run_seed: int, run_idx: int):
        cfg = self.cfg
        torch.manual_seed(run_seed)
        set_random_seed(run_seed)

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
        run_id = f"{cfg.data}_{cfg.deliver_to}_{cfg.decoder}_{run_seed}_{run_idx}_{os.getpid()}"
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
            "data_cache": self.data_cache,
        }

    def setup(self):
        if not self._data_ready:
            self._prepare_data()
        self._build_run(self.cfg.seed, 0)
        self._built = True
        self._test_consumed = False
        return self

    def train(self, epochs: int | None = None):
        if not self._data_ready:
            self._prepare_data()
        self._test_consumed = False
        epochs = self.cfg.num_epoch if epochs is None else epochs
        all_runs = []
        for run_idx in range(self.cfg.num_runs):
            run_seed = self.cfg.seed + run_idx
            self._build_run(run_seed, run_idx)
            self._test_consumed = False
            history = {"loss": [], "val": [], "time": []}
            train_val_start = timeit.default_timer()
            train_only_time = 0.0
            metric_name = self.dataset.get("metric", "mrr")
            print("-------------------------------------------------------------------------------")
            print(f"INFO: >>>>> Run: {run_idx} <<<<<")
            print(f"INFO: fixed random seed: {run_seed}")
            self.phase = "train"
            self.phase_max_seen_eid = {"train": -1, "val": -1, "test": -1}
            self.memory_progress_eid = -1
            for epoch in range(1, epochs + 1):
                epoch_start = timeit.default_timer()
                # Keep parity with main.py: trainer expects epoch-local eid offset.
                train_start = timeit.default_timer()
                loss, max_seen_eid = actrain(self.targs, -1)
                epoch_train_time = timeit.default_timer() - train_start
                train_only_time += epoch_train_time
                val, val_end_eid = test(self.targs, max_seen_eid, split_mode="val")
                self.phase_max_seen_eid["train"] = max_seen_eid
                self.phase_max_seen_eid["val"] = val_end_eid
                self.memory_progress_eid = val_end_eid
                history["loss"].append(loss)
                history["val"].append(val)
                epoch_elapsed = timeit.default_timer() - epoch_start
                history["time"].append(epoch_elapsed)
                train_val_elapsed = timeit.default_timer() - train_val_start
                print(
                    f"Epoch: {epoch:02d}, Loss: {loss:.4f}, "
                    f"{metric_name}: {val:.4f}, "
                    f"Train Time (s): {epoch_train_time:.4f}, "
                    f"Epoch Elapsed (s): {epoch_elapsed:.4f}, "
                    f"Train+Val Elapsed (s): {train_val_elapsed:.4f}"
                )
                if self.early_stopper.step_check(val, self.model):
                    break
            self.early_stopper.load_checkpoint(self.model)
            best_val = max(history["val"]) if history["val"] else float("nan")
            train_val_time = timeit.default_timer() - train_val_start
            run_result = {
                "run_idx": run_idx,
                "seed": run_seed,
                "history": history,
                "best_val": best_val,
                "train_time": train_only_time,
                "train_val_time": train_val_time,
                "phase_max_seen_eid": dict(self.phase_max_seen_eid),
            }
            if self.cfg.run_test:
                self.phase = "test"
                test_start_eid = self.phase_max_seen_eid["val"]
                if test_start_eid < 0:
                    test_start_eid = self.phase_max_seen_eid["train"]
                test_metric, test_end_eid = test(self.targs, test_start_eid, split_mode="test")
                self.phase_max_seen_eid["test"] = test_end_eid
                self.memory_progress_eid = test_end_eid
                run_result["test"] = test_metric
                run_result["phase_max_seen_eid"] = dict(self.phase_max_seen_eid)
                # Test events have been consumed for this run state.
                self._test_consumed = True
            all_runs.append(
                run_result
            )

        if self.cfg.num_runs == 1:
            self.last_train_result = all_runs[0]["history"]
            if self.cfg.run_test:
                self.last_test_result = all_runs[0]["test"]
            return self.last_train_result

        best_vals = [r["best_val"] for r in all_runs]
        train_only_times = [r["train_time"] for r in all_runs]
        train_times = [r["train_val_time"] for r in all_runs]
        sample_val_std = statistics.stdev(best_vals) if len(best_vals) > 1 else 0.0
        sample_train_only_std = statistics.stdev(train_only_times) if len(train_only_times) > 1 else 0.0
        sample_train_time_std = statistics.stdev(train_times) if len(train_times) > 1 else 0.0
        self.last_train_result = {
            "num_runs": self.cfg.num_runs,
            "runs": all_runs,
            "summary": {
                "best_val_mean": statistics.mean(best_vals),
                "best_val_std": sample_val_std,
                "train_time_mean": statistics.mean(train_only_times),
                "train_time_std": sample_train_only_std,
                "train_val_time_mean": statistics.mean(train_times),
                "train_val_time_std": sample_train_time_std,
            },
        }
        if self.cfg.run_test:
            tests = [r["test"] for r in all_runs]
            sample_test_std = statistics.stdev(tests) if len(tests) > 1 else 0.0
            self.last_train_result["summary"]["test_mean"] = statistics.mean(tests)
            self.last_train_result["summary"]["test_std"] = sample_test_std
            self.last_test_result = self.last_train_result["summary"]["test_mean"]
        return self.last_train_result

    def evaluate(self, epochs: int | None = None):
        """
        Train+validate, and optionally test if cfg.run_test=True.
        Returns per-run stats and summaries from train().
        """
        return self.train(epochs=epochs)

    # Backward-compatible typo alias requested by user prompt.
    def evalute(self, epochs: int | None = None):
        return self.evaluate(epochs=epochs)

    def validate(self):
        if not self._built:
            self.setup()
        self.phase = "val"
        val_start_eid = self.phase_max_seen_eid["train"]
        val, val_end_eid = test(self.targs, val_start_eid, split_mode="val")
        self.phase_max_seen_eid["val"] = val_end_eid
        self.memory_progress_eid = val_end_eid
        return val

    def test(self):
        if not self._built:
            self.setup()
        if self._test_consumed:
            raise RuntimeError(
                "test() already executed for the current trained state. "
                "This would re-consume test events and leak memory state. "
                "Start a new experiment/run or retrain before testing again."
            )
        self.early_stopper.load_checkpoint(self.model)
        self.phase = "test"
        # test starts after train and val memory progression
        test_start_eid = self.phase_max_seen_eid["val"]
        if test_start_eid < 0:
            test_start_eid = self.phase_max_seen_eid["train"]
        tst, test_end_eid = test(self.targs, test_start_eid, split_mode="test")
        self.phase_max_seen_eid["test"] = test_end_eid
        self.memory_progress_eid = test_end_eid
        self.last_test_result = tst
        self._test_consumed = True
        return tst

    def get_state(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "phase_max_seen_eid": dict(self.phase_max_seen_eid),
            "memory_progress_eid": self.memory_progress_eid,
        }

    @staticmethod
    def _format_mean_std(mean_val: float, std_val: float) -> str:
        # Matches TGB form expectation: "mean, std"
        return f"{mean_val:.10f}, {std_val:.10f}"

    def leaderboard_submission_report(
        self,
        contact_email: str,
        primary_contact_name: str,
        tgb_version: str,
        method_name: str,
        external_data: str,
        dataset_name: str,
        code_access: str,
        paper_link: str,
        tuned_hyperparameters: str,
        implementation: str,
        num_parameters: str,
        hardware: str,
    ) -> dict[str, Any]:
        """
        Create a form-ready report dictionary for TGB leaderboard submission.
        Call this after `train()`.
        """
        if not hasattr(self, "last_train_result"):
            raise RuntimeError("Run train() before generating submission report.")

        result = self.last_train_result
        if isinstance(result, dict) and "summary" in result:
            val_mean = result["summary"]["best_val_mean"]
            val_std = result["summary"]["best_val_std"]
            test_mean = getattr(self, "last_test_result", float("nan"))
            test_std = 0.0
        else:
            # Single-run fallback: std = 0
            val_hist = result.get("val", [])
            val_mean = max(val_hist) if val_hist else float("nan")
            val_std = 0.0
            test_mean = getattr(self, "last_test_result", float("nan"))
            test_std = 0.0

        return {
            "Contact Email": contact_email,
            "Primary Contact Name": primary_contact_name,
            "TGB Package Version": tgb_version,
            "Name of Your Method": method_name,
            "External data": external_data,
            "Dataset": dataset_name,
            "Test Performance": self._format_mean_std(test_mean, test_std),
            "Validation Performance": self._format_mean_std(val_mean, val_std),
            "Code Access": code_access,
            "Paper Link": paper_link,
            "Tuned Hyper-parameters": tuned_hyperparameters,
            "Implementation": implementation,
            "# of Parameters": num_parameters,
            "Hardware": hardware,
        }
