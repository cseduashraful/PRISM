import os
import os.path as osp
import timeit
from pathlib import Path
from typing import Any

import torch

from modules.early_stopping import EarlyStopMonitor
from modules.runtime_entry import (
    PrismRuntimeConfig,
    build_dataset_bundle,
    build_model_bundle,
    build_sampler,
    build_train_runtime,
    make_run_tag,
)
from modules.train_utils import train as actrain, test_new as test
from tgb.utils.utils import set_random_seed


class SingleGPUExperiment:
    def __init__(self, config: PrismRuntimeConfig, device: torch.device | None = None):
        self.config = config
        self.args = config.args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_tag = make_run_tag(config)
        self.dataset_bundle = None
        self.sampler_bundle = None
        self.model_bundle = None
        self.runtime = None
        self.early_stopper = None
        self.max_seen_eid = -1
        self.max_seen_id = -1
        self.results_path = f"{osp.dirname(osp.abspath(__file__))}/saved_results"
        self.save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"

    def setup(self):
        self.dataset_bundle = build_dataset_bundle(self.config, self.device)
        self.sampler_bundle = build_sampler(self.config, self.dataset_bundle, self.device)

        if not osp.exists(self.results_path):
            os.mkdir(self.results_path)
            print(f"INFO: Create directory {self.results_path}")
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        return self

    def build_run(self, run_idx: int):
        if self.dataset_bundle is None or self.sampler_bundle is None:
            self.setup()

        run_seed = run_idx + self.config.seed
        torch.manual_seed(run_seed)
        set_random_seed(run_seed)

        self.model_bundle = build_model_bundle(self.config, self.dataset_bundle, self.device)
        save_model_id = (
            f"{self.config.model_name}_{self.config.data}_{self.args.deliver_to}_"
            f"{self.args.decoder}_{self.args.embedding}_{self.config.seed}_{run_idx}_{self.run_tag}"
        )
        self.early_stopper = EarlyStopMonitor(
            save_model_dir=self.save_model_dir,
            save_model_id=save_model_id,
            tolerance=self.config.tolerance,
            patience=self.config.patience,
        )
        self.runtime = build_train_runtime(
            self.config,
            self.model_bundle,
            self.dataset_bundle,
            self.sampler_bundle,
            self.device,
        )
        self.max_seen_eid = -1
        self.max_seen_id = -1
        return self

    def train(self, run_idx: int = 0, epochs: int | None = None) -> dict[str, Any]:
        if self.runtime is None or self.model_bundle is None or self.early_stopper is None:
            self.build_run(run_idx)

        num_epochs = self.config.num_epoch if epochs is None else epochs
        val_perf_list = []
        losses = []
        tims = []
        t_tims = 0.0
        e_tims = 0.0
        start_train_val = timeit.default_timer()

        for epoch in range(1, num_epochs + 1):
            start_epoch_train = timeit.default_timer()
            loss, self.max_seen_eid = actrain(self.runtime, -1)
            tim = timeit.default_timer() - start_epoch_train
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, "
                f"Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
            )

            tims.append(tim)
            losses.append(loss)
            t_tims += tim

            if not self.config.debug:
                perf_metric_val, self.max_seen_id = self.validate(self.max_seen_eid)
                print(f"\tValidation {self.dataset_bundle.dataset['metric']}: {perf_metric_val: .4f}")
                val_perf_list.append(perf_metric_val)

                if self.early_stopper.step_check(perf_metric_val, self.model_bundle.model):
                    break
                if t_tims > self.config.max_train_seconds:
                    break

                e_tims += timeit.default_timer() - start_epoch_train
                if e_tims > self.config.max_exec_seconds:
                    break

            self._maybe_switch_sampler_backend()

        train_val_time = timeit.default_timer() - start_train_val
        return {
            "val": val_perf_list,
            "loss": losses,
            "time": tims,
            "train_val_time": train_val_time,
            "max_seen_eid": self.max_seen_eid,
            "max_seen_id": self.max_seen_id,
        }

    def validate(self, max_seen_eid: int | None = None):
        eval_start = self.max_seen_eid if max_seen_eid is None else max_seen_eid
        return test(self.runtime, eval_start, split_mode="val")

    def test(self, max_seen_id: int | None = None):
        if self.early_stopper is None or self.model_bundle is None:
            raise RuntimeError("Call build_run() and train() before test().")
        self.early_stopper.load_checkpoint(self.model_bundle.model)
        test_start = self.max_seen_id if max_seen_id is None else max_seen_id
        return test(self.runtime, test_start, split_mode="test")

    def _maybe_switch_sampler_backend(self):
        if (
            self.args.offload_mode == "auto"
            and self.sampler_bundle.backend == "in-memory"
            and self.args.auto_offload_gpu_mem_pct >= 0
            and self.device.type == "cuda"
        ):
            dev_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device=dev_index)
            used_pct = 100.0 * (1.0 - (float(free_bytes) / float(total_bytes)))
            if used_pct >= self.args.auto_offload_gpu_mem_pct:
                print(
                    f"INFO: GPU memory usage {used_pct:.2f}% >= "
                    f"{self.args.auto_offload_gpu_mem_pct:.2f}% -> switching sampler to disk-offload."
                )
                self.args.offload_mode = "on"
                self.sampler_bundle = build_sampler(self.config, self.dataset_bundle, self.device)
                self.runtime.sampler_runtime.sampler = self.sampler_bundle.sampler
                self.runtime.sampler_runtime.backend = self.sampler_bundle.backend
                print(f"INFO: Sampler backend: {self.sampler_bundle.backend}")
