from modules.train_utils import train as actrain, test_new as test, train_with_custom_neg_sampler
from modules.early_stopping import EarlyStopMonitor
from modules.runtime_entry import (
    build_dataset_bundle,
    build_model_bundle,
    build_sampler,
    build_train_args,
    make_run_tag,
    parse_runtime_config,
)

# from torch.optim.lr_scheduler import StepLR

from tgb.utils.utils import set_random_seed
import torch

import timeit
import os
import os.path as osp
from pathlib import Path

def main():
    config = parse_runtime_config()
    args = config.args

    print("INFO: Arguments:", args)
    print(f"INFO: Offload mode: {args.offload_mode}")
    print(f"INFO: Auto offload GPU mem threshold (%): {args.auto_offload_gpu_mem_pct}")
    print(f"INFO: PRISM tensor store mode: {args.tensor_store_mode}")
    print(f"INFO: Cache data on GPU: {args.cache_data_on_gpu}")

    run_tag = make_run_tag(config)
    print(f"INFO: Run tag: {run_tag}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_bundle = build_dataset_bundle(config, device)
    sampler_bundle = build_sampler(config, dataset_bundle, device)

    print(f"Done. Conversion  Time (s): {sampler_bundle.build_time_seconds: .4f}")
    print(f"INFO: Sampler backend: {sampler_bundle.backend}")
       # for saving the results...
    results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
    if not osp.exists(results_path):
        os.mkdir(results_path)
        print('INFO: Create directory {}'.format(results_path))
    Path(results_path).mkdir(parents=True, exist_ok=True)

    for run_idx in range(config.num_runs):
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + config.seed)
        set_random_seed(run_idx + config.seed)
        model_bundle = build_model_bundle(config, dataset_bundle, device)

        # Helper vector to map global node indices to local ones.
        # assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

        # define an early stopper
        save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
        save_model_id = (
            f'{config.model_name}_{config.data}_{args.deliver_to}_{args.decoder}_{args.embedding}_{config.seed}_{run_idx}_{run_tag}'
        )
        early_stopper = EarlyStopMonitor(
            save_model_dir=save_model_dir,
            save_model_id=save_model_id,
            tolerance=config.tolerance,
            patience=config.patience,
        )
        targs = build_train_args(
            config,
            model_bundle,
            dataset_bundle,
            sampler_bundle.sampler,
            device,
        )
        val_perf_list = []
        start_train_val = timeit.default_timer()
        losses = []
        tims = []
        t_tims = 0
        e_tims = 0
        mrrs = []
        for epoch in range(1, config.num_epoch + 1):
            # training
            start_epoch_train = timeit.default_timer()
            loss, max_seen_eid = actrain(targs, -1)
            tim = timeit.default_timer() - start_epoch_train
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
            )

            tims.append(tim)
            losses.append(loss)
            t_tims += tim

            if not config.debug:
            
                perf_metric_val, max_seen_id = test(targs, max_seen_eid, split_mode="val")
                print(f"\tValidation {dataset_bundle.dataset['metric']}: {perf_metric_val: .4f}")
                # # print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
                val_perf_list.append(perf_metric_val)
                # check for early stopping
                
                if early_stopper.step_check(perf_metric_val, model_bundle.model):
                    break
                if t_tims > config.max_train_seconds:
                    break

                e_tims += timeit.default_timer() - start_epoch_train
                if e_tims > config.max_exec_seconds:
                    break

            # Optional runtime switch: in auto mode, move to disk-offload when
            # GPU memory pressure crosses a configured threshold.
            if (
                args.offload_mode == "auto"
                and sampler_bundle.backend == "in-memory"
                and args.auto_offload_gpu_mem_pct >= 0
                and device.type == "cuda"
            ):
                dev_index = device.index if device.index is not None else torch.cuda.current_device()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device=dev_index)
                used_pct = 100.0 * (1.0 - (float(free_bytes) / float(total_bytes)))
                if used_pct >= args.auto_offload_gpu_mem_pct:
                    print(
                        f"INFO: GPU memory usage {used_pct:.2f}% >= "
                        f"{args.auto_offload_gpu_mem_pct:.2f}% -> switching sampler to disk-offload."
                    )
                    args.offload_mode = "on"
                    sampler_bundle = build_sampler(config, dataset_bundle, device)
                    targs["sampler"] = sampler_bundle.sampler
                    print(f"INFO: Sampler backend: {sampler_bundle.backend}")
            
        train_val_time = timeit.default_timer() - start_train_val
        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")
        print("'mrr' : ", val_perf_list, ",")
        print("'loss' : ",losses,",")
        print("'time' : ",tims, ",")
        # ==================================================== Test
        if not config.debug:
            # first, load the best model
            early_stopper.load_checkpoint(model_bundle.model)
            # final testing
            start_test = timeit.default_timer()
            perf_metric_test, max_seen_eid = test(targs, max_seen_id, split_mode="test")

            print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
            print(f"\tTest: {dataset_bundle.dataset['metric']}: {perf_metric_test: .4f}")


if __name__ == "__main__":
    main()
