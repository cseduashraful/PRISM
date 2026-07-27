import timeit

from modules.runtime_entry import parse_runtime_config
from single_gpu_experiment import SingleGPUExperiment


def main():
    config = parse_runtime_config()
    args = config.args

    print("INFO: Arguments:", args)
    print(f"INFO: Offload mode: {args.offload_mode}")
    print(f"INFO: Auto offload GPU mem threshold (%): {args.auto_offload_gpu_mem_pct}")
    print(f"INFO: PRISM tensor store mode: {args.tensor_store_mode}")
    print(f"INFO: Cache data on GPU: {args.cache_data_on_gpu}")

    exp = SingleGPUExperiment(config)
    exp.setup()

    print(f"INFO: Run tag: {exp.run_tag}")
    print(f"INFO: Sampler backend: {exp.sampler_bundle.backend}")

    for run_idx in range(config.num_runs):
        print("-------------------------------------------------------------------------------")
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        exp.build_run(run_idx=run_idx)
        train_result = exp.train(run_idx=run_idx)

        print(f"Train & Validation: Elapsed Time (s): {train_result['train_val_time']: .4f}")
        print("'mrr' : ", train_result["val"], ",")
        print("'loss' : ", train_result["loss"], ",")
        print("'time' : ", train_result["time"], ",")

        if not config.debug:
            perf_metric_test, max_seen_eid = exp.test()
            print("INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
            print(f"\tTest: {exp.dataset_bundle.dataset['metric']}: {perf_metric_test: .4f}")

        print(f"INFO: Run elapsed Time (s): {timeit.default_timer() - start_run: .4f}")


if __name__ == "__main__":
    main()
