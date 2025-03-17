import json
import os
import pathlib
import sys
import time
import multiprocessing as mp
from src.motivational.proactive import get_model_locations_direct
from src.motivational.proactiveparalleldiffworkloads import execute_proactive, save_results
from src.placement.model import DataclassJSONEncoder


def save_single_stats(results_dir, rps, stats, output_infra, results_postfix):
    results_folder = os.path.join(results_dir, f"{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    save_results(results_folder, stats)
    with open(os.path.join(results_folder, f"peak-config.json"), "w") as outfile:
        json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder

def proactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, exp_id, model_infra = args
    stats = execute_proactive(base_dir, output_infra, workload_config, sim_input_path, model_locations)
    results_postfix = f'{exp_id}/{output_infra}/{config_idx}'
    save_single_stats(results_dir, workload_config, stats, output_infra, results_postfix)

def main():
    out_dir = pathlib.Path(sys.argv[1])
    exp_id = sys.argv[2]
    opt_path = out_dir / "optimization_results" / exp_id
    path_fine_tuned_models = opt_path / "fine_tuned_models"
    model_locations = get_model_locations_direct(path_fine_tuned_models)
    assert(len(model_locations) > 0)
    sim_input_path = pathlib.Path("data/nofs-ids")  # Base path for simulation input files
    base_dir = pathlib.Path("data/nofs-ids")  # Base path for simulation input files

    workload_config_file = sys.argv[3]
    num_cores = int(sys.argv[4])

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    result_dir = out_dir / "validation_results"
    repetitions = 1
    infra = sys.argv[5]
    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, model_locations, result_dir, exp_id,
         infra)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as proactive_pool:
        proactive_pool.map(proactive_worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")

    end_ts = time.time()
    print(f'Duration: {end_ts - start_ts} seconds')

if __name__ == '__main__':
    main()
