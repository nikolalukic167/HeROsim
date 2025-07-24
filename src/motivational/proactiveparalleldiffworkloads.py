import json
import os.path
import sys
import time
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

from src.executeinitial import load_simulation_inputs, setup_logging, flatten_workloads
from src.motivational.constants import QUEUE_LENGTH, KEEP_ALIVE, PROACTIVE_RECONCILE_INTERVAL
from src.motivational.proactive import get_model_locations
from src.motivational.reactive import save_stats
from src.motivational.reactiveparalleldiffworkloads import save_results
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder


def save_single_stats(results_dir, rps, stats, output_infra, results_postfix, save_raw_results: bool=True):
    results_folder = os.path.join(results_dir, f"infra-{output_infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    save_results(results_folder, stats)
    if save_raw_results:
        with open(os.path.join(results_folder, f"peak-config.json"), "w") as outfile:
            json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def execute_proactive(base_dir, infra, workload_config, sim_input_path, model_locations):
    sim_inputs = load_simulation_inputs(sim_input_path)
    with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
        infrastructure = json.load(fd)
    with open(workload_config, 'r') as fd:
        workload = json.load(fd)
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'prohetkn_prohetkn'
        simulation_data = SimulationData(
            platform_types=sim_inputs['platform_types'],
            storage_types=sim_inputs['storage_types'],
            qos_types=sim_inputs['qos_types'],
            application_types=sim_inputs['application_types'],
            task_types=sim_inputs['task_types'],
        )
        print(f'Start simulation: peak_config - {infra}, {model_locations}')
        stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, task_priority,
                            queue_length,
                            scheduling_strategy, workload, 'workload-mine', model_locations=model_locations,
                            reconcile_interval=PROACTIVE_RECONCILE_INTERVAL)
        print(f'End simulation: peak_config - {infra}')
        return stats


def worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra = args
    stats = execute_proactive(base_dir, output_infra, workload_config, sim_input_path, model_locations)
    results_postfix = f'proactive/origin-{model_infra}-target-{output_infra}-model-{model_dir}/{start_time}/{str(rep_idx)}/{config_idx}'
    save_single_stats(results_dir, workload_config, stats, output_infra, results_postfix)


def main():
    import multiprocessing as mp
    # Set this at the beginning of your script
    mp.set_start_method('spawn')
    if len(sys.argv) != 8:
        print(
            "Usage: script.py <results_dir> <model_infra> <output_infra> <model_dir> <workload_config_file> <repetitions> <num_cores>")
        sys.exit(1)

    results_dir = sys.argv[1]
    model_infra = sys.argv[2]
    output_infra = sys.argv[3]
    model_dir = sys.argv[4]
    workload_config_file = sys.argv[5]
    repetitions = int(sys.argv[6])
    num_cores = int(sys.argv[7])

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    print("Fetching model locations")
    model_locations = get_model_locations(results_dir, model_infra, model_dir)
    print(f"Model locations: {model_locations}")
    print(
        f"Starting proactive simulation with infra-{output_infra} workload_config: {workload_config_file} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time,
         model_dir, model_infra)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as pool:
        pool.map(worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")
    print(f"Results saved under {os.path.join(results_dir, f'infra-{output_infra}', 'results-proactive')}")


if __name__ == '__main__':
    main()
