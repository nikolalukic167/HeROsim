import json
import os
import sys
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from functools import partial

from src.executeinitial import load_simulation_inputs, setup_logging, flatten_workloads
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder


def save_stats(output_dir, rps, stats, infra, results_postfix):
    results_folder = os.path.join(output_dir, f"infra-{infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, f"peak-config.json"), "w") as outfile:
        json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def execute_reactive(base_dir, infra, workload_config, sim_input_path):
    sim_inputs = load_simulation_inputs(sim_input_path)
    with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
        infrastructure = json.load(fd)

    with open(workload_config, 'r') as fd:
        workload = json.load(fd)
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'kn_kn'
        simulation_data = SimulationData(
            platform_types=sim_inputs['platform_types'],
            storage_types=sim_inputs['storage_types'],
            qos_types=sim_inputs['qos_types'],
            application_types=sim_inputs['application_types'],
            task_types=sim_inputs['task_types'],
        )
        print(f'Start simulation: peak-config - {infra}')
        stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, task_priority,
                            queue_length,
                            scheduling_strategy, workload, 'workload-mine',reconcile_interval=1)
        print(f'End simulation: peak-config - {infra}')
        return stats


def worker_function(args):
    rep_idx, workload_config, base_dir, infra, sim_input_path, output_dir, start_time = args
    stats = execute_reactive(base_dir, infra, workload_config, sim_input_path)
    save_stats(output_dir, workload_config, stats, infra, f'reactive/{start_time}/{str(rep_idx)}')


def main():
    if len(sys.argv) != 6:
        print("Usage: script.py <output_dir> <infra> <peak_config> <repetitions> <num_cores>")
        sys.exit(1)

    output_dir = sys.argv[1]
    infra = sys.argv[2]
    workload_config_file = sys.argv[3]
    repetitions = int(sys.argv[4])
    num_cores = int(sys.argv[5])

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    print(f'Loading infra {infra}, executing with {workload_config_file} using {num_cores} cores')
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create all combinations of repetitions and RPS values
    work_items = [
        (rep_idx, config, base_dir, infra, sim_input_path, output_dir, start_time)
        for rep_idx in range(repetitions)
        for config in workload_configs
    ]

    # Create a pool of workers and map the work items
    with mp.Pool(num_cores) as pool:
        pool.map(worker_function, work_items)


if __name__ == '__main__':
    main()
