import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.executeinitial import load_simulation_inputs, setup_logging
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH, REACTIVE_RECONCILE_INTERVAL
from src.motivational.proactiveparalleldiffworkloads import execute_proactive, save_single_stats
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.train import train_model, train_model_reactive_then_proactive, save_models


def save_results(results_folder, stats):
    scenario_statistics = {
        "averageQueueTime": stats["averageQueueTime"],
        "penaltyProportion": stats["penaltyProportion"],
        "averageExecutionTime": stats["averageExecutionTime"],
        "averageComputerTime": stats["averageComputeTime"],
        "averageWaitTime": stats["averageWaitTime"],
        "endTime": stats["endTime"]
    }

    with open(os.path.join(results_folder, "results.json"), "w") as results_file:
        json.dump(scenario_statistics, results_file)

    list1, list2 = zip(*stats['penaltyDistributionOverTime'])
    penalty_over_time = pd.DataFrame({'time': list1, 'penalty': list2})
    system_events_df = pd.DataFrame(stats['systemEvents'])
    system_events_df.to_csv(os.path.join(results_folder, "system_events.csv"))
    penalty_over_time.to_csv(os.path.join(results_folder, "penalty_over_time.csv"))

    sns.scatterplot(x='time', y='penalty', data=penalty_over_time)
    plt.savefig(os.path.join(results_folder, "penalties.pdf"))
    plt.close()
    sns.scatterplot(x='timestamp', y='count', hue='name', data=system_events_df)
    plt.savefig(os.path.join(results_folder, "system_events.pdf"))
    plt.close()


def save_stats(output_dir, rps, stats, infra, results_postfix):
    results_folder = os.path.join(output_dir, f"infra-{infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    save_results(results_folder, stats)
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
                            scheduling_strategy, workload, 'workload-mine',
                            reconcile_interval=REACTIVE_RECONCILE_INTERVAL)
        print(f'End simulation: peak-config - {infra}')
        return stats


def reactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, infra, sim_input_path, output_dir, start_time = args
    stats = execute_reactive(base_dir, infra, workload_config, sim_input_path)
    return save_stats(output_dir, workload_config, stats, infra,
                      f'reactive/{start_time}/{str(rep_idx)}/{str(config_idx)}')


def proactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra = args
    stats = execute_proactive(base_dir, output_infra, workload_config, sim_input_path, model_locations)
    results_postfix = f'proactive/{model_dir}-origin-{model_infra}-target-{output_infra}/{start_time}/{str(rep_idx)}/{config_idx}'
    save_single_stats(results_dir, workload_config, stats, output_infra, results_postfix)


def main():
    if len(sys.argv) != 6:
        print("Usage: script.py <output_dir> <infra> <peak_config> <repetitions> <num_cores>")
        sys.exit(1)

    output_dir = sys.argv[1]
    infra = sys.argv[2]
    workload_config_file = sys.argv[3]
    repetitions = int(sys.argv[4])
    num_cores = int(sys.argv[5])
    region = sys.argv[6]
    fn = sys.argv[7]
    os.makedirs(output_dir, exist_ok=True)

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
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, output_dir, start_time)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    # Create a pool of workers and map the work items
    start_ts = time.time()
    with mp.Pool(num_cores) as pool:
        result_folders = pool.map(reactive_worker_function, work_items)

    result_files = [f'{x}/peak-config.json' for x in result_folders[:2]]
    models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False)
    model_paths = save_models(models, output_dir)

    model_locations = model_paths
    print(f"Model locations: {model_locations}")
    print(
        f"Starting proactive simulation with infra-{infra} workload_config: {workload_config_file} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, model_locations, output_dir, start_time,
         f'{region}-{fn}', infra)
        for rep_idx in range(repetitions)
        for config_idx, config in enumerate(workload_configs)
    ]

    start_ts = time.time()
    # Create a pool of workers and map the work items
    print(f"Start time: {start_ts}")
    with mp.Pool(num_cores) as pool:
        pool.map(proactive_worker_function, work_items)
    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds passed')
    print(f"Finished simulation")
    print(f"Results saved under {os.path.join(output_dir, f'infra-{infra}', 'results-proactive')}")

    end_ts = time.time()
    print(f'Duration: {end_ts - start_ts} seconds')

if __name__ == '__main__':
    main()
