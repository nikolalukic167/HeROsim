import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.executeinitial import load_simulation_inputs, setup_logging
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH, REACTIVE_RECONCILE_INTERVAL
from src.motivational.proactive import get_model_locations_direct
from src.motivational.proactiveparalleldiffworkloads import execute_proactive, save_single_stats
from src.motivationalhetero.encoders import get_platform_type_encoder, PLATFORM_TYPES
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.preprocessing import create_inputs_outputs_seperated_per_app_windowed_per_device_type, \
    calculate_metrics_combined_by_platform_type
from src.train import train_model, train_model_reactive_then_proactive, save_models, \
    train_model_reactive_then_proactive_per_device_type


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
    melted_df = system_events_df.melt(id_vars=['timestamp'], value_vars=[x for  x in PLATFORM_TYPES if x in system_events_df.columns],
                        var_name='platform', value_name='pods_count')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted_df, x='timestamp', y='pods_count', hue='platform', marker='o')
    plt.title('Number of Pods per Platform Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Number of Pods')
    plt.legend(title='Platform')
    plt.savefig(os.path.join(results_folder, "system_events_by_platform.pdf"))
    plt.close()

    app_definitions = {}
    for task in stats['taskResults']:
        app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())

    metrics = calculate_metrics_combined_by_platform_type(stats['applicationResults'], stats['systemEvents'], window_size=5, application_to_task_map=app_definitions)
    rows = []
    for function, platforms in metrics.items():
        for platform, windows in platforms.items():
            for window, stats in windows.items():
                row = {
                    'function': function,
                    'platform': platform,
                    'window_start': stats.get('window_start'),
                    'window_end': stats.get('window_end'),
                    'avg_pods': stats.get('avg_pods'),
                    'avg_queue_length': stats.get('avg_queue_length'),
                    'avg_throughput': stats.get('avg_throughput'),
                    'penalty_rate': stats.get('penalty_rate'),
                    'total_requests': stats.get('total_requests')
                }
                rows.append(row)

    # Create DataFrame
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(results_folder, "performance_metrics_over_time.csv"))

    sns.lineplot(x='window_start', y='avg_throughput', hue='platform', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "throughput_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='penalty_rate', hue='platform', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "penalty_rate_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='total_requests', hue='platform', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "total_requests_by_platform.pdf"))
    plt.close()
    sns.lineplot(x='window_start', y='avg_queue_length', hue='platform', data=df_metrics)
    plt.savefig(os.path.join(results_folder, "queue_length_by_platform.pdf"))
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
                      f'hetero-reactive/{start_time}/{str(rep_idx)}/{str(config_idx)}')


def main():

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
    reactive_pool = mp.Pool(num_cores)
    try:
        result_folders = reactive_pool.map(reactive_worker_function, work_items)
    finally:
        reactive_pool.close()
        reactive_pool.join()

    test_size = 0.01

    encoder = get_platform_type_encoder()

    one_day_until = 172
    result_files = [f'{x}/peak-config.json' for x in result_folders[:1]]
    models = train_model_reactive_then_proactive_per_device_type(result_files, include_queue_length=False, test_size=test_size, until=one_day_until, encoder=encoder)
    dir_first_second = pathlib.Path(output_dir) / 'one_day'
    os.makedirs(dir_first_second, exist_ok=True)
    print('here')
    model_paths = save_models(models, dir_first_second)

    # first week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:1]]
    models, eval_results = train_model_reactive_then_proactive_per_device_type(result_files, include_queue_length=False, test_size=test_size, encoder=encoder)
    dir_first_second = pathlib.Path(output_dir) / 'first'
    os.makedirs(dir_first_second, exist_ok=True)
    model_paths = save_models(models, dir_first_second)

    # first & second week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:2]]
    models, eval_results = train_model_reactive_then_proactive_per_device_type(result_files, include_queue_length=False, test_size=test_size, encoder=encoder)
    dir_first_second = pathlib.Path(output_dir) / 'first_second'
    os.makedirs(dir_first_second, exist_ok=True)
    model_paths = save_models(models, dir_first_second)

    # first & second & third week training
    result_files = [f'{x}/peak-config.json' for x in result_folders[:3]]
    models, eval_results = train_model_reactive_then_proactive_per_device_type(result_files, include_queue_length=False, test_size=test_size, encoder=encoder)
    dir_all = pathlib.Path(output_dir) / 'first_second_third'
    os.makedirs(dir_all, exist_ok=True)
    model_paths = save_models(models, dir_all)

    for result_folder in result_folders:
        os.remove(f'{result_folder}/peak-config.json')

if __name__ == '__main__':
    main()
