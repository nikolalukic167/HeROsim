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
from src.motivationalhetero.encoders import PLATFORM_TYPES
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder
from src.preprocessing import calculate_metrics_combined_by_platform_type
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





def proactive_worker_function(args):
    rep_idx, workload_config, config_idx, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra = args
    stats = execute_proactive(base_dir, output_infra, workload_config, sim_input_path, model_locations)
    results_postfix = f'hetero-proactive/{model_dir}-origin-{model_infra}-target-{output_infra}/{start_time}/{str(rep_idx)}/{config_idx}'
    save_single_stats(results_dir, workload_config, stats, output_infra, results_postfix, save_raw_results=False)


def main():

    output_dir = sys.argv[1]
    infra = sys.argv[2]
    workload_config_file = sys.argv[3]
    repetitions = int(sys.argv[4])
    num_cores = int(sys.argv[5])
    region = sys.argv[6]
    fn = sys.argv[7]
    model_folder = sys.argv[8]
    os.makedirs(output_dir, exist_ok=True)

    with open(workload_config_file, 'r') as fd:
        workload_configs = json.load(fd)

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")
    #
    # print(f'Loading infra {infra}, executing with {workload_config_file} using {num_cores} cores')
    # start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    #
    # # Create all combinations of repetitions and RPS values
    # work_items = [
    #     (rep_idx, config, config_idx, base_dir, infra, sim_input_path, output_dir, start_time)
    #     for rep_idx in range(repetitions)
    #     for config_idx, config in enumerate(workload_configs)
    # ]
    #
    # # Create a pool of workers and map the work items
    # start_ts = time.time()
    # reactive_pool = mp.Pool(num_cores)
    # try:
    #     result_folders = reactive_pool.map(reactive_worker_function, work_items)
    # finally:
    #     reactive_pool.close()
    #     reactive_pool.join()
    #
    #
    # result_files = [f'{x}/peak-config.json' for x in result_folders[:2]]
    # models, eval_results = train_model_reactive_then_proactive(result_files, include_queue_length=False)
    # model_paths = save_models(models, pathlib.Path(output_dir))
    # del models
    model_locations = get_model_locations_direct(pathlib.Path(output_dir) / model_folder)
    print(f"Model locations: {model_locations}")
    print(
        f"Starting proactive simulation with infra-{infra} workload_config: {workload_config_file} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, config, config_idx, base_dir, infra, sim_input_path, model_locations, f'{output_dir}/{model_folder}', start_time,
         f'{region}-{fn}', infra)
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
    print(f"Results saved under {os.path.join(output_dir, f'infra-{infra}', 'results-proactive', model_folder)}")

    end_ts = time.time()
    print(f'Duration: {end_ts - start_ts} seconds')

if __name__ == '__main__':
    main()
