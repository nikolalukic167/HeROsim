import json
import os
import sys
from datetime import datetime
from pathlib import Path

from src.executeinitial import load_simulation_inputs, setup_logging
from src.motivational.constants import KEEP_ALIVE, QUEUE_LENGTH
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder


def save_stats(output_dir, all_stats, infra, results_postfix):
    results_folder = os.path.join(output_dir, f"infra-{infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    for rps, stats in all_stats.items():
        with open(os.path.join(results_folder, f"{rps}.json"), "w") as outfile:
            json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def main():
    output_dir = sys.argv[1]
    infra = sys.argv[2]
    rpss = sys.argv[3].split('-')
    repetitions = int(sys.argv[4])
    logger = setup_logging(Path("data/nofs-ids"))
    # execute reactive with 30, 60, 90 and infra 1
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files

    print(f'Loading infra {infra}, executing with rpss {rpss}')
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for i in range(repetitions):
        all_stats = execute_reactive(base_dir, infra, rpss, sim_input_path)
        save_stats(output_dir, all_stats, infra, f'reactive/{start_time}/{str(i)}')


def execute_reactive(base_dir, infra, rpss, sim_input_path):
    all_stats = {}
    for rps in rpss:
        sim_inputs = load_simulation_inputs(sim_input_path)
        with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
            infrastructure = json.load(fd)

        with open(base_dir / f'traces/workload-{rps}-600.json', 'r') as fd:
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
            print(f'Start simulation: {rps} - {infra}')
            stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, task_priority,
                                queue_length,
                                scheduling_strategy, workload, 'workload-mine',reconcile_interval=1)
            print(f'End simulation: {rps} - {infra}')
            all_stats[str(rps)] = stats
    return all_stats


if __name__ == '__main__':
    main()
