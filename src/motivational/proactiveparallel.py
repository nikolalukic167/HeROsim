import json
import os.path
import sys
from pathlib import Path
import multiprocessing as mp
from datetime import datetime

from src.executeinitial import load_simulation_inputs, setup_logging
from src.motivational.constants import QUEUE_LENGTH, KEEP_ALIVE
from src.motivational.proactive import get_model_locations
from src.motivational.reactive import save_stats
from src.placement.executor import execute_sim
from src.placement.model import SimulationData, DataclassJSONEncoder



def save_single_stats(results_dir, rps, stats, output_infra, results_postfix):
    results_folder = os.path.join(results_dir, f"infra-{output_infra}", f"results-{results_postfix}")
    os.makedirs(results_folder, exist_ok=True)
    with open(os.path.join(results_folder, f"{rps}.json"), "w") as outfile:
        json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)
    return results_folder


def execute_proactive(base_dir, infra, rps, sim_input_path, model_locations):
    sim_inputs = load_simulation_inputs(sim_input_path)
    with open(base_dir / f'motivational-infrastructures/{infra}.json', 'r') as fd:
        infrastructure = json.load(fd)

    with open(base_dir / f'traces/workload-{rps}-600.json', 'r') as fd:
        workload = json.load(fd)
        cache_policy = 'fifo'
        task_priority = 'fifo'
        keep_alive = KEEP_ALIVE
        queue_length = QUEUE_LENGTH
        scheduling_strategy = 'prokn_prokn'
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
                            scheduling_strategy, workload, 'workload-mine', model_locations=model_locations,
                            reconcile_interval=5)
        print(f'End simulation: {rps} - {infra}')
        return stats


def worker_function(args):
    rep_idx, rps, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra= args
    stats = execute_proactive(base_dir, output_infra, rps, sim_input_path, model_locations)
    results_postfix = f'proactive/origin-{model_infra}-target-{output_infra}-model-{model_dir}/{start_time}/{str(rep_idx)}'
    save_single_stats(results_dir, rps, stats, output_infra, results_postfix)


def main():
    if len(sys.argv) != 8:
        print("Usage: script.py <results_dir> <model_infra> <output_infra> <model_dir> <rpss> <repetitions> <num_cores>")
        sys.exit(1)

    results_dir = sys.argv[1]
    model_infra = sys.argv[2]
    output_infra = sys.argv[3]
    model_dir = sys.argv[4]
    rpss = sys.argv[5].split('-')
    repetitions = int(sys.argv[6])
    num_cores = int(sys.argv[7])

    # Limit number of cores to available cores
    num_cores = min(num_cores, mp.cpu_count())

    logger = setup_logging(Path("data/nofs-ids"))
    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")

    print("Fetching model locations")
    model_locations = get_model_locations(results_dir, model_infra, model_dir)
    print(f"Model locations: {model_locations}")
    print(f"Starting proactive simulation with infra-{output_infra} rpss: {rpss} using {num_cores} cores")

    start_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    # Create work items for each combination of repetition and RPS
    work_items = [
        (rep_idx, rps, base_dir, output_infra, sim_input_path, model_locations, results_dir, start_time, model_dir, model_infra)
        for rep_idx in range(repetitions)
        for rps in rpss
    ]

    # Create a pool of workers and map the work items
    with mp.Pool(num_cores) as pool:
        pool.map(worker_function, work_items)

    print(f"Finished simulation")
    print(f"Results saved under {os.path.join(results_dir, f'infra-{output_infra}', 'results-proactive')}")


if __name__ == '__main__':
    main()
