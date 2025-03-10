import json
import os.path
import pathlib
import sys
from pathlib import Path

from src.executeinitial import load_simulation_inputs, setup_logging
from src.motivational.constants import QUEUE_LENGTH, KEEP_ALIVE
from src.motivational.reactive import save_stats
from src.placement.executor import execute_sim
from src.placement.model import SimulationData


def get_model_locations_direct(model_dir: pathlib.Path):
    model_locations = {}
    for file in model_dir.iterdir():
        if not '_model' in Path(file).stem:
            continue
        fn = Path(file).stem.replace('_model', '')
        model_locations[fn] = file
    return model_locations

def get_model_locations(input_dir, infra, model_dir):
    dir = Path(str(os.path.join(input_dir, f"infra-{infra}", 'models', model_dir)))
    model_locations = {}
    for file in dir.iterdir():
        fn = Path(file).stem.replace('_model', '')
        model_locations[fn] = file
    return model_locations

def main():
    logger = setup_logging(Path("data/nofs-ids"))
    results_dir = sys.argv[1]
    model_infra = sys.argv[2]
    output_infra = sys.argv[3]
    model_dir = sys.argv[4]
    rpss = sys.argv[5].split('-')

    base_dir = Path("data/nofs-ids")
    sim_input_path = Path("data/nofs-ids")  # Base path for simulation input files
    print("Fetching model locations")
    model_locations = get_model_locations(results_dir, model_infra, model_dir)
    print(f"Model locations: {model_locations}")
    print(f"Starting proactive simulation with infra-{output_infra} rpss: {rpss}")
    all_stats = execute_proactive(base_dir, output_infra, rpss, sim_input_path, model_locations)
    print(f"Finished simulation")
    print("Saving results...")
    # results_folder = save_stats(results_dir, all_stats, output_infra, f'proactive-{model_dir}-{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    results_folder = save_stats(results_dir, all_stats, output_infra,
                                f'proactive-origin-{model_infra}-target-{output_infra}-model-{model_dir}')
    print(f"Saved results under {results_folder}")


def execute_proactive(base_dir, infra, rpss, sim_input_path, model_locations):
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
            all_stats[rps] = stats
    return all_stats


if __name__ == '__main__':
    main()
