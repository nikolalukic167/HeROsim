import json
from pathlib import Path

from src.executeinitial import execute_simulation, setup_logging


def main():
    # Configuration paths
    base_dir = Path("/home/app")
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    # Setup logging
    logger = setup_logging(output_dir)
    try:

        logger.info("Starting simulation preparation")

        sim_inputs = {}
        files_to_read = ['platform_types',
                         'storage_types',
                         'qos_types',
                         'application_types',
                         'task_types']
        for file_to_read in files_to_read:
            with open(input_dir / f'{file_to_read}.json', 'r') as fd:
                sim_inputs[file_to_read] = json.load(fd)
        with open(input_dir / 'simulation_config.json', 'r') as fd:
            full_config = json.load(fd)

        result = execute_simulation(full_config, sim_inputs)

        # Save result
        result_file = output_dir / f"result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info("Completed simulation")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
