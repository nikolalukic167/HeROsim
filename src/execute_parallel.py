import pickle
import subprocess
import shutil
from pathlib import Path
import tempfile
import logging
import sys
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from src.executeinitial import load_simulation_inputs, setup_logging, prepare_simulation_config, prepare_workloads, \
    flatten_workloads


class SimulationRunner:
    def __init__(
            self,
            sim_inputs: dict,
            infrastructure_config: dict,
            workload: dict,
            output_dir: Path,
            max_parallel: int = 4
    ):
        self.sim_inputs = sim_inputs
        self.infrastructure_config = infrastructure_config
        self.workload = workload
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel

        # Setup logging
        self.logger = logging.getLogger('simulation_runner')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def run_single_simulation(self, config: dict, sim_id: str) -> dict:
        """Run a single simulation in a container."""
        # Create temporary directories for this simulation
        temp_dir = Path(tempfile.mkdtemp())
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        try:
            # Write configuration files
            for name, data in self.sim_inputs.items():
                with open(input_dir / f"{name}.json", 'w') as f:
                    json.dump(data, f)

            # Write simulation specific config
            with open(input_dir / "simulation_config.json", 'w') as f:
                json.dump(config, f)

            # Run container
            cmd = [
                "docker", "run",
                "--rm",  # Remove container after completion
                "-v", f"{input_dir}:/home/app/inputs:ro",
                "-v", f"{output_dir}:/home/app/outputs:rw",
                "ecdc/sim"
            ]

            # Execute simulation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Read results
            result_file = output_dir / "result.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    simulation_result = json.load(f)
            else:
                raise FileNotFoundError("Simulation did not produce results")

            # Copy results to final output directory
            shutil.copy2(result_file, self.output_dir / f"simulation_{sim_id}.json")

            return simulation_result

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    def run_simulations(self, configs: list) -> list:
        """Run multiple simulations in parallel."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all simulations
            future_to_id = {
                executor.submit(
                    self.run_single_simulation,
                    config,
                    str(uuid.uuid4())
                ): i
                for i, config in enumerate(configs)
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                sim_id = future_to_id[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Simulation {sim_id} completed successfully")
                except Exception as e:
                    self.logger.error(f"Simulation {sim_id} failed: {str(e)}")
                    results.append({"status": "failed", "error": str(e)})

        return results

def main():
    # Configuration paths
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples.npy"
    mapping_file = base_dir / "lhs_samples_mapping.pkl"
    config_file = base_dir / "infrastructure_config.json"
    workload_base_file = "data/ids/traces/workload-83-10.json"
    output_dir = base_dir / "results"
    # Setup logging
    logger = setup_logging(output_dir)
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting simulation preparation")

        # Load simulation inputs
        logger.info("Loading simulation input files")
        sim_inputs = load_simulation_inputs(sim_input_path)

        # Load samples and mapping
        logger.info("Loading samples and mapping")
        samples = np.load(samples_file)
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)

        # Load infrastructure config
        logger.info("Loading infrastructure configuration")
        with open(config_file, 'r') as f:
            infra_config = json.load(f)

        # Load workload base
        logger.info("Loading workload base")
        with open(workload_base_file, 'r') as f:
            workload_base = json.load(f)


        configs = []

        # Get list of applications from config
        apps = [app for app in infra_config['wsc'].keys()]

        # Process each sample
        for i, sample in enumerate(samples[:1]):
            logger.info(f"Processing sample {i + 1}/{len(samples)}")

            # Prepare infrastructure configuration
            sim_config = prepare_simulation_config(sample, mapping, infra_config)

            # Prepare workloads
            workloads = prepare_workloads(sample, mapping, workload_base, apps)
            # Flatten workloads into single sorted list
            flattened_workloads = flatten_workloads(workloads)

            # Combine infrastructure and workload configurations
            config = {
                "infrastructure": sim_config,
                "workload": flattened_workloads
            }
            configs.append(config)

        # Create runner and execute simulations
        runner = SimulationRunner(
            sim_inputs=sim_inputs,
            infrastructure_config=infra_config,
            workload=workload_base,
            output_dir=Path("simulation_results"),
            max_parallel=1
        )

        results = runner.run_simulations(configs)

        # Process results
        print(f"Completed {len(results)} simulations")
        failed_count = sum(1 for r in results if r.get("status") == "failed")
        print(f"Failed simulations: {failed_count}")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
