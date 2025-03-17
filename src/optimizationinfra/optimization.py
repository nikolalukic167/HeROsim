import json
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.executeoptimization import setup_logging, process_sample, NumpyEncoder

logger = setup_logging(Path('/tmp'))

def main():
    base_dir = Path(sys.argv[6])

    exp_id = sys.argv[7]
    output_dir = base_dir / "optimization_results" / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger.info("Starting optimization process")

    try:
        # Load LHS samples and their results
        logger.info("Loading initial samples and results")
        samples = np.load(base_dir / "lhs_samples.npy")

        # Load space
        logger.info("Loading space")
        space_file = base_dir / 'space.json'
        with open(space_file, 'r') as fd:
            space = json.load(fd)

        # Load initial simulation results
        logger.info("Loading initial simulation results")
        simulation_results = []
        results_dir = base_dir / "initial_results"
        for result_file in sorted(results_dir.glob("simulation_*.json")):
            with open(result_file, 'r') as f:
                simulation_results.append(json.load(f))


        # Calculate penalties for initial samples
        logger.info("Calculating initial penalties")
        initial_penalties = np.array([result['stats']['penaltyProportion'] for result in simulation_results])

        # Identify high-penalty samples
        penalty_threshold = np.max(initial_penalties)  # Top 10% worst cases
        high_penalty_mask = initial_penalties == penalty_threshold
        high_penalty_indices = np.where(high_penalty_mask)[0]
        high_penalty_results = np.array(simulation_results)[high_penalty_indices]
        high_penalty_samples = samples[high_penalty_indices]

        logger.info(f"Found {len(high_penalty_samples)} high-penalty samples to optimize")


        # Determine number of cores to use
        num_cores = multiprocessing.cpu_count()
        # You might want to use fewer cores to avoid overloading the system
        n_jobs = min(int(sys.argv[5]), num_cores - 1)
        models_dir = base_dir / "initial_results"

        # Run the optimization in parallel
        optimization_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_sample)(
                idx, sample, state, space, initial_penalties, high_penalty_indices, base_dir, output_dir, models_dir
            ) for idx, (sample, state) in enumerate(zip(high_penalty_samples, high_penalty_results))
        )
        # Save optimization results
        logger.info("Saving optimization results")

        # Save summary
        summary = {
            'n_initial_samples': len(samples),
            'n_high_penalty_samples': len(high_penalty_samples),
            'penalty_threshold': float(penalty_threshold),
            'optimization_results': optimization_results
        }

        with open(output_dir / "optimization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)


        # Calculate improvement statistics
        improvements = [
            (r['original_penalty'] - r['final_penalty']) / r['original_penalty']
            for r in optimization_results
        ]

        logger.info("Optimization completed")
        logger.info(f"Average improvement: {np.mean(improvements) * 100:.2f}%")
        logger.info(f"Maximum improvement: {np.max(improvements) * 100:.2f}%")
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
