import multiprocessing

import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import sys
from datetime import datetime

from joblib import Parallel, delayed

from src.optimizer import ProactiveParallelOptimizer


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('optimization')
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(output_dir / 'optimization.log')
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging(Path('/tmp'))

# Optimize each high-penalty sample
def process_sample(idx, sample, state, space, initial_penalties, high_penalty_indices, base_dir, output_dir, models_dir):
    logger.info(f"Optimizing sample {idx + 1}")
    logger.info(f"Original penalty: {initial_penalties[high_penalty_indices[idx]]}")

    try:
        # Load initial models
        logger.info("Loading initial models")
        initial_models = {}
        for model_file in models_dir.glob("*_model.json"):
            task_name = model_file.stem.replace("_model", "")
            model = xgb.XGBRegressor()
            model.load_model(str(model_file))
            initial_models[task_name] = model
        assert len(initial_models) > 0
        # Initialize optimizer
        optimizer = ProactiveParallelOptimizer(
            initial_models=initial_models,
            target_penalty=float(sys.argv[1]),  # Set your target penalty,
            n_iterations=int(sys.argv[2]),
            n_parallel=int(sys.argv[3])
        )


        best_params, iterations = optimizer.optimize_sample(sample, state, space)

        # The structure of best_params has changed in the new implementation
        optimization_result = {
            'sample_index': int(high_penalty_indices[idx]),
            'original_sample': sample.tolist(),
            'optimized_params': best_params,  # Now directly contains the parameters dictionary
            'original_penalty': float(initial_penalties[high_penalty_indices[idx]]),
            'final_penalty': float(optimizer.best_proactive_penalty),  # Get penalty from optimizer
            'iterations': iterations
        }

        logger.info(f"Optimization completed in {len(iterations)} iterations")
        logger.info(f"Final penalty: {optimizer.best_proactive_penalty}")

        # Save models and improvement datasets
        optimizer.save_optimization_results(output_dir / str(idx) / "models")

        return optimization_result

    except Exception as e:
        logger.error(f"Error optimizing sample {idx}: {str(e)}")
        raise e


# 1st argument: target penalty
# 2nd argument: n_iterations
# 3rd argument: n_parallel
# 4th argument: percentile to determine high violation samples
# 5th argument: how many samples should be looked at in parallel
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    # Configuration
    base_dir = Path("simulation_data")
    output_dir = base_dir / "optimization_simple_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger.info("Starting optimization process")

    try:
        # Load LHS samples and their results
        logger.info("Loading initial samples and results")
        samples = np.load(base_dir / "lhs_samples_simple.npy")

        # Load space
        logger.info("Loading space")
        space_file = base_dir / 'space_simple.json'
        with open(space_file, 'r') as fd:
            space = json.load(fd)

        # Load initial simulation results
        logger.info("Loading initial simulation results")
        simulation_results = []
        results_dir = base_dir / "initial_results_simple"
        for result_file in sorted(results_dir.glob("simulation_*.json")):
            with open(result_file, 'r') as f:
                simulation_results.append(json.load(f))


        # Calculate penalties for initial samples
        logger.info("Calculating initial penalties")
        initial_penalties = np.array([result['stats']['penaltyProportion'] for result in simulation_results])

        # Identify high-penalty samples
        penalty_threshold = np.percentile(initial_penalties, int(sys.argv[4]))  # Top 10% worst cases
        high_penalty_mask = initial_penalties > penalty_threshold
        # high_penalty_mask = [True]
        high_penalty_indices = np.where(high_penalty_mask)[0]
        high_penalty_results = np.array(simulation_results)[high_penalty_indices]
        high_penalty_samples = samples[high_penalty_indices]

        logger.info(f"Found {len(high_penalty_samples)} high-penalty samples to optimize")


        # Determine number of cores to use
        num_cores = multiprocessing.cpu_count()
        # You might want to use fewer cores to avoid overloading the system
        n_jobs = min(int(sys.argv[5]), num_cores - 1)
        models_dir = base_dir / "initial_results_simple"

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


if __name__ == "__main__":
    main()
