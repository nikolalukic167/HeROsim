import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import sys
from datetime import datetime

from src.optimizer import ProactiveOptimizer


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

def main():
    # Configuration
    base_dir = Path("simulation_data")
    output_dir = base_dir / "optimization_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("Starting optimization process")

    try:
        # Load LHS samples and their results
        logger.info("Loading initial samples and results")
        samples = np.load(base_dir / "lhs_samples.npy")
        # with open(base_dir / "lhs_samples_mapping.pkl", 'rb') as f:
        #     samples_mapping = pickle.load(f)

        # Load initial simulation results
        logger.info("Loading initial simulation results")
        simulation_results = []
        results_dir = base_dir / "initial_results"
        for result_file in sorted(results_dir.glob("simulation_*.json")):
            with open(result_file, 'r') as f:
                simulation_results.append(json.load(f))

        # Load initial models
        logger.info("Loading initial models")
        initial_models = {}
        models_dir = base_dir / "initial_results"
        for model_file in models_dir.glob("*_model.json"):
            task_name = model_file.stem.replace("_model", "")
            model = xgb.XGBRegressor()
            model.load_model(str(model_file))
            initial_models[task_name] = model

        # Calculate penalties for initial samples
        logger.info("Calculating initial penalties")
        initial_penalties = np.array([result['stats']['penaltyProportion'] for result in simulation_results])

        # Identify high-penalty samples
        penalty_threshold = np.percentile(initial_penalties, 0)  # Top 10% worst cases
        high_penalty_mask = initial_penalties > penalty_threshold
        high_penalty_mask = [True]
        high_penalty_samples = samples[:1]
        high_penalty_results = simulation_results[:1]
        high_penalty_indices = np.where(high_penalty_mask)[0]

        logger.info(f"Found {len(high_penalty_samples)} high-penalty samples to optimize")

        # Initialize optimizer
        optimizer = ProactiveOptimizer(
            initial_models=initial_models,
            target_penalty=0.1  # Set your target penalty
        )

        # Optimize each high-penalty sample
        optimization_results = []
        for idx, (sample, state) in enumerate(zip(high_penalty_samples, high_penalty_results)):
            logger.info(f"Optimizing sample {idx+1}/{len(high_penalty_samples)}")
            logger.info(f"Original penalty: {initial_penalties[high_penalty_indices[idx]]}")

            try:
                result, iterations = optimizer.optimize_sample(sample, state)

                optimization_results.append({
                    'sample_index': high_penalty_indices[idx],
                    'original_sample': sample.tolist(),
                    'optimized_params': result['params'],
                    'original_penalty': float(initial_penalties[high_penalty_indices[idx]]),
                    'final_penalty': float(-result['target']),
                    'iterations': iterations
                })

                logger.info(f"Optimization completed in {iterations} iterations")
                logger.info(f"Final penalty: {-result['target']}")

            except Exception as e:
                logger.error(f"Error optimizing sample {idx}: {str(e)}")
                raise e

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
            json.dump(summary, f, indent=2)

        # Save models and improvement datasets
        optimizer.save_optimization_results(output_dir / "models")

        # Calculate improvement statistics
        improvements = [
            (r['original_penalty'] - r['final_penalty']) / r['original_penalty']
            for r in optimization_results
        ]

        logger.info("Optimization completed")
        logger.info(f"Average improvement: {np.mean(improvements)*100:.2f}%")
        logger.info(f"Maximum improvement: {np.max(improvements)*100:.2f}%")
        logger.info(f"Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

# def create_initial_models(samples, simulation_results, tasks):
#     initial_models = {}
#     for task in tasks:
#         # Prepare data for the task
#         X = []
#         y = []
#         for sample, result in zip(samples, simulation_results):
#             X.append(sample)
#             y.append(result['penalty'])
#
#         # Convert to numpy arrays
#         X = np.array(X)
#         y = np.array(y)
#
#         # Split data into training and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#         # Train XGBoost model
#         model = xgb.XGBRegressor()
#         model.fit(X_train, y_train)
#
#         initial_models[task] = model
#
#     return initial_models


if __name__ == "__main__":
    main()
