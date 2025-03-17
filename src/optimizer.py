import json
import logging
import os
import pathlib
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import sklearn
import xgboost as xgb
from joblib import Parallel, delayed
from skopt import Optimizer
from skopt.callbacks import VerboseCallback
from skopt.space import Integer, Real

from src.bayesian import optimize_parameters
from src.executeinitial import prepare_simulation_config, prepare_workloads, flatten_workloads, execute_simulation
from src.motivational.constants import PROACTIVE_RECONCILE_INTERVAL, REACTIVE_RECONCILE_INTERVAL, \
    PREPARE_PREDICTION_WINDOW_SIZE
from src.preprocessing import create_inputs_outputs_seperated_per_app_windowed
from src.train import save_models

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStep:
    params: Dict[str, float]
    X: np.ndarray
    y: Dict[str, float]  # Task-specific targets
    penalty: float
    improvement: bool


def run_reactive_simulation(state, params):
    apps = state['apps']
    mapping = state['mapping']
    infra_config = state['infra_config']
    workload_base = state['workload_base']
    sim_inputs = state['sim_inputs']
    cache_policy = state['cache_policy']
    task_priority = state['task_priority']
    keep_alive = state['keep_alive']
    queue_length = state['queue_length']

    sample = [0] * len(params)
    for idx, param in mapping.items():
        sample[int(idx)] = params[param]

    sample = np.array(sample)

    # Prepare infrastructure configuration
    sim_config = prepare_simulation_config(sample, mapping, infra_config)

    # Prepare workloads
    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    # Flatten workloads into single sorted list
    flattened_workloads = flatten_workloads(workloads)

    # Combine infrastructure and workload configurations
    full_config = {
        "infrastructure": sim_config,
        "workload": flattened_workloads
    }

    try:
        scheduling_strategy = 'kn_kn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, cache_policy=cache_policy,
                                    task_priority=task_priority,
                                    keep_alive=keep_alive,
                                    queue_length=queue_length, reconcile_interval=REACTIVE_RECONCILE_INTERVAL)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        return result
    except Exception as e:
        logger.error(f"Error in simulation")
        logger.exception(e)


def run_proactive_simulation(state, models, params):
    apps = state['apps']
    mapping = state['mapping']
    infra_config = state['infra_config']
    workload_base = state['workload_base']
    sim_inputs = state['sim_inputs']
    cache_policy = state['cache_policy']
    task_priority = state['task_priority']
    keep_alive = state['keep_alive']
    queue_length = state['queue_length']

    sample = [0] * len(params)
    for idx, param in mapping.items():
        sample[int(idx)] = params[param]

    sample = np.array(sample)

    # Prepare infrastructure configuration
    sim_config = prepare_simulation_config(sample, mapping, infra_config)

    # Prepare workloads
    workloads = prepare_workloads(sample, mapping, workload_base, apps)
    # Flatten workloads into single sorted list
    flattened_workloads = flatten_workloads(workloads)

    # Combine infrastructure and workload configurations
    full_config = {
        "infrastructure": sim_config,
        "workload": flattened_workloads
    }

    try:
        scheduling_strategy = 'prokn_prokn'
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, cache_policy=cache_policy,
                                    task_priority=task_priority,
                                    keep_alive=keep_alive,
                                    queue_length=queue_length, models=models,
                                    reconcile_interval=PROACTIVE_RECONCILE_INTERVAL)
        result['sample'] = {
            'apps': apps,
            'sample': sample.tolist(),
            'mapping': mapping,
            'infra_config': infra_config,
            'workload_base': workload_base,
            'sim_inputs': sim_inputs,
            'scheduling_strategy': scheduling_strategy,
            'cache_policy': cache_policy,
            'task_priority': task_priority,
            'keep_alive': keep_alive,
            'queue_length': queue_length
        }

        return result
    except Exception as e:
        logger.error(f"Error in simulation")
        logger.exception(e)


class TabularPrintCallback:
    def __init__(self, param_names):
        self.param_names = param_names
        self.iteration = 0
        self.header_printed = False

    def __call__(self, res):
        # Print header on first call
        if not self.header_printed:
            # Create header row with parameter names (truncated if needed)
            param_headers = [name[:8] + "..." if len(name) > 8 else name for name in self.param_names]
            header = f"|   iter    |  target   | " + " | ".join(param_headers) + " |"

            # Print separator line
            separator = "=" * len(header)
            print(separator)
            print(header)
            print("-" * len(header))

            self.header_printed = True

        # Get best result so far
        best_idx = np.argmin(res.func_vals)
        best_value = res.func_vals[best_idx]
        best_x = res.x_iters[best_idx]

        # Format the row
        row = f"| {self.iteration:<9} | {-best_value:<9.2f} | " + " | ".join(f"{val:<9.3f}" for val in best_x) + " |"
        print(row)

        self.iteration += 1
        return False  # Continue optimization


def create_device_constraint_function(dimensions):
    """
    Creates a constraint function that ensures all parameters starting with 'device_' sum to 1

    Args:
        dimensions: List of dimension objects with names
    """
    # Create a mapping of parameter names to indices
    param_indices = {dim.name: i for i, dim in enumerate(dimensions)}

    # Filter for parameters that start with 'device_'
    device_params = [name for name in param_indices.keys() if name.startswith('device_')]

    def constraint(x):
        # If no device parameters, constraint is satisfied
        if not device_params:
            return True

        # Sum the values of the device parameters
        device_sum = sum(x[param_indices[param]] for param in device_params)

        # Check if sum equals 1 (with small tolerance for floating point precision)
        return abs(device_sum - 1.0) < 1e-6

    return constraint

class ProactiveParallelOptimizer:
    def __init__(self, initial_models: Dict[str, xgb.XGBRegressor], target_penalty=0.1,
                 param_bounds_range_factor=0.5, n_iterations=10, n_parallel=4):
        self.target_penalty = target_penalty
        # Track best penalties per task
        self.best_penalties = {task: float('inf') for task in initial_models.keys()}
        # Store optimization steps that led to improvements
        self.improvement_history = {task: [] for task in initial_models.keys()}
        # Determines the bounds of the parameters
        self.param_bounds_range_factor = param_bounds_range_factor
        self.best_proactive_penalty = float('inf')
        # Keep best models separately
        self.best_models = {task: deepcopy(model) for task, model in initial_models.items()}
        self.n_iterations = n_iterations
        self.n_parallel = n_parallel


    def optimize_sample(self, initial_sample, state, space):
        # Create the search space based on initial sample
        dimensions, param_names = self.create_space(initial_sample, state, space)


        # Create constraint function for device parameters
        constraint = create_device_constraint_function(dimensions)

        # Initialize the optimizer
        opt = Optimizer(
            dimensions=dimensions,
            base_estimator="GP",  # Gaussian Process
            acq_func="EI",  # Expected Improvement
            acq_optimizer="sampling",
            initial_point_generator="lhs",
            random_state=42
        )

        iterations = []
        # Print header for tabular output
        param_headers = [name[:8] + "..." if len(name) > 8 else name for name in param_names]
        header = f"|   iter    |  target   | " + " | ".join(param_headers) + " |"
        separator = "=" * len(header)
        print(separator)
        print(header)
        print("-" * len(header))
        max_i = self.n_iterations
        i = 0
        while i < max_i:
            # Ask for points to evaluate in parallel
            points = opt.ask(n_points=self.n_parallel)
            print(points)
            valid_points = []
            for point in points:
                device_params = [param for param in param_names if param.startswith('device_')]
                device_sum = sum(point[param_names.index(param)] for param in device_params)
                print(device_sum)
                if abs(device_sum - 1.0) < 1e-1:
                    valid_points.append(point)
            if len(valid_points) == 0:
                # Tell optimizer the results
                opt.tell(points, [-1e6 for _ in points])
                print(f'no valid points')
                continue

            # Evaluate points in parallel
            eval_results = Parallel(n_jobs=self.n_parallel)(
                delayed(self.evaluate_parameters_wrapper)(x, state['sample'], {task: deepcopy(model) for task, model in
                                                                               self.best_models.items()}, param_names)
                for x in points
            )

            # Extract penalties for optimizer
            penalties = [result['penalty'] for result in eval_results if result['penalty']]

            # Tell optimizer the results
            opt.tell(points, penalties)

            # Find the best result from this batch
            best_batch_result = None
            best_batch_penalty = float('inf')
            for result in eval_results:
                if 'proactive_penalty' in result and result['proactive_penalty'] < best_batch_penalty:
                    logger.info("Found best result in batch")
                    best_batch_penalty = result['proactive_penalty']
                    best_batch_result = result

            # Only update if the batch's best result is better than our overall best
            if best_batch_result is not None and best_batch_penalty < self.best_proactive_penalty:
                print("Best in batch is global best for now")
                self.best_proactive_penalty = best_batch_penalty


            for eval_result in eval_results:
                if all(key in eval_result for key in ['X_new', 'y_new', 'params']):
                    for task, model in self.best_models.items():
                        X = [[x] for x in eval_result['X_new'][task]]
                        y = np.array(eval_result['y_new'][task])
                        if len(X) == 0:
                            continue
                        model.fit(X, y, xgb_model=model)

                    for task in self.best_models.keys():
                        if eval_result['X_new'] is not None and task in eval_result['X_new']:
                            self.improvement_history[task].append(
                                OptimizationStep(
                                    params=eval_result['params'].copy(),
                                    X=eval_result['X_new'][task].copy(),
                                    y={task: eval_result['y_new'][task]},
                                    penalty=best_batch_penalty,
                                    improvement=True
                                )
                            )
            # Track iteration
            best_idx = np.argmin(opt.yi)
            best_value = opt.yi[best_idx]
            best_params = {param_names[j]: opt.Xi[best_idx][j] for j in range(len(param_names))}
            best_x = opt.Xi[best_idx]

            if -best_value != -1000000.00:
                iterations.append({
                    'iteration': i,
                    'best_penalty': -best_value,
                    'best_params': best_params
                })

                row = f"| {i:<9} | {-best_value:<9.2f} | " + " | ".join(f"{val:<9.3f}" for val in best_x) + " |"
                print(row)
                i = i + 1
            else:
                print("invalid config")


        # Return best parameters found
        best_idx = np.argmin(opt.yi)
        best_params = {param_names[j]: opt.Xi[best_idx][j] for j in range(len(param_names))}

        return best_params, iterations

    def create_space(self, initial_sample, state, space):
        """Convert parameter bounds to skopt space definition"""
        param_bounds = self.get_bounds(initial_sample, state, space)

        dimensions = []
        param_names = []

        for param_name, (lower, upper) in param_bounds.items():
            param_names.append(param_name)
            if param_name == 'cluster_size':
                dimensions.append(Integer(lower, upper, name=param_name))
            else:
                dimensions.append(Real(lower, upper, name=param_name))

        return dimensions, param_names

    def evaluate_parameters_wrapper(self, x, state, models, param_names):
        """Convert list of parameters to dictionary for evaluation"""
        params = {param_names[i]: x[i] for i in range(len(x))}
        return self.evaluate_parameters(state, models, **params)

    def get_bounds(self, initial_sample, state, space):
        mapping = state['sample']['mapping']

        param_bounds = {}
        for idx, param in mapping.items():
            param_value = initial_sample[int(idx)]
            # TODO decide whether we need to clamp/abort on specific values (i.e., out of "range")
            param_up = param_value + (param_value * self.param_bounds_range_factor)
            param_down = param_value - (param_value * self.param_bounds_range_factor)
            if param == 'cluster_size':
                param_down = int(param_down)
                param_up = int(param_up)
                cluste_max = space['csc']['max']
                cluster_min = space['csc']['min']

                if param_up > cluste_max:
                    param_up = cluste_max
                if param_down < cluster_min:
                    param_down = cluster_min
                param_up = int(cluste_max)
                param_down = int(cluster_min)

            if 'device' in param:
                device = param.split('_')[-1]
                device_max_proportion = space['pci'][device]['max']
                device_min_proportion = space['pci'][device]['min']
                if param_up > device_max_proportion:
                    param_up = device_max_proportion
                if param_down < device_min_proportion:
                    param_down = device_min_proportion

                param_up = device_max_proportion
                param_down = device_min_proportion
            if param == 'network_bandwidth':
                network_max_bandwidth = space['nwc']['max']
                network_min_bandwidth = space['nwc']['min']
                if param_up > network_max_bandwidth:
                    param_up = network_max_bandwidth
                if param_down < network_min_bandwidth:
                    param_down = network_min_bandwidth

                param_up = network_max_bandwidth
                param_down = network_min_bandwidth
            if 'workload' in param:
                app = param.split('_')[-1]
                workload_min = space['wsc'][app]['min']
                workload_max = space['wsc'][app]['max']
                if param_up > workload_max:
                    param_up = workload_max
                if param_down < workload_min:
                    param_down = workload_min

                param_up = workload_max
                param_down = workload_min

            if param != 'cluster_size' and param_up == param_down:
                param_up += 0.0000001

            if param == 'cluster_size' and param_up == param_down:
                param_up += 1

            param_bounds[param] = (param_down, param_up)
        print(param_bounds)
        return param_bounds

    def evaluate_parameters(self, state, models, **params):
        # We want to avoid parameters which sum is higher than 1
        total_prop = 0
        for k in params.keys():
            if k.startswith('device_'):
                total_prop += params[k]


        # Cluster size parameter must be discrete
        params['cluster_size'] = round(params['cluster_size'])

        # Create temporary models for fine-tuning
        temp_models = {task: deepcopy(model) for task, model in models.items()}

        if not np.isclose(total_prop, 1.0, atol=0.01):
            return {
                'penalty': 1e6,
                'proactive_penalty': 1e6,
                'params': params,
            }

        # reactive_result = None
        reactive_result = run_reactive_simulation(state, params)
        app_definitions = {}
        for task in reactive_result['stats']['taskResults']:
            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
        # Prepare data for all tasks
        X_new, y_new = create_inputs_outputs_seperated_per_app_windowed(reactive_result['stats'],
                                                                        PREPARE_PREDICTION_WINDOW_SIZE,
                                                                        app_definitions)

        # Fine-tune models and track improvements

        # Fine-tune temporary models
        for task, model in temp_models.items():
            X = [[x] for x in X_new[task]]
            y = np.array(y_new[task])
            if len(X) == 0:
                continue
            model.fit(X, y, xgb_model=model)

        # Run proactive simulation with fine-tuned models
        proactive_result = run_proactive_simulation(state, temp_models, params)

        # Run proactive simulation with updated models
        proactive_penalty = proactive_result['stats']['penaltyProportion']
        # Return all the data needed to track improvements
        return {
            'penalty': -proactive_penalty,
            'proactive_penalty': proactive_penalty,
            'params': params,
            'temp_models': temp_models,
            'X_new': X_new,
            'y_new': y_new
        }

    def fine_tune_model(self, model: xgb.XGBRegressor, new_data: Tuple, task: str):
        """Fine-tune specific task model with new data."""
        X_new, y_new = new_data
        model.fit(
            X_new, y_new,
            xgb_model=model
        )
        return model

    def save_optimization_results(self, output_dir: Path):
        """Save models and their improvement datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for task, model in self.best_models.items():
            task_dir = output_dir / task
            task_dir.mkdir(exist_ok=True)

            # Save model
            model.save_model(task_dir / "model.json")

            # Save improvement datasets
            improvement_data = self.improvement_history[task]
            if improvement_data:
                # Combine all improvement steps
                X_combined = np.vstack([np.array(step.X).reshape(-1, 1) for step in improvement_data])
                y_combined = np.vstack([step.y[task].reshape(-1, 1) for step in improvement_data])

                # Save datasets
                np.save(task_dir / "X_improvements.npy", X_combined)
                np.save(task_dir / "y_improvements.npy", y_combined)

                # Save metadata
                metadata = {
                    "n_improvements": len(improvement_data),
                    "final_penalty": self.best_penalties[task],
                    "improvement_steps": [
                        {
                            "params": step.params,
                            "penalty": step.penalty
                        }
                        for step in improvement_data
                    ]
                }

                with open(task_dir / "optimization_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)


def load_optimization_results(input_dir: Path) -> Dict[str, Any]:
    """Load models and their improvement datasets."""
    results = {}

    for task_dir in input_dir.iterdir():
        if task_dir.is_dir():
            task_name = task_dir.name
            results[task_name] = {
                "model": xgb.XGBRegressor(),
                "X_improvements": np.load(task_dir / "X_improvements.npy"),
                "y_improvements": np.load(task_dir / "y_improvements.npy")
            }
            results[task_name]["model"].load_model(task_dir / "model.json")

            with open(task_dir / "optimization_metadata.json", "r") as f:
                results[task_name]["metadata"] = json.load(f)

    return results


def read_and_finetune_opt_results(opt_path: pathlib.Path):
    with open(opt_path / 'optimization_summary.json', 'r') as fd:
        optimization_summary = json.load(fd)
        n_samples = len(optimization_summary['optimization_results'])
        X_improvements_by_task = defaultdict(list)
        y_improvements_by_task = defaultdict(list)
        for i in range(n_samples):
            input_dir = opt_path / str(i) / "models"
            for task_dir in input_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name
                    X_improvements_by_task[task_name].extend(np.load(task_dir / "X_improvements.npy"))
                    y_improvements_by_task[task_name].extend(np.load(task_dir / "y_improvements.npy").reshape(-1, 1))
    return X_improvements_by_task, y_improvements_by_task


def finetune_initial_models(models_path: pathlib.Path, opt_path: pathlib.Path):
    X_improvements_by_task, y_improvements_by_task = read_and_finetune_opt_results(opt_path)
    fine_tuned_models = {}

    for model_file in models_path.glob("*_model.json"):
        task_name = model_file.stem.replace("_model", "")
        model = xgb.XGBRegressor()
        model.load_model(str(model_file))
        X_new = X_improvements_by_task[task_name]
        y_new = y_improvements_by_task[task_name]
        model.fit(
            X_new, y_new,
            xgb_model=model
        )
        fine_tuned_models[task_name] = model
    return fine_tuned_models


def main():
    models_path = Path("simulation_data/initial_results_simple")
    opt_path = Path("simulation_data/optimization_simple_results/20250309_203222")
    fine_tuned_models = finetune_initial_models(models_path=models_path, opt_path=opt_path)
    path_fine_tuned_models = opt_path / "fine_tuned_models"
    os.makedirs(path_fine_tuned_models, exist_ok=True)
    save_models(fine_tuned_models, path_fine_tuned_models)


if __name__ == '__main__':
    main()
