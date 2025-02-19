import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import xgboost as xgb

from src.bayesian import optimize_parameters
from src.executeinitial import prepare_simulation_config, prepare_workloads, flatten_workloads, execute_simulation
from src.preprocessing import create_inputs_outputs_seperated_per_app_windowed

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
                                    queue_length=queue_length)
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
        result = execute_simulation(full_config, sim_inputs, scheduling_strategy, models, cache_policy=cache_policy,
                                    task_priority=task_priority,
                                    keep_alive=keep_alive,
                                    queue_length=queue_length, models=models)
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


class ProactiveOptimizer:
    def __init__(self, initial_models: Dict[str, xgb.XGBRegressor], target_penalty=0.1, param_bounds_range_factor=0.25,
                 n_iterations=10):
        self.target_penalty = target_penalty
        # Track best penalties per task
        self.best_penalties = {task: float('inf') for task in initial_models.keys()}
        # Store optimization steps that led to improvements
        self.improvement_history = {task: [] for task in initial_models.keys()}
        # Determines the bounds of the parameters
        self.param_bounds_range_factor = param_bounds_range_factor
        self.best_proactive_penalty = float('inf')
        # Keep best models separately
        self.best_models = {task: model.copy() for task, model in initial_models.items()}
        self.improvement_history = {task: [] for task in initial_models.keys()}
        self.n_iterations = n_iterations

    def optimize_sample(self, initial_sample, state):
        param_bounds = self.get_bounds(initial_sample, state)
        # Create evaluation function with fixed state
        eval_func = partial(self.evaluate_parameters, state=state['sample'])
        result, iterations = optimize_parameters(
            evaluate_function=eval_func,
            param_bounds=param_bounds,
            target_penalty=self.target_penalty,
            n_iterations=self.n_iterations
        )
        return result, iterations

    def fine_tune_model(self, model: xgb.XGBRegressor, new_data: Tuple, task: str):
        """Fine-tune specific task model with new data."""
        X_new, y_new = new_data
        model.fit(
            X_new, y_new,
            xgb_model=model
        )
        return model

    def evaluate_parameters(self, state, **params):
        # Run reactive simulation
        reactive_result = run_reactive_simulation(state, params)
        app_definitions = {}
        for task in reactive_result['stats']['taskResults']:
            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
        # Prepare data for all tasks
        X_new, y_new = create_inputs_outputs_seperated_per_app_windowed(reactive_result['stats'], 5, app_definitions)

        # Fine-tune models and track improvements
        # Create temporary models for fine-tuning
        temp_models = {task: model.copy() for task, model in self.best_models.items()}

        # Fine-tune temporary models
        for task, model in temp_models.items():
            model.fit(X_new[task], y_new[task], xgb_model=model)

        # Run proactive simulation with fine-tuned models
        proactive_result = run_proactive_simulation(state, temp_models, params)

        # current_penalties = {}
        # for task, model in self.models.items():
        #     # Fine-tune model
        #     self.models[task] = self.fine_tune_model(
        #         model,
        #         X_new[task],
        #         y_new[task]
        #     )

        # Run proactive simulation with updated models
        proactive_penalty = proactive_result['stats']['penaltyProportion']

        # Check if this led to an improvement in proactive performance
        if proactive_penalty < self.best_proactive_penalty:
            self.best_proactive_penalty = proactive_penalty
            # Update best models
            self.best_models = temp_models
            # Store the improvement data for all tasks
            for task in self.best_models.keys():
                self.improvement_history[task].append(
                    OptimizationStep(
                        params=params.copy(),
                        X=X_new[task].copy(),
                        y={task: y_new[task]},
                        penalty=proactive_penalty,
                        improvement=True
                    )
                )

        return -proactive_penalty

    def get_bounds(self, initial_sample, state):
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
            param_bounds[param] = (param_down, param_up)
        return param_bounds

    def save_optimization_results(self, output_dir: Path):
        """Save models and their improvement datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for task, model in self.models.items():
            task_dir = output_dir / task
            task_dir.mkdir(exist_ok=True)

            # Save model
            model.save_model(task_dir / "model.json")

            # Save improvement datasets
            improvement_data = self.improvement_history[task]
            if improvement_data:
                # Combine all improvement steps
                X_combined = np.vstack([step.X for step in improvement_data])
                y_combined = np.array([step.y[task] for step in improvement_data])

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
