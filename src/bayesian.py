import time

import numpy as np


def optimize_parameters_pygopt(
        # def optimize_parameters(
        evaluate_function,
        param_bounds,
        target_penalty=0.1,  # Define acceptable penalty threshold
        n_iterations=50,
        init_points=5,
        random_state=42,
        constraints=None
):
    from GPyOpt.methods import BayesianOptimization

    if constraints is None:
        constraints = []
    # Convert param_bounds from dict to GPyOpt format
    domain = []
    for param_name, bound in param_bounds.items():
        domain.append({
            'name': param_name,
            'type': 'continuous',
            'domain': bound
        })

    # Wrapper for the objective function to match GPyOpt's expected format
    # GPyOpt expects function that takes 2D array and returns column vector
    def objective_wrapper(x):
        # Convert to the format expected by the original function if needed
        return np.array([[evaluate_function(x[0])]])

    # Create the GPyOpt optimizer
    bo = BayesianOptimization(
        f=objective_wrapper,
        domain=domain,
        constraints=constraints,
        initial_design_numdata=init_points,
        exact_feval=False,
        model_type='GP',
        acquisition_type='EI',
        normalize_Y=True,
        maximize=True,  # Set to True if your function should be maximized
        verbosity=False,
        random_seed=random_state
    )

    # Run the optimization with early stopping based on target penalty
    for i in range(n_iterations):
        bo.run_optimization(max_iter=1)

        # Check if we've reached our target
        current_best = -bo.fx_opt  # Negate if your original function was maximizing
        if current_best <= target_penalty:
            # Format the result to match the original function's output format
            param_values = bo.x_opt
            param_dict = {name: value for name, value in zip([d['name'] for d in domain], param_values)}

            result = {
                "target": -bo.fx_opt,  # Negate if your original function was maximizing
                "params": param_dict
            }
            return result, i + init_points + 1

    # If we reach here, we've completed all iterations without meeting the target
    param_values = bo.x_opt
    param_dict = {name: value for name, value in zip([d['name'] for d in domain], param_values)}

    result = {
        "target": -bo.fx_opt,  # Negate if your original function was maximizing
        "params": param_dict
    }
    return result, n_iterations + init_points


# def optimize_parameters_bayesian(
def optimize_parameters(
        evaluate_function,
        param_bounds,
        target_penalty=0.1,  # Define acceptable penalty threshold
        n_iterations=50,
        init_points=5,
        random_state=42
):
    from bayes_opt import BayesianOptimization

    optimizer = BayesianOptimization(
        f=evaluate_function,
        pbounds=param_bounds,
        random_state=random_state
    )

    # Custom optimization loop with early stopping
    interim_durations = []
    start_ts = time.time()
    for i in range(init_points):
        start_interim_ts = time.time()
        optimizer.maximize(init_points=1, n_iter=0)
        end_interim_ts = time.time()
        interim_durations.append(end_interim_ts - start_interim_ts)
        if -optimizer.max["target"] <= target_penalty:  # Convert back to penalty
            return optimizer.max, i + 1  # Return iterations needed

    for i in range(n_iterations):
        start_interim_ts = time.time()
        optimizer.maximize(init_points=0, n_iter=1)
        end_interim_ts = time.time()
        interim_durations.append(end_interim_ts - start_interim_ts)
        if -optimizer.max["target"] <= target_penalty:
            return optimizer.max, i + init_points + 1

    end_ts = time.time()
    print(f'{end_ts - start_ts} seconds')
    print(f'Average optimization time: {np.mean(interim_durations)}')

    return optimizer.max, n_iterations + init_points


# Example usage:
def example_usage():
    # Define parameter bounds
    param_bounds = {
        'network_bandwidth': (100, 1000),
        'cluster_size': (2, 10),
        'device_prop_rpi': (0.0, 1.0),
        'device_prop_xavier': (0.0, 1.0),
        'workload_app1': (1, 5),
        'workload_app2': (0, 3)
    }

    # Define evaluation function
    def evaluate_parameters(**params):
        # Ensure device proportions sum to 1
        total_prop = params['device_prop_rpi'] + params['device_prop_xavier']
        if not np.isclose(total_prop, 1.0, atol=0.01):
            return float('-inf')

        # Run simulation and return negative penalty (for maximization)
        penalty = run_proactive_simulation(params)
        return -penalty

    # Run optimization
    result = optimize_parameters(
        evaluate_function=evaluate_parameters,
        param_bounds=param_bounds,
        n_iterations=50
    )

    print("Best parameters:", result["params"])
    print("Best score:", -result["target"])  # Convert back to penalty
