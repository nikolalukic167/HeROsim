from bayes_opt import BayesianOptimization
import numpy as np

def optimize_parameters(
        evaluate_function,
        param_bounds,
        target_penalty=0.1,  # Define acceptable penalty threshold
        n_iterations=50,
        init_points=5,
        random_state=42
):
    optimizer = BayesianOptimization(
        f=evaluate_function,
        pbounds=param_bounds,
        random_state=random_state
    )

    # Custom optimization loop with early stopping
    for i in range(init_points):
        optimizer.maximize(init_points=1, n_iter=0)
        if -optimizer.max["target"] <= target_penalty:  # Convert back to penalty
            return optimizer.max, i+1  # Return iterations needed

    for i in range(n_iterations):
        optimizer.maximize(init_points=0, n_iter=1)
        if -optimizer.max["target"] <= target_penalty:
            return optimizer.max, i+init_points+1

    return optimizer.max, n_iterations+init_points

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
