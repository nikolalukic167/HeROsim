from src.placement.model import (
    PriorityPolicy,
    SimulationPolicy,
    TimeSeries, SimulationStats,
)
from src.placement.model import scheduling_strategies
from src.placement.simulation import start_simulation


def execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, policy, queue_length, strategy,
                workload_trace,
                workload_trace_name, model_locations=None) -> SimulationStats:
    simulation_policy = SimulationPolicy(
        priority=PriorityPolicy(tasks=policy),
        scheduling=strategy,
        cache=cache_policy,
        keep_alive=keep_alive,
        queue_length=queue_length,
        short_name=scheduling_strategies[strategy],
    )

    # Read time series
    time_series: TimeSeries = TimeSeries.from_dict(workload_trace)
    # Run simulation
    stats = start_simulation(simulation_data, simulation_policy, infrastructure, time_series, workload_trace_name, model_locations)
    return stats
