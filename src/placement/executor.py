from src.placement.model import (
    PriorityPolicy,
    SimulationPolicy,
    TimeSeries, SimulationStats,
)
from src.placement.model import scheduling_strategies
from src.placement.simulation import start_simulation
from src.train import load_models


def execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, policy, queue_length, strategy,
                workload_trace,
                workload_trace_name, model_locations=None,models=None, reconcile_interval=5) -> SimulationStats:
    simulation_policy = SimulationPolicy(
        priority=PriorityPolicy(tasks=policy),
        scheduling=strategy,
        cache=cache_policy,
        keep_alive=keep_alive,
        queue_length=queue_length,
        short_name=scheduling_strategies[strategy],
        reconcile_interval=reconcile_interval,
        forced_placements=infrastructure.get('forced_placements') if isinstance(infrastructure, dict) else None,
        forced_placements_sequence=infrastructure.get('forced_placements_sequence') if isinstance(infrastructure, dict) else None
    )
    # if models is None and model_locations is not None:
    #    models = load_models(model_locations)
    # Read time series
    time_series: TimeSeries = TimeSeries.from_dict(workload_trace)
    print(f"Number of events in time series (tasks): {len(time_series.events)}")
    # Run simulation
    stats = start_simulation(simulation_data, simulation_policy, infrastructure, time_series, workload_trace_name, models)
    return stats
