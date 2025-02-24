"""
Copyright 2024 b<>com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random

from typing import List

from src.placement.model import (
    ApplicationType,
    QoSType,
    SimulationData,
    TimeSeries,
    WorkloadEvent,
)


def poisson_process(lambd: int, duration_time: int) -> List[float]:
    arrivals: List[float] = []
    current_time = 0.0

    while current_time < duration_time:
        inter_arrival_time = random.expovariate(lambd)
        current_time += inter_arrival_time

        if current_time < duration_time:
            arrivals.append(current_time)

    return arrivals


def exponential_arrivals(mean_rate: float, duration: int) -> List[float]:
    arrivals = []
    current_time = 0.0
    while current_time < duration:
        inter_arrival = random.expovariate(1.0 / mean_rate)
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    return arrivals

def gamma_distribution(alpha: float, beta: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        # Using sum of exponentials approximation
        sample = sum(random.expovariate(beta) for _ in range(int(alpha)))
        samples.append(sample)
    return samples

def bimodal_normal(mu1: float, sigma1: float, mu2: float, sigma2: float,
                   weight: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        if random.random() < weight:
            samples.append(random.gauss(mu1, sigma1))
        else:
            samples.append(random.gauss(mu2, sigma2))
    return samples

def beta_distribution(alpha: float, beta: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        x = random.random()
        y = random.random()
        if x + y <= 1:
            samples.append(x / (x + y))
    return samples[:size]

def uniform_arrivals(min_rate: float, max_rate: float, duration: int) -> List[float]:
    arrivals = []
    current_time = 0.0
    while current_time < duration:
        rate = random.uniform(min_rate, max_rate)
        inter_arrival = 1.0 / rate
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    return arrivals


def generate_time_series(
        data: SimulationData, rps: int, duration_time: int
) -> TimeSeries:
    # Generate Poisson process arrivals
    arrivals = poisson_process(rps, duration_time)

    events: List[WorkloadEvent] = []
    qos_levels_per_app = {}
    for application_type in data.application_types.keys():
        qos_type_count: int = len(data.qos_types)
        qos_type_index: int = random.randint(0, qos_type_count - 1)
        qos_type_name: str = list(data.qos_types)[qos_type_index]
        qos_type: QoSType = data.qos_types[qos_type_name]
        qos_levels_per_app[str(application_type)] = qos_type

    for timestamp in arrivals:
        application_type_count: int = len(data.application_types)
        application_type_index: int = random.randint(0, application_type_count - 1)
        application_type_name: str = list(data.application_types)[
            application_type_index
        ]
        application_type: ApplicationType = data.application_types[
            application_type_name
        ]

        workload_event: WorkloadEvent = {
            "timestamp": timestamp,
            "application": application_type,
            "qos": qos_levels_per_app[application_type_name],
        }

        events.append(workload_event)

    time_series = TimeSeries(rps=rps, duration=duration_time, events=events)

    return time_series
