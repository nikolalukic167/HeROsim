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

import argparse
import glob
import json
import os
import sys
from datetime import datetime

from src.parser.parser import parse_simulation_data
from src.placement.executor import execute_sim
from src.placement.model import (
    Infrastructure,
    dir_path,
    restricted_float,
    positive_int, SimulationData, DataclassJSONEncoder,
)
from src.placement.model import priority_policies, scheduling_strategies, cache_policies


def main() -> int:
    # Parser
    parser = argparse.ArgumentParser(
        description=(
            "☁️ Tasks Scheduling on Heterogeneous Resources for Serverless Cloud"
            " Computing"
        )
    )
    parser.add_argument(
        "--clear", help="Clear log and result folders", action="store_true"
    )
    parser.add_argument(
        "-i",
        "--infrastructure",
        help="Infrastructure JSON filename",
        type=argparse.FileType("r"),
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        help="Simulation data JSON files directory",
        type=dir_path,
        required=True,
    )
    parser.add_argument(
        "-w",
        "--workload-trace",
        help="Workload trace JSON filename",
        type=argparse.FileType("r"),
        required=True,
    )
    parser.add_argument(
        "-s",
        "--scheduling-strategy",
        help="Select the scheduling strategy for task placement",
        choices=scheduling_strategies,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--task-priority-policy",
        help="Select the priority policy for task selection",
        choices=priority_policies["tasks"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cache-policy",
        help="Select the cache eviction policy for node storage",
        choices=cache_policies,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-k",
        "--keep-alive",
        help="Select the replica keep-alive duration",
        type=restricted_float,
        required=True,
    )
    parser.add_argument(
        "-q",
        "--queue-length",
        help="Select the queue length (# tasks) on baseline platform",
        type=positive_int,
        required=True,
    )
    args = parser.parse_args()

    # Check if output directories need to be created
    if not os.path.exists("log"):
        os.makedirs("log")
    if not os.path.exists("result"):
        os.makedirs("result")

    # Clear log and result folders if so flagged
    if args.clear:
        for log in glob.glob("log/*.log"):
            os.remove(log)
        for result in glob.glob("result/*.json"):
            os.remove(result)

    policy = args.task_priority_policy
    strategy = args.scheduling_strategy
    cache_policy = args.cache_policy
    keep_alive = args.keep_alive
    queue_length = args.queue_length
    workload_trace = args.workload_trace
    workload_trace_name = args.workload_trace.name
    args_infrastructure = args.infrastructure
    # Read infrastructure and policy
    with args_infrastructure as infile:
        infrastructure: Infrastructure = json.load(infile)

    simulation_data: SimulationData = parse_simulation_data(args.data_directory)

    with workload_trace as infile:
        workload = json.load(infile)
    simulation_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    stats = execute_sim(simulation_data, infrastructure, cache_policy, keep_alive, policy, queue_length, strategy,
                       workload, workload_trace_name)

    with open(os.path.join("result", f"{simulation_time}.json"), "w") as outfile:
        json.dump(stats, outfile, indent=2, cls=DataclassJSONEncoder)

    return os.EX_OK



if __name__ == "__main__":
    sys.exit(main())
