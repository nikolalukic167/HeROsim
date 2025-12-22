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

from __future__ import annotations

from typing import Generator, Set, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import SystemState

from src.placement.scheduler import Scheduler


class KnativeScheduler(Scheduler):
    def placement(self, system_state: SystemState, task: Task) -> Generator:
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        replicas: Set[Tuple[Node, Platform]] = system_state.replicas[task.type["name"]]

        # DEBUG: dump current per-platform queue status including internal (prewarm) tasks
        try:
            for node, plat in replicas:
                q_len = len(plat.queue.items)
                if q_len > 0:
                    internal = sum(1 for t in plat.queue.items if getattr(t, 'is_internal', False))
                    print(f"[ {self.env.now} ] DEBUG: Platform {plat.id}@{node.node_name} q_len={q_len} internal={internal}")
        except Exception:
            pass

        # Least Connected
        bounded_concurrency = min(
            replicas, key=lambda couple: len(couple[1].queue.items)
        )

        print(f"task: {task.id}")
        print(f"bounded_concurrency: {bounded_concurrency}")

        return bounded_concurrency
