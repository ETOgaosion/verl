# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import torch

from .config import ProfilerConfig, TorchProfilerToolConfig
from .profile import DistProfiler


def get_dist_topology() -> dict:
    """Best-effort snapshot of the current process's distributed topology.

    Used to make per-process profiler trace files self-describing. The returned dict
    may contain ``rank``/``world_size`` (from ``torch.distributed``) and the
    ``tp``/``pp``/``dp``/``cp`` parallel ranks (from Megatron's ``parallel_state`` when
    initialized). Every lookup is guarded, so this never raises and simply omits the
    pieces that are unavailable (e.g. plain FSDP data parallelism only exposes rank).
    """
    info: dict = {}
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            info["rank"] = dist.get_rank()
            info["world_size"] = dist.get_world_size()
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            info["tp"] = mpu.get_tensor_model_parallel_rank()
            info["pp"] = mpu.get_pipeline_model_parallel_rank()
            info["dp"] = mpu.get_data_parallel_rank()
            try:
                info["cp"] = mpu.get_context_parallel_rank()
            except Exception:
                pass
    except Exception:
        pass

    return info


def _sanitize_name_part(text: str) -> str:
    """Make an arbitrary label safe to embed in a filename."""
    return re.sub(r"[^0-9A-Za-z.=+-]+", "-", str(text)).strip("-")


def build_trace_basename(
    rank: int,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    topology: Optional[dict] = None,
    profile_step: Optional[int] = None,
) -> str:
    """Build a descriptive, per-process trace filename stem.

    Encodes -- when available -- the worker role (``save_file_prefix``, e.g. ``actor``),
    the profiling scope role (``role``, e.g. ``train``), the RL step
    (``profile_step``), the global rank and world size, and the
    tensor/pipeline/data/context parallel ranks, followed by pid and a timestamp so that
    files written by different processes never collide.
    """
    topology = get_dist_topology() if topology is None else topology
    current_time = datetime.now(tz=timezone.utc).astimezone()
    timestamp = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    pid = os.getpid()

    parts: list[str] = []
    if save_file_prefix:
        parts.append(_sanitize_name_part(save_file_prefix))
    if role:
        parts.append(_sanitize_name_part(role))
    if profile_step is not None:
        parts.append(f"step{_sanitize_name_part(profile_step)}")

    global_rank = topology.get("rank", rank)
    world_size = topology.get("world_size")
    rank_part = f"rank{global_rank}"
    if world_size:
        rank_part += f"-of-{world_size}"
    parts.append(rank_part)

    parallel_part = "-".join(f"{dim}{topology[dim]}" for dim in ("tp", "pp", "dp", "cp") if dim in topology)
    if parallel_part:
        parts.append(parallel_part)

    parts.append(f"pid{pid}")
    parts.append(timestamp)
    return "_".join(parts)


def get_torch_profiler(
    contents: list[str],
    save_path: str,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    rank: int = 0,
    profile_step: Optional[int] = None,
):
    """Build a ``torch.profiler.profile`` instance.

    Args:
        contents: Selects the other ``torch.profiler.profile`` arguments -- ``cuda`` maps to
            ``activities``, ``shapes`` to ``record_shapes``, ``memory`` to ``profile_memory`` and
            ``stack`` to ``with_stack``. CPU activity is always on, since verl's per-stage
            ``record_function`` markers are CPU-side events.
        save_path: Directory to write chrome traces to.
        role: Optional logical scope name (e.g. ``train`` for a worker's whole-step window, or
            a stage name in discrete mode), embedded in the filename.
        save_file_prefix: Optional filename prefix, typically the worker role (``actor``/
            ``critic``/``ref``) so per-process traces are distinguishable.
        rank: Global rank, embedded in the trace filename (a fallback when
            ``torch.distributed`` is not initialized).
        profile_step: Optional RL step being profiled, embedded in the filename.
    """
    # All traces land directly in save_path: the role is already part of the filename, so an
    # extra directory level would only scatter one step's traces across sibling dirs and hide
    # them from finish_hook_cmd, which is handed save_path.
    os.makedirs(save_path, exist_ok=True)

    base_file_name = build_trace_basename(
        rank=rank, role=role, save_file_prefix=save_file_prefix, profile_step=profile_step
    )

    # One collection window writes one file, but keep an invocation counter so a second
    # flush on the same profiler cannot overwrite the first.
    handler_state = {"count": 0}

    def _trace_handler(prof):
        idx = handler_state["count"]
        handler_state["count"] += 1
        suffix = f"_part{idx}" if idx else ""
        out_path = os.path.join(save_path, f"{base_file_name}{suffix}.json.gz")
        print(f"[Profiler] Saving trace to {out_path}")
        prof.export_chrome_trace(out_path)

    contents = set(contents) if contents else set()
    # CPU activity is always collected, whatever `contents` selects: verl marks each stage with
    # record_function, and those markers -- like operator names -- are CPU-side events, so a
    # device-only trace would be bare kernels that cannot be attributed to any stage.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if not contents or "cuda" in contents:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # No torch.profiler.schedule: collection runs from start() to stop(), which is one RL step
    # (or a run of consecutive ones with global_profiler.profile_continuous_steps).
    return torch.profiler.profile(
        activities=activities,
        with_stack="stack" in contents,
        record_shapes="shapes" in contents,
        profile_memory="memory" in contents,
        on_trace_ready=_trace_handler,
    )


class Profiler(DistProfiler):
    """A PyTorch profiler wrapper class for collecting performance metrics.

    This profiler provides a convenient interface for profiling PyTorch operations,
    with support for:

    - CPU and CUDA activity profiling
    - One profiler step per training step, with stages and mini-batches annotated inside it
    - Multi-rank profiling support
    - Chrome trace export

    Args:
        config: Configuration object containing profiling parameters
    """

    _define_count = 0
    # Process-global handle to the currently running torch profiler. torch.profiler is
    # process-wide, so a step() issued by one Profiler instance (e.g. an inner
    # TrainingWorker) must advance the profiler that another instance started (e.g. the
    # outer ActorRolloutRefWorker).
    _active_prof = None
    # The instance that opened _active_prof. Colocated roles (actor and a reference model in
    # one process) each own a Profiler and are each told when a training step ends, but the
    # trace has a single timeline: only the owner may close a step in it, or one step boundary
    # per role would show up as several.
    _owner = None

    def __init__(
        self,
        rank,
        config: ProfilerConfig,
        tool_config: Optional[TorchProfilerToolConfig] = None,
        save_file_prefix=None,
    ):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        config = config or ProfilerConfig(ranks=[], enable=False)
        self.save_file_prefix = save_file_prefix

        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"

        self.prof = None
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self.contents = self.tool_config.contents
        self.save_path = self.config.save_path
        # Align with other profilers: read discrete mode, default to False for torch profiler
        self.discrete = getattr(self.tool_config, "discrete", False)
        # RL step of the profiled window, reported by the trainer on start().
        self._profile_step = None

    def check(self):
        return self.prof is not None

    def start(self, **kwargs):
        role = kwargs.get("role", None)
        # Recorded outside the discrete gate: discrete mode opens its profilers later, from
        # annotate(), and still needs to know which RL step it is collecting.
        profile_step = kwargs.get("profile_step", kwargs.get("global_step"))
        if not self.discrete and Profiler._define_count == 0:
            self._profile_step = profile_step
            self.prof = get_torch_profiler(
                contents=self.contents,
                save_path=self.save_path,
                role=role,
                save_file_prefix=self.save_file_prefix,
                rank=self.rank,
                profile_step=self._profile_step,
            )
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()
            Profiler._active_prof = self.prof
            Profiler._owner = self
            Profiler._define_count += 1
            return

        self._profile_step = profile_step

    def step(self):
        """End the current training step's window in the trace and open the next one.

        This is torch's ``ProfilerStep#<n>``, which verl advances once per training step -- the
        whole RL cycle, not the mini-batches it is made of. Collection is unaffected: without a
        ``torch.profiler.schedule`` every step is recorded, so this only labels the boundary.

        No-op when no torch profiler is currently running, or when this instance does not own
        the running one.
        """
        if Profiler._active_prof is not None and Profiler._owner in (None, self):
            Profiler._active_prof.step()

    def stop(self):
        if not self.discrete and Profiler._define_count == 1:
            # Close the last training step's window before tearing the profiler down.
            self.step()
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()
            Profiler._active_prof = None
            Profiler._owner = None
            Profiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with Profiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                # Prefer the stage label (`role`, e.g. "actor_compute_log_prob"), which names both
                # the role and the function it ran: in a colocated worker the method name alone
                # cannot say whether a log-prob forward belongs to the actor or the reference
                # model. Fall back to the method name for stages that declare no role.
                profile_name = message or role or func.__name__

                if not self.discrete:
                    # In continuous mode, we just record function, profiler started globally
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)

                # In discrete mode, we start/stop profiler around the function.
                # torch.profiler is process-global, so wrap the call in try/finally:
                # if func raises, we must still stop the profiler. Otherwise it leaks
                # and the next stage's prof.start() fails with "Profiler is already
                # enabled on this thread", plus the process aborts at teardown.
                prof = get_torch_profiler(
                    contents=self.contents,
                    save_path=self.save_path,
                    # Without an explicit role the stage is still identified by the wrapped
                    # function, which is what the reader needs to attribute the trace.
                    role=role or profile_name,
                    save_file_prefix=self.save_file_prefix,
                    rank=self.rank,
                    profile_step=self._profile_step,
                )
                prof.start()
                try:
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)
                finally:
                    prof.stop()

            return wrapper

        return decorator
