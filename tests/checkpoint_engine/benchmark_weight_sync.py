# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Benchmark: filesystem weight sync vs verl checkpoint-engine RDMA backends.

Two weight-sync paths are measured on the same synthetic payload:

1. ``fs`` -- the mono_rl / alpha-seed style filesystem path: every trainer rank
   writes its own safetensors shard to a shared filesystem (CFS / HDFS-fuse /
   tmpfs) in parallel, then every rollout rank mmaps all shards and stages them
   onto GPU. No collective communication, no rendezvous.
2. ``nccl`` / ``nixl`` / ``mooncake`` -- the verl checkpoint-engine backends,
   driven through their public ``prepare -> build_topology -> init_process_group
   -> send_weights / receive_weights -> finalize`` lifecycle. NCCL does
   NVLink-relayed broadcast; NIXL (UCX/RDMA) and Mooncake do ring p2p RDMA.

Typical setup is 2 nodes x 4 GPUs (e.g. GB200 NVL7200-ish slice): 4 senders on
node 0, 4 receivers on node 1, so every payload byte actually crosses the
fabric exactly once on both paths.

Run on a Ray cluster (``ray start --head`` on node 0, ``ray start --address=...``
on node 1) from the repo root::

    python tests/checkpoint_engine/benchmark_weight_sync.py \
        --backends nccl nixl mooncake fs \
        --total-gb 8 --iters 5 \
        --fs-dir /mnt/hdfs/ckpt/weight_sync_bench

Or via pytest (opt-in, needs 8 GPUs)::

    BENCH_WEIGHT_SYNC=1 pytest tests/checkpoint_engine/benchmark_weight_sync.py -s

The FS round-trip helpers also have a CPU-only correctness test that runs
without GPUs or Ray.
"""
import argparse
import dataclasses
import os
import shutil
import sys
import tempfile
import time
from typing import Optional

import numpy as np
import torch

try:
    import pytest
except ImportError:
    # Run as a plain script (e.g. inside the uv venv, which carries no pytest).
    # The test decorators below are evaluated at import time, so pytest has to
    # resolve to something; this is just enough of it to keep the module usable.
    class _PytestShim:
        class mark:
            @staticmethod
            def skipif(*_args, **_kwargs):
                return lambda fn: fn

        @staticmethod
        def skip(reason):
            raise RuntimeError(f"skipped: {reason}")

    pytest = _PytestShim()

GIB = 1024**3

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "uint8": torch.uint8,
}


@dataclasses.dataclass
class WeightSpec:
    """A synthetic weight tensor: name, shape and dtype fully determine it."""

    name: str
    shape: tuple
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize


def build_weight_specs(total_bytes: int, num_tensors: int, dtype: torch.dtype, prefix: str = "model") -> list[WeightSpec]:
    """Split ``total_bytes`` into ``num_tensors`` equally sized 2D weights."""
    bytes_per_tensor = total_bytes // num_tensors
    numel_per_tensor = max(1, bytes_per_tensor // dtype.itemsize)
    cols = 4096
    rows = max(1, numel_per_tensor // cols)
    specs = []
    for i in range(num_tensors):
        specs.append(WeightSpec(name=f"{prefix}.layer_{i}.weight", shape=(rows, cols), dtype=dtype))
    return specs


def fill_pattern(tensor: torch.Tensor, tensor_idx: int) -> torch.Tensor:
    """Deterministic in-place fill with small integers, exact in bf16/fp16.

    Value at flat position p of tensor i is ``(p + i * 104729) % 97``. Small
    integers (< 128) are exactly representable in every floating dtype we use,
    so sender and receiver checksums agree regardless of accumulation order.
    """
    n = tensor.numel()
    pos = torch.arange(n, device=tensor.device, dtype=torch.int64)
    values = ((pos + tensor_idx * 104729) % 97).to(tensor.dtype)
    tensor.view(-1).copy_(values)
    return tensor


def spot_check_positions(numel: int) -> list[int]:
    """Cheap, deterministic set of positions to verify end to end."""
    if numel <= 8:
        return list(range(numel))
    return [0, 1, numel // 4, numel // 2, (3 * numel) // 4, numel - 2, numel - 1]


def expected_spot_values(spec: WeightSpec, tensor_idx: int) -> list[int]:
    positions = spot_check_positions(int(np.prod(spec.shape)))
    return [(p + tensor_idx * 104729) % 97 for p in positions]


def compute_spot_checksum(tensor: torch.Tensor, tensor_idx: int) -> tuple[str, list[int]]:
    positions = spot_check_positions(tensor.numel())
    flat = tensor.view(-1)
    vals = [int(flat[p].to(torch.float32).item()) for p in positions]
    return tensor_idx, vals


def iter_weight_tensors(specs: list[WeightSpec], device: str):
    """Generator yielding ``(name, tensor)`` matching the engines' expected input."""
    for i, spec in enumerate(specs):
        t = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        fill_pattern(t, i)
        yield spec.name, t
        del t


# ---------------------------------------------------------------------------
# Filesystem weight sync (safetensors shards; mirrors mono_rl/alpha-seed path)
# ---------------------------------------------------------------------------


def _step_dir(fs_dir: str, step: int) -> str:
    return os.path.join(fs_dir, f"step_{step}")


def _shard_file(step_dir: str, shard_rank: int, num_shards: int) -> str:
    return os.path.join(step_dir, f"model_{shard_rank:05d}-{num_shards:05d}.safetensors")


def _ready_marker(step_dir: str, shard_rank: int) -> str:
    return os.path.join(step_dir, f"shard_{shard_rank:05d}.ready")


def fs_write_shard(fs_dir: str, step: int, shard_rank: int, num_shards: int, named_tensors: dict) -> str:
    """Write one safetensors shard atomically: tmp file -> rename -> ready marker.

    ``named_tensors`` must be CPU tensors (the caller stages GPU tensors into
    pinned host memory first, same as the mono_rl actor writer).

    Serialize to bytes and write them sequentially rather than calling
    ``save_file``: ``save_file`` memory-maps its output, and an HDFS-fuse mount
    does not implement that, so it fails outright with
    ``I/O error: Function not implemented (os error 38)``. Production does not
    hit this because mono_rl streams the shard out
    (``save_sharded_safetensors_streaming``), so a sequential write is both the
    thing that works and the faithful model of it.
    """
    from safetensors.torch import save

    step_dir = _step_dir(fs_dir, step)
    os.makedirs(step_dir, exist_ok=True)
    final_path = _shard_file(step_dir, shard_rank, num_shards)
    tmp_path = final_path + f".tmp{os.getpid()}"
    blob = save(named_tensors, metadata={"shard_rank": str(shard_rank)})
    with open(tmp_path, "wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    del blob
    os.replace(tmp_path, final_path)
    marker = _ready_marker(step_dir, shard_rank)
    with open(marker, "w") as f:
        f.write(str(time.time()))
        f.flush()
        os.fsync(f.fileno())
    return final_path


def fs_wait_ready(fs_dir: str, step: int, num_shards: int, timeout: float = 600.0, poll: float = 0.2) -> None:
    """Block until every shard's ready marker is visible on this node.

    Discover the markers by listing the directory, NOT by stat-ing their known
    paths. On an HDFS-fuse mount a stat of a not-yet-existing file caches the
    negative result for the mount's attribute TTL (30s by default), so polling
    with os.path.exists reports the step as invisible for a flat 30s no matter
    how fast the write actually landed -- an artifact of the probe, not a
    property of the filesystem. readdir refreshes the directory entry cache, and
    it is also what production does: alpha-seed's
    ``latest_sampler_weights_step_dir`` scans for step dirs rather than probing
    one path.
    """
    step_dir = _step_dir(fs_dir, step)
    wanted = {os.path.basename(_ready_marker(step_dir, r)) for r in range(num_shards)}
    deadline = time.time() + timeout
    seen = set()
    while time.time() < deadline:
        try:
            seen = wanted & set(os.listdir(step_dir))
        except FileNotFoundError:
            # The step dir itself is not visible yet; listing the parent
            # refreshes its entry too.
            try:
                os.listdir(os.path.dirname(step_dir))
            except OSError:
                pass
            seen = set()
        if seen == wanted:
            return
        time.sleep(poll)
    raise TimeoutError(f"markers {sorted(wanted - seen)} not visible under {step_dir} "
                       f"within {timeout}s")


def fs_read_shards(
    fs_dir: str,
    step: int,
    num_shards: int,
    stage_device: Optional[str] = None,
) -> dict:
    """Read all shards of a step via mmap (safetensors) and optionally stage to GPU.

    Returns ``{name: tensor}`` on ``stage_device`` when given, else CPU mmap views.
    """
    from safetensors import safe_open

    step_dir = _step_dir(fs_dir, step)
    out = {}
    for shard_rank in range(num_shards):
        path = _shard_file(step_dir, shard_rank, num_shards)
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                shard = {name: f.get_tensor(name) for name in f.keys()}
        except Exception:
            # Same story as the write side: mmap is not available on every
            # shared mount. Fall back to a sequential read so the path is
            # measurable at all; the phase is still "get all bytes into host
            # memory", just without demand paging.
            from safetensors.torch import load

            with open(path, "rb") as fh:
                shard = load(fh.read())
        for name, cpu_t in shard.items():
            if stage_device is not None and stage_device != "cpu":
                gpu_t = torch.empty(cpu_t.shape, dtype=cpu_t.dtype, device=stage_device)
                gpu_t.copy_(cpu_t.to(stage_device, non_blocking=True))
                out[name] = gpu_t
            else:
                out[name] = cpu_t
    if stage_device is not None and stage_device != "cpu":
        torch.cuda.synchronize()
    return out


def fs_cleanup(fs_dir: str, step: int) -> None:
    shutil.rmtree(_step_dir(fs_dir, step), ignore_errors=True)


# ---------------------------------------------------------------------------
# Ray worker: one per GPU, drives one engine backend or the FS path
# ---------------------------------------------------------------------------

try:
    import ray

    @ray.remote(num_gpus=1)
    class WeightSyncWorker:
        """Runs inside a Ray actor with exactly one GPU assigned."""

        def __init__(self, global_rank: int):
            self.global_rank = global_rank
            self.device = "cuda"
            torch.cuda.set_device(0)
            self.engine = None
            self.backend = None
            self.bucket_size = None
            self.gpu_buffers = {}

        def info(self) -> dict:
            import socket
            import importlib

            deps = {}
            for m in ("torch", "ray", "cupy", "zmq", "safetensors", "tensordict", "nixl", "mooncake"):
                try:
                    mod = importlib.import_module(m)
                    deps[m] = getattr(mod, "__version__", "ok")
                except Exception as e:
                    deps[m] = f"MISS({type(e).__name__})"
            return {
                "global_rank": self.global_rank,
                "node_id": ray.get_runtime_context().get_node_id(),
                "ip": socket.gethostbyname(socket.gethostname()),
                "gpu": torch.cuda.get_device_name(0),
                "python": sys.executable,
                "deps": deps,
            }

        # -- RDMA checkpoint engines ---------------------------------------

        def engine_create(self, backend: str, bucket_bytes: int) -> None:
            from verl.checkpoint_engine import CheckpointEngineRegistry

            os.environ["RANK"] = str(self.global_rank)
            self.backend = backend
            self.bucket_size = bucket_bytes
            engine_kwargs = {"nccl": {"multi_sender": True}}.get(backend, {})
            if backend in ("nixl", "mooncake"):
                engine_kwargs["device"] = "cuda"
            if backend in ("nccl", "delta_sharded"):
                # One collective group per backend. These two share
                # NCCLCheckpointEngine.init_process_group, which only assigns
                # self.rank on the branch that actually creates the group; with a
                # shared name the second engine in this process takes the "already
                # initialized" branch and asserts on an attribute it never set
                # ('DeltaShardedCheckpointEngine' object has no attribute 'rank').
                engine_kwargs["group_name"] = f"ws_bench_{backend}"
                engine_kwargs["rebuild_group"] = False
            self.engine = CheckpointEngineRegistry.new(
                backend, bucket_size=bucket_bytes, is_master=(self.global_rank == 0), **engine_kwargs
            )

        def engine_prepare(self):
            return self.engine.prepare()

        def engine_init(self, **kwargs) -> None:
            self.engine.init_process_group(**kwargs)

        async def engine_send(self, specs_serialized: list[dict]) -> dict:
            specs = [WeightSpec(s["name"], tuple(s["shape"]), _DTYPE_MAP[s["dtype"]]) for s in specs_serialized]

            def gen():
                for name, t in iter_weight_tensors(specs, self.device):
                    yield name, t

            total_bytes = sum(s.nbytes for s in specs)
            start = time.time()
            await self.engine.send_weights(gen())
            torch.cuda.synchronize()
            return {"rank": self.global_rank, "role": "send", "seconds": time.time() - start, "bytes": total_bytes}

        async def engine_recv(self, specs_serialized: list[dict]) -> dict:
            name_to_idx = {s["name"]: i for i, s in enumerate(specs_serialized)}
            total_bytes = 0
            checksums = {}
            start = time.time()
            async for name, tensor in self.engine.receive_weights():
                # Buckets are double-buffered and reused, so materialize a
                # private GPU copy before the next bucket lands (same as the
                # real server adapter's weight.clone()).
                owned = tensor.clone()
                _, vals = compute_spot_checksum(owned, name_to_idx[name])
                checksums[name] = vals
                total_bytes += owned.numel() * owned.element_size()
                del owned
            torch.cuda.synchronize()
            return {
                "rank": self.global_rank,
                "role": "recv",
                "seconds": time.time() - start,
                "bytes": total_bytes,
                "checksums": checksums,
                "num_weights": len(checksums),
            }

        def engine_finalize(self) -> None:
            if self.engine is not None:
                try:
                    self.engine.finalize()
                except Exception as e:
                    print(f"[rank {self.global_rank}] finalize warning: {type(e).__name__}: {str(e)[:200]}")
                self.engine = None
            torch.cuda.empty_cache()

        # -- Filesystem path ------------------------------------------------

        def fs_write(
            self, fs_dir: str, step: int, shard_rank: int, num_shards: int, specs_serialized: list[dict]
        ) -> dict:
            named_cpu = {}
            total_bytes = 0
            stage_start = time.time()
            for i, s in enumerate(specs_serialized):
                if i % num_shards != shard_rank:
                    continue
                spec = WeightSpec(s["name"], tuple(s["shape"]), _DTYPE_MAP[s["dtype"]])
                gpu_t = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
                fill_pattern(gpu_t, i)
                cpu_t = torch.empty(spec.shape, dtype=spec.dtype, device="cpu", pin_memory=True)
                cpu_t.copy_(gpu_t, non_blocking=True)
                named_cpu[spec.name] = cpu_t
                total_bytes += spec.nbytes
                del gpu_t
            torch.cuda.synchronize()
            stage_seconds = time.time() - stage_start

            start = time.time()
            fs_write_shard(fs_dir, step, shard_rank, num_shards, named_cpu)
            return {
                "rank": self.global_rank,
                "shard": shard_rank,
                "seconds": time.time() - start,
                "stage_seconds": stage_seconds,
                "bytes": total_bytes,
            }

        def fs_read(self, fs_dir: str, step: int, num_shards: int, specs_serialized: list[dict]) -> dict:
            wait_start = time.time()
            fs_wait_ready(fs_dir, step, num_shards)
            wait_seconds = time.time() - wait_start
            start = time.time()
            weights = fs_read_shards(fs_dir, step, num_shards, stage_device=self.device)
            read_seconds = time.time() - start
            checksums = {}
            for i, s in enumerate(specs_serialized):
                if s["name"] in weights:
                    _, vals = compute_spot_checksum(weights[s["name"]], i)
                    checksums[s["name"]] = vals
            return {
                "rank": self.global_rank,
                "role": "fs_read",
                "wait_seconds": wait_seconds,
                "seconds": read_seconds,
                "bytes": sum(w.numel() * w.element_size() for w in weights.values()),
                "checksums": checksums,
                "num_weights": len(checksums),
            }

except ImportError:
    ray = None
    WeightSyncWorker = None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _specs_to_json(specs: list[WeightSpec]) -> list[dict]:
    dtype_inv = {v: k for k, v in _DTYPE_MAP.items()}
    return [{"name": s.name, "shape": list(s.shape), "dtype": dtype_inv[s.dtype]} for s in specs]


def _expected_checksums(specs: list[WeightSpec]) -> dict:
    return {s.name: expected_spot_values(s, i) for i, s in enumerate(specs)}


def _gib_per_sec(num_bytes: float, seconds: float) -> float:
    return num_bytes / GIB / max(seconds, 1e-9)


def run_benchmark(
    backends: list[str],
    total_gb: float = 4.0,
    num_tensors: int = 64,
    iters: int = 3,
    warmup: int = 1,
    bucket_mb: int = 2048,
    fs_dir: Optional[str] = None,
    fs_read_dir: Optional[str] = None,
    num_senders: int = 4,
    num_receivers: int = 4,
    dtype: str = "bf16",
    allow_single_node: bool = False,
    json_out: Optional[str] = None,
) -> dict:
    """Run the comparison on a Ray cluster; returns a structured results dict."""
    if ray is None:
        raise RuntimeError("ray is not installed")

    # Ray on Arnold / managed Ray already injects a Job-level runtime_env
    # (working_dir, git_private_key, env_vars such as PYTHONPATH/PATH/LD_LIBRARY_PATH).
    # Passing the same env keys from the driver raises a merge-conflict ValueError.
    # RAY_OVERRIDE_JOB_RUNTIME_ENV=1 tells Ray to merge Job + Driver runtime envs.
    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")

    # Torch-free bootstrap module: ships a worker_process_setup_hook (fixes
    # sys.path so the image's user-site torch is visible in raylet-spawned
    # worker processes) plus probe()/ensure_deps() remote tasks that install
    # cupy/nixl/mooncake ON EACH node (the entrypoint only runs on head).
    setup_mod = None
    setup_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ws_setup.py")
    setup_pymod_dir = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _ws_setup as setup_mod

        # CRITICAL: Ray deserializes worker_process_setup_hook at worker-process
        # bootstrap, BEFORE py_modules paths are usable. A by-reference hook
        # requires `import _ws_setup` at that moment -> ModuleNotFoundError ->
        # the worker dies. Force pickle-by-value so the whole (torch-free,
        # stdlib-only) module body is embedded in the hook payload itself.
        try:
            import cloudpickle as _cp

            _cp.register_pickle_by_value(setup_mod)
        except Exception as _e:
            print(f"[setup] note: standalone cloudpickle register skipped: {_e}")
        try:
            import ray.cloudpickle as _ray_cp

            _ray_cp.register_pickle_by_value(setup_mod)
        except Exception:
            pass

        # Ray py_modules only accepts a directory or a .whl (NOT a bare .py);
        # ship the module dir too as a belt-and-braces second path.
        import shutil
        import tempfile

        setup_pymod_dir = tempfile.mkdtemp(prefix="ws_setup_pkg_")
        shutil.copy(setup_py, os.path.join(setup_pymod_dir, "_ws_setup.py"))
        print(f"[setup] _ws_setup loaded from {setup_py}; py_modules dir={setup_pymod_dir}; "
              "hook registered pickle-by-value")
    except Exception as e:
        print(f"[setup] WARNING: _ws_setup not importable ({type(e).__name__}: {e}); "
              "worker processes may fail to find torch/cupy")

    # Only forward vars that the managed Job env does not already set; the base
    # PYTHONPATH/PATH/LD_LIBRARY_PATH are inherited cluster-wide and already include
    # the verl working dir, so re-specifying them would either conflict or duplicate.
    env_vars = {
        "UCX_TLS": os.environ.get("UCX_TLS", "rc,tcp,cuda"),
        "UCX_MAX_RNDV_RAILS": os.environ.get("UCX_MAX_RNDV_RAILS", "4"),
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "VERL_LOGGING_LEVEL": os.environ.get("VERL_LOGGING_LEVEL", "INFO"),
    }
    for prefix in ("MOONCAKE_", "NIXL_"):
        for k, v in os.environ.items():
            if k.startswith(prefix) and k not in env_vars:
                env_vars[k] = v
    # Ray on Arnold starts raylet workers with a platform Python that has ray but
    # may not have torch (the driver's `python3` does). Force every Ray worker to
    # use the driver interpreter (which has both ray + torch + cupy + verl).
    for inherit in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH",
                    "WS_UV_PROJECT", "WS_UV_EXTRAS", "WS_DEPS_DIR"):
        if inherit in os.environ and os.environ[inherit]:
            env_vars[inherit] = os.environ[inherit]
    runtime_env = {"py_executable": sys.executable, "env_vars": env_vars}
    if setup_mod is not None and setup_pymod_dir:
        runtime_env["worker_process_setup_hook"] = setup_mod.worker_process_setup_hook
        runtime_env["py_modules"] = [setup_pymod_dir]
    try:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env=runtime_env, ignore_reinit_error=True)
    except (TypeError, ValueError) as e:
        # Older Ray without py_executable/setup-hook/py_modules-dir support:
        # drop the unsupported keys and retry env-only once. A repeated failure
        # is fatal -- re-calling ray.init() after a half-done connect leaves Ray
        # in a broken state (Worker.core_worker missing), so do not retry blindly.
        print(f"[setup] WARNING: ray.init with full runtime_env failed ({type(e).__name__}: {str(e)[:200]}); "
              "retrying env-only without py_executable/setup_hook/py_modules")
        runtime_env = {"env_vars": env_vars}
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), runtime_env=runtime_env, ignore_reinit_error=True)

    alive_nodes = [n for n in ray.nodes() if n["Alive"]]
    gpu_nodes = sorted(
        ((n["NodeID"], int(n.get("Resources", {}).get("GPU", 0))) for n in alive_nodes),
        key=lambda x: -x[1],
    )
    gpu_nodes = [(nid, g) for nid, g in gpu_nodes if g > 0]

    def _fill(nodes, count):
        """Lay `count` ranks onto whole nodes, returning one node id per rank."""
        slots, nodes_used = [], 0
        for nid, gpus in nodes:
            if len(slots) >= count:
                break
            slots.extend([nid] * min(gpus, count - len(slots)))
            nodes_used += 1
        return slots, nodes_used

    # Senders fill whole nodes first, then receivers take DISJOINT nodes, so the
    # payload always crosses the fabric. This is what lets the run mirror a real
    # deployment's shape -- e.g. a 32-node trainer feeding an 8-node rollout --
    # instead of being capped at one node per side.
    sender_slots, sender_node_count = _fill(gpu_nodes, num_senders)
    receiver_slots, _ = _fill(gpu_nodes[sender_node_count:], num_receivers)

    if len(sender_slots) < num_senders or len(receiver_slots) < num_receivers:
        total = sum(g for _, g in gpu_nodes)
        if not allow_single_node:
            raise RuntimeError(
                f"need {num_senders} sender + {num_receivers} receiver GPUs on disjoint nodes; "
                f"cluster has {len(gpu_nodes)} GPU node(s) totalling {total} GPUs. "
                "Pass allow_single_node=True to co-locate them for a smoke run."
            )
        # Smoke mode: let both sides share whatever is available.
        sender_slots, _ = _fill(gpu_nodes, num_senders)
        receiver_slots, _ = _fill(gpu_nodes, num_receivers)
        if len(sender_slots) < num_senders or len(receiver_slots) < num_receivers:
            raise RuntimeError(
                f"cluster has {total} GPUs, need {num_senders} + {num_receivers}"
            )

    node_sender, node_receiver = sender_slots[0], receiver_slots[0]
    print(f"[setup] senders on {len(set(sender_slots))} node(s), "
          f"receivers on {len(set(receiver_slots))} node(s); "
          f"overlap={bool(set(sender_slots) & set(receiver_slots))}")

    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # -- Per-node bootstrap: probe the real worker env, then install cupy /
    # nixl / mooncake ON EACH node (entrypoint pip installs land on head only).
    # Must finish before GPU actors are created: actor class module imports
    # torch at top level, and the setup hook only fixes sys.path -- the deps
    # themselves are put in place here.
    if setup_mod is not None and getattr(setup_mod, "probe", None) is not None:
        try:
            boot_nodes = sorted(set(sender_slots) | set(receiver_slots))
            print(f"[bootstrap] probing {len(boot_nodes)} nodes before actor creation ...")
            probe_refs = [
                setup_mod.probe.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
                ).remote()
                for nid in boot_nodes
            ]
            for p in ray.get(probe_refs, timeout=300):
                print(
                    f"[probe] host={p['hostname']} py={p['python']} ({p['version']}) "
                    f"user={p['user']} HOME={p['HOME']} mach={p['machine']}"
                )
                print(f"[probe]   user_site={p['user_site']} deps_target={p['deps_target']}")
                for m, d in p["deps"].items():
                    if d.get("ok"):
                        print(f"[probe]   {m:<12} OK   {str(d.get('version', ''))[:20]:<20} {d.get('file') or ''}")
                    else:
                        print(f"[probe]   {m:<12} MISS {str(d.get('error', ''))[:110]}")
                print(f"[probe]   torch_cuda={p.get('torch_cuda')}")
            print("[bootstrap] installing RDMA deps on each node (cupy can take a few minutes) ...")
            deps_refs = [
                setup_mod.ensure_deps.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
                ).remote()
                for nid in boot_nodes
            ]
            reports = ray.get(deps_refs, timeout=3000)
            for rep in reports:
                host = rep.get("probe", {}).get("hostname", "?")
                print(f"[deps] node={host} target={rep.get('target')} cached={rep.get('cached', False)}")
                for pkg, r in rep.get("results", {}).items():
                    print(f"[deps]   {pkg:<26} {str(r.get('status')):<26} {r.get('version', '')}")
                    if r.get("tail"):
                        print(f"[deps]      tail: {str(r['tail'])[-220:]}")
                for m, d in rep.get("final", {}).items():
                    mark = "OK  " if d.get("ok") else "MISS"
                    extra = d.get("version", "") if d.get("ok") else ""
                    print(f"[deps]   final {m:<12} {mark} {extra}")
        except Exception as e:
            print(f"[bootstrap] WARNING: per-node bootstrap error: {type(e).__name__}: {str(e)[:300]}")

    senders = [
        WeightSyncWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=sender_slots[i], soft=False)
        ).remote(global_rank=i)
        for i in range(num_senders)
    ]
    receivers = [
        WeightSyncWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=receiver_slots[i], soft=False)
        ).remote(global_rank=num_senders + i)
        for i in range(num_receivers)
    ]

    infos = ray.get([w.info.remote() for w in senders + receivers])
    sender_infos = infos[:num_senders]
    receiver_infos = infos[num_senders:]
    print(f"[setup] senders node={sender_infos[0]['node_id'][:8]} gpu={sender_infos[0]['gpu']}")
    print(f"[setup] receivers node={receiver_infos[0]['node_id'][:8]} gpu={receiver_infos[0]['gpu']}")
    if sender_infos[0]["node_id"] == receiver_infos[0]["node_id"]:
        print("[setup] WARNING: senders and receivers share one node; cross-fabric path not exercised")

    def _dep_line(tag: str, info: dict) -> None:
        print(f"[setup] {tag} python={info['python']}")
        d = info["deps"]
        print(f"[setup] {tag} deps: " + " ".join(f"{m}={v}" for m, v in d.items()))

    _dep_line("sender  ", sender_infos[0])
    _dep_line("receiver", receiver_infos[0])
    for crit, hint in (("cupy", "nccl/nixl RDMA engines"), ("zmq", "nccl control plane")):
        sv = sender_infos[0]["deps"].get(crit, "")
        rv = receiver_infos[0]["deps"].get(crit, "")
        if str(sv).startswith("MISS") or str(rv).startswith("MISS"):
            print(f"[setup] WARNING: '{crit}' missing on a worker (sender={sv} receiver={rv}); {hint} will fail")

    torch_dtype = _DTYPE_MAP[dtype]
    total_bytes = int(total_gb * GIB)
    specs = build_weight_specs(total_bytes, num_tensors, torch_dtype)
    specs_json = _specs_to_json(specs)
    expected = _expected_checksums(specs)
    actual_payload = sum(s.nbytes for s in specs)
    print(f"[setup] payload={actual_payload / GIB:.2f} GiB across {len(specs)} tensors, dtype={dtype}")

    results = {
        "payload_gib": actual_payload / GIB,
        "num_tensors": len(specs),
        "dtype": dtype,
        "num_senders": num_senders,
        "num_receivers": num_receivers,
        "bucket_mb": bucket_mb,
        "iters": iters,
        "warmup": warmup,
        "backends": {},
    }

    def verify_checksums(recv_results: list[dict], backend_tag: str) -> None:
        for r in recv_results:
            assert r["num_weights"] == len(specs), (
                f"{backend_tag} rank {r['rank']} got {r['num_weights']}/{len(specs)} weights"
            )
            for name, vals in r["checksums"].items():
                assert vals == expected[name], f"{backend_tag} rank {r['rank']} checksum mismatch on {name}: {vals}"
        print(f"[{backend_tag}] correctness OK ({len(specs)} weights x {num_receivers} receivers verified)")

    # -- RDMA engines -------------------------------------------------------
    from verl.checkpoint_engine import CheckpointEngineRegistry

    rdma_backends = [b for b in backends if b != "fs"]
    for backend in rdma_backends:
        try:
            engine_cls = CheckpointEngineRegistry.get(backend)
        except ValueError as e:
            print(f"[{backend}] SKIP: {e}")
            results["backends"][backend] = {"error": f"not registered: {str(e)[:200]}"}
            continue
        if backend == "naive" and node_sender != node_receiver:
            # The colocated engine hands weights over in-process and does not
            # implement prepare/build_topology at all; across two nodes it can
            # only raise NotImplementedError.
            print("[naive] SKIP: colocated backend, same-node only")
            results["backends"][backend] = {"error": "colocated backend, same-node only"}
            continue

        all_workers = senders + receivers
        iter_seconds = []
        backend_error = None
        try:
            try:
                ray.get([w.engine_create.remote(backend, bucket_mb << 20) for w in all_workers])
                for it in range(warmup + iters):
                    metadata = ray.get([w.engine_prepare.remote() for w in all_workers])
                    sender_kwargs, rollout_kwargs = engine_cls.build_topology(num_senders, num_receivers, metadata)
                    init_refs = []
                    for i, w in enumerate(senders):
                        kwargs = {k: v[i] for k, v in sender_kwargs.items() if k != "method"}
                        init_refs.append(w.engine_init.remote(**kwargs))
                    for j, w in enumerate(receivers):
                        kwargs = {k: v[j] for k, v in rollout_kwargs.items() if k != "method"}
                        init_refs.append(w.engine_init.remote(**kwargs))
                    ray.get(init_refs)

                    # Receivers first, then senders, then one wall-clock stop:
                    # e2e is "dispatch -> every receiver holds the full payload
                    # on GPU", measured exactly like the fs path below so the
                    # two families can be compared on one number.
                    t0 = time.time()
                    recv_refs = [w.engine_recv.remote(specs_json) for w in receivers]
                    send_refs = [w.engine_send.remote(specs_json) for w in senders]
                    send_res, recv_res = ray.get(send_refs), ray.get(recv_refs)
                    e2e = time.time() - t0

                    if it >= warmup:
                        recv_max = max(r["seconds"] for r in recv_res)
                        send_max = max(r["seconds"] for r in send_res)
                        iter_seconds.append({"send_max": send_max, "recv_max": recv_max,
                                             "e2e": e2e, "recv_all": recv_res})
                        print(
                            f"[{backend}] iter {it - warmup + 1}: send={send_max:.2f}s recv={recv_max:.2f}s "
                            f"e2e={e2e:.2f}s bw={_gib_per_sec(actual_payload, recv_max):.2f} GiB/s"
                        )
                    if it == warmup + iters - 1:
                        verify_checksums(recv_res, backend)
            finally:
                try:
                    ray.get([w.engine_finalize.remote() for w in all_workers])
                except Exception as fe:
                    print(f"[{backend}] finalize warning: {type(fe).__name__}: {str(fe)[:200]}")
        except Exception as e:
            backend_error = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[{backend}] ERROR (skipping backend, continuing): {backend_error}")
            results["backends"][backend] = {"error": backend_error}
            continue

        if iter_seconds:
            recv_times = [t["recv_max"] for t in iter_seconds]
            send_times = [t["send_max"] for t in iter_seconds]
            e2e_times = [t["e2e"] for t in iter_seconds]
            results["backends"][backend] = {
                "send_seconds_mean": float(np.mean(send_times)),
                "recv_seconds_mean": float(np.mean(recv_times)),
                "recv_seconds_max": float(np.max(recv_times)),
                "e2e_seconds_mean": float(np.mean(e2e_times)),
                "gib_per_s_mean": _gib_per_sec(actual_payload, np.mean(recv_times)),
                "e2e_gib_per_s_mean": _gib_per_sec(actual_payload, np.mean(e2e_times)),
            }
        elif backend_error is None:
            backend_error = "no completed iterations"
            results["backends"][backend] = {"error": backend_error}

    # -- Filesystem path ----------------------------------------------------
    if "fs" in backends:
        if fs_dir is None:
            if sender_infos[0]["node_id"] != receiver_infos[0]["node_id"]:
                print("[fs] SKIP: cross-node run needs --fs-dir pointing to a shared CFS/HDFS-fuse mount")
            else:
                fs_dir = tempfile.mkdtemp(prefix="weight_sync_bench_")
                print(f"[fs] single-node smoke run using local tmp dir {fs_dir}")
        if fs_dir is not None:
            os.makedirs(fs_dir, exist_ok=True)
        # Production writes through a RW mount and reads the SAME backing bytes
        # through a second, read-optimised mount (alpha-seed's rollout_read_root,
        # a FAFUSE mount over the same HDFS path). Model that split when given;
        # otherwise read back through the write mount.
        fs_read_root = fs_read_dir or fs_dir
        if fs_dir is not None:
            print(f"[fs] write root {fs_dir}")
            print(f"[fs] read  root {fs_read_root}"
                  f"{' (same mount)' if fs_read_root == fs_dir else ' (separate read mount)'}")

        fs_iter = []
        if fs_dir is not None:
            for it in range(warmup + iters):
                step = 1000 + it
                t0 = time.time()
                # Readers start polling now, concurrently with the writers, so
                # their visibility wait is a real measurement rather than zero,
                # and e2e overlaps write and read the way production does.
                read_refs = [w.fs_read.remote(fs_read_root, step, num_senders, specs_json)
                             for w in receivers]
                write_refs = [
                    w.fs_write.remote(fs_dir, step, shard_rank=r, num_shards=num_senders,
                                      specs_serialized=specs_json)
                    for r, w in enumerate(senders)
                ]
                write_res = ray.get(write_refs)
                read_res = ray.get(read_refs)
                end = time.time()

                write_max = max(r["seconds"] for r in write_res)
                wait_max = max(r["wait_seconds"] for r in read_res)
                read_max = max(r["seconds"] for r in read_res)
                e2e = end - t0
                if it >= warmup:
                    fs_iter.append(
                        {
                            "write_max": write_max,
                            "wait_max": wait_max,
                            "read_max": read_max,
                            "e2e": e2e,
                        }
                    )
                    print(
                        f"[fs] iter {it - warmup + 1}: write={write_max:.2f}s (agg {_gib_per_sec(actual_payload, write_max):.2f} GiB/s) "
                        f"visibility_wait={wait_max:.2f}s read={read_max:.2f}s "
                        f"({_gib_per_sec(actual_payload, read_max):.2f} GiB/s per receiver) e2e={e2e:.2f}s"
                    )
                if it == warmup + iters - 1:
                    verify_checksums(read_res, "fs")
                fs_cleanup(fs_dir, step)

            tmp_fs_dir = fs_dir if fs_dir.startswith(tempfile.gettempdir()) else None
            if tmp_fs_dir is not None:
                shutil.rmtree(tmp_fs_dir, ignore_errors=True)

        if fs_iter:
            results["backends"]["fs"] = {
                "write_seconds_mean": float(np.mean([t["write_max"] for t in fs_iter])),
                "write_gib_per_s": _gib_per_sec(actual_payload, np.mean([t["write_max"] for t in fs_iter])),
                "visibility_wait_seconds_mean": float(np.mean([t["wait_max"] for t in fs_iter])),
                "read_seconds_mean": float(np.mean([t["read_max"] for t in fs_iter])),
                "read_gib_per_s": _gib_per_sec(actual_payload, np.mean([t["read_max"] for t in fs_iter])),
                "e2e_seconds_mean": float(np.mean([t["e2e"] for t in fs_iter])),
            }

    _print_summary(results)
    if json_out:
        import json

        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[result] json written to {json_out}")
    return results


def _print_summary(results: dict) -> None:
    payload = results["payload_gib"] * GIB
    print("\n" + "=" * 84)
    print(f" weight sync benchmark summary  (payload {results['payload_gib']:.2f} GiB, "
          f"{results['num_senders']} senders -> {results['num_receivers']} receivers)")
    print("=" * 84)

    # Headline table first: one directly comparable number per backend.
    print(f" {'backend':<18} {'e2e s':>10} {'GiB/s (e2e)':>13}   note")
    print("-" * 84)
    for backend, r in results["backends"].items():
        if "error" in r:
            print(f" {backend:<18} {'-':>10} {'-':>13}   ERROR/SKIP: {r['error'][:38]}")
            continue
        e2e = r.get("e2e_seconds_mean")
        if e2e is None:
            print(f" {backend:<18} {'-':>10} {'-':>13}   no e2e recorded")
            continue
        kind = "shared filesystem" if backend == "fs" else "in-memory transport"
        print(f" {backend:<18} {e2e:>10.2f} {_gib_per_sec(payload, e2e):>13.2f}   {kind}")
    print("-" * 84)
    print(" e2e = driver wall clock, dispatch -> all receivers hold the full payload on GPU.")
    print(" Intended asymmetry: fs has N writers each persisting 1/N of the payload in")
    print(" parallel (FSDP stripes), while the engines ship one full copy from the root")
    print(" rank. Both mirror their own production path, so e2e is the fair comparison.")

    # Per-backend phase breakdown, for reading where the time actually goes.
    print("\n" + "=" * 84)
    print(f" phase breakdown")
    print("=" * 84)
    print(f" {'backend':<18} {'phase':<28} {'seconds':>10} {'GiB/s':>10}")
    print("-" * 84)
    for backend, r in results["backends"].items():
        if "error" in r:
            continue
        if backend == "fs":
            rows = [
                ("parallel write (aggregate)", r["write_seconds_mean"], r["write_gib_per_s"]),
                ("visibility wait", r["visibility_wait_seconds_mean"], None),
                ("mmap read + GPU stage", r["read_seconds_mean"], r["read_gib_per_s"]),
                ("end-to-end sync", r["e2e_seconds_mean"], _gib_per_sec(payload, r["e2e_seconds_mean"])),
            ]
        else:
            rows = [
                ("send (sender side)", r["send_seconds_mean"], None),
                ("recv (critical path)", r["recv_seconds_mean"], r["gib_per_s_mean"]),
                ("end-to-end sync", r["e2e_seconds_mean"], r.get("e2e_gib_per_s_mean")),
            ]
        for i, (phase, sec, bw) in enumerate(rows):
            bw_s = f"{bw:.2f}" if bw is not None else "-"
            print(f" {backend if i == 0 else '':<18} {phase:<28} {sec:>10.2f} {bw_s:>10}")
    print("=" * 84)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fs_weight_sync_roundtrip_cpu(tmp_path):
    """CPU-only correctness test for the FS shard write/read helpers."""
    specs = build_weight_specs(total_bytes=4 * 1024 * 1024, num_tensors=8, dtype=torch.bfloat16, prefix="ut")
    num_shards = 2
    step = 1
    for shard_rank in range(num_shards):
        named = {}
        for i, spec in enumerate(specs):
            if i % num_shards != shard_rank:
                continue
            t = torch.empty(spec.shape, dtype=spec.dtype)
            fill_pattern(t, i)
            named[spec.name] = t
        fs_write_shard(str(tmp_path), step, shard_rank, num_shards, named)

    fs_wait_ready(str(tmp_path), step, num_shards, timeout=5)
    loaded = fs_read_shards(str(tmp_path), step, num_shards, stage_device=None)
    assert len(loaded) == len(specs)
    for i, spec in enumerate(specs):
        expected_vals = expected_spot_values(spec, i)
        _, vals = compute_spot_checksum(loaded[spec.name], i)
        assert vals == expected_vals, f"mismatch on {spec.name}"


@pytest.mark.skipif(
    os.environ.get("BENCH_WEIGHT_SYNC") != "1",
    reason="set BENCH_WEIGHT_SYNC=1 to run the GPU weight-sync benchmark (needs a Ray cluster)",
)
def test_weight_sync_benchmark_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 8 and os.environ.get("BENCH_ALLOW_SMALL") != "1":
        pytest.skip("needs >=8 GPUs; set BENCH_ALLOW_SMALL=1 to force a small run")
    fs_dir = os.environ.get("BENCH_FS_DIR")
    results = run_benchmark(
        backends=["nccl", "nixl", "mooncake", "fs"],
        total_gb=float(os.environ.get("BENCH_TOTAL_GB", "2")),
        iters=int(os.environ.get("BENCH_ITERS", "2")),
        fs_dir=fs_dir,
        allow_single_node=os.environ.get("BENCH_ALLOW_SMALL") == "1",
    )
    assert results["backends"], "no backend produced results"
    for backend, r in results["backends"].items():
        if backend == "fs":
            assert r["e2e_seconds_mean"] > 0
        else:
            assert r["recv_seconds_mean"] > 0


def main():
    parser = argparse.ArgumentParser(description="FS vs RDMA weight sync benchmark")
    parser.add_argument(
        "--backends", nargs="+",
        default=["fs", "naive", "nccl", "nixl", "mooncake", "kimi_ckpt_engine", "delta_sharded"],
        help="every registered checkpoint-engine backend plus 'fs'; unavailable ones SKIP",
    )
    parser.add_argument("--total-gb", type=float, default=4.0)
    parser.add_argument("--num-tensors", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--bucket-mb", type=int, default=2048)
    parser.add_argument("--fs-dir", default=None, help="RW mount the writers persist shards to")
    parser.add_argument("--fs-read-dir", default=None,
                        help="read mount over the same bytes (alpha-seed rollout_read_root / FAFUSE); "
                             "defaults to --fs-dir")
    parser.add_argument("--num-senders", type=int, default=4)
    parser.add_argument("--num-receivers", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    parser.add_argument("--allow-single-node", action="store_true")
    parser.add_argument("--json-out", default=None, help="path for machine-readable results")
    args = parser.parse_args()
    run_benchmark(
        backends=args.backends,
        total_gb=args.total_gb,
        num_tensors=args.num_tensors,
        iters=args.iters,
        warmup=args.warmup,
        bucket_mb=args.bucket_mb,
        fs_dir=args.fs_dir,
        fs_read_dir=args.fs_read_dir,
        num_senders=args.num_senders,
        num_receivers=args.num_receivers,
        dtype=args.dtype,
        allow_single_node=args.allow_single_node,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
