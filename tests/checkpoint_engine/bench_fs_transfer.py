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
"""Benchmark the alpha-seed / mono_rl FILESYSTEM weight-sync path ONLY.

Production path being mirrored (see alpha_seed/workers/rollout/sampler_weights/):

  layout  {fs_dir}/sampler_weights/trial_{ARNOLD_TRIAL_ID}/run_{ROBUST_RUN_ID}/
              step_{N}/model-0000r-of-0000R.safetensors   (one shard per actor rank)
              step_{N}/model.safetensors.index.json       (written at publish)
              step_{N}/skip_keys.json                     (publish marker)

  write   every actor rank materialises its FSDP stripe on GPU, DMAs it into
          pinned host memory and writes its own safetensors shard to the shared
          filesystem (CFS / HDFS-fuse / FAFUSE) in parallel;
  publish after the driver has observed ALL shard writes, rank 0 writes
          model.safetensors.index.json and finally skip_keys.json (fsynced) —
          readers gate step visibility on skip_keys.json, exactly like
          ``latest_sampler_weights_step_dir``;
  read    every rollout replica polls for skip_keys.json, mmaps every shard
          (``safetensors.safe_open``), pages the full payload in (the real
          reload_weights_from_path touches every byte) and optionally stages it
          onto GPU with non-blocking H2D copies.

What is measured, per iteration:
  writer side  stage  GPU -> pinned-host DMA;
               write  tmp shard -> os.replace (serialise + FS write);
               publish index + skip_keys.json + fsync;
  reader side  wait     marker visibility (FUSE metadata propagation);
               read_cpu mmap + full page-in from the network mount;
               stage_h2d CPU -> GPU upload (skip with --no-gpu-stage);
               e2e    driver wall time from dispatch to all replicas loaded.

This deliberately does NOT exercise FSDP all-gather or the production
streaming/pinned-pool overlap: it isolates the *filesystem transfer* of a fixed
payload, which is what you compare against the verl checkpoint-engine numbers
from ``bench_checkpoint_engine.py``. Both scripts accept --total-gb/--iters/
--dtype and emit the same JSON shape.

Cross-node run (2 nodes x N GPUs) from the repo root on a Ray cluster::

    python tests/checkpoint_engine/bench_fs_transfer.py \
        --fs-dir /mnt/hdfs/your_mount/weight_sync_bench \
        --total-gb 8 --iters 5

CPU-only logic check (no Ray/GPU/shared FS)::

    python tests/checkpoint_engine/bench_fs_transfer.py --cpu-selftest
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Optional


def _bootstrap_sys_path() -> None:
    """Make the node-local deps dir importable before torch is imported.

    This module is imported by Ray when it loads the actor class, in a process
    we do not control: Ray-on-Arnold merges a job-level runtime_env whose
    PYTHONPATH can win over the driver's, and worker_process_setup_hook is not
    honoured everywhere. Either way the actor would die with "No module named
    'torch'" while the driver is perfectly healthy, so repair the path here
    rather than relying on the environment reaching us intact.
    """
    try:
        import _ws_setup

        _ws_setup.fix_sys_path()
        return
    except Exception:  # noqa: BLE001
        # py_modules can be dropped by the same runtime_env merge, so the
        # fallback below must not depend on _ws_setup being importable.
        pass
    import glob

    vtag = "python%d.%d" % sys.version_info[:2]
    candidates = [os.environ.get("WS_DEPS_DIR"), "/opt/tiger/ws_deps", "/tmp/ws_deps"]
    # torch ships in a user-site on these images, and raylet spawns workers with a
    # different HOME than the job driver, so the owning user's site dir has to be
    # found by globbing rather than through site.getusersitepackages().
    for pattern in ("/home/*/.local/lib/python*/site-packages",
                    "/root/.local/lib/python*/site-packages",
                    "/usr/local/lib/python*/dist-packages",
                    "/usr/local/lib/python*/site-packages"):
        candidates.extend(sorted(glob.glob(pattern)))
    candidates.append("/usr/local/lib/%s/dist-packages" % vtag)
    for d in candidates:
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)


_bootstrap_sys_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

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


def build_weight_specs(total_bytes: int, num_tensors: int, dtype: torch.dtype, prefix: str = "model"):
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
    """Deterministic in-place fill with small integers, exact in bf16/fp16."""
    n = tensor.numel()
    pos = torch.arange(n, device=tensor.device, dtype=torch.int64)
    values = ((pos + tensor_idx * 104729) % 97).to(tensor.dtype)
    tensor.view(-1).copy_(values)
    return tensor


def spot_check_positions(numel: int):
    if numel <= 8:
        return list(range(numel))
    return [0, 1, numel // 4, numel // 2, (3 * numel) // 4, numel - 2, numel - 1]


def expected_spot_values(spec: WeightSpec, tensor_idx: int):
    positions = spot_check_positions(int(np.prod(spec.shape)))
    return [(p + tensor_idx * 104729) % 97 for p in positions]


def compute_spot_checksum(tensor: torch.Tensor, tensor_idx: int):
    positions = spot_check_positions(tensor.numel())
    flat = tensor.view(-1)
    vals = [int(flat[p].to(torch.float32).item()) for p in positions]
    return tensor_idx, vals


# ---------------------------------------------------------------------------
# On-disk layout -- mirrors alpha_seed .../sampler_weights/fs_store.py and
# mono_rl .../checkpoint/sharded_safetensors.py naming.
# ---------------------------------------------------------------------------

SKIP_MANIFEST_NAME = "skip_keys.json"
INDEX_NAME = "model.safetensors.index.json"


def trial_root(fs_dir: str, trial_id: str) -> str:
    return os.path.join(fs_dir, "sampler_weights", f"trial_{trial_id}")


def run_root(fs_dir: str, trial_id: str, run_id: int) -> str:
    return os.path.join(trial_root(fs_dir, trial_id), f"run_{run_id}")


def step_dir(fs_dir: str, trial_id: str, run_id: int, step: int) -> str:
    return os.path.join(run_root(fs_dir, trial_id, run_id), f"step_{step}")


def shard_file(d: str, shard_rank: int, num_shards: int) -> str:
    return os.path.join(d, f"model-{shard_rank + 1:05d}-of-{num_shards:05d}.safetensors")


def write_shard(d: str, shard_rank: int, num_shards: int, named_tensors: dict) -> int:
    """Write one safetensors shard atomically: tmp file -> rename."""
    from safetensors.torch import save_file

    os.makedirs(d, exist_ok=True)
    final_path = shard_file(d, shard_rank, num_shards)
    tmp_path = final_path + f".tmp{os.getpid()}"
    save_file(named_tensors, tmp_path, metadata={"shard_rank": str(shard_rank)})
    os.replace(tmp_path, final_path)
    return os.path.getsize(final_path)


def publish_step(d: str, num_shards: int, weight_map: dict, total_bytes: int) -> None:
    """Rank 0 publishes: HF index first, then the fsynced skip_keys.json marker.

    Readers gate on the marker (``latest_sampler_weights_step_dir``), so the
    index must already be on disk when the marker becomes visible.
    """
    index = {
        "metadata": {"total_size": int(total_bytes)},
        "weight_map": {
            name: os.path.basename(shard_file(d, rank, num_shards))
            for name, rank in weight_map.items()
        },
    }
    idx_tmp = os.path.join(d, INDEX_NAME + ".tmp")
    with open(idx_tmp, "w") as f:
        json.dump(index, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(idx_tmp, os.path.join(d, INDEX_NAME))

    # Marker publish must be atomic: a polling reader must never observe a
    # half-written skip_keys.json on a FUSE mount.
    marker = os.path.join(d, SKIP_MANIFEST_NAME)
    marker_tmp = marker + f".tmp{os.getpid()}"
    with open(marker_tmp, "w") as f:
        json.dump({"skip_keys": [], "published_at": time.time()}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(marker_tmp, marker)


def wait_published(d: str, timeout: float = 1800.0, poll: float = 0.1) -> float:
    """Block until the publish marker is visible; return seconds spent waiting."""
    marker = os.path.join(d, SKIP_MANIFEST_NAME)
    start = time.time()
    deadline = start + timeout
    while time.time() < deadline:
        if os.path.isfile(marker):
            return time.time() - start
        time.sleep(poll)
    raise TimeoutError(f"{marker} not visible within {timeout}s")


def read_all_shards(d: str, num_shards: int, stage_device: Optional[str]):
    """Mmap every shard, page all bytes into private CPU tensors, optionally H2D.

    Returns (weights, phase_seconds) where phase_seconds breaks the read into
    ``read_cpu`` (open/mmap/full page-in over the network mount) and
    ``stage_h2d`` (non-blocking CPU->GPU copies + synchronize).
    """
    from safetensors import safe_open

    cpu_weights = {}
    t_read0 = time.time()
    for r in range(num_shards):
        path = shard_file(d, r, num_shards)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                # get_tensor on an mmaped safetensors faults pages in lazily;
                # clone() forces a FULL read, exactly what a real weight load
                # pays on a network mount.
                cpu_weights[name] = f.get_tensor(name).clone()
    read_cpu = time.time() - t_read0

    stage_h2d = 0.0
    if stage_device is not None:
        weights = {}
        t_stage0 = time.time()
        for name, cpu_t in cpu_weights.items():
            gpu_t = torch.empty(cpu_t.shape, dtype=cpu_t.dtype, device=stage_device)
            gpu_t.copy_(cpu_t, non_blocking=True)
            weights[name] = gpu_t
        torch.cuda.synchronize()
        stage_h2d = time.time() - t_stage0
        del cpu_weights
    else:
        weights = cpu_weights
    return weights, {"read_cpu": read_cpu, "stage_h2d": stage_h2d}


def _rmtree_background(d: str) -> None:
    """Mirror the production store: retention deletes off the timed path."""
    def _rm():
        shutil.rmtree(d, ignore_errors=True)

    threading.Thread(target=_rm, name="fs-bench-cleanup", daemon=True).start()


# ---------------------------------------------------------------------------
# CPU-only self test
# ---------------------------------------------------------------------------


def cpu_selftest(total_mb: float = 64.0, num_tensors: int = 8) -> None:
    print(f"[selftest] CPU-only FS round-trip, payload={total_mb} MiB, shards=2")
    tmp = tempfile.mkdtemp(prefix="fs_transfer_selftest_")
    num_shards = 2
    step = 1
    d = step_dir(tmp, "selftest", 0, step)
    specs = build_weight_specs(int(total_mb * 1024 * 1024), num_tensors, torch.bfloat16, prefix="ut")
    expected = {s.name: expected_spot_values(s, i) for i, s in enumerate(specs)}
    weight_map = {}
    t0 = time.time()
    for r in range(num_shards):
        named = {}
        for i, spec in enumerate(specs):
            if i % num_shards != r:
                continue
            t = torch.empty(spec.shape, dtype=spec.dtype)
            fill_pattern(t, i)
            named[spec.name] = t
            weight_map[spec.name] = r
        write_shard(d, r, num_shards, named)
    total_bytes = sum(s.nbytes for s in specs)
    publish_step(d, num_shards, weight_map, total_bytes)
    write_s = time.time() - t0

    wait_s = wait_published(d, timeout=5.0)
    weights, phases = read_all_shards(d, num_shards, None)
    assert len(weights) == len(specs), f"got {len(weights)}/{len(specs)} tensors"
    for i, spec in enumerate(specs):
        _, vals = compute_spot_checksum(weights[spec.name], i)
        assert vals == expected[spec.name], f"checksum mismatch on {spec.name}"
    print(f"[selftest] OK  write+publish={write_s:.3f}s wait={wait_s:.3f}s "
          f"read_cpu={phases['read_cpu']:.3f}s ({_gib(total_bytes) / max(phases['read_cpu'], 1e-9):.2f} GiB/s)")
    print(f"[selftest] dir was {d}")
    shutil.rmtree(tmp, ignore_errors=True)


def _gib(nbytes: float) -> float:
    return nbytes / GIB


# ---------------------------------------------------------------------------
# Ray worker: one per GPU, FS writer (actor) or FS reader (rollout replica)
# ---------------------------------------------------------------------------

try:
    import ray

    @ray.remote(num_gpus=1)
    class FSWeightWorker:
        def __init__(self, global_rank: int):
            self.global_rank = global_rank
            self.device = "cuda"
            torch.cuda.set_device(0)

        def info(self) -> dict:
            import importlib
            import socket

            deps = {}
            for m in ("torch", "ray", "safetensors", "numpy"):
                try:
                    mod = importlib.import_module(m)
                    deps[m] = getattr(mod, "__version__", "ok")
                except Exception as e:  # noqa: BLE001
                    deps[m] = f"MISS({type(e).__name__})"
            return {
                "global_rank": self.global_rank,
                "node_id": ray.get_runtime_context().get_node_id(),
                "ip": socket.gethostbyname(socket.gethostname()),
                "gpu": torch.cuda.get_device_name(0),
                "python": sys.executable,
                "deps": deps,
            }

        def write(
            self,
            target_dir: str,
            shard_rank: int,
            num_shards: int,
            specs_serialized: list,
            pin_memory: bool,
        ) -> dict:
            named_cpu = {}
            total_bytes = 0
            weight_map = {}
            stage_start = time.time()
            for i, s in enumerate(specs_serialized):
                if i % num_shards != shard_rank:
                    continue
                spec = WeightSpec(s["name"], tuple(s["shape"]), _DTYPE_MAP[s["dtype"]])
                gpu_t = torch.empty(spec.shape, dtype=spec.dtype, device=self.device)
                fill_pattern(gpu_t, i)
                cpu_t = torch.empty(spec.shape, dtype=spec.dtype, device="cpu", pin_memory=pin_memory)
                cpu_t.copy_(gpu_t, non_blocking=True)
                named_cpu[spec.name] = cpu_t
                weight_map[spec.name] = shard_rank
                total_bytes += spec.nbytes
                del gpu_t
            torch.cuda.synchronize()
            stage_seconds = time.time() - stage_start

            start = time.time()
            file_bytes = write_shard(target_dir, shard_rank, num_shards, named_cpu)
            return {
                "rank": self.global_rank,
                "shard": shard_rank,
                "stage_seconds": stage_seconds,
                "seconds": time.time() - start,
                "bytes": total_bytes,
                "file_bytes": file_bytes,
                "weight_map": weight_map,
            }

        def publish(self, target_dir: str, num_shards: int, weight_map: dict, total_bytes: int) -> dict:
            start = time.time()
            publish_step(target_dir, num_shards, weight_map, total_bytes)
            return {"rank": self.global_rank, "seconds": time.time() - start}

        def read(
            self,
            target_dir: str,
            num_shards: int,
            specs_serialized: list,
            gpu_stage: bool,
        ) -> dict:
            wait_seconds = wait_published(target_dir)
            stage_device = self.device if gpu_stage else None
            weights, phases = read_all_shards(target_dir, num_shards, stage_device)
            name_to_idx = {s["name"]: i for i, s in enumerate(specs_serialized)}
            checksums = {}
            for name, t in weights.items():
                _, vals = compute_spot_checksum(t, name_to_idx[name])
                checksums[name] = vals
            total_bytes = sum(t.numel() * t.element_size() for t in weights.values())
            return {
                "rank": self.global_rank,
                "role": "fs_read",
                "wait_seconds": wait_seconds,
                "read_cpu_seconds": phases["read_cpu"],
                "stage_h2d_seconds": phases["stage_h2d"],
                "seconds": phases["read_cpu"] + phases["stage_h2d"],
                "bytes": total_bytes,
                "checksums": checksums,
                "num_weights": len(checksums),
            }

except ImportError:
    ray = None
    FSWeightWorker = None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _specs_to_json(specs):
    dtype_inv = {v: k for k, v in _DTYPE_MAP.items()}
    return [{"name": s.name, "shape": list(s.shape), "dtype": dtype_inv[s.dtype]} for s in specs]


def _gib_per_sec(num_bytes: float, seconds: float) -> float:
    return num_bytes / GIB / max(seconds, 1e-9)


def _init_ray():
    if ray is None:
        raise RuntimeError("ray is not installed")
    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")

    # Reuse ONLY the torch-free sys.path fixup hook from _ws_setup (no RDMA dep
    # install needed for the FS path). Raylet-spawned workers may otherwise not
    # see the image's user-site torch.
    setup_hook = None
    setup_pymod_dir = None
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        import _ws_setup as setup_mod  # noqa: PLC0415

        try:
            import cloudpickle as _cp

            _cp.register_pickle_by_value(setup_mod)
        except Exception:  # noqa: BLE001
            pass
        try:
            import ray.cloudpickle as _ray_cp

            _ray_cp.register_pickle_by_value(setup_mod)
        except Exception:  # noqa: BLE001
            pass
        setup_pymod_dir = tempfile.mkdtemp(prefix="ws_setup_pkg_")
        shutil.copy(os.path.join(here, "_ws_setup.py"), os.path.join(setup_pymod_dir, "_ws_setup.py"))
        setup_hook = setup_mod.worker_process_setup_hook
        print(f"[setup] using _ws_setup worker_process_setup_hook from {here}")
    except Exception as e:  # noqa: BLE001
        print(f"[setup] note: _ws_setup hook unavailable ({type(e).__name__}: {e}); continuing without it")

    env_vars = {"VERL_LOGGING_LEVEL": os.environ.get("VERL_LOGGING_LEVEL", "INFO")}
    # WS_* must reach the worker pods: they decide where _ws_setup installs and
    # which torch spec it pins, and a worker that disagrees with the head would
    # build a different environment than the one under test.
    for inherit in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH",
                    "WS_DEPS_DIR", "WS_TORCH_SPEC", "WS_PIP_PROXY"):
        if inherit in os.environ and os.environ[inherit]:
            env_vars[inherit] = os.environ[inherit]
    runtime_env = {"py_executable": sys.executable, "env_vars": env_vars}
    if setup_hook is not None:
        runtime_env["worker_process_setup_hook"] = setup_hook
        runtime_env["py_modules"] = [setup_pymod_dir]
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "auto"),
        runtime_env=runtime_env,
        ignore_reinit_error=True,
    )


def _select_nodes(num_senders: int, num_receivers: int, allow_single_node: bool):
    alive_nodes = [n for n in ray.nodes() if n["Alive"]]
    gpu_nodes = []
    for n in alive_nodes:
        n_gpu = n.get("Resources", {}).get("GPU", 0)
        if n_gpu >= max(num_senders, num_receivers):
            gpu_nodes.append((n["NodeID"], int(n_gpu)))
    gpu_nodes.sort(key=lambda x: -x[1])
    if len(gpu_nodes) < 2:
        if not allow_single_node:
            raise RuntimeError(
                f"need >=2 GPU nodes for a real cross-mount FS transfer, found {len(gpu_nodes)}; "
                "pass --allow-single-node for a local-tmp smoke run"
            )
        return gpu_nodes[0][0], gpu_nodes[0][0]
    return gpu_nodes[0][0], gpu_nodes[1][0]


def run_benchmark(
    fs_dir: Optional[str],
    total_gb: float = 4.0,
    num_tensors: int = 64,
    iters: int = 3,
    warmup: int = 1,
    num_senders: int = 4,
    num_receivers: int = 4,
    dtype: str = "bf16",
    pin_memory: bool = True,
    gpu_stage: bool = True,
    trial_id: str = "bench",
    run_id: Optional[int] = None,
    cleanup: bool = True,
    allow_single_node: bool = False,
    json_out: Optional[str] = None,
) -> dict:
    _init_ray()
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_sender, node_receiver = _select_nodes(num_senders, num_receivers, allow_single_node)
    tmp_fs_dir = None
    if fs_dir is None:
        if node_sender != node_receiver:
            raise RuntimeError(
                "cross-node run needs --fs-dir pointing at a shared CFS/HDFS-fuse/FAFUSE mount"
            )
        fs_dir = tempfile.mkdtemp(prefix="fs_transfer_bench_")
        tmp_fs_dir = fs_dir
        print(f"[setup] single-node smoke run using local tmp dir {fs_dir}")
    if run_id is None:
        env_run = str(os.environ.get("ROBUST_RUN_ID", "0") or "0")
        run_id = int(env_run) if env_run.isdigit() else 0
    os.makedirs(run_root(fs_dir, trial_id, run_id), exist_ok=True)

    senders = [
        FSWeightWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_sender, soft=False)
        ).remote(global_rank=i)
        for i in range(num_senders)
    ]
    receivers = [
        FSWeightWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_receiver, soft=False)
        ).remote(global_rank=num_senders + i)
        for i in range(num_receivers)
    ]
    infos = ray.get([w.info.remote() for w in senders + receivers])
    s_info, r_info = infos[0], infos[num_senders]
    print(f"[setup] writers  node={s_info['node_id'][:8]} gpu={s_info['gpu']} deps={s_info['deps']}")
    print(f"[setup] readers  node={r_info['node_id'][:8]} gpu={r_info['gpu']} deps={r_info['deps']}")
    if s_info["node_id"] == r_info["node_id"]:
        print("[setup] WARNING: writers and readers share one node; network-mount path not exercised")

    torch_dtype = _DTYPE_MAP[dtype]
    total_bytes = int(total_gb * GIB)
    specs = build_weight_specs(total_bytes, num_tensors, torch_dtype)
    specs_json = _specs_to_json(specs)
    expected = {s.name: expected_spot_values(s, i) for i, s in enumerate(specs)}
    actual_payload = sum(s.nbytes for s in specs)
    print(f"[setup] payload={actual_payload / GIB:.2f} GiB across {len(specs)} tensors, dtype={dtype}, "
          f"{num_senders} writer shards, {num_receivers} reader replicas, pin_memory={pin_memory}, "
          f"gpu_stage={gpu_stage}")
    print(f"[setup] layout root: {run_root(fs_dir, trial_id, run_id)}")

    results = {
        "kind": "alphaseed_filesystem",
        "fs_dir": fs_dir,
        "payload_gib": actual_payload / GIB,
        "num_tensors": num_tensors,
        "num_senders": num_senders,
        "num_receivers": num_receivers,
        "dtype": dtype,
        "pin_memory": pin_memory,
        "gpu_stage": gpu_stage,
        "iterations": [],
    }

    for it in range(warmup + iters):
        step = 1000 + it
        d = step_dir(fs_dir, trial_id, run_id, step)
        t0 = time.time()
        # Dispatch readers concurrently with writers: their poll clock starts now.
        read_refs = [w.read.remote(d, num_senders, specs_json, gpu_stage) for w in receivers]
        write_refs = [
            w.write.remote(d, shard_rank=r, num_shards=num_senders,
                           specs_serialized=specs_json, pin_memory=pin_memory)
            for r, w in enumerate(senders)
        ]
        write_res = ray.get(write_refs)

        # Driver barrier over every shard, then rank-0 publishes (production
        # order: skip_keys.json only after all shards have landed).
        weight_map = {}
        for wr in write_res:
            weight_map.update(wr["weight_map"])
        pub_res = ray.get(senders[0].publish.remote(d, num_senders, weight_map, actual_payload))

        read_res = ray.get(read_refs)
        e2e = time.time() - t0

        stage_max = max(r["stage_seconds"] for r in write_res)
        write_max = max(r["seconds"] for r in write_res)
        wait_max = max(r["wait_seconds"] for r in read_res)
        read_cpu_max = max(r["read_cpu_seconds"] for r in read_res)
        h2d_max = max(r["stage_h2d_seconds"] for r in read_res)
        read_total_max = max(r["seconds"] for r in read_res)
        shard_file_bytes = sum(r["file_bytes"] for r in write_res)

        if it >= warmup:
            tag = it - warmup + 1
            record = {
                "iter": tag,
                "stage_d2h_max": stage_max,
                "write_max": write_max,
                "publish_seconds": pub_res["seconds"],
                "visibility_wait_max": wait_max,
                "read_cpu_max": read_cpu_max,
                "stage_h2d_max": h2d_max,
                "read_total_max": read_total_max,
                "e2e": e2e,
                "write_gib_per_s": _gib_per_sec(actual_payload, write_max),
                "read_gib_per_s_per_replica": _gib_per_sec(actual_payload, read_total_max),
                "read_aggregate_gib_per_s": _gib_per_sec(actual_payload * num_receivers, read_total_max),
                "shard_bytes_on_disk": shard_file_bytes,
            }
            results["iterations"].append(record)
            print(
                f"[fs] iter {tag}/{iters}: d2h={stage_max:.2f}s write={write_max:.2f}s "
                f"(agg {_gib_per_sec(actual_payload, write_max):.2f} GiB/s) "
                f"publish={pub_res['seconds']:.2f}s wait={wait_max:.2f}s "
                f"read_cpu={read_cpu_max:.2f}s h2d={h2d_max:.2f}s "
                f"({_gib_per_sec(actual_payload, read_total_max):.2f} GiB/s per replica, "
                f"agg {_gib_per_sec(actual_payload * num_receivers, read_total_max):.2f} GiB/s) "
                f"e2e={e2e:.2f}s"
            )
        if it == warmup + iters - 1:
            for r in read_res:
                assert r["num_weights"] == len(specs), \
                    f"reader rank {r['rank']} got {r['num_weights']}/{len(specs)} weights"
                for name, vals in r["checksums"].items():
                    assert vals == expected[name], f"reader rank {r['rank']} mismatch on {name}: {vals}"
            print(f"[fs] correctness OK ({len(specs)} weights x {num_receivers} replicas verified)")

        if cleanup:
            _rmtree_background(d)

    if results["iterations"]:
        def _mean(key):
            return float(np.mean([r[key] for r in results["iterations"]]))

        results["summary"] = {
            "stage_d2h_seconds_mean": _mean("stage_d2h_max"),
            "write_seconds_mean": _mean("write_max"),
            "write_gib_per_s_mean": _gib_per_sec(actual_payload, _mean("write_max")),
            "publish_seconds_mean": _mean("publish_seconds"),
            "visibility_wait_seconds_mean": _mean("visibility_wait_max"),
            "read_cpu_seconds_mean": _mean("read_cpu_max"),
            "stage_h2d_seconds_mean": _mean("stage_h2d_max"),
            "read_seconds_mean": _mean("read_total_max"),
            "read_gib_per_s_mean": _gib_per_sec(actual_payload, _mean("read_total_max")),
            "read_aggregate_gib_per_s_mean": _gib_per_sec(
                actual_payload * num_receivers, _mean("read_total_max")),
            "e2e_seconds_mean": _mean("e2e"),
        }

    _print_summary(results)
    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[result] json written to {json_out}")
    if tmp_fs_dir is not None and cleanup:
        shutil.rmtree(tmp_fs_dir, ignore_errors=True)
    return results


def _print_summary(results: dict) -> None:
    s = results.get("summary")
    print("\n" + "=" * 90)
    print(f" filesystem weight-sync benchmark summary  (payload {results['payload_gib']:.2f} GiB, "
          f"{results['num_senders']} writers -> {results['num_receivers']} replicas)")
    print(f" fs_dir={results['fs_dir']}  pin_memory={results['pin_memory']}  gpu_stage={results['gpu_stage']}")
    print("=" * 90)
    if s is None:
        print(" no completed iterations")
        print("=" * 90)
        return
    rows = [
        ("GPU->pinned host (d2h)", s["stage_d2h_seconds_mean"], None),
        ("parallel write (aggregate)", s["write_seconds_mean"], s["write_gib_per_s_mean"]),
        ("publish (index+marker+fsync)", s["publish_seconds_mean"], None),
        ("marker visibility wait", s["visibility_wait_seconds_mean"], None),
        ("mmap + page-in (read_cpu)", s["read_cpu_seconds_mean"], None),
        ("CPU->GPU stage (h2d)", s["stage_h2d_seconds_mean"], None),
        ("read total per replica", s["read_seconds_mean"], s["read_gib_per_s_mean"]),
        ("read aggregate all replicas", s["read_seconds_mean"], s["read_aggregate_gib_per_s_mean"]),
        ("end-to-end sync", s["e2e_seconds_mean"],
         _gib_per_sec(results["payload_gib"] * GIB, s["e2e_seconds_mean"])),
    ]
    print(f" {'phase':<32} {'seconds':>10} {'GiB/s':>10}")
    print("-" * 90)
    for name, sec, bw in rows:
        bw_s = f"{bw:.2f}" if bw is not None else "-"
        print(f" {name:<32} {sec:>10.2f} {bw_s:>10}")
    print("=" * 90)
    print(" 'end-to-end sync' is the driver wall clock, dispatch -> all replicas hold the full")
    print(" payload on GPU; compare it against the e2e column in bench_checkpoint_engine.py.")
    print(" Note the intended asymmetry: here N writers each persist 1/N of the payload in")
    print(" parallel (FSDP stripes), whereas the checkpoint engine ships one full copy from")
    print(" the root rank. Both mirror their own production path, so e2e is the fair metric.")


def main():
    parser = argparse.ArgumentParser(description="alpha-seed filesystem weight transfer benchmark")
    parser.add_argument("--fs-dir", default=None,
                        help="shared CFS/HDFS-fuse/FAFUSE mount root (required for cross-node runs)")
    parser.add_argument("--total-gb", type=float, default=4.0)
    parser.add_argument("--num-tensors", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-senders", type=int, default=4, help="writer ranks = safetensors shards")
    parser.add_argument("--num-receivers", type=int, default=4, help="reader replicas, each reads ALL shards")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--gpu-stage", action="store_true", default=True,
                        help="stage mmaped weights onto GPU (mirrors real rollout reload)")
    parser.add_argument("--no-gpu-stage", dest="gpu_stage", action="store_false",
                        help="measure pure filesystem read, skip CPU->GPU upload")
    parser.add_argument("--trial-id", default="bench")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--cleanup", action="store_true", default=True)
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    parser.add_argument("--allow-single-node", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--cpu-selftest", action="store_true",
                        help="CPU-only round-trip check without Ray/GPU/cluster, then exit")
    args = parser.parse_args()
    if args.cpu_selftest:
        cpu_selftest()
        return
    run_benchmark(
        fs_dir=args.fs_dir,
        total_gb=args.total_gb,
        num_tensors=args.num_tensors,
        iters=args.iters,
        warmup=args.warmup,
        num_senders=args.num_senders,
        num_receivers=args.num_receivers,
        dtype=args.dtype,
        pin_memory=args.pin_memory,
        gpu_stage=args.gpu_stage,
        trial_id=args.trial_id,
        run_id=args.run_id,
        cleanup=args.cleanup,
        allow_single_node=args.allow_single_node,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
