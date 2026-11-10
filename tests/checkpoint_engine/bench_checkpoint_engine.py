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
"""Benchmark the verl checkpoint-engine weight-transfer path ONLY.

This is the online actor -> rollout weight transport implemented in
``verl.checkpoint_engine`` (it never touches a filesystem): every selected
backend is driven through its real public lifecycle
``prepare -> build_topology -> init_process_group -> send_weights /
receive_weights -> finalize``.

Backends:
  * ``naive``    -- same-process / CUDA-IPC copy, single node only (colocated);
  * ``nccl``     -- NCCL broadcast relayed over NVLink/NVSwitch;
  * ``nixl``     -- ring p2p over UCX/RDMA;
  * ``mooncake`` -- ring p2p over the Mooncake transfer engine;
  * ``kimi_ckpt_engine`` / ``delta_sharded`` -- registered when their extra
    deps are importable, otherwise the registry reports them as missing and
    this script SKIPs them.

Typical fabric-realistic setup: 2 nodes x N GPUs, N senders on node 0 and
N receivers on node 1, so every payload byte crosses the fabric exactly once.

Run on a Ray cluster (``ray start --head`` on node 0,
``ray start --address=...`` on node 1) from the repo root::

    python tests/checkpoint_engine/bench_checkpoint_engine.py \
        --backends nccl nixl mooncake \
        --total-gb 8 --iters 5 --warmup 1

Optional JSON dump for side-by-side comparison with the filesystem benchmark::

    python tests/checkpoint_engine/bench_checkpoint_engine.py \
        --backends nccl nixl mooncake --total-gb 8 \
        --json-out /tmp/ce_result.json

The companion ``bench_fs_transfer.py`` measures the alpha-seed / mono_rl
shared-filesystem path on the same payload; the two scripts are intentionally
independent so each can run inside its own image (verl cu130 image here,
alpha-seed th29 image there).
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
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


def spot_check_positions(numel: int):
    """Cheap, deterministic set of positions to verify end to end."""
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


def iter_weight_tensors(specs, device: str):
    """Generator yielding ``(name, tensor)`` matching the engines' expected input."""
    for i, spec in enumerate(specs):
        t = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        fill_pattern(t, i)
        yield spec.name, t
        del t


# ---------------------------------------------------------------------------
# Ray worker: one per GPU, drives one checkpoint-engine backend
# ---------------------------------------------------------------------------

try:
    import ray

    @ray.remote(num_gpus=1)
    class CheckpointEngineWorker:
        """Runs inside a Ray actor with exactly one GPU assigned."""

        def __init__(self, global_rank: int):
            self.global_rank = global_rank
            self.device = "cuda"
            torch.cuda.set_device(0)
            self.engine = None
            self.backend = None

        def info(self) -> dict:
            import importlib
            import socket

            deps = {}
            for m in ("torch", "ray", "cupy", "zmq", "tensordict", "nixl", "mooncake", "verl"):
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

        def engine_create(self, backend: str, bucket_bytes: int) -> None:
            from verl.checkpoint_engine import CheckpointEngineRegistry

            os.environ["RANK"] = str(self.global_rank)
            self.backend = backend
            engine_kwargs = {"nccl": {"multi_sender": True, "rebuild_group": False}}.get(backend, {})
            if backend in ("nixl", "mooncake"):
                engine_kwargs["device"] = "cuda"
            self.engine = CheckpointEngineRegistry.new(
                backend,
                bucket_size=bucket_bytes,
                is_master=(self.global_rank == 0),
                **engine_kwargs,
            )

        def engine_prepare(self):
            return self.engine.prepare()

        def engine_init(self, **kwargs) -> None:
            self.engine.init_process_group(**kwargs)

        async def engine_send(self, specs_serialized: list) -> dict:
            specs = [WeightSpec(s["name"], tuple(s["shape"]), _DTYPE_MAP[s["dtype"]]) for s in specs_serialized]

            def gen():
                for name, t in iter_weight_tensors(specs, self.device):
                    yield name, t

            total_bytes = sum(s.nbytes for s in specs)
            start = time.time()
            await self.engine.send_weights(gen())
            torch.cuda.synchronize()
            return {
                "rank": self.global_rank,
                "role": "send",
                "seconds": time.time() - start,
                "bytes": total_bytes,
            }

        async def engine_recv(self, specs_serialized: list) -> dict:
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
                except Exception as e:  # noqa: BLE001
                    print(f"[rank {self.global_rank}] finalize warning: {type(e).__name__}: {str(e)[:200]}")
                self.engine = None
            torch.cuda.empty_cache()

except ImportError:
    ray = None
    CheckpointEngineWorker = None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _specs_to_json(specs):
    dtype_inv = {v: k for k, v in _DTYPE_MAP.items()}
    return [{"name": s.name, "shape": list(s.shape), "dtype": dtype_inv[s.dtype]} for s in specs]


def _expected_checksums(specs):
    return {s.name: expected_spot_values(s, i) for i, s in enumerate(specs)}


def _gib_per_sec(num_bytes: float, seconds: float) -> float:
    return num_bytes / GIB / max(seconds, 1e-9)


def _init_ray():
    """Connect to Ray with the same torch-free worker bootstrap the combined
    benchmark uses (sys.path fixup + per-node cupy/nixl/mooncake install)."""
    if ray is None:
        raise RuntimeError("ray is not installed")

    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")

    setup_mod = None
    setup_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ws_setup.py")
    setup_pymod_dir = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _ws_setup as setup_mod  # noqa: PLC0415

        # Pickle-by-value: the setup hook is deserialized before py_modules are
        # importable, so it cannot import _ws_setup by reference at that point.
        try:
            import cloudpickle as _cp

            _cp.register_pickle_by_value(setup_mod)
        except Exception as _e:  # noqa: BLE001
            print(f"[setup] note: standalone cloudpickle register skipped: {_e}")
        try:
            import ray.cloudpickle as _ray_cp

            _ray_cp.register_pickle_by_value(setup_mod)
        except Exception:  # noqa: BLE001
            pass

        setup_pymod_dir = tempfile.mkdtemp(prefix="ws_setup_pkg_")
        shutil.copy(setup_py, os.path.join(setup_pymod_dir, "_ws_setup.py"))
        print(f"[setup] _ws_setup loaded from {setup_py}; py_modules dir={setup_pymod_dir}")
    except Exception as e:  # noqa: BLE001
        print(f"[setup] WARNING: _ws_setup not importable ({type(e).__name__}: {e})")

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
    # WS_* must reach the worker pods: they decide where _ws_setup installs and
    # which torch spec it pins, and a worker that disagrees with the head would
    # build a different environment than the one under test.
    for inherit in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH",
                    "WS_DEPS_DIR", "WS_TORCH_SPEC", "WS_PIP_PROXY"):
        if inherit in os.environ and os.environ[inherit]:
            env_vars[inherit] = os.environ[inherit]
    runtime_env = {"py_executable": sys.executable, "env_vars": env_vars}
    if setup_mod is not None and setup_pymod_dir:
        runtime_env["worker_process_setup_hook"] = setup_mod.worker_process_setup_hook
        runtime_env["py_modules"] = [setup_pymod_dir]
    try:
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except (TypeError, ValueError) as e:
        print(f"[setup] WARNING: full runtime_env ray.init failed ({type(e).__name__}: {str(e)[:200]}); "
              "retrying env-only")
        runtime_env = {"env_vars": env_vars}
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    return setup_mod


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
                f"need >=2 GPU nodes (senders and receivers on different nodes), found {len(gpu_nodes)}; "
                "pass --allow-single-node for an intra-node smoke run (only meaningful with --backends naive)"
            )
        return gpu_nodes[0][0], gpu_nodes[0][0]
    return gpu_nodes[0][0], gpu_nodes[1][0]


def run_benchmark(
    backends,
    total_gb: float = 4.0,
    num_tensors: int = 64,
    iters: int = 3,
    warmup: int = 1,
    bucket_mb: int = 2048,
    num_senders: int = 4,
    num_receivers: int = 4,
    dtype: str = "bf16",
    allow_single_node: bool = False,
    json_out: Optional[str] = None,
) -> dict:
    setup_mod = _init_ray()

    node_sender, node_receiver = _select_nodes(num_senders, num_receivers, allow_single_node)
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Per-node bootstrap: probe env and install cupy/nixl/mooncake on every node
    # (the job entrypoint pip-installs on head only).
    if setup_mod is not None and getattr(setup_mod, "probe", None) is not None:
        try:
            boot_nodes = sorted({node_sender, node_receiver})
            print(f"[bootstrap] probing {len(boot_nodes)} nodes before actor creation ...")
            probe_refs = [
                setup_mod.probe.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
                ).remote()
                for nid in boot_nodes
            ]
            for p in ray.get(probe_refs, timeout=300):
                print(f"[probe] host={p['hostname']} py={p['python']} torch_cuda={p.get('torch_cuda')}")
            print("[bootstrap] installing RDMA deps on each node (cupy can take a few minutes) ...")
            deps_refs = [
                setup_mod.ensure_deps.options(
                    num_cpus=0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
                ).remote()
                for nid in boot_nodes
            ]
            for rep in ray.get(deps_refs, timeout=3000):
                host = rep.get("probe", {}).get("hostname", "?")
                print(f"[deps] node={host} target={rep.get('target')} cached={rep.get('cached', False)}")
                ce = rep.get("verl_checkpoint_engine", {})
                print(f"[deps]   verl.checkpoint_engine ok={ce.get('ok')} "
                      f"{'' if ce.get('ok') else str(ce.get('error'))[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"[bootstrap] WARNING: per-node bootstrap error: {type(e).__name__}: {str(e)[:300]}")

    senders = [
        CheckpointEngineWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_sender, soft=False)
        ).remote(global_rank=i)
        for i in range(num_senders)
    ]
    receivers = [
        CheckpointEngineWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_receiver, soft=False)
        ).remote(global_rank=num_senders + i)
        for i in range(num_receivers)
    ]

    infos = ray.get([w.info.remote() for w in senders + receivers])
    sender_infos, receiver_infos = infos[:num_senders], infos[num_senders:]
    print(f"[setup] senders node={sender_infos[0]['node_id'][:8]} gpu={sender_infos[0]['gpu']}")
    print(f"[setup] receivers node={receiver_infos[0]['node_id'][:8]} gpu={receiver_infos[0]['gpu']}")
    if sender_infos[0]["node_id"] == receiver_infos[0]["node_id"]:
        print("[setup] WARNING: senders and receivers share one node; cross-fabric path not exercised")
    print(f"[setup] sender deps:   " + " ".join(f"{m}={v}" for m, v in sender_infos[0]["deps"].items()))
    print(f"[setup] receiver deps: " + " ".join(f"{m}={v}" for m, v in receiver_infos[0]["deps"].items()))

    torch_dtype = _DTYPE_MAP[dtype]
    total_bytes = int(total_gb * GIB)
    specs = build_weight_specs(total_bytes, num_tensors, torch_dtype)
    specs_json = _specs_to_json(specs)
    expected = _expected_checksums(specs)
    actual_payload = sum(s.nbytes for s in specs)
    print(f"[setup] payload={actual_payload / GIB:.2f} GiB across {len(specs)} tensors, dtype={dtype}")

    from verl.checkpoint_engine import CheckpointEngineRegistry

    results = {
        "kind": "verl_checkpoint_engine",
        "payload_gib": actual_payload / GIB,
        "num_senders": num_senders,
        "num_receivers": num_receivers,
        "bucket_mb": bucket_mb,
        "backends": {},
    }

    for backend in backends:
        try:
            engine_cls = CheckpointEngineRegistry.get(backend)
        except ValueError as e:
            print(f"[{backend}] SKIP: {e}")
            results["backends"][backend] = {"error": str(e)}
            continue
        if backend == "naive" and node_sender != node_receiver:
            print("[naive] SKIP: naive backend is same-node only; rerun with --allow-single-node")
            results["backends"][backend] = {"error": "naive is same-node only"}
            continue

        all_workers = senders + receivers
        iter_records = []
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

                    # e2e is the driver wall clock from dispatch to "every
                    # receiver holds the full payload on GPU" -- the same
                    # boundary bench_fs_transfer.py reports, so the two
                    # benchmarks' headline numbers are directly comparable.
                    t0 = time.time()
                    recv_refs = [w.engine_recv.remote(specs_json) for w in receivers]
                    send_refs = [w.engine_send.remote(specs_json) for w in senders]
                    send_res, recv_res = ray.get(send_refs), ray.get(recv_refs)
                    e2e = time.time() - t0

                    recv_max = max(r["seconds"] for r in recv_res)
                    send_max = max(r["seconds"] for r in send_res)
                    if it >= warmup:
                        iter_records.append({
                            "iter": it - warmup + 1,
                            "send_max": send_max,
                            "recv_max": recv_max,
                            "e2e": e2e,
                            "gib_per_s": _gib_per_sec(actual_payload, recv_max),
                            "aggregate_gib_per_s": _gib_per_sec(actual_payload * num_receivers, recv_max),
                        })
                        print(
                            f"[{backend}] iter {it - warmup + 1}/{iters}: "
                            f"send={send_max:.3f}s recv={recv_max:.3f}s e2e={e2e:.3f}s "
                            f"bw={_gib_per_sec(actual_payload, recv_max):.2f} GiB/s per receiver "
                            f"(aggregate {_gib_per_sec(actual_payload * num_receivers, recv_max):.2f} GiB/s)"
                        )
                    if it == warmup + iters - 1:
                        for r in recv_res:
                            assert r["num_weights"] == len(specs), \
                                f"{backend} rank {r['rank']} got {r['num_weights']}/{len(specs)} weights"
                            for name, vals in r["checksums"].items():
                                assert vals == expected[name], \
                                    f"{backend} rank {r['rank']} checksum mismatch on {name}: {vals}"
                        print(f"[{backend}] correctness OK ({len(specs)} weights x {num_receivers} receivers)")
            finally:
                try:
                    ray.get([w.engine_finalize.remote() for w in all_workers])
                except Exception as fe:  # noqa: BLE001
                    print(f"[{backend}] finalize warning: {type(fe).__name__}: {str(fe)[:200]}")
        except Exception as e:  # noqa: BLE001
            backend_error = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"[{backend}] ERROR (skipping backend, continuing): {backend_error}")
            results["backends"][backend] = {"error": backend_error}
            continue

        if iter_records:
            recv_times = [t["recv_max"] for t in iter_records]
            send_times = [t["send_max"] for t in iter_records]
            e2e_times = [t["e2e"] for t in iter_records]
            results["backends"][backend] = {
                "send_seconds_mean": float(np.mean(send_times)),
                "recv_seconds_mean": float(np.mean(recv_times)),
                "recv_seconds_max": float(np.max(recv_times)),
                "e2e_seconds_mean": float(np.mean(e2e_times)),
                "gib_per_s_mean": _gib_per_sec(actual_payload, np.mean(recv_times)),
                "aggregate_gib_per_s_mean": _gib_per_sec(actual_payload * num_receivers, np.mean(recv_times)),
                "e2e_gib_per_s_mean": _gib_per_sec(actual_payload, np.mean(e2e_times)),
                "iterations": iter_records,
            }
        elif backend_error is None:
            results["backends"][backend] = {"error": "no completed iterations"}

    _print_summary(results)
    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[result] json written to {json_out}")
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 86)
    print(f" checkpoint-engine benchmark summary  (payload {results['payload_gib']:.2f} GiB, "
          f"{results['num_senders']} senders -> {results['num_receivers']} receivers)")
    print("=" * 86)
    print(f" {'backend':<18} {'recv s (mean)':>14} {'e2e s (mean)':>14} "
          f"{'GiB/s/receiver':>16} {'GiB/s aggregate':>18}")
    print("-" * 86)
    for backend, r in results["backends"].items():
        if "error" in r:
            print(f" {backend:<18} {'-':>14} {'-':>14} {'-':>16} {'-':>18}  "
                  f"ERROR/SKIP: {r['error'][:60]}")
            continue
        print(
            f" {backend:<18} {r['recv_seconds_mean']:>14.3f} {r['e2e_seconds_mean']:>14.3f} "
            f"{r['gib_per_s_mean']:>16.2f} {r['aggregate_gib_per_s_mean']:>18.2f}"
        )
    print("=" * 86)
    print(" e2e = driver wall clock, dispatch -> all receivers hold the full payload on GPU;")
    print("       compare directly against 'end-to-end sync' in bench_fs_transfer.py.")


def main():
    parser = argparse.ArgumentParser(description="verl checkpoint-engine weight transfer benchmark")
    parser.add_argument("--backends", nargs="+", default=["nccl", "nixl", "mooncake"],
                        help="registered checkpoint-engine backends (naive is single-node only)")
    parser.add_argument("--total-gb", type=float, default=4.0)
    parser.add_argument("--num-tensors", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3, help="measured iterations after warmup")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--bucket-mb", type=int, default=2048)
    parser.add_argument("--num-senders", type=int, default=4)
    parser.add_argument("--num-receivers", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP))
    parser.add_argument("--allow-single-node", action="store_true")
    parser.add_argument("--json-out", default=None, help="optional path for machine-readable results")
    args = parser.parse_args()
    run_benchmark(
        backends=args.backends,
        total_gb=args.total_gb,
        num_tensors=args.num_tensors,
        iters=args.iters,
        warmup=args.warmup,
        bucket_mb=args.bucket_mb,
        num_senders=args.num_senders,
        num_receivers=args.num_receivers,
        dtype=args.dtype,
        allow_single_node=args.allow_single_node,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
