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
"""Declarative, three-layer device topology for RL roles (RFC issue #7269).

This module is intentionally dependency-light (standard library only) so the
data model, pool allocation, mode derivation, validation and reporting can be
imported and unit-tested without Ray/torch. The trainer-facing resolver that
turns a hydra ``topology:`` block (or the legacy knobs) into
``resource_pool_spec`` / ``mapping`` lives in
``verl/trainer/ppo/topology_resolver.py``.

Three layers::

    topology:
      clusters:                                              # L1 hardware
        - {name: default, nnodes: 2, n_gpus_per_node: 8}
      device_pools:                                          # L2 GPU groups
        - {name: hybrid_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}
      models:                                                # L3 model instances
        - {name: actor,   worker: actor,   config_key: actor_rollout_ref, resource_pool: hybrid_pool}
        - {name: rollout, worker: rollout, config_key: actor_rollout_ref, resource_pool: hybrid_pool}
        - {name: ref,     worker: ref,     config_key: actor_rollout_ref, resource_pool: hybrid_pool}
        - {name: critic,  worker: critic,  config_key: critic,            resource_pool: hybrid_pool}

* **Cluster** (L1): one homogeneous machine group ``{name, nnodes,
  n_gpus_per_node}``. Its ``name`` doubles as the Ray custom resource that pins
  its machines, which is what makes heterogeneous hardware expressible. Each
  cluster owns a flat GPU index space ``0 .. nnodes*n_gpus_per_node - 1``; flat
  id ``g`` lives on node ``g // n_gpus_per_node``.
* **DevicePool** (L2): a named group of GPUs carved from exactly one cluster,
  sized ``{nnodes, n_gpus_per_node}`` and allocated node-by-node in declaration
  order (so pools within a cluster are disjoint).
* **Model** (L3): one model instance ``{name, worker, config_key,
  resource_pool, device_range?}``. Models that share a ``(config_key,
  resource_pool, device_range)`` are *fused* into a single worker process.

``RolloutMode`` (hybrid / colocated / standalone) is *derived* from placement,
never hand-written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "RolloutMode",
    "Cluster",
    "Device",
    "DevicePool",
    "ModelPlacement",
    "FusionGroup",
    "ClusterTopology",
    "build_topology",
    "build_topology_from_models",
    "parse_device_range",
    "compact_gpu_ids",
]

# Workers that host an inference server whose placement selects a RolloutMode.
_SERVER_WORKERS = ("rollout", "rm", "teacher")
# The trainable policy worker (owns the actor pool).
_ACTOR_WORKERS = ("actor",)
# Recognized L3 worker names.
_KNOWN_WORKERS = ("actor", "rollout", "ref", "critic", "rm", "teacher", "env")
# Fallback config subtree for a worker when ``config_key`` is omitted.
_DEFAULT_CONFIG_KEY = {
    "actor": "actor_rollout_ref",
    "rollout": "actor_rollout_ref",
    "ref": "actor_rollout_ref",
    "critic": "critic",
    "rm": "reward.reward_model",
    "teacher": "distillation",
    "env": "env",
}
# Pool attribute keys the framework reserves for future enforcement (carried
# through and surfaced in the report today; see RFC section 6).
_RESERVED_POOL_ATTRS = {
    "elastic": bool,
    "min_gpus": int,
    "max_gpus": int,
    "scale_signal": str,
    "standby": bool,
    "fault_tolerant": bool,
    "protects": list,
}


class RolloutMode(Enum):
    """Canonical placement primitive shared by the rollout stack and topology.

    Defined here (rather than in ``verl/workers/rollout/replica.py``) so it is
    importable without Ray; ``replica.py`` re-exports it for backward
    compatibility.
    """

    # Rollout engine and training engine (fsdp/megatron) fused in same process.
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but
    # in separate process. Rollout and hybrid processes share GPUs, switch context
    # without weight synchronization. Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated
    # architecture. Usage scenarios: off-policy training.
    STANDALONE = "standalone"


@dataclass(frozen=True)
class Cluster:
    """L1: one homogeneous machine group; ``name`` is also the Ray resource key."""

    name: str
    nnodes: int
    n_gpus_per_node: int

    @property
    def size(self) -> int:
        return self.nnodes * self.n_gpus_per_node


@dataclass(frozen=True)
class Device:
    """A single GPU, addressed by its flat index *within* its cluster."""

    cluster: str
    gpu: int
    node: Optional[int] = None
    local_gpu: Optional[int] = None
    node_id: Optional[str] = None
    node_ip: Optional[str] = None


@dataclass
class DevicePool:
    """L2: a named GPU group carved from one cluster, sized by nnodes x n_gpus_per_node."""

    name: str
    cluster: str
    nnodes: int
    n_gpus_per_node: int
    gpus: list[int]  # resolved flat cluster GPU ids
    attributes: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.nnodes * self.n_gpus_per_node

    def process_on_nodes(self, cluster_n_gpus_per_node: int) -> list[int]:
        """Per-node GPU counts across the nodes this pool touches (the ``ResourcePoolManager`` shape)."""
        by_node: dict[int, int] = {}
        for g in self.gpus:
            by_node[g // cluster_n_gpus_per_node] = by_node.get(g // cluster_n_gpus_per_node, 0) + 1
        return [by_node[n] for n in sorted(by_node)]


@dataclass
class ModelPlacement:
    """L3: one model instance placed on a pool (optionally a device_range slice of it)."""

    name: str
    worker: str
    config_key: str
    pool: str
    device_range: Optional[tuple[int, int]] = None
    gpus: list[int] = field(default_factory=list)
    devices: list[Device] = field(default_factory=list)
    rollout_mode: Optional[RolloutMode] = None
    group: int = 0


@dataclass
class FusionGroup:
    """One worker process: the models fused by a shared (config_key, pool, device_range)."""

    index: int
    config_key: str
    pool: str
    device_range: Optional[tuple[int, int]]
    models: list[ModelPlacement] = field(default_factory=list)

    @property
    def workers(self) -> list[str]:
        return [m.worker for m in self.models]

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.models]

    @property
    def gpus(self) -> list[int]:
        return list(self.models[0].gpus) if self.models else []

    @property
    def is_fused(self) -> bool:
        return len(self.models) > 1

    @property
    def rollout_mode(self) -> Optional[RolloutMode]:
        for m in self.models:
            if m.rollout_mode is not None:
                return m.rollout_mode
        return None


@dataclass
class ClusterTopology:
    """The resolved, validated topology plus queries and reporting."""

    clusters: dict[str, Cluster]
    pools: dict[str, DevicePool]
    models: list[ModelPlacement]
    groups: list[FusionGroup] = field(default_factory=list)

    # convenience alias so callers reading "placements" still work
    @property
    def placements(self) -> list[ModelPlacement]:
        return self.models

    # ------------------------------ queries ------------------------------
    def model_of_worker(self, worker: str) -> Optional[ModelPlacement]:
        for model in self.models:
            if model.worker == worker:
                return model
        return None

    def pool_of(self, worker: str) -> Optional[str]:
        model = self.model_of_worker(worker)
        return model.pool if model else None

    def actor_pool(self) -> Optional[str]:
        for model in self.models:
            if model.worker in _ACTOR_WORKERS:
                return model.pool
        return None

    def actor_group(self) -> Optional[FusionGroup]:
        for model in self.models:
            if model.worker in _ACTOR_WORKERS:
                return self.groups[model.group]
        return None

    def worker_pool_names(self) -> dict[str, str]:
        """Raw ``worker -> pool name`` (first occurrence of each worker)."""
        out: dict[str, str] = {}
        for model in self.models:
            out.setdefault(model.worker, model.pool)
        return out

    def resource_pool_spec(self, include_roleless: bool = False) -> dict[str, list[int]]:
        """Emit ``{pool: process_on_nodes}`` (per-node GPU counts) for model-bearing pools.

        This is exactly the ``resource_pool_spec`` shape ``ResourcePoolManager`` consumes.
        """
        pools_with_models = {m.pool for m in self.models}
        spec: dict[str, list[int]] = {}
        for name, pool in self.pools.items():
            if not include_roleless and name not in pools_with_models:
                continue
            cluster = self.clusters[pool.cluster]
            spec[name] = pool.process_on_nodes(cluster.n_gpus_per_node)
        return spec

    def accelerator_type_map(self) -> dict[str, str]:
        """``{pool: cluster}`` used as the Ray custom-resource key, only when heterogeneous.

        A single (implicit) cluster returns ``{}`` so single-cluster runs keep today's
        behavior (no custom-resource demand threaded into placement groups).
        """
        if len(self.clusters) <= 1:
            return {}
        return {name: pool.cluster for name, pool in self.pools.items()}

    def process_rows(self) -> list[dict]:
        """One row per worker process (fusion group), in declaration order."""
        rows: list[dict] = []
        for group in self.groups:
            pool = self.pools[group.pool]
            rows.append(
                {
                    "cluster": pool.cluster,
                    "pool": group.pool,
                    "workers": list(group.workers),
                    "names": list(group.names),
                    "config_key": group.config_key,
                    "index": group.index,
                    "process": _process_label(self, group),
                    "mode": group.rollout_mode,
                    "gpus": list(group.gpus),
                    "device_range": group.device_range,
                }
            )
        return rows

    # ------------------------------ reporting ------------------------------
    def to_dict(self) -> dict:
        return {
            "clusters": {
                name: {"nnodes": c.nnodes, "n_gpus_per_node": c.n_gpus_per_node, "size": c.size}
                for name, c in self.clusters.items()
            },
            "device_pools": {
                name: {
                    "cluster": p.cluster,
                    "nnodes": p.nnodes,
                    "n_gpus_per_node": p.n_gpus_per_node,
                    "gpus": list(p.gpus),
                    "attributes": dict(p.attributes),
                }
                for name, p in self.pools.items()
            },
            "processes": [
                {
                    "cluster": row["cluster"],
                    "pool": row["pool"],
                    "workers": row["workers"],
                    "names": row["names"],
                    "config_key": row["config_key"],
                    "process": row["process"],
                    "mode": row["mode"].value if row["mode"] is not None else None,
                    "gpus": row["gpus"],
                }
                for row in self.process_rows()
            ],
        }

    def describe(self) -> str:
        rows = self.process_rows()
        header = ["CLUSTER", "POOL", "GPUS", "NODE:local", "WORKERS", "CONFIG_KEY", "PROCESS", "ROLLOUT MODE"]
        table: list[list[str]] = [header]
        for row in rows:
            cluster = self.clusters[row["cluster"]]
            table.append(
                [
                    row["cluster"],
                    row["pool"],
                    compact_gpu_ids(row["gpus"]),
                    _node_local(row["gpus"], cluster.n_gpus_per_node),
                    "+".join(row["workers"]),
                    row["config_key"],
                    row["process"],
                    row["mode"].value.upper() if row["mode"] is not None else _dedicated_or_dash(row),
                ]
            )
        widths = [max(len(r[i]) for r in table) for i in range(len(header))]
        lines = ["Resolved topology"]
        for r in table:
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip())

        attr_lines = [
            f"  {name} -> {_format_attrs(pool.attributes)}" for name, pool in self.pools.items() if pool.attributes
        ]
        if attr_lines:
            lines.append("pool attributes:")
            lines.extend(attr_lines)

        warnings = self.validate()
        lines.append("Warnings: " + ("none" if not warnings else ""))
        lines.extend(f"  - {w}" for w in warnings)
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for cname, cluster in self.clusters.items():
            lines.append(f'  cluster_{_san(cname)}["{cname} ({cluster.nnodes}x{cluster.n_gpus_per_node} GPU)"]')
        for pname, pool in self.pools.items():
            lines.append(f'  pool_{_san(pname)}["{pname}: {compact_gpu_ids(pool.gpus)}"]')
            lines.append(f"  cluster_{_san(pool.cluster)} --> pool_{_san(pname)}")
        for row in self.process_rows():
            label = "+".join(row["workers"])
            mode = f" ({row['mode'].value})" if row["mode"] is not None else ""
            node = f"proc_{row['index']}"
            lines.append(f'  {node}["{label}{mode}"]')
            lines.append(f"  pool_{_san(row['pool'])} --> {node}")
        return "\n".join(lines)

    # ------------------------------ validation ------------------------------
    def validate(self) -> list[str]:
        """Return non-fatal warnings. Fatal errors are raised during ``build_topology``."""
        warnings: list[str] = []
        actor_pool = self.actor_pool()
        actor_gpus = _pool_gpu_set(self.pools[actor_pool]) if actor_pool else set()

        pools_with_models = {m.pool for m in self.models}
        for name, pool in self.pools.items():
            # roleless pool: allowed only as an explicit reserve (standby / fault_tolerant)
            if name not in pools_with_models:
                if not (pool.attributes.get("standby") or pool.attributes.get("fault_tolerant")):
                    warnings.append(f"pool '{name}' has no models and no reserve attribute; its GPUs are idle")

            cluster = self.clusters[pool.cluster]
            # a pool spanning >1 node but not filling whole nodes straddles a boundary
            nodes = {g // cluster.n_gpus_per_node for g in pool.gpus}
            if len(nodes) > 1 and (len(pool.gpus) % cluster.n_gpus_per_node) != 0:
                warnings.append(f"pool '{name}' spans nodes {sorted(nodes)} but does not fill whole nodes")

            # partial overlap with the actor pool (some but not all GPUs shared)
            if actor_pool and name != actor_pool:
                overlap = _pool_gpu_set(pool) & actor_gpus
                if overlap and overlap != _pool_gpu_set(pool) and overlap != actor_gpus:
                    warnings.append(f"pool '{name}' partially overlaps actor pool '{actor_pool}'")

            # attribute hygiene
            for key, value in pool.attributes.items():
                if key not in _RESERVED_POOL_ATTRS:
                    warnings.append(f"pool '{name}' has unknown attribute '{key}' (carried through, not enforced)")
                else:
                    expected = _RESERVED_POOL_ATTRS[key]
                    if not isinstance(value, expected):
                        warnings.append(
                            f"pool '{name}' attribute '{key}' expected {expected.__name__}, got {type(value).__name__}"
                        )
        return warnings

    def require_ray_resources(self) -> None:
        """Assert each multi-cluster name exists as a Ray custom resource.

        No-op when Ray is unavailable or when there is a single (implicit)
        cluster, so it is safe to call unconditionally at startup.
        """
        if len(self.clusters) <= 1:
            return
        try:
            import ray  # noqa: PLC0415
        except Exception:
            return
        if not ray.is_initialized():
            return
        available = ray.cluster_resources()
        for name, cluster in self.clusters.items():
            have = available.get(name, 0)
            if have < cluster.size:
                raise ValueError(
                    f"cluster '{name}' needs {cluster.size} units of Ray custom resource '{name}', "
                    f"found {have}. Start those machines with "
                    f"ray start --resources '{{\"{name}\": {cluster.n_gpus_per_node}}}'."
                )


# =============================== builders ===============================
def build_topology(
    clusters: list | dict | None,
    device_pools: list | dict | None,
    models: list | None,
) -> ClusterTopology:
    """Parse + resolve + derive a :class:`ClusterTopology`.

    Raises ``ValueError`` on fatal structural problems (unknown references,
    an unknown worker name, a pool that does not fit its cluster, an
    out-of-range ``device_range``, or a policy rollout placed as a separate
    process on the actor's GPUs).
    """
    clusters_map = _normalize_clusters(clusters)
    pools_in = _normalize_pools(device_pools)
    models_in = _normalize_models(models)

    pool_objs = _resolve_pools(clusters_map, pools_in)
    placements = _build_placements(clusters_map, pool_objs, models_in)
    groups = _build_groups(placements)
    _derive_modes(pool_objs, placements, groups)

    return ClusterTopology(clusters=clusters_map, pools=pool_objs, models=placements, groups=groups)


def build_topology_from_models(
    resource_pool_spec: dict[str, list[int]],
    model_entries: list,
    n_gpus_per_node: int,
    cluster_name: str = "default",
) -> ClusterTopology:
    """Build a single (implicit) cluster topology from a legacy ``resource_pool_spec`` + models.

    Used by the legacy adapter (zero-migration report path): each pool becomes a
    node-shaped :class:`DevicePool` tiled sequentially in one homogeneous cluster, and
    ``model_entries`` are the reconstructed L3 models (``{name, worker, config_key,
    resource_pool}``).
    """
    ngpn = int(n_gpus_per_node) if n_gpus_per_node else 1
    total_nodes = sum(len(procs) for procs in resource_pool_spec.values()) or 1
    clusters = [{"name": cluster_name, "nnodes": total_nodes, "n_gpus_per_node": ngpn}]

    pools: list[dict] = []
    for name, procs in resource_pool_spec.items():
        procs = list(procs)
        if not procs or sum(procs) <= 0:
            continue
        pools.append(
            {
                "name": name,
                "cluster": cluster_name,
                "nnodes": len(procs),
                "n_gpus_per_node": procs[0],
            }
        )
    return build_topology(clusters, pools, model_entries)


def parse_device_range(device_range, pool_len: int) -> tuple[int, int]:
    """Resolve a model ``device_range`` into a ``[start, end)`` slice of its pool's GPU list.

    Accepts ``[start, end]`` (end exclusive) or a bare count ``n`` (== ``[0, n]``).
    """
    if isinstance(device_range, int):
        start, end = 0, int(device_range)
    else:
        seq = list(device_range)
        if len(seq) != 2:
            raise ValueError(f"device_range must be [start, end] or a count, got {device_range!r}")
        start, end = int(seq[0]), int(seq[1])
    if start < 0 or end > pool_len or start >= end:
        raise ValueError(f"device_range [{start}, {end}) out of the pool's {pool_len} GPU(s)")
    return start, end


def compact_gpu_ids(ids: list[int]) -> str:
    """Render sorted ids as compact ranges, e.g. ``[0..7, 16..23] -> '0-7,16-23'``."""
    if not ids:
        return "-"
    ids = sorted(ids)
    parts: list[str] = []
    start = prev = ids[0]
    for g in ids[1:]:
        if g == prev + 1:
            prev = g
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = g
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


# =============================== internals ===============================
def _normalize_clusters(clusters) -> dict[str, Cluster]:
    if not clusters:
        return {}
    if isinstance(clusters, dict):
        items = [{"name": name, **_as_dict(spec)} for name, spec in clusters.items()]
    else:
        items = [_as_dict(c) for c in clusters]
    out: dict[str, Cluster] = {}
    for spec in items:
        name = str(spec["name"])
        if name in out:
            raise ValueError(f"duplicate cluster name '{name}'")
        nnodes = int(spec["nnodes"])
        n_gpus_per_node = int(spec["n_gpus_per_node"])
        if nnodes <= 0 or n_gpus_per_node <= 0:
            raise ValueError(f"cluster '{name}' must have positive nnodes and n_gpus_per_node")
        out[name] = Cluster(name=name, nnodes=nnodes, n_gpus_per_node=n_gpus_per_node)
    return out


def _normalize_pools(pools) -> list[dict]:
    if not pools:
        return []
    if isinstance(pools, dict):
        items = [{"name": name, **_as_dict(spec)} for name, spec in pools.items()]
    else:
        items = [_as_dict(p) for p in pools]
    seen = set()
    for spec in items:
        name = str(spec["name"])
        if name in seen:
            raise ValueError(f"duplicate pool name '{name}'")
        seen.add(name)
    return items


def _normalize_models(models) -> list[dict]:
    if not models:
        return []
    out = []
    for entry in models:
        spec = _as_dict(entry)
        worker = spec.get("worker")
        if not worker:
            raise ValueError(f"model entry {spec!r} must set 'worker'")
        worker = str(worker)
        if worker not in _KNOWN_WORKERS:
            raise ValueError(f"model entry {spec!r} has unknown worker '{worker}' (expected one of {_KNOWN_WORKERS})")
        pool = spec.get("resource_pool") or spec.get("pool")
        if not pool:
            raise ValueError(f"model entry {spec!r} must set 'resource_pool'")
        name = spec.get("name") or worker
        config_key = spec.get("config_key") or _DEFAULT_CONFIG_KEY.get(worker, worker)
        out.append(
            {
                "name": str(name),
                "worker": worker,
                "config_key": str(config_key),
                "resource_pool": str(pool),
                "device_range": spec.get("device_range"),
            }
        )
    return out


def _resolve_pools(clusters_map: dict[str, Cluster], pools_in: list[dict]) -> dict[str, DevicePool]:
    # Free local GPU ids per (cluster, node); pools claim them node-by-node in declaration order.
    free: dict[str, dict[int, list[int]]] = {
        name: {n: list(range(c.n_gpus_per_node)) for n in range(c.nnodes)} for name, c in clusters_map.items()
    }
    resolved: dict[str, DevicePool] = {}

    for spec in pools_in:
        name = str(spec["name"])
        cluster_name = str(spec["cluster"])
        if cluster_name not in clusters_map:
            raise ValueError(f"pool '{name}' references unknown cluster '{cluster_name}'")
        cluster = clusters_map[cluster_name]
        p_nnodes = int(spec["nnodes"])
        p_ngpn = int(spec["n_gpus_per_node"])
        if p_nnodes <= 0 or p_ngpn <= 0:
            raise ValueError(f"pool '{name}' must have positive nnodes and n_gpus_per_node")
        if p_ngpn > cluster.n_gpus_per_node:
            raise ValueError(
                f"pool '{name}' n_gpus_per_node ({p_ngpn}) exceeds cluster '{cluster_name}' "
                f"n_gpus_per_node ({cluster.n_gpus_per_node})"
            )
        chosen_nodes = [n for n in range(cluster.nnodes) if len(free[cluster_name][n]) >= p_ngpn][:p_nnodes]
        if len(chosen_nodes) < p_nnodes:
            raise ValueError(
                f"pool '{name}' needs {p_nnodes} node(s) with {p_ngpn} free GPU(s) in cluster "
                f"'{cluster_name}', not enough free (pools within a cluster must be disjoint)"
            )
        gpus: list[int] = []
        for node in chosen_nodes:
            take = sorted(free[cluster_name][node])[:p_ngpn]
            for local in take:
                free[cluster_name][node].remove(local)
                gpus.append(node * cluster.n_gpus_per_node + local)
        gpus.sort()
        attributes = _as_dict(spec.get("attributes", {})) if spec.get("attributes") else {}
        resolved[name] = DevicePool(
            name=name,
            cluster=cluster_name,
            nnodes=p_nnodes,
            n_gpus_per_node=p_ngpn,
            gpus=gpus,
            attributes=attributes,
        )
    return resolved


def _build_placements(
    clusters_map: dict[str, Cluster],
    pool_objs: dict[str, DevicePool],
    models_in: list[dict],
) -> list[ModelPlacement]:
    placements: list[ModelPlacement] = []
    for spec in models_in:
        pool_name = spec["resource_pool"]
        if pool_name not in pool_objs:
            raise ValueError(f"model '{spec['name']}' references unknown resource_pool '{pool_name}'")
        pool = pool_objs[pool_name]
        cluster = clusters_map[pool.cluster]
        device_range = None
        gpus = list(pool.gpus)
        if spec.get("device_range") is not None:
            start, end = parse_device_range(spec["device_range"], len(pool.gpus))
            device_range = (start, end)
            gpus = pool.gpus[start:end]
        devices = [
            Device(
                cluster=pool.cluster,
                gpu=g,
                node=g // cluster.n_gpus_per_node,
                local_gpu=g % cluster.n_gpus_per_node,
            )
            for g in gpus
        ]
        placements.append(
            ModelPlacement(
                name=spec["name"],
                worker=spec["worker"],
                config_key=spec["config_key"],
                pool=pool_name,
                device_range=device_range,
                gpus=gpus,
                devices=devices,
            )
        )
    return placements


def _build_groups(placements: list[ModelPlacement]) -> list[FusionGroup]:
    groups: list[FusionGroup] = []
    index_by_key: dict[tuple, int] = {}
    for model in placements:
        key = (model.config_key, model.pool, model.device_range)
        if key not in index_by_key:
            index_by_key[key] = len(groups)
            groups.append(
                FusionGroup(
                    index=len(groups),
                    config_key=model.config_key,
                    pool=model.pool,
                    device_range=model.device_range,
                )
            )
        gi = index_by_key[key]
        model.group = gi
        groups[gi].models.append(model)
    return groups


def _derive_modes(
    pool_objs: dict[str, DevicePool],
    placements: list[ModelPlacement],
    groups: list[FusionGroup],
) -> None:
    actor = next((m for m in placements if m.worker in _ACTOR_WORKERS), None)
    if actor is None:
        return
    actor_cluster = pool_objs[actor.pool].cluster
    actor_gpu_set = {(actor_cluster, g) for g in actor.gpus}
    actor_group = actor.group

    for model in placements:
        if model.worker not in _SERVER_WORKERS:
            continue
        model_cluster = pool_objs[model.pool].cluster
        model_gpu_set = {(model_cluster, g) for g in model.gpus}
        overlaps_actor = bool(model_gpu_set & actor_gpu_set)
        in_actor_group = model.group == actor_group

        if model.worker == "rollout":
            if in_actor_group:
                model.rollout_mode = RolloutMode.HYBRID
            elif not overlaps_actor:
                model.rollout_mode = RolloutMode.STANDALONE
            else:
                raise ValueError(
                    "rollout placed as a separate process on the actor's GPUs is invalid: fuse it with "
                    "the actor (same config_key + resource_pool) for HYBRID, or put it on a disjoint pool "
                    "for STANDALONE."
                )
        else:  # rm / teacher
            model.rollout_mode = RolloutMode.COLOCATED if overlaps_actor else None


def _pool_gpu_set(pool: DevicePool) -> set[tuple[str, int]]:
    return {(pool.cluster, g) for g in pool.gpus}


def _process_label(topology: ClusterTopology, group: FusionGroup) -> str:
    if group.is_fused:
        return "fused"
    my_gpus = {(topology.pools[group.pool].cluster, g) for g in group.gpus}
    for other in topology.groups:
        if other.index == group.index:
            continue
        other_gpus = {(topology.pools[other.pool].cluster, g) for g in other.gpus}
        if my_gpus & other_gpus:
            return "colocated"
    return "own"


def _dedicated_or_dash(row: dict) -> str:
    for worker in row["workers"]:
        if worker in ("rm", "teacher"):
            return "dedicated"
    return "-"


def _node_local(ids: list[int], n_gpus_per_node: int) -> str:
    by_node: dict[int, list[int]] = {}
    for g in sorted(ids):
        by_node.setdefault(g // n_gpus_per_node, []).append(g % n_gpus_per_node)
    return " ".join(f"n{n}:{compact_gpu_ids(locals_)}" for n, locals_ in sorted(by_node.items()))


def _format_attrs(attributes: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in attributes.items())


def _san(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _as_dict(obj) -> dict:
    """Accept plain dicts or omegaconf DictConfig (converted lazily to avoid a hard dep)."""
    if isinstance(obj, dict):
        return dict(obj)
    try:
        from omegaconf import OmegaConf  # noqa: PLC0415

        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass
    if hasattr(obj, "items"):
        return dict(obj.items())
    raise TypeError(f"expected a mapping, got {type(obj).__name__}")
