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
"""Turn a hydra ``topology:`` block (or today's legacy knobs) into the
``role_worker_mapping`` / ``resource_pool_spec`` / ``mapping`` that
``ResourcePoolManager`` already consumes.

Two front-ends produce the same :class:`ClusterTopology`:

* :func:`build_config_topology` — from the declarative ``topology:`` block.
* :func:`resolve_legacy_placement` — reconstructed from the legacy
  ``resource_pool_spec`` + ``role_pool_mapping`` a trainer already built, so
  the introspection report works with zero migration.

Only the declarative front-end also drives placement (:func:`resolve_placement`).
The runtime seam is unchanged: the emitted structures plug straight into
``ResourcePoolManager(resource_pool_spec, mapping, accelerator_type=...)``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import ray
from omegaconf import OmegaConf

from verl.single_controller.topology import (
    ClusterTopology,
    RolloutMode,
    build_topology,
    build_topology_from_models,
)
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy

logger = logging.getLogger(__name__)


@dataclass
class ResolvedPlacement:
    """Everything the trainer needs to build a ``ResourcePoolManager`` + worker groups.

    Each declared model maps to its own atomic worker class in ``role_worker_mapping``
    (``Role.Actor`` -> ``ActorWorker``, ``Role.Rollout`` -> ``RolloutWorker``, ``Role.RefPolicy`` ->
    ``RefWorker``, ``Role.Critic`` -> ``TrainingWorker``) — no fused ``ActorRolloutRef`` collapse.
    ``role_pool_mapping`` (``Role -> pool name``), ``resource_pool_spec`` and ``accelerator_type``
    feed ``ResourcePoolManager`` unchanged; the trainer colocates workers that share a pool into one
    ``WorkerDict`` (so a HYBRID actor+rollout+ref share one process but stay three handles).
    """

    topology: ClusterTopology
    role_worker_mapping: dict = field(default_factory=dict)
    resource_pool_spec: dict = field(default_factory=dict)
    role_pool_mapping: dict = field(default_factory=dict)
    accelerator_type: dict = field(default_factory=dict)

    @property
    def mapping(self) -> dict:
        """Backward-compatible alias for ``role_pool_mapping``; use ``role_pool_mapping`` directly."""
        warnings.warn(
            "ResolvedPlacement.mapping is deprecated; use role_pool_mapping instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.role_pool_mapping


def topology_enabled(config) -> bool:
    """True only when the opt-in ``topology:`` block declares clusters and models."""
    topo = _get(config, "topology")
    if not topo:
        return False
    return bool(_get(topo, "clusters")) and bool(_get(topo, "models"))


def single_cluster_topology_dims(config) -> Optional[tuple[int, int, str]]:
    """``(nnodes, n_gpus_per_node, cluster_name)`` implied by a *single-cluster* ``topology:``.

    Returns ``None`` when topology is disabled or spans multiple clusters: a heterogeneous
    topology has no single machine shape, so the legacy scalars cannot be derived from it.
    """
    if not topology_enabled(config):
        return None
    clusters = _get(_get(config, "topology"), "clusters")
    if OmegaConf.is_config(clusters):
        clusters = OmegaConf.to_container(clusters, resolve=True)
    if not clusters or len(clusters) != 1:
        return None
    cluster = clusters[0]
    try:
        return int(cluster["nnodes"]), int(cluster["n_gpus_per_node"]), str(cluster["name"])
    except (KeyError, TypeError, ValueError):
        return None


def reconcile_trainer_dims_with_topology(config, user_set_dims: tuple[str, ...] = ()) -> None:
    """Make ``topology`` the single source of truth for placement scale (RFC #7269).

    ``trainer.nnodes`` / ``trainer.n_gpus_per_node`` are **deprecated** once a ``topology:`` block
    is set. Many non-placement components (rollout servers, reward / teacher loops, batch-size
    validation) still read these scalars, so rather than dropping them we *derive* them from a
    single-cluster topology and overwrite them in place — the topology wins. A warning is emitted
    when the user set them by hand (``user_set_dims``, from the CLI) or when the prior values
    disagree with the topology. Multi-cluster topologies have no single machine shape, so the
    scalars are left untouched (with a warning if the user set them). No-op without ``topology``.
    """
    if not topology_enabled(config):
        return
    trainer = _get(config, "trainer")
    if trainer is None:
        return
    knobs = (
        " / ".join(f"trainer.{d}" for d in user_set_dims)
        if user_set_dims
        else "trainer.nnodes / trainer.n_gpus_per_node"
    )
    dims = single_cluster_topology_dims(config)
    if dims is None:
        if user_set_dims:
            logger.warning(
                "%s are deprecated and ignored for placement when `topology` is set; a multi-cluster "
                "topology has no single (nnodes, n_gpus_per_node) to derive them from.",
                knobs,
            )
        return
    nnodes, n_gpus_per_node, cluster_name = dims
    try:
        current = (int(config.trainer.nnodes), int(config.trainer.n_gpus_per_node))
    except Exception:
        current = None
    if user_set_dims:
        logger.warning(
            "%s are deprecated when `topology` is set and are ignored for placement: the topology "
            "drives it. Deriving nnodes=%d, n_gpus_per_node=%d from cluster '%s'.",
            knobs,
            nnodes,
            n_gpus_per_node,
            cluster_name,
        )
    elif current is not None and current != (nnodes, n_gpus_per_node):
        logger.warning(
            "`topology` cluster '%s' implies nnodes=%d, n_gpus_per_node=%d; overriding the current "
            "trainer.nnodes / trainer.n_gpus_per_node (%dx%d) so non-placement components stay "
            "consistent with the declared placement.",
            cluster_name,
            nnodes,
            n_gpus_per_node,
            current[0],
            current[1],
        )
    config.trainer.nnodes = nnodes
    config.trainer.n_gpus_per_node = n_gpus_per_node


def build_config_topology(config) -> ClusterTopology:
    """Build a :class:`ClusterTopology` from the declarative ``topology:`` block."""
    topo = OmegaConf.to_container(config.topology, resolve=True)
    topology = build_topology(topo.get("clusters"), topo.get("device_pools"), topo.get("models"))
    validate_pool_parallelism(topology, config)
    return topology


def bind_topology_ray_nodes(topology: ClusterTopology, resource_pool_manager) -> None:
    """Fill ``Device.node_id`` / ``node_ip`` from Ray PG bindings after pools are created.

    No-op when Ray is unavailable or pools have not yet materialized placement groups.
    """
    if not topology.pools:
        return
    for pool_name, device_pool in topology.pools.items():
        ray_pool = resource_pool_manager.resource_pool_dict.get(pool_name)
        if ray_pool is None or not getattr(ray_pool, "bundle_bindings", None):
            continue
        cluster = topology.clusters.get(device_pool.cluster)
        if cluster is None:
            continue
        gpu_binding_map = _gpu_to_bundle_bindings(device_pool.gpus, cluster.n_gpus_per_node, ray_pool.bundle_bindings)
        for placement in topology.placements:
            if placement.pool != pool_name:
                continue
            for device in placement.devices:
                binding = gpu_binding_map.get(device.gpu)
                if binding is None:
                    continue
                device.node_id = binding["node_id"]
                device.node_ip = binding["node_ip"]


def validate_pool_parallelism(topology: ClusterTopology, config) -> None:
    """Raise if a server-worker's GPU count is not sized for its TP*DP*PP footprint."""
    for placement in topology.placements:
        if placement.worker not in ("rollout", "rm", "teacher"):
            continue
        n_gpus = len(placement.gpus) or len(topology.pools[placement.pool].gpus)
        if n_gpus == 0:
            continue
        world_size = _rollout_world_size_for_worker(placement.worker, config)
        if world_size <= 0:
            continue
        if n_gpus % world_size != 0:
            raise ValueError(
                f"pool '{placement.pool}' has {n_gpus} GPU(s) for worker '{placement.worker}', "
                f"which is not a multiple of the worker's rollout world size {world_size} "
                f"(tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size)"
            )


def _rollout_world_size_for_worker(worker: str, config) -> int:
    rollout_cfg = _get(config, "actor_rollout_ref")
    rollout_cfg = _get(rollout_cfg, "rollout") if rollout_cfg else None
    if worker == "rollout" and rollout_cfg is not None:
        return _rollout_world_size_from_cfg(rollout_cfg)
    if worker == "rm":
        rm = _get(_get(config, "reward"), "reward_model")
        rm_rollout = _get(rm, "rollout") if rm else None
        if rm_rollout is not None:
            return _rollout_world_size_from_cfg(rm_rollout)
    if worker == "teacher":
        # Multi-teacher: validate against the largest teacher footprint when enabled.
        dist = _get(config, "distillation")
        teachers = _get(dist, "teacher_models") if dist else None
        if not teachers:
            return 0
        sizes = []
        for _name, tcfg in _as_teacher_map(teachers).items():
            inf = tcfg.get("inference") if isinstance(tcfg, dict) else None
            if inf:
                sizes.append(_rollout_world_size_from_cfg(inf) * int(tcfg.get("num_replicas", 1) or 1))
        return max(sizes) if sizes else 0
    return 0


def _rollout_world_size_from_cfg(rollout_cfg) -> int:
    tp = int(_get(rollout_cfg, "tensor_model_parallel_size") or 1)
    dp = int(_get(rollout_cfg, "data_parallel_size") or 1)
    pp = int(_get(rollout_cfg, "pipeline_model_parallel_size") or 1)
    return tp * dp * pp


def _gpu_to_bundle_bindings(gpus: list[int], gpus_per_machine: int, bundle_bindings: list[dict]) -> dict[int, dict]:
    """Map flat cluster GPU ids to Ray bundle bindings in PG / machine order."""
    by_machine: dict[int, list[int]] = {}
    for g in sorted(gpus):
        by_machine.setdefault(g // gpus_per_machine, []).append(g)
    gpu_binding_map: dict[int, dict] = {}
    cursor = 0
    for machine in sorted(by_machine):
        for g in sorted(by_machine[machine]):
            if cursor < len(bundle_bindings):
                gpu_binding_map[g] = bundle_bindings[cursor]
                cursor += 1
    return gpu_binding_map


def _as_teacher_map(teachers) -> dict:
    if isinstance(teachers, dict):
        return teachers
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(teachers):
            return OmegaConf.to_container(teachers, resolve=True)
    except Exception:
        pass
    return dict(teachers) if hasattr(teachers, "items") else {}


def resolve_placement(config, topology: Optional[ClusterTopology] = None) -> ResolvedPlacement:
    """Emit the per-model placement plan + ``ResourcePoolManager`` inputs from the topology.

    Each declared model becomes its own atomic worker (``ActorWorker`` / ``RolloutWorker`` /
    ``RefWorker`` / ``TrainingWorker`` for critic) with its own handle, rather than being collapsed
    onto a fused ``ActorRolloutRef`` role. The trainer colocates workers that share a pool into one
    ``WorkerDict``, so a HYBRID actor+rollout+ref share one process but stay three handles
    (``actor_wg`` / ``rollout_wg`` / ``ref_wg``).

    Raises ``NotImplementedError`` for placements the declarative path does not yet wire
    end-to-end (e.g. a standalone rollout pool), so misconfigurations fail closed rather than
    silently misplacing models. The report still reflects those placements.
    """
    topology = topology or build_config_topology(config)
    topology.require_ray_resources()

    worker_to_pool = topology.worker_pool_names()
    resource_pool_spec = topology.resource_pool_spec()
    accelerator_type = topology.accelerator_type_map()

    actor_group = topology.actor_group()
    if actor_group is None:
        raise ValueError("topology.models must declare a model with worker: actor")

    # Fail closed for placements the declarative resolver does not wire end-to-end yet, so a model is
    # never silently misplaced. The startup report still shows the intended placement.
    for placement in topology.placements:
        if placement.worker == "rollout" and placement.rollout_mode == RolloutMode.STANDALONE:
            raise NotImplementedError(
                "Standalone rollout placement declared in topology.models is not wired through the "
                "declarative resolver yet; use the separate_async trainer knobs for now. The startup "
                "report still shows the intended STANDALONE placement."
            )
        if placement.device_range is not None and placement.gpus != topology.pools[placement.pool].gpus:
            raise NotImplementedError(
                "device_range (sub-pool GPU carving) is declared in topology.models but not wired "
                "through the declarative resolver yet; it is shown in the startup report. Give each "
                "model group its own resource_pool for now."
            )

    # Atomic per-model worker classes. Actor is required; rollout/ref/critic are added only when the
    # topology declares them and the config enables them. ref is folded into the actor (no separate
    # RefWorker) when the reference policy is not needed or lives inside the actor (LoRA), matching
    # the fused worker's behavior.
    role_worker_mapping = _build_atomic_role_worker_mapping(
        config,
        has_actor=True,
        has_rollout=("rollout" in worker_to_pool),
        has_ref=("ref" in worker_to_pool),
        has_critic=("critic" in worker_to_pool),
    )

    # Role -> pool for ResourcePoolManager (pool allocation is driven by resource_pool_spec; this
    # map serves get_resource_pool(role), including the reward / teacher managers).
    role_pool_mapping: dict = {Role.Actor: worker_to_pool["actor"]}
    if "rollout" in worker_to_pool:
        role_pool_mapping[Role.Rollout] = worker_to_pool["rollout"]
    if "ref" in worker_to_pool:
        role_pool_mapping[Role.RefPolicy] = worker_to_pool["ref"]
    if "critic" in worker_to_pool:
        role_pool_mapping[Role.Critic] = worker_to_pool["critic"]
    if "rm" in worker_to_pool:
        role_pool_mapping[Role.RewardModel] = worker_to_pool["rm"]
    if "teacher" in worker_to_pool:
        role_pool_mapping[Role.TeacherModel] = worker_to_pool["teacher"]

    return ResolvedPlacement(
        topology=topology,
        role_worker_mapping=role_worker_mapping,
        resource_pool_spec=resource_pool_spec,
        role_pool_mapping=role_pool_mapping,
        accelerator_type=accelerator_type,
    )


def resolve_legacy_placement(config, resource_pool_spec: dict, role_pool_mapping: dict) -> ResolvedPlacement:
    """Resolve the legacy trainer knobs into the same placement shape as topology mode."""
    # Legacy always sets RewardModel (to global_pool when colocated); only show it in the report
    # when a reward model is actually enabled.
    effective_mapping = dict(role_pool_mapping)
    if Role.RewardModel in effective_mapping and not _reward_enabled(config):
        effective_mapping.pop(Role.RewardModel)

    model_entries = _legacy_model_entries(effective_mapping)
    n_gpus_per_node = int(config.trainer.n_gpus_per_node)
    try:
        topology = build_topology_from_models(resource_pool_spec, model_entries, n_gpus_per_node)
    except Exception as e:  # never let the report break startup
        logger.warning("Could not build legacy topology report: %s", e)
        topology = ClusterTopology(clusters={}, pools={}, models=[], groups=[])

    return ResolvedPlacement(
        topology=topology,
        role_worker_mapping=_legacy_role_worker_mapping(config, effective_mapping),
        resource_pool_spec=resource_pool_spec,
        role_pool_mapping=role_pool_mapping,
        accelerator_type={},
    )


def build_legacy_topology(config, resource_pool_spec: dict, mapping: dict) -> ClusterTopology:
    """Backward-compatible helper returning only the reconstructed topology report object."""
    return resolve_legacy_placement(config, resource_pool_spec, mapping).topology


def _legacy_model_entries(mapping: dict) -> list[dict]:
    """Expand a ``Role -> pool`` mapping into L3 model entries (one per worker).

    The actor's ``actor_rollout_ref`` config_key multiplexes rollout (and the reference policy
    unless it is folded into the actor via LoRA); critic / rm / teacher are each their own
    process (own ``config_key``), even when they share the actor pool (colocated).
    """
    entries: list[dict] = []

    def add(workers: list[str], pool: str, config_key: str) -> None:
        for worker in workers:
            entries.append({"name": worker, "worker": worker, "config_key": config_key, "resource_pool": pool})

    if Role.ActorRolloutRef in mapping:
        add(["actor", "ref", "rollout"], mapping[Role.ActorRolloutRef], "actor_rollout_ref")
    elif Role.ActorRollout in mapping:
        add(["actor", "rollout"], mapping[Role.ActorRollout], "actor_rollout_ref")

    for role_key, worker, config_key in (
        (Role.Critic, "critic", "critic"),
        (Role.RewardModel, "rm", "reward.reward_model"),
        (Role.TeacherModel, "teacher", "distillation"),
    ):
        if role_key in mapping:
            add([worker], mapping[role_key], config_key)
    return entries


def _legacy_role_worker_mapping(config, role_pool_mapping: dict) -> dict:
    """Atomic legacy worker classes keyed by role, mirroring topology-mode output."""
    return _build_atomic_role_worker_mapping(
        config,
        has_actor=(Role.Actor in role_pool_mapping),
        has_rollout=(Role.Rollout in role_pool_mapping),
        has_ref=(Role.RefPolicy in role_pool_mapping),
        has_critic=(Role.Critic in role_pool_mapping),
    )


def _build_atomic_role_worker_mapping(config, *, has_actor=True, has_rollout=False, has_ref=False, has_critic=False):
    """Build ``Role -> ray.remote(Worker)`` mapping from per-worker presence flags.

    Shared by both the declarative resolver (``resolve_placement``) and the legacy adapter
    (``_legacy_role_worker_mapping``) to avoid duplicating the worker-class wiring.
    """
    from verl.workers.engine_workers import ActorWorker, RefWorker, RolloutWorker, TrainingWorker

    role_worker_mapping: dict = {}
    if has_actor:
        role_worker_mapping[Role.Actor] = ray.remote(ActorWorker)
    if has_rollout:
        role_worker_mapping[Role.Rollout] = ray.remote(RolloutWorker)
    if has_ref and need_reference_policy(config) and not _ref_in_actor(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(RefWorker)
    if has_critic and need_critic(config):
        role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
    return role_worker_mapping


def _reward_enabled(config) -> bool:
    try:
        return bool(config.reward.reward_model.enable)
    except Exception:
        return False


def _ref_in_actor(config) -> bool:
    lora = config.actor_rollout_ref.model.get("lora", {}) or {}
    lora_rank = lora.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    return lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None


def _get(obj, key):
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except (TypeError, AttributeError):
            return None
        except Exception as e:
            logger.warning("Unexpected error calling %r.get(%r): %s", type(obj).__name__, key, e)
            return None
    return getattr(obj, key, None)
