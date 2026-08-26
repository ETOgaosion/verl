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
"""Resolve opt-in topology declarations through the existing v1 worker interfaces."""

from dataclasses import dataclass

from omegaconf import OmegaConf

from verl.single_controller.topology import ClusterTopology, build_topology, build_topology_from_models
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy


@dataclass
class ResolvedPlacement:
    """Validated topology and the inputs consumed by ResourcePoolManager."""

    topology: ClusterTopology
    resource_pool_spec: dict
    role_pool_mapping: dict
    actor_role: Role


def topology_enabled(config) -> bool:
    """Enable declarations when any topology list is nonempty, including incomplete input."""
    return any(OmegaConf.select(config, f"topology.{key}") for key in ("clusters", "device_pools", "models"))


def build_config_topology(config) -> ClusterTopology:
    """Parse declarative topology and validate server parallelism."""
    topo = OmegaConf.to_container(config.topology, resolve=True)
    topology = build_topology(topo.get("clusters"), topo.get("device_pools"), topo.get("models"))
    for placement in topology.placements:
        if placement.worker != "rollout":
            continue
        cfg = config.actor_rollout_ref.rollout
        world_size = 1
        for key in ("tensor_model_parallel_size", "data_parallel_size", "pipeline_model_parallel_size"):
            value = int(cfg.get(key, 1))
            if value <= 0:
                raise ValueError(f"{key} must be positive")
            world_size *= value
        if len(placement.gpus) % world_size:
            raise ValueError(f"rollout pool size is not a multiple of rollout world size {world_size}")
    return topology


def resolve_placement(config, topology=None) -> ResolvedPlacement:
    """Resolve supported declarations without replacing the fused actor/rollout/ref worker.

    Initially support whole-node pools in one homogeneous cluster on v1 sync/colocate_async.
    Reject unsupported configurations before allocation instead of ignoring placement intent.
    """
    topology = topology or build_config_topology(config)
    if config.trainer.get("use_v1", True) is False:
        raise NotImplementedError("topology placement requires trainer.use_v1=true")
    mode = OmegaConf.select(config, "trainer.v1.trainer_mode", default="sync")
    if mode not in ("sync", "colocate_async"):
        raise NotImplementedError(
            "topology placement currently supports v1 sync/colocate_async; use separate_async knobs otherwise"
        )
    if not config.actor_rollout_ref.get("hybrid_engine", True):
        raise NotImplementedError("topology placement currently requires the hybrid engine")
    if len(topology.clusters) != 1:
        raise NotImplementedError("multi-cluster runtime placement is not wired yet")
    workers = {}
    defaults = {
        "actor": "actor_rollout_ref",
        "rollout": "actor_rollout_ref",
        "ref": "actor_rollout_ref",
        "critic": "critic",
    }
    for model in topology.models:
        if model.worker not in defaults:
            raise NotImplementedError(f"runtime placement for worker {model.worker!r} is not wired yet")
        if model.worker in workers:
            raise ValueError(f"only one model per worker is supported: {model.worker}")
        if model.config_key != defaults[model.worker]:
            raise NotImplementedError(f"custom config_key for {model.worker} is not wired yet")
        if model.device_range is not None:
            raise NotImplementedError("device_range runtime placement is not wired yet")
        workers[model.worker] = model
    if not {"actor", "rollout"} <= workers.keys():
        raise ValueError("topology.models must declare actor and rollout")
    actor_pool = workers["actor"].pool
    if any(workers[w].pool != actor_pool for w in ("rollout", "ref") if w in workers):
        raise NotImplementedError("standalone rollout or reference placement is not wired yet")
    for pool in topology.pools.values():
        cluster = topology.clusters[pool.cluster]
        if pool.n_gpus_per_node != cluster.n_gpus_per_node:
            raise NotImplementedError("sub-node pool runtime placement is not wired yet")
        if pool.attributes:
            raise NotImplementedError("pool attributes are report-only and cannot drive runtime placement yet")
    if OmegaConf.select(config, "reward.reward_model.enable", default=False):
        raise NotImplementedError("reward-model topology placement is not wired yet")
    from verl.trainer.distillation import is_distillation_enabled

    if is_distillation_enabled(config.get("distillation")):
        raise NotImplementedError("teacher topology placement is not wired yet")
    model = config.actor_rollout_ref.model
    rank = (model.get("lora") or {}).get("rank", 0)
    if rank <= 0:
        rank = model.get("lora_rank", 0)
    ref_in_actor = rank > 0 or model.get("lora_adapter_path") is not None
    needs_ref = need_reference_policy(config) and not ref_in_actor
    if needs_ref and "ref" not in workers:
        raise ValueError("reference policy is enabled but topology.models has no ref")
    if need_critic(config) != ("critic" in workers):
        raise ValueError("critic declaration must match the configured critic requirement")
    actor_role = Role.ActorRolloutRef if needs_ref else Role.ActorRollout
    mapping = {actor_role: actor_pool, Role.RewardModel: actor_pool}
    if "critic" in workers:
        mapping[Role.Critic] = workers["critic"].pool
    return ResolvedPlacement(topology, topology.resource_pool_spec(), mapping, actor_role)


def reconcile_trainer_dims_with_topology(config) -> None:
    """Derive legacy training dimensions from the actor pool after validation."""
    if not topology_enabled(config):
        return
    resolved = resolve_placement(config)
    pool = resolved.topology.pools[resolved.topology.actor_pool()]
    config.trainer.nnodes = pool.nnodes
    config.trainer.n_gpus_per_node = pool.n_gpus_per_node


def build_legacy_topology(config, resource_pool_spec, mapping) -> ClusterTopology:
    """Describe existing pools without changing worker registration or scheduling.

    GPU indices are logical positions in each pool; Ray chooses physical devices.
    """
    entries = []
    expanded = {
        Role.ActorRolloutRef: ("actor", "rollout", "ref"),
        Role.ActorRollout: ("actor", "rollout"),
        Role.Actor: ("actor",),
        Role.RefPolicy: ("ref",),
        Role.Critic: ("critic",),
        Role.RewardModel: ("rm",),
        Role.TeacherModel: ("teacher",),
    }
    for role, pool in mapping.items():
        if role == Role.RewardModel and not OmegaConf.select(config, "reward.reward_model.enable", default=False):
            continue
        for worker in expanded.get(role, ()):
            entries.append({"name": worker, "worker": worker, "resource_pool": pool})
    return build_topology_from_models(resource_pool_spec, entries, int(config.trainer.n_gpus_per_node))
