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
"""Validate declarative placement through the existing fused-worker seam."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo import topology_resolver as resolver
from verl.trainer.ppo.utils import Role


def _colocated_config():
    return OmegaConf.create(
        {
            "topology": {
                "clusters": [{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
                "device_pools": [{"name": "hybrid_pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
                "models": [
                    {
                        "name": "actor",
                        "worker": "actor",
                        "config_key": "actor_rollout_ref",
                        "resource_pool": "hybrid_pool",
                    },
                    {
                        "name": "rollout",
                        "worker": "rollout",
                        "config_key": "actor_rollout_ref",
                        "resource_pool": "hybrid_pool",
                    },
                    {"name": "ref", "worker": "ref", "config_key": "actor_rollout_ref", "resource_pool": "hybrid_pool"},
                    {"name": "critic", "worker": "critic", "config_key": "critic", "resource_pool": "hybrid_pool"},
                ],
            },
            "actor_rollout_ref": {
                "model": {},
                "actor": {"use_kl_loss": True},
                "rollout": {
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
            },
            "algorithm": {"use_kl_in_reward": False, "adv_estimator": "gae"},
            "critic": {"enable": True},
            "reward": {"reward_model": {"enable": False}},
            "trainer": {"nnodes": 1, "n_gpus_per_node": 8},
        }
    )


def test_empty_config_preserves_legacy_path():
    assert not resolver.topology_enabled(OmegaConf.create({}))
    assert not resolver.topology_enabled(
        OmegaConf.create({"topology": {"models": [], "clusters": [], "device_pools": []}})
    )


def test_incomplete_config_fails_validation():
    config = _colocated_config()
    config.topology.clusters = []
    assert resolver.topology_enabled(config)
    with pytest.raises(ValueError):
        resolver.resolve_placement(config)


@pytest.mark.parametrize("mode", ["sync", "colocate_async"])
def test_colocated_mapping_preserves_fused_worker(mode):
    config = _colocated_config()
    config.trainer.v1 = {"trainer_mode": mode}
    resolved = resolver.resolve_placement(config)
    assert resolved.actor_role == Role.ActorRolloutRef
    assert resolved.resource_pool_spec == {"hybrid_pool": [8]}
    assert resolved.role_pool_mapping == {
        Role.ActorRolloutRef: "hybrid_pool",
        Role.Critic: "hybrid_pool",
        Role.RewardModel: "hybrid_pool",
    }


def test_no_reference_uses_actor_rollout():
    config = _colocated_config()
    config.actor_rollout_ref.actor.use_kl_loss = False
    assert resolver.resolve_placement(config).actor_role == Role.ActorRollout


@pytest.mark.parametrize("rank,legacy_rank", [(8, 0), (0, 8), (-1, 8)])
def test_lora_reference_stays_in_actor(rank, legacy_rank):
    config = _colocated_config()
    config.actor_rollout_ref.model.lora = {"rank": rank}
    config.actor_rollout_ref.model.lora_rank = legacy_rank
    assert resolver.resolve_placement(config).actor_role == Role.ActorRollout


def test_actor_pool_sizes_training_instead_of_entire_cluster():
    config = _colocated_config()
    config.topology.clusters[0].nnodes = 4
    resolver.reconcile_trainer_dims_with_topology(config)
    assert config.trainer.nnodes == 1
    assert config.trainer.n_gpus_per_node == 8


def test_separate_critic_pool_uses_existing_seam():
    config = _colocated_config()
    config.topology.clusters[0].nnodes = 2
    config.topology.device_pools.append(
        {"name": "critic_pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}
    )
    config.topology.models[3].resource_pool = "critic_pool"
    resolved = resolver.resolve_placement(config)
    assert resolved.role_pool_mapping[Role.Critic] == "critic_pool"
    assert resolved.resource_pool_spec == {"hybrid_pool": [8], "critic_pool": [8]}


@pytest.mark.parametrize("field,value", [("use_v1", False), ("v1", {"trainer_mode": "separate_async"})])
def test_unsupported_trainer_rejected(field, value):
    config = _colocated_config()
    config.trainer[field] = value
    with pytest.raises(NotImplementedError):
        resolver.resolve_placement(config)


@pytest.mark.parametrize(
    "mutation", ["device_range", "config_key", "duplicate", "missing_ref", "missing_critic", "subnode", "reward"]
)
def test_unsupported_or_inconsistent_models_rejected(mutation):
    config = _colocated_config()
    if mutation == "device_range":
        config.topology.models[0].device_range = [0, 8]
    elif mutation == "config_key":
        config.topology.models[0].config_key = "another_model"
    elif mutation == "duplicate":
        config.topology.models.append({"name": "actor2", "worker": "actor", "resource_pool": "hybrid_pool"})
    elif mutation == "missing_ref":
        del config.topology.models[2]
    elif mutation == "missing_critic":
        del config.topology.models[3]
    elif mutation == "subnode":
        config.topology.device_pools[0].n_gpus_per_node = 4
    else:
        config.reward.reward_model.enable = True
    with pytest.raises((ValueError, NotImplementedError)):
        resolver.resolve_placement(config)


def test_bad_parallelism_rejected():
    config = _colocated_config()
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 3
    with pytest.raises(ValueError, match="not a multiple"):
        resolver.resolve_placement(config)


def test_legacy_report_retains_actor_only_and_reference_roles():
    config = _colocated_config()
    topology = resolver.build_legacy_topology(
        config, {"train": [8]}, {Role.Actor: "train", Role.RefPolicy: "train", Role.RewardModel: "train"}
    )
    assert set(topology.worker_pool_names()) == {"actor", "ref"}


def test_no_topology_does_not_change_training_dimensions():
    config = OmegaConf.create({"trainer": {"nnodes": 3, "n_gpus_per_node": 4}})
    resolver.reconcile_trainer_dims_with_topology(config)
    assert config.trainer.nnodes == 3
    assert config.trainer.n_gpus_per_node == 4


@pytest.mark.parametrize("declarative", [False, True])
def test_trainer_registration_keeps_existing_worker_classes(declarative):
    from types import SimpleNamespace

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    config = _colocated_config()
    config.reward.reward_model.enable_resource_pool = False
    if not declarative:
        config.topology = {"clusters": [], "device_pools": [], "models": []}
    trainer = SimpleNamespace(config=config, _log_topology_report=lambda: None)
    PPOTrainer._init_resource_pool_mgr(trainer)
    assert set(trainer.role_worker_mapping) == {Role.ActorRolloutRef, Role.Critic}
    actor_class = trainer.role_worker_mapping[Role.ActorRolloutRef].__ray_metadata__.modified_class
    assert actor_class.__name__ == "ActorRolloutRefWorker"
    pool = "hybrid_pool" if declarative else "global_pool"
    assert trainer.resource_pool_manager.resource_pool_spec == {pool: [8]}
    assert trainer.resource_pool_manager.mapping == {
        Role.ActorRolloutRef: pool,
        Role.Critic: pool,
        Role.RewardModel: pool,
    }
