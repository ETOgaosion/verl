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
"""Resolver tests: the declarative ``topology:`` block (clusters / device_pools /
models, RFC #7269) emits the same ``resource_pool_spec`` / ``mapping`` seam that
``ResourcePoolManager`` consumes, and it is the single source of truth for the
legacy ``trainer.nnodes`` / ``n_gpus_per_node`` scalars. Skipped automatically
where the full ``verl`` stack (Ray/torch) is unavailable.
"""

import pytest
from omegaconf import OmegaConf

resolver = pytest.importorskip("verl.trainer.ppo.topology_resolver")
Role = pytest.importorskip("verl.trainer.ppo.utils").Role


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


def test_topology_enabled_toggle():
    assert resolver.topology_enabled(_colocated_config()) is True
    empty = OmegaConf.create({"topology": {"clusters": [], "device_pools": [], "models": []}})
    assert resolver.topology_enabled(empty) is False
    assert resolver.topology_enabled(OmegaConf.create({})) is False


def test_build_config_topology_emits_spec():
    topo = resolver.build_config_topology(_colocated_config())
    assert topo.resource_pool_spec() == {"hybrid_pool": [8]}
    assert topo.accelerator_type_map() == {}


def test_resolve_placement_per_model_atomic():
    resolved = resolver.resolve_placement(_colocated_config())
    # colocated single cluster: same resource_pool_spec seam as legacy _init_resource_pool_mgr
    assert resolved.resource_pool_spec == {"hybrid_pool": [8]}
    assert resolved.accelerator_type == {}
    # per-model atomic roles (no fused ActorRolloutRef collapse)
    assert set(resolved.role_worker_mapping) == {Role.Actor, Role.Rollout, Role.RefPolicy, Role.Critic}
    assert resolved.role_pool_mapping == {
        Role.Actor: "hybrid_pool",
        Role.Rollout: "hybrid_pool",
        Role.RefPolicy: "hybrid_pool",
        Role.Critic: "hybrid_pool",
    }


def test_validate_pool_parallelism_rejects_bad_sizing():
    config = OmegaConf.create(
        {
            "topology": {
                "clusters": [{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
                "device_pools": [{"name": "pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 7}],
                "models": [
                    {"name": "actor", "worker": "actor", "resource_pool": "pool"},
                    {"name": "rollout", "worker": "rollout", "resource_pool": "pool"},
                ],
            },
            "actor_rollout_ref": {
                "model": {},
                "rollout": {
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                },
            },
        }
    )
    with pytest.raises(ValueError, match="not a multiple"):
        resolver.build_config_topology(config)


def test_gpu_to_bundle_bindings_maps_in_machine_order():
    bindings = [
        {"node_id": "n0", "node_ip": "10.0.0.1"},
        {"node_id": "n0", "node_ip": "10.0.0.1"},
        {"node_id": "n1", "node_ip": "10.0.0.2"},
        {"node_id": "n1", "node_ip": "10.0.0.2"},
    ]
    mapped = resolver._gpu_to_bundle_bindings([0, 1, 8, 9], gpus_per_machine=8, bundle_bindings=bindings)
    assert mapped[0]["node_ip"] == "10.0.0.1"
    assert mapped[8]["node_ip"] == "10.0.0.2"


def test_role_from_string_env_and_teacher():
    assert str(Role.from_string("env")) == "env"
    assert str(Role.from_string("teacher")) == "teacher"


def test_resolve_placement_standalone_rollout_not_wired_yet():
    config = OmegaConf.create(
        {
            "topology": {
                "clusters": [{"name": "default", "nnodes": 4, "n_gpus_per_node": 8}],
                "device_pools": [
                    {"name": "train_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
                    {"name": "infer_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
                ],
                "models": [
                    {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
                    {"name": "ref", "worker": "ref", "resource_pool": "train_pool"},
                    {"name": "rollout", "worker": "rollout", "resource_pool": "infer_pool"},
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
            "algorithm": {"use_kl_in_reward": False, "adv_estimator": "grpo"},
            "critic": {"enable": False},
            "reward": {"reward_model": {"enable": False}},
        }
    )
    with pytest.raises(NotImplementedError):
        resolver.resolve_placement(config)


# ------------------- trainer.nnodes / n_gpus_per_node deprecation -------------------
def test_single_cluster_topology_dims():
    assert resolver.single_cluster_topology_dims(_colocated_config()) == (1, 8, "default")


def test_single_cluster_topology_dims_none_for_multi_cluster():
    config = OmegaConf.create(
        {
            "topology": {
                "clusters": [
                    {"name": "h100", "nnodes": 2, "n_gpus_per_node": 8},
                    {"name": "a100", "nnodes": 2, "n_gpus_per_node": 8},
                ],
                "device_pools": [
                    {"name": "train_pool", "cluster": "h100", "nnodes": 2, "n_gpus_per_node": 8},
                    {"name": "teacher_pool", "cluster": "a100", "nnodes": 2, "n_gpus_per_node": 8},
                ],
                "models": [
                    {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
                    {"name": "rollout", "worker": "rollout", "resource_pool": "train_pool"},
                    {"name": "teacher", "worker": "teacher", "resource_pool": "teacher_pool"},
                ],
            }
        }
    )
    assert resolver.single_cluster_topology_dims(config) is None


def test_reconcile_trainer_dims_derives_from_topology():
    config = _colocated_config()
    config.topology.clusters[0].nnodes = 2
    config.topology.clusters[0].n_gpus_per_node = 4
    # user left trainer.* at the (stale) defaults; topology is the source of truth
    resolver.reconcile_trainer_dims_with_topology(config)
    assert int(config.trainer.nnodes) == 2
    assert int(config.trainer.n_gpus_per_node) == 4


def test_reconcile_trainer_dims_noop_without_topology():
    config = OmegaConf.create(
        {"topology": {"clusters": [], "device_pools": [], "models": []}, "trainer": {"nnodes": 3, "n_gpus_per_node": 8}}
    )
    resolver.reconcile_trainer_dims_with_topology(config, user_set_dims=("nnodes",))
    assert int(config.trainer.nnodes) == 3  # untouched
    assert int(config.trainer.n_gpus_per_node) == 8
