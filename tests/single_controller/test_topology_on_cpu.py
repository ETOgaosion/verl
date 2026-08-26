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
"""CPU tests for the declarative device-topology data model (RFC #7269).

The module under test (``verl/single_controller/topology.py``) is deliberately
stdlib-only; these tests import it directly by path when the full ``verl``
package (which pulls in Ray/torch) is unavailable, so the pure logic is testable
in any environment.
"""

import importlib.util
import pathlib
import sys

import pytest

try:  # normal import path (CI, with ray/torch installed)
    from verl.single_controller import topology as topo
except Exception:  # minimal env: load the stdlib-only module directly
    _path = pathlib.Path(__file__).resolve().parents[2] / "verl" / "single_controller" / "topology.py"
    _spec = importlib.util.spec_from_file_location("verl_topology_cpu", _path)
    topo = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = topo
    _spec.loader.exec_module(topo)

RolloutMode = topo.RolloutMode
build_topology = topo.build_topology
build_topology_from_models = topo.build_topology_from_models
parse_device_range = topo.parse_device_range
compact_gpu_ids = topo.compact_gpu_ids


def _workers_of(topology):
    """Map ``tuple(worker names) -> process row`` for compact assertions."""
    return {tuple(r["workers"]): r for r in topology.process_rows()}


# ------------------------------ small helpers ------------------------------
def test_parse_device_range_pair():
    assert parse_device_range([0, 4], 8) == (0, 4)


def test_parse_device_range_count():
    assert parse_device_range(4, 8) == (0, 4)


def test_parse_device_range_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_device_range([0, 9], 8)


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], "0-7"),
        ([0, 1, 2, 3, 4, 5, 6, 7, 16, 17], "0-7,16-17"),
        ([5], "5"),
        ([], "-"),
    ],
)
def test_compact_gpu_ids(ids, expected):
    assert compact_gpu_ids(ids) == expected


# ------------------------------ pool allocation ------------------------------
def test_pools_allocate_disjoint_nodes_in_order():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 4, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "infer_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "infer_pool"},
        ],
    )
    assert t.pools["train_pool"].gpus == list(range(16))
    assert t.pools["infer_pool"].gpus == list(range(16, 32))


def test_sub_node_pools_share_a_node():
    """Two half-node pools on a single node get the low / high local GPUs."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 4},
            {"name": "infer", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 4},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "infer"},
        ],
    )
    assert t.pools["train"].gpus == [0, 1, 2, 3]
    assert t.pools["infer"].gpus == [4, 5, 6, 7]
    assert t.resource_pool_spec() == {"train": [4], "infer": [4]}
    # rollout on a disjoint sub-node pool derives STANDALONE
    assert _workers_of(t)[("rollout",)]["mode"] == RolloutMode.STANDALONE


def test_pool_larger_than_cluster_raises():
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8}],
            models=[{"name": "actor", "worker": "actor", "resource_pool": "p"}],
        )


def test_pool_n_gpus_per_node_exceeds_cluster_raises():
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 16}],
            models=[{"name": "actor", "worker": "actor", "resource_pool": "p"}],
        )


# ------------------------------ scenarios / derivation ------------------------------
def test_colocated_hybrid_and_critic():
    """RFC 1: actor+rollout+ref fused (HYBRID) + critic as its own process on the same pool."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 2, "n_gpus_per_node": 8}],
        device_pools=[{"name": "hybrid_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "config_key": "actor_rollout_ref", "resource_pool": "hybrid_pool"},
            {"name": "rollout", "worker": "rollout", "config_key": "actor_rollout_ref", "resource_pool": "hybrid_pool"},
            {"name": "ref", "worker": "ref", "config_key": "actor_rollout_ref", "resource_pool": "hybrid_pool"},
            {"name": "critic", "worker": "critic", "config_key": "critic", "resource_pool": "hybrid_pool"},
        ],
    )
    rows = t.process_rows()
    assert rows[0]["workers"] == ["actor", "rollout", "ref"]
    assert rows[0]["config_key"] == "actor_rollout_ref"
    assert rows[0]["process"] == "fused"
    assert rows[0]["mode"] == RolloutMode.HYBRID
    assert rows[1]["workers"] == ["critic"]
    assert rows[1]["process"] == "colocated"
    assert rows[1]["mode"] is None
    assert t.resource_pool_spec() == {"hybrid_pool": [8, 8]}


def test_default_config_key_fuses_actor_rollout_ref():
    """Omitting config_key: actor/rollout/ref default to actor_rollout_ref and fuse to HYBRID."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[{"name": "p", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "p"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "p"},
            {"name": "ref", "worker": "ref", "resource_pool": "p"},
        ],
    )
    rows = t.process_rows()
    assert len(rows) == 1  # single fused process
    assert rows[0]["mode"] == RolloutMode.HYBRID


def test_disaggregated_standalone_rollout():
    """Rollout on a disjoint pool derives STANDALONE."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 4, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "infer_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
            {"name": "ref", "worker": "ref", "resource_pool": "train_pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "infer_pool"},
        ],
    )
    rows = _workers_of(t)
    assert rows[("rollout",)]["mode"] == RolloutMode.STANDALONE
    assert rows[("rollout",)]["process"] == "own"
    assert t.resource_pool_spec() == {"train_pool": [8, 8], "infer_pool": [8, 8]}


def test_opd_single_teacher_dedicated():
    """A teacher on its own pool is dedicated (no derived rollout mode)."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 3, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "teacher_pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "train_pool"},
            {"name": "ref", "worker": "ref", "resource_pool": "train_pool"},
            {"name": "teacher", "worker": "teacher", "resource_pool": "teacher_pool"},
        ],
    )
    rows = _workers_of(t)
    assert rows[("actor", "rollout", "ref")]["mode"] == RolloutMode.HYBRID
    assert rows[("teacher",)]["mode"] is None
    assert rows[("teacher",)]["process"] == "own"
    assert t.pools["teacher_pool"].gpus == list(range(16, 24))


def test_reward_model_colocated_derives_colocated_mode():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[{"name": "pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "pool"},
            {"name": "rm", "worker": "rm", "resource_pool": "pool"},
        ],
    )
    rows = _workers_of(t)
    assert rows[("rm",)]["mode"] == RolloutMode.COLOCATED
    assert rows[("rm",)]["process"] == "colocated"


def test_gpu_level_placement_with_device_range():
    """RFC 2.3/5.2: actor on GPU 0-3, rollout on GPU 4-7 of one node via device_range."""
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[{"name": "node_pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "node_pool", "device_range": [0, 4]},
            {"name": "rollout", "worker": "rollout", "resource_pool": "node_pool", "device_range": [4, 8]},
        ],
    )
    actor = t.model_of_worker("actor")
    rollout = t.model_of_worker("rollout")
    assert actor.gpus == [0, 1, 2, 3]
    assert rollout.gpus == [4, 5, 6, 7]
    # disjoint GPUs (different device_range) => separate processes, rollout STANDALONE
    assert rollout.rollout_mode == RolloutMode.STANDALONE


# ------------------------------ heterogeneous ------------------------------
def test_heterogeneous_accelerator_type():
    """Two clusters => each pool carries its cluster name as accelerator_type."""
    t = build_topology(
        clusters=[
            {"name": "h100", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "a100", "nnodes": 2, "n_gpus_per_node": 8},
        ],
        device_pools=[
            {"name": "train_pool", "cluster": "h100", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "teacher_pool", "cluster": "a100", "nnodes": 2, "n_gpus_per_node": 8},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train_pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "train_pool"},
            {"name": "teacher", "worker": "teacher", "resource_pool": "teacher_pool"},
        ],
    )
    assert t.accelerator_type_map() == {"train_pool": "h100", "teacher_pool": "a100"}


def test_single_cluster_has_no_accelerator_type():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[{"name": "pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "pool"},
        ],
    )
    assert t.accelerator_type_map() == {}


# ------------------------------ errors ------------------------------
def test_rollout_separate_process_on_actor_gpus_rejected():
    """A rollout that overlaps the actor's GPUs but is not fused with it is invalid."""
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            models=[
                {"name": "actor", "worker": "actor", "resource_pool": "p"},
                # different device_range => separate process, but overlaps actor's GPUs
                {"name": "rollout", "worker": "rollout", "resource_pool": "p", "device_range": [0, 4]},
            ],
        )


def test_unknown_cluster_ref_rejected():
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "nope", "nnodes": 1, "n_gpus_per_node": 8}],
            models=[{"name": "actor", "worker": "actor", "resource_pool": "p"}],
        )


def test_unknown_pool_ref_rejected():
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            models=[{"name": "actor", "worker": "actor", "resource_pool": "missing"}],
        )


def test_unknown_worker_rejected():
    with pytest.raises(ValueError):
        build_topology(
            clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            device_pools=[{"name": "p", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
            models=[{"name": "x", "worker": "bogus", "resource_pool": "p"}],
        )


# ------------------------------ validation warnings ------------------------------
def test_roleless_pool_warns_without_reserve_attribute():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 3, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "idle", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "train"},
        ],
    )
    assert any("idle" in w for w in t.validate())


def test_standby_reserve_pool_does_not_warn():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 3, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {
                "name": "spare",
                "cluster": "default",
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "attributes": {"standby": True, "fault_tolerant": True},
            },
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "train"},
        ],
    )
    assert not any("spare" in w for w in t.validate())


def test_unknown_attribute_warns_but_is_carried_through():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 4, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {"name": "infer", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8, "attributes": {"mystery": 7}},
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train"},
            {"name": "ref", "worker": "ref", "resource_pool": "train"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "infer"},
        ],
    )
    assert any("mystery" in w for w in t.validate())
    assert t.pools["infer"].attributes == {"mystery": 7}


def test_elastic_attribute_type_error_warns():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 4, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "train", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8},
            {
                "name": "infer",
                "cluster": "default",
                "nnodes": 2,
                "n_gpus_per_node": 8,
                "attributes": {"min_gpus": "eight"},
            },
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "train"},
            {"name": "ref", "worker": "ref", "resource_pool": "train"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "infer"},
        ],
    )
    assert any("min_gpus" in w for w in t.validate())


# ------------------------------ legacy adapter (report path) ------------------------------
def test_legacy_spec_round_trip_colocated():
    spec = {"global_pool": [8, 8]}
    entries = [
        {"name": "actor", "worker": "actor", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "ref", "worker": "ref", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "rollout", "worker": "rollout", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "critic", "worker": "critic", "config_key": "critic", "resource_pool": "global_pool"},
    ]
    t = build_topology_from_models(spec, entries, n_gpus_per_node=8)
    assert t.clusters["default"].nnodes == 2
    assert t.pools["global_pool"].gpus == list(range(16))
    assert t.resource_pool_spec() == spec  # round-trip invariant
    rows = _workers_of(t)
    assert rows[("actor", "ref", "rollout")]["mode"] == RolloutMode.HYBRID


def test_legacy_spec_round_trip_separate_reward():
    spec = {"global_pool": [8], "reward_pool": [8]}
    entries = [
        {"name": "actor", "worker": "actor", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "ref", "worker": "ref", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "rollout", "worker": "rollout", "config_key": "actor_rollout_ref", "resource_pool": "global_pool"},
        {"name": "rm", "worker": "rm", "config_key": "reward.reward_model", "resource_pool": "reward_pool"},
    ]
    t = build_topology_from_models(spec, entries, n_gpus_per_node=8)
    assert t.resource_pool_spec() == spec
    assert t.pools["global_pool"].gpus == list(range(8))
    assert t.pools["reward_pool"].gpus == list(range(8, 16))
    rows = _workers_of(t)
    assert rows[("rm",)]["mode"] is None  # dedicated (disjoint pool)
    assert rows[("rm",)]["process"] == "own"


# ------------------------------ reporting ------------------------------
def test_describe_contains_columns_and_rows():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 2, "n_gpus_per_node": 8}],
        device_pools=[{"name": "hybrid_pool", "cluster": "default", "nnodes": 2, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "hybrid_pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "hybrid_pool"},
            {"name": "ref", "worker": "ref", "resource_pool": "hybrid_pool"},
            {"name": "critic", "worker": "critic", "resource_pool": "hybrid_pool"},
        ],
    )
    report = t.describe()
    assert "CLUSTER" in report and "ROLLOUT MODE" in report and "CONFIG_KEY" in report
    assert "actor+rollout+ref" in report
    assert "HYBRID" in report
    assert "n0:0-7 n1:0-7" in report
    assert "Warnings: none" in report


def test_to_dict_structure():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[
            {"name": "pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8, "attributes": {"elastic": True}}
        ],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "pool"},
        ],
    )
    d = t.to_dict()
    assert d["clusters"]["default"] == {"nnodes": 1, "n_gpus_per_node": 8, "size": 8}
    assert d["device_pools"]["pool"]["attributes"] == {"elastic": True}
    assert d["processes"][0]["mode"] == "hybrid"
    assert d["processes"][0]["workers"] == ["actor", "rollout"]


def test_to_mermaid_smoke():
    t = build_topology(
        clusters=[{"name": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        device_pools=[{"name": "pool", "cluster": "default", "nnodes": 1, "n_gpus_per_node": 8}],
        models=[
            {"name": "actor", "worker": "actor", "resource_pool": "pool"},
            {"name": "rollout", "worker": "rollout", "resource_pool": "pool"},
        ],
    )
    m = t.to_mermaid()
    assert m.startswith("flowchart TD")
    assert "cluster_default" in m and "pool_pool" in m
