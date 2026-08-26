Declarative Device Topology
===========================

Last updated: 10/29/2026.

The opt-in ``topology`` block describes clusters, device pools and models in one
place, following `RFC #7269 <https://github.com/verl-project/verl/issues/7269>`_.
The resolver uses the existing fused ``ActorRolloutRefWorker`` and
``ResourcePoolManager`` interfaces. Empty topology lists preserve existing
registration and scheduling.

Supported runtime placement
---------------------------

The initial integration supports v1 ``sync`` and ``colocate_async`` with the hybrid engine,
one homogeneous cluster, whole-node pools, and one instance per worker kind.
Actor, rollout and the optional reference share a pool and the existing fused
worker. A critic can share that pool or use another whole-node pool.
``trainer.nnodes`` and ``trainer.n_gpus_per_node`` are derived from the **actor
pool**, so spare cluster capacity does not inflate training batch-size checks.

.. code-block:: yaml

   topology:
     clusters:
       - {name: default, nnodes: 2, n_gpus_per_node: 8}
     device_pools:
       - {name: hybrid_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}
     models:
       - {name: actor, worker: actor, resource_pool: hybrid_pool}
       - {name: rollout, worker: rollout, resource_pool: hybrid_pool}
       - {name: ref, worker: ref, resource_pool: hybrid_pool}

This example assumes reference KL is enabled and a critic is not required
(e.g. GRPO). Add a ``worker: critic`` entry when the algorithm needs a critic.
The default config keys are ``actor_rollout_ref`` for actor/rollout/ref and
``critic`` for the critic. Custom config keys are rejected by the runtime resolver.
The optional reference declaration is ignored by worker registration when KL is
disabled or LoRA serves the reference inside the actor.

See ``examples/topology/run_qwen3_8b_fsdp.sh`` for a launch example. Existing
training examples keep their legacy configuration.

Logical placement report
------------------------

V1 logs a logical placement report, including for legacy training pools.
GPU indices in this report are **logical pool/cluster positions**, not physical
CUDA device IDs or observed Ray node bindings. Ray chooses the actual devices.
The report is produced during training-pool setup; standalone replicas created
later by separate-async are outside this report.

The stdlib-only ``verl/single_controller/topology.py`` module provides
``describe()``, ``to_dict()`` and ``to_mermaid()`` for inspecting placement plans.
Models with the same config key and placement form one fusion group. The report
can represent layouts beyond those supported by the runtime resolver.

Explicit runtime limits
-----------------------

The runtime resolver rejects multi-cluster placement, sub-node pools,
``device_range`` slicing, separate rollout/reference pools, reward models,
teachers, multiple instances of a worker kind, custom config keys and pool
attributes. It also rejects v0 and separate-async trainers. Use their existing
configuration interfaces until those paths are integrated. Partial topology
input raises an error instead of silently falling back to legacy settings.

The data model can describe heterogeneous clusters, device ranges and reserve
attributes, but these capabilities do not imply runtime support. Elasticity,
fault tolerance, physical GPU pinning and multi-teacher orchestration require
follow-up work. No multi-node GPU training validation is claimed here.
