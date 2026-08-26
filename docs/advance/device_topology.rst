Declarative Device Topology
===========================

Last updated: 08/24/2026.

The ``topology:`` config block lets you declare, in one place, **which GPU runs which
model** (actor, rollout, ref, critic, reward model, teacher) — including heterogeneous
hardware. It is an opt-in refactor of resource placement proposed in
`RFC #7269 <https://github.com/verl-project/verl/issues/7269>`_.

Without it, placement is assembled at runtime from ``role_worker_mapping``,
``resource_pool_spec`` and several scattered knobs (``trainer.*``,
``reward.reward_model.*``, ``distillation.*``), and the rollout mode is decided by
whether an argument happened to be passed. The topology block makes placement a single
declarative input, and derives the rollout mode from it.

Data model lives in ``verl/single_controller/topology.py`` (stdlib-only, unit-tested in
``tests/single_controller/test_topology_on_cpu.py``); the resolver that turns the block
into the structures ``ResourcePoolManager`` consumes lives in
``verl/trainer/ppo/topology_resolver.py``.

Quick start
-----------

The block ships **empty**, so existing configs are unchanged. You declare the topology
**directly** — there is no preset and no ``topology=<mode>`` to select; you state the
placement and the trainer *infers* the rollout mode. Set the three lists on the CLI (they
scale with your run via shell variables):

.. code-block:: bash

   python3 -m verl.trainer.main_ppo \
       "topology.clusters=[{name: default, nnodes: $NNODES, n_gpus_per_node: $NGPUS_PER_NODE}]" \
       "topology.device_pools=[{name: hybrid_pool, cluster: default, nnodes: $NNODES, n_gpus_per_node: $NGPUS_PER_NODE}]" \
       "topology.models=[{name: actor, worker: actor, config_key: actor_rollout_ref, resource_pool: hybrid_pool}, {name: rollout, worker: rollout, config_key: actor_rollout_ref, resource_pool: hybrid_pool}, {name: ref, worker: ref, config_key: actor_rollout_ref, resource_pool: hybrid_pool}]" \
       algorithm.adv_estimator=grpo \
       ...  # your usual data / model / trainer overrides

(HYBRID is *derived* from actor+rollout+ref sharing a pool — it is never a name you pass.)
Several flagship examples already declare it inline this way (delete the ``topology`` lines
to fall back to the legacy path): ``examples/grpo_trainer/run_qwen3_8b_fsdp.sh``,
``examples/grpo_trainer/run_qwen3_8b_megatron.sh``,
``examples/gspo_trainer/run_qwen3_8b_fsdp.sh``,
``examples/ppo_trainer/run_qwen3_8b_megatron.sh``, and the self-contained
``examples/topology/run_qwen3_8b_fsdp.sh``.

At startup the trainer logs a report (see `The startup report`_) showing the resolved
placement. This report prints **whether or not** you enable the block: with the legacy
knobs it is reconstructed from the built resource pools (zero migration).

The three layers
----------------

.. code-block:: yaml

   topology:
     # Layer 1 - CLUSTERS: one entry per homogeneous machine group.
     clusters:
       - {name: default, nnodes: 2, n_gpus_per_node: 8}      # cluster GPU ids 0..15

     # Layer 2 - DEVICE_POOLS: named GPU groups, each carved from one cluster.
     device_pools:
       - {name: hybrid_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}

     # Layer 3 - MODELS: one model instance each; models sharing (config_key, resource_pool) fuse.
     models:
       - {name: actor,   worker: actor,   config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: rollout, worker: rollout, config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: ref,     worker: ref,     config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: critic,  worker: critic,  config_key: critic,            resource_pool: hybrid_pool}

**Layer 1 — clusters.** Each cluster is one machine type. Its ``name`` doubles as the Ray
custom resource that pins its machines, which is what makes heterogeneous hardware
expressible (see `Heterogeneous hardware`_). With a single GPU type, name it ``default``.
Every cluster owns a flat GPU index space ``0 .. nnodes*n_gpus_per_node - 1``.

- ``name``: unique name; also the Ray custom-resource key for its machines.
- ``nnodes``: number of machines (nodes) of this type.
- ``n_gpus_per_node``: GPUs per node. Flat id ``g`` is on node ``g // n_gpus_per_node``.

**Layer 2 — device_pools.** A pool references one cluster and is sized by ``nnodes`` and
``n_gpus_per_node``. Pools are allocated node-by-node in declaration order, so pools within
a cluster are disjoint. Optional ``attributes`` (see `Pool attributes`_) are carried
through and shown in the report.

**Layer 3 — models.** One entry = one model instance:

- ``name``: instance name (free-form; distinguishes e.g. multiple teachers).
- ``worker``: which worker class runs it — ``actor`` (required), ``rollout``, ``ref``, ``critic``, ``rm``, ``teacher``.
- ``config_key``: which config subtree it reads (e.g. ``actor_rollout_ref``, ``critic``, ``reward.reward_model``, ``distillation``). Omit it to accept the per-worker default.
- ``resource_pool``: the pool it runs on.
- ``device_range`` (optional): pins it to a sub-slice of the pool (see `Sizing pools and device_range`_).

Models that share a ``(config_key, resource_pool, device_range)`` are **fused** into one
worker process — e.g. the ``ActorRolloutRefWorker`` that multiplexes actor + rollout + ref.

Sizing pools and device_range
-----------------------------

A pool claims ``n_gpus_per_node`` GPUs from each of ``nnodes`` nodes of its cluster, taking
the lowest free local ids on the first nodes that still have room. So two whole-node pools
tile the cluster's nodes; two half-node pools share a node's low / high GPUs.

To place a model on a *sub-slice* of its pool, add ``device_range: [start, end]`` (end
exclusive; a bare count ``n`` means ``[0, n]``). It indexes the pool's sorted GPU list, so
``device_range: [0, 4]`` is the pool's first four GPUs. This is how "actor on GPU 0-3,
rollout on GPU 4-7 of one node" becomes expressible.

.. note::

   ``device_range`` is **declared, validated and reported** today, but the declarative
   resolver does not yet carve a sub-pool through ``ResourcePoolManager``; if a model's
   ``device_range`` selects fewer GPUs than its pool, the resolver raises
   ``NotImplementedError`` (fail-closed). Give each model group its own ``resource_pool``
   to run today.

Fusion and derived rollout mode
-------------------------------

``RolloutMode`` (``hybrid`` / ``colocated`` / ``standalone``) is **derived** from how a
model's GPUs overlap the actor's — never hand-written:

- ``rollout`` fused with the actor (same ``config_key`` + ``resource_pool``) → ``HYBRID`` (``init_hybrid``).
- ``rollout`` on GPUs disjoint from the actor → ``STANDALONE`` (``init_standalone``).
- ``rm`` / ``teacher`` sharing the actor's GPUs as their own process → ``COLOCATED`` (``init_colocated``).
- ``rm`` / ``teacher`` on their own pool → dedicated (no rollout mode).

Placing ``rollout`` as a *separate* process on the actor's GPUs is rejected: the policy
rollout must be fused with the actor (HYBRID), or live on disjoint GPUs (STANDALONE).

The startup report
------------------

.. code-block:: text

   Resolved topology
   CLUSTER  POOL         GPUS  NODE:local     WORKERS            CONFIG_KEY         PROCESS    ROLLOUT MODE
   default  hybrid_pool  0-15  n0:0-7 n1:0-7  actor+rollout+ref  actor_rollout_ref  fused      HYBRID
   default  hybrid_pool  0-15  n0:0-7 n1:0-7  critic             critic             colocated  -
   Warnings: none

``ClusterTopology`` also exposes ``to_dict()`` (JSON-friendly) and ``to_mermaid()`` (a
flowchart of clusters → pools → processes) for programmatic use.

Scenarios
---------

Colocated RL (runnable)
~~~~~~~~~~~~~~~~~~~~~~~~~

Actor, rollout and ref fused (HYBRID); optionally a critic for PPO. This is what the
flagship examples declare inline.

.. code-block:: yaml

   topology:
     clusters:
       - {name: default, nnodes: 2, n_gpus_per_node: 8}
     device_pools:
       - {name: hybrid_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}
     models:
       - {name: actor,   worker: actor,   config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: rollout, worker: rollout, config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: ref,     worker: ref,     config_key: actor_rollout_ref, resource_pool: hybrid_pool}
       - {name: critic,  worker: critic,  config_key: critic,            resource_pool: hybrid_pool}  # PPO only

Disaggregated rollout (report-only for now)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rollout on a pool disjoint from training derives ``STANDALONE``.

.. code-block:: yaml

   topology:
     clusters:
       - {name: default, nnodes: 4, n_gpus_per_node: 8}
     device_pools:
       - {name: train_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}
       - {name: infer_pool, cluster: default, nnodes: 2, n_gpus_per_node: 8}
     models:
       - {name: actor,   worker: actor,   resource_pool: train_pool}
       - {name: ref,     worker: ref,     resource_pool: train_pool}
       - {name: rollout, worker: rollout, resource_pool: infer_pool}

.. note::

   The declarative resolver does not yet wire a standalone rollout end-to-end; it raises
   ``NotImplementedError`` (fail-closed) so a model is never silently misplaced. Use the
   existing async / separated trainer knobs for disaggregated runs today. The startup
   report still shows the intended ``STANDALONE`` placement.

OPD — single teacher
~~~~~~~~~~~~~~~~~~~~~~

A teacher on its own pool is dedicated (no rollout mode).

.. code-block:: yaml

   topology:
     clusters:
       - {name: default, nnodes: 3, n_gpus_per_node: 8}
     device_pools:
       - {name: train_pool,   cluster: default, nnodes: 2, n_gpus_per_node: 8}
       - {name: teacher_pool, cluster: default, nnodes: 1, n_gpus_per_node: 8}
     models:
       - {name: actor,   worker: actor,   resource_pool: train_pool}
       - {name: rollout, worker: rollout, resource_pool: train_pool}
       - {name: ref,     worker: ref,     resource_pool: train_pool}
       - {name: teacher, worker: teacher, config_key: distillation, resource_pool: teacher_pool}

GPU-level placement inside one node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``device_range`` a single node's pool splits between models (actor on GPU 0-3, rollout
on GPU 4-7):

.. code-block:: yaml

   topology:
     clusters:
       - {name: default, nnodes: 1, n_gpus_per_node: 8}
     device_pools:
       - {name: node_pool, cluster: default, nnodes: 1, n_gpus_per_node: 8}
     models:
       - {name: actor,   worker: actor,   resource_pool: node_pool, device_range: [0, 4]}
       - {name: rollout, worker: rollout, resource_pool: node_pool, device_range: [4, 8]}

Heterogeneous hardware
----------------------

Declare one cluster per GPU type. Each cluster ``name`` must exist as a Ray custom
resource, so start those machines with a matching ``--resources`` value:

.. code-block:: bash

   # on each h100 machine
   ray start --resources '{"h100": 8}'
   # on each a100 machine
   ray start --resources '{"a100": 8}'

.. code-block:: yaml

   topology:
     clusters:
       - {name: h100, nnodes: 2, n_gpus_per_node: 8}
       - {name: a100, nnodes: 2, n_gpus_per_node: 8}
     device_pools:
       - {name: train_pool,   cluster: h100, nnodes: 2, n_gpus_per_node: 8}
       - {name: teacher_pool, cluster: a100, nnodes: 2, n_gpus_per_node: 8}
     models:
       - {name: actor,   worker: actor,   resource_pool: train_pool}
       - {name: rollout, worker: rollout, resource_pool: train_pool}
       - {name: teacher, worker: teacher, config_key: distillation, resource_pool: teacher_pool}

The resolver threads each pool's cluster name into ``RayResourcePool.accelerator_type`` and
validates via ``ray.cluster_resources()`` that enough of each custom resource is present.
Single-cluster (``default``) runs need no ``--resources`` and behave exactly as today.

Pool attributes
---------------

Pools accept an open-ended ``attributes`` map for capabilities that are genuinely placement
concerns. Reserved keys are type-checked and surfaced in the report; unknown keys are
carried through with a warning (forward-compatible).

.. code-block:: yaml

   device_pools:
     - {name: infer, cluster: default, nnodes: 1, n_gpus_per_node: 8, attributes: {elastic: true, min_gpus: 8, max_gpus: 16}}
     - {name: spare, cluster: default, nnodes: 1, n_gpus_per_node: 8, attributes: {standby: true, fault_tolerant: true}}  # roleless reserve

Reserved keys: ``elastic`` / ``min_gpus`` / ``max_gpus`` / ``scale_signal`` (elastic
scaling), ``standby`` / ``fault_tolerant`` / ``protects`` (fault tolerance). These are
**carried through and warned on today, not yet enforced**; enforcing them is follow-up work.
Dynamic resource scheduling stays in ``async_training.*`` — it is a training-side behavior,
not placement.

Validation
----------

``build_topology`` raises on impossible placements: unknown cluster/pool references, an
unknown ``worker`` name, a pool that does not fit its cluster, an out-of-range
``device_range``, or a policy rollout as a separate process on the actor's GPUs. The
resolver additionally checks that each server worker's GPU count (``rollout`` / ``rm`` /
``teacher``) is a multiple of that worker's rollout world size
(``tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size``).
Non-fatal issues — a modelless pool without a reserve attribute, a pool straddling node
boundaries, partial overlap with the actor pool — are returned as warnings and printed in
the report.

``trainer.nnodes`` / ``trainer.n_gpus_per_node``
------------------------------------------------

When ``topology`` is set it is the single source of truth for placement scale. Many
non-placement components (rollout servers, reward / teacher loops, batch-size validation)
still read the legacy scalars ``trainer.nnodes`` / ``trainer.n_gpus_per_node``, so rather
than dropping them the trainer **derives** them from a single-cluster topology and
overwrites them in place before validation. Setting them by hand alongside ``topology`` is
**deprecated** and warns:

- Single-cluster topology: ``trainer.nnodes`` / ``n_gpus_per_node`` are set from that
  cluster's ``nnodes`` / ``n_gpus_per_node``. A warning is logged if you also set them on
  the CLI, or if the prior values disagreed.
- Multi-cluster topology: there is no single machine shape, so the scalars are left as-is
  (with a warning if you set them by hand).

So the flagship examples omit ``trainer.nnodes`` / ``n_gpus_per_node`` and let the topology
drive them. To use the legacy path, drop the ``topology`` lines and add the scalars back.

Backward compatibility
-----------------------

* The block ships empty (``clusters: []``, ``device_pools: []``, ``models: []``); empty
  means the legacy knobs drive placement, so existing configs are byte-for-byte unaffected.
* A non-empty block (both ``clusters`` and ``models`` set) drives placement instead.
* The startup report is emitted in both cases.

Parallelism (TP/PP/DP) stays in the existing model configs (e.g.
``actor_rollout_ref.rollout.tensor_model_parallel_size``); the topology only validates that a
pool's GPU count is a multiple of it.
