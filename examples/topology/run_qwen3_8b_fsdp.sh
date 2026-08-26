#!/usr/bin/env bash
# Declarative device topology (RFC #7269) | Qwen3-8B | GRPO | FSDP | 1 node x 8 GPUs.
#
# This is an ordinary GRPO run with ONE addition: a `topology` block declared directly on the CLI
# (topology.clusters / device_pools / models). There is no preset to select and no `topology=<mode>`:
# you state the placement and the trainer INFERS the rollout mode. Here actor+rollout+ref share
# (config_key: actor_rollout_ref, resource_pool: hybrid_pool), so they fuse into one process and rollout
# derives HYBRID; the pool spans all GPUs. The trainer logs a "Resolved topology" report at startup
# showing which cluster/GPU runs which model. Because `topology` is the single source of truth,
# trainer.nnodes / n_gpus_per_node are derived from the cluster (setting them by hand is deprecated)
# and are intentionally omitted below. To use the legacy path instead, drop the TOPOLOGY overrides and
# add `trainer.nnodes=${NNODES} trainer.n_gpus_per_node=${NGPUS_PER_NODE}`.
#
# Knobs:
#   INFER_BACKEND   rollout backend: vllm | sglang        (default: vllm)
#   MODEL_PATH      HF model id or local path             (default: Qwen/Qwen3-8B)

set -xeuo pipefail

INFER_BACKEND=${INFER_BACKEND:-vllm}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}

# Declare the topology directly: one cluster sized NNODES x NGPUS_PER_NODE, one pool over all of it, and
# actor+rollout+ref fused on it (rollout -> HYBRID, inferred). Edit `models` / add `device_pools` for
# other layouts; the rollout mode is always derived, never named.
TOPOLOGY=(
    "topology.clusters=[{name: default, nnodes: ${NNODES}, n_gpus_per_node: ${NGPUS_PER_NODE}}]"
    "topology.device_pools=[{name: hybrid_pool, cluster: default, nnodes: ${NNODES}, n_gpus_per_node: ${NGPUS_PER_NODE}}]"
    "topology.models=[\
        {name: actor, worker: actor, config_key: actor_rollout_ref, resource_pool: hybrid_pool}, \
        {name: rollout, worker: rollout, config_key: actor_rollout_ref, resource_pool: hybrid_pool}, \
        {name: ref, worker: ref, config_key: actor_rollout_ref, resource_pool: hybrid_pool}]"
)

PROJECT_NAME=${PROJECT_NAME:-verl_topology_demo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_topology_${INFER_BACKEND}_$(date +%Y%m%d_%H%M)}

# uv (set VERL_USE_UV=0 for system python): run the driver and every Ray worker through `uv run`
# on the matching extras. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra "${INFER_BACKEND}" --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${INFER_BACKEND} --extra fsdp")
fi

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${TOPOLOGY[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$HOME/data/gsm8k/train.parquet" \
    data.val_files="$HOME/data/gsm8k/test.parquet" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "${RAY[@]}" \
    "$@"
