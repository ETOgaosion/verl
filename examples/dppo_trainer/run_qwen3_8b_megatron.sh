
# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/gaoziyuan/models/Qwen/Qwen3-8B-Base}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

WANDB_API_KEY=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

# LOSS_MODE selects DPPO variant: dppo_tv | dppo_kl (or vanilla for GRPO baseline)
LOSS_MODE=${LOSS_MODE:-dppo_tv}
case $LOSS_MODE in
    dppo_tv)  CLIP_DEFAULT=0.15 ;;
    dppo_kl)  CLIP_DEFAULT=0.05 ;;
    vanilla)  CLIP_DEFAULT=0.20 ;;
    *) echo "Unknown LOSS_MODE: $LOSS_MODE"; exit 1 ;;
esac
clip_ratio_low=${CLIP_LOW:-$CLIP_DEFAULT}
clip_ratio_high=${CLIP_HIGH:-$CLIP_DEFAULT}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-2}
actor_cp=${ACTOR_CP:-2}
actor_pp=${ACTOR_PP:-1}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
rollout_n=${ROLLOUT_N:-16}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_async_exp}
experiment_name=${EXPERIMENT_NAME:-verl_sync_qwen3_8b_${LOSS_MODE}_vllm_megatron}
# ---- end user-adjustable ----

train_file=${TRAIN_FILE:-/mnt/hdfs/gaoziyuan/data/dapo/dapo-math-17k.parquet}
val_file=()
val_file+=("/mnt/hdfs/gaoziyuan/data/dapo/aime-2024.parquet")
val_file+=("/mnt/hdfs/gaoziyuan/data/dapo/aime-2025-1-verl.parquet")
val_file+=("/mnt/hdfs/gaoziyuan/data/dapo/aime-2025-2-verl.parquet")
val_file+=("/mnt/hdfs/gaoziyuan/data/dapo/math_lv3to5_test.parquet")

val_files_str="["
for f in "${val_file[@]}"; do
    val_files_str+="'$f',"
done
val_files_str="${val_files_str%,}]"

custom_verify_fn_src_path="examples/rewards/custome_reward_fn/math_dapo.py"
custom_verify_fn_name="compute_score_math_dapo_boxed"
custom_dataset_src_path="examples/rewards/custom_dataset/boxed.py"
custom_dataset_name="BoxedRLHFDataset"

reward_manager="dapo"
enable_overlong_buffer=True
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-"1024"}
overlong_buffer_penalty_factor=1.0
overlong_buffer_log=True

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=False
    data.train_files="['$train_file']"
    data.val_files="${val_files_str}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.custom_cls.path="${custom_dataset_src_path}"
    data.custom_cls.name="${custom_dataset_name}"
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10000.0
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    +trainer.wandb_proxy=http://sys-proxy-rd-relay.byted.org:8118
)

EXTRA=(
    model_engine=megatron
    reward.reward_manager.name=${reward_manager} \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_buffer_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=${overlong_buffer_log} \
    custom_reward_function.path="${custom_verify_fn_src_path}"
    custom_reward_function.name="${custom_verify_fn_name}"
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}"
