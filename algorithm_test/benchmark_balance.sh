
#!/bin/bash
set -e

# ========================================
# DP Load Balance Benchmark Script
# 负载均衡算法对比实验脚本
# ========================================


# ========== 环境配置 ==========
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export PYTHONPATH="/home/wsl/projects/verl-enhanced:${PYTHONPATH}"
export VERL_LOG_LEVEL=INFO

# ========== 路径配置 ==========
DATA_DIR="${DATA_DIR:-/data/common/dataset/gsm8k}"
MODEL_PATH="${MODEL_PATH:-/data/common/models/Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wsl/dp_balance_benchmark}"
N_GPUS="${N_GPUS:-8}"

# ========== 并行策略配置 ==========
# 在 8 卡环境下: TP=2, PP=2, DP=2 (DP = N_GPUS / (TP * PP))
# 修改建议：
#   - 如果主要测试 DP 负载均衡: TP=1, PP=1, DP=8 (最大化 DP)
#   - 如果模型较大需要模型并行: TP=2, PP=2, DP=2
#   - 如果想平衡: TP=2, PP=1, DP=4
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-2}"

# 计算实际的 DP size
DP_SIZE=$((N_GPUS / (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)))
echo "📊 Parallel Strategy: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DP_SIZE"

if [ $DP_SIZE -lt 2 ]; then
    echo "⚠️  Warning: DP size is $DP_SIZE, load balancing effects may not be significant."
    echo "   Consider reducing TP/PP to increase DP size for better load balancing test."
fi

# 日志和结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$OUTPUT_DIR/logs/$TIMESTAMP"
RESULT_DIR="$OUTPUT_DIR/results"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CHECKPOINT_DIR"

# ========== 环境检查 ==========
echo "🔍 Checking environment..."

# 检查数据集
if [ ! -f "$DATA_DIR/verl_format/train.parquet" ]; then
    echo "❌ Error: Training data not found at $DATA_DIR/verl_format/train.parquet"
    exit 1
fi
echo "✓ Training data found"

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_PATH"
    exit 1
fi
echo "✓ Model found"

# 检查GPU
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ $AVAILABLE_GPUS -lt $N_GPUS ]; then
    echo "⚠️  Warning: Need $N_GPUS GPUs, but only $AVAILABLE_GPUS available"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ GPU check passed ($AVAILABLE_GPUS GPUs available)"
fi

# 检查 verl 包
python -c "import verl; print(f'✓ verl version: {verl.__version__}')" 2>/dev/null || {
    echo "❌ Error: verl package not found"
    exit 1
}

# ========== 算法配置 ==========
declare -A ALGORITHM_CONFIGS
ALGORITHM_CONFIGS["karmarkar_karp"]="Karmarkar-Karp (Baseline)"
ALGORITHM_CONFIGS["balanced"]="Stratified Balanced (Enhanced/Ours)"
ALGORITHM_CONFIGS["multiway"]="Multiway Greedy (LPT)"
ALGORITHM_CONFIGS["snake_full"]="Snake Pattern (Full)"
ALGORITHM_CONFIGS["greedy"]="Simple Greedy"

# 消融实验变体（可选，用于深入分析）
# ALGORITHM_CONFIGS["balanced_no_v"]="Balanced without V-sort (Ablation)"
# ALGORITHM_CONFIGS["balanced_align"]="Balanced with Cross-DP Align (Ablation)"

# 实验种子（用于多次运行）
SEEDS=(42 2024 9527)

# ========== 公共训练参数 ==========
# 使用较小的配置以便快速对比
# 并行策略已在上方配置: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE, DP=$DP_SIZE
COMMON_ARGS="
    algorithm.adv_estimator=grpo
    data.train_files=$DATA_DIR/verl_format/train.parquet
    data.val_files=$DATA_DIR/verl_format/test.parquet
    data.return_raw_chat=True
    data.train_batch_size=64
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path=$MODEL_PATH
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PIPELINE_PARALLEL_SIZE
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 
     actor_rollout_ref.rollout.max_model_len=1024
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE
    trainer.logger=[\"tensorboard\",\"file\",\"console\"]
    trainer.val_before_train=False
    trainer.n_gpus_per_node=$N_GPUS
    trainer.nnodes=1
    trainer.total_epochs=1
    trainer.test_freq=5
    trainer.default_local_dir=$OUTPUT_DIR/checkpoints
    trainer.balance_batch=true
"

# ========== 辅助函数 ==========

# 运行单个实验
run_experiment() {
    local method="$1"
    local seed="$2"
    local description="${ALGORITHM_CONFIGS[$method]}"
    local exp_name="${method}_seed${seed}"
    local log_file="$LOG_DIR/${exp_name}.log"
    local checkpoint_dir="$CHECKPOINT_DIR/${exp_name}_${TIMESTAMP}"
    
    echo ""
    echo "=========================================="
    echo "📊 Running Experiment"
    echo "=========================================="
    echo "Method: $method ($description)"
    echo "Seed: $seed"
    echo "Log: $log_file"
    echo "Checkpoint: $checkpoint_dir"
    echo "=========================================="
    
    # 通过环境变量设置算法（无需修改源文件）
    export VERL_PARTITION_METHOD="$method"
    echo "📝 Set partition method via env var: VERL_PARTITION_METHOD=$method"
    
    # 运行训练（带重试机制）
    local max_retries=2
    local retry=0
    local success=false
    
    while [ $retry -le $max_retries ] && [ "$success" = false ]; do
        if [ $retry -gt 0 ]; then
            echo "⚠️  Retry attempt $retry/$max_retries..."
            sleep 5
        fi
        
        # 执行训练（使用 Megatron 配置文件）
        if python -m verl.trainer.main_ppo \
            --config-name=ppo_megatron_trainer \
            $COMMON_ARGS \
            trainer.project_name="dp_balance_benchmark" \
            trainer.experiment_name="$exp_name" \
            trainer.default_local_dir="$checkpoint_dir" \
            data.seed=$seed \
            2>&1 | tee "$log_file"; then
            
            success=true
            echo "✓ Experiment completed successfully"
        else
            retry=$((retry + 1))
            echo "❌ Experiment failed (attempt $retry)"
        fi
    done
    
    if [ "$success" = false ]; then
        echo "❌ Experiment failed after $max_retries retries"
        return 1
    fi
    
    # 验证日志包含关键指标
    echo "🔍 Validating metrics in log..."
    local metric_count=$(grep -c "global_seqlen/compute_imbalance_ratio" "$log_file" 2>/dev/null || echo "0")
    # 清理可能的换行符和空格
    metric_count=$(echo "$metric_count" | tr -d '\n' | tr -d ' ')
    if [ "$metric_count" -lt 5 ] 2>/dev/null; then
        echo "⚠️  Warning: Only found $metric_count metric entries (expected > 5)"
    else
        echo "✓ Found $metric_count metric entries"
    fi
    
    return 0
}

# 提取单个实验的指标
extract_experiment_metrics() {
    local log_file="$1"
    local method="$2"
    local seed="$3"
    
    if [ ! -f "$log_file" ]; then
        echo "N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        return 1
    fi
    
    # 提取最终指标（取最后一次记录）
    # 修复正则：使用 \\s 匹配空白符，[\d.eE+-]+ 支持科学计数法
    local memory_imb=$(grep -oP 'global_seqlen/memory_imbalance_ratio["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local compute_imb=$(grep -oP 'global_seqlen/compute_imbalance_ratio["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local var_disc=$(grep -oP 'global_seqlen/variance_discrepancy["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local balance_score=$(grep -oP 'global_seqlen/balance_quality_score["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local pipeline_time=$(grep -oP 'pipeline/update_actor_time_ms_mean["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local pipeline_cv=$(grep -oP 'pipeline/update_actor_cv["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    local sync_eff=$(grep -oP 'cross_dp_mb/sync_efficiency["\\s:]+[\d.eE+-]+' "$log_file" | tail -1 | grep -oP '[\d.eE+-]+$' || echo "N/A")
    
    echo "${memory_imb},${compute_imb},${var_disc},${balance_score},${pipeline_time},${pipeline_cv},${sync_eff}"
}

# 导出所有实验的指标到CSV
export_metrics_to_csv() {
    local csv_file="$RESULT_DIR/metrics_${TIMESTAMP}.csv"
    
    echo "📊 Exporting metrics to: $csv_file"
    
    # CSV header
    echo "Method,Description,Seed,Memory_Imbalance,Compute_Imbalance,Variance_Discrepancy,Balance_Score,Pipeline_Time_ms,Pipeline_CV,Sync_Efficiency" > "$csv_file"
    
    # 只导出实际存在的日志文件
    local found_count=0
    local total_count=0
    
    for method in "${!ALGORITHM_CONFIGS[@]}"; do
        local description="${ALGORITHM_CONFIGS[$method]}"
        
        for seed in "${SEEDS[@]}"; do
            local exp_name="${method}_seed${seed}"
            local log_file="$LOG_DIR/${exp_name}.log"
            total_count=$((total_count + 1))
            
            # 只处理存在的日志文件
            if [ -f "$log_file" ]; then
                local metrics=$(extract_experiment_metrics "$log_file" "$method" "$seed")
                echo "${method},\"${description}\",${seed},${metrics}" >> "$csv_file"
                found_count=$((found_count + 1))
            fi
        done
    done
    
    echo "✓ Metrics exported: $found_count/$total_count experiments found"
    echo ""
    
    if [ $found_count -eq 0 ]; then
        echo "⚠️  Warning: No log files found in $LOG_DIR"
        echo "   Make sure experiments have been run before exporting metrics."
    else
        echo "Preview:"
        head -n 10 "$csv_file"
    fi
}

# 生成统计摘要（均值±标准差）
generate_statistics() {
    local csv_file="$RESULT_DIR/metrics_${TIMESTAMP}.csv"
    local stats_file="$RESULT_DIR/statistics_${TIMESTAMP}.csv"
    
    echo "📈 Generating statistics: $stats_file"
    
    # 使用 Python 计算统计量
    python3 << EOF
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('$csv_file')

# 按 Method 分组计算统计量
stats = df.groupby('Method').agg({
    'Memory_Imbalance': ['mean', 'std'],
    'Compute_Imbalance': ['mean', 'std'],
    'Variance_Discrepancy': ['mean', 'std'],
    'Balance_Score': ['mean', 'std'],
    'Pipeline_Time_ms': ['mean', 'std'],
    'Pipeline_CV': ['mean', 'std'],
    'Sync_Efficiency': ['mean', 'std']
}).round(4)

# 保存统计结果
stats.to_csv('$stats_file')
print("✓ Statistics saved to: $stats_file")
print()
print("Summary:")
print(stats)
EOF
}

# 生成汇总报告
generate_summary_report() {
    local summary_file="$RESULT_DIR/summary_${TIMESTAMP}.txt"
    
    echo "📝 Generating summary report: $summary_file"
    
    cat > "$summary_file" << EOF
========================================
DP Load Balance Benchmark Summary
========================================
Date: $(date)
Timestamp: $TIMESTAMP
Output Directory: $OUTPUT_DIR
Number of GPUs: $N_GPUS
Model: $MODEL_PATH
Dataset: $DATA_DIR

Parallel Strategy:
  Tensor Parallel (TP): $TENSOR_PARALLEL_SIZE
  Pipeline Parallel (PP): $PIPELINE_PARALLEL_SIZE
  Data Parallel (DP): $DP_SIZE

Algorithms Tested:
EOF
    
    for method in "${!ALGORITHM_CONFIGS[@]}"; do
        echo "  - $method: ${ALGORITHM_CONFIGS[$method]}" >> "$summary_file"
    done
    
    echo "" >> "$summary_file"
    echo "Seeds: ${SEEDS[*]}" >> "$summary_file"
    echo "Experiments per algorithm: ${#SEEDS[@]}" >> "$summary_file"
    echo "Total experiments: $((${#ALGORITHM_CONFIGS[@]} * ${#SEEDS[@]}))" >> "$summary_file"
    echo "" >> "$summary_file"
    
    echo "========================================" >> "$summary_file"
    echo "Results (Mean ± Std)" >> "$summary_file"
    echo "========================================" >> "$summary_file"
    
    # 如果统计文件存在，追加到报告
    if [ -f "$RESULT_DIR/statistics_${TIMESTAMP}.csv" ]; then
        echo "" >> "$summary_file"
        cat "$RESULT_DIR/statistics_${TIMESTAMP}.csv" >> "$summary_file"
    fi
    
    echo "" >> "$summary_file"
    echo "========================================" >> "$summary_file"
    echo "Files Generated" >> "$summary_file"
    echo "========================================" >> "$summary_file"
    echo "Logs: $LOG_DIR" >> "$summary_file"
    echo "Metrics CSV: $RESULT_DIR/metrics_${TIMESTAMP}.csv" >> "$summary_file"
    echo "Statistics CSV: $RESULT_DIR/statistics_${TIMESTAMP}.csv" >> "$summary_file"
    echo "TensorBoard: $OUTPUT_DIR/tensorboard_log/" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "To view TensorBoard:" >> "$summary_file"
    echo "  tensorboard --logdir=$OUTPUT_DIR/tensorboard_log/" >> "$summary_file"
    echo "========================================" >> "$summary_file"
    
    cat "$summary_file"
}

# ========== 主执行逻辑 ==========

# 解析命令行参数
MODE="${1:-compare}"  # all, single, compare
SELECTED_METHOD="${2:-balanced}"
SELECTED_SEED="${3:-42}"

echo ""
echo "=========================================="
echo "🚀 DP Load Balance Benchmark"
echo "=========================================="
echo "Mode: $MODE"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

case "$MODE" in
    "single")
        # 运行单个实验
        echo "Running single experiment: $SELECTED_METHOD (seed=$SELECTED_SEED)"
        run_experiment "$SELECTED_METHOD" "$SELECTED_SEED"
        export_metrics_to_csv
        generate_summary_report
        ;;
        
    "compare")
        # 运行baseline和enhanced对比（多种子）
        echo "Running baseline vs enhanced comparison (${#SEEDS[@]} seeds each)"
        
        failed_count=0
        total_count=$((2 * ${#SEEDS[@]}))
        
        for method in "karmarkar_karp" "balanced"; do
            for seed in "${SEEDS[@]}"; do
                if ! run_experiment "$method" "$seed"; then
                    failed_count=$((failed_count + 1))
                    echo "⚠️  Experiment failed, continuing..."
                fi
            done
        done
        
        echo ""
        echo "Completed: $((total_count - failed_count))/$total_count experiments"
        
        export_metrics_to_csv
        generate_statistics
        generate_summary_report
        ;;
        
    "all")
        # 运行所有算法（多种子）
        echo "Running all algorithms (${#SEEDS[@]} seeds each)"
        
        failed_count=0
        total_count=$((${#ALGORITHM_CONFIGS[@]} * ${#SEEDS[@]}))
        
        for method in "${!ALGORITHM_CONFIGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if ! run_experiment "$method" "$seed"; then
                    failed_count=$((failed_count + 1))
                    echo "⚠️  Experiment failed, continuing..."
                fi
            done
        done
        
        echo ""
        echo "Completed: $((total_count - failed_count))/$total_count experiments"
        
        export_metrics_to_csv
        generate_statistics
        generate_summary_report
        ;;
        
    *)
        echo "Usage: $0 {all|single|compare} [method] [seed]"
        echo ""
        echo "Modes:"
        echo "  compare   - Run baseline vs enhanced comparison (default, 3 seeds each)"
        echo "  all       - Run all algorithms (3 seeds each)"
        echo "  single    - Run a single experiment (specify method and seed)"
        echo ""
        echo "Available methods:"
        for method in "${!ALGORITHM_CONFIGS[@]}"; do
            echo "  - $method: ${ALGORITHM_CONFIGS[$method]}"
        done
        echo ""
        echo "Example:"
        echo "  $0 compare                    # Baseline vs Enhanced (recommended)"
        echo "  $0 single balanced 42         # Single run with balanced method"
        echo "  $0 all                        # Full benchmark (all methods)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Benchmark Complete!"
echo "=========================================="
echo "Results saved to: $RESULT_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Review summary: cat $RESULT_DIR/summary_${TIMESTAMP}.txt"
echo "  2. View metrics: cat $RESULT_DIR/metrics_${TIMESTAMP}.csv"
echo "  3. TensorBoard: tensorboard --logdir=$OUTPUT_DIR/tensorboard_log/"
echo "=========================================="
