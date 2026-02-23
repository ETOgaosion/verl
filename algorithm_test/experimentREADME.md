# 🚀 DP负载均衡实验脚本使用指南

## ⚠️ 开始前必读

### 🎯 并行策略配置（影响实验效果）

**默认配置**：TP=2, PP=2, DP=2  
**⚠️ 问题**：DP=2时负载均衡效果**不够显著**！

**✅ 推荐配置**（测试负载均衡）：

```bash

# 修改为推荐配置（最大化DP）
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=1
# 现在 DP=8，负载均衡效果最显著！

# 运行实验
./benchmark_balance.sh compare
```

**为什么 DP 大小很重要？**
- DP=2: 只有2个GPU参与数据并行，负载不均衡不明显
- DP=8: 8个GPU参与数据并行，负载差异和改进效果非常显著
- **论文实验建议 DP≥4**

## 📁 文件说明

- **`benchmark_balance.sh`**: 主实验脚本，自动化运行多算法对比实验

## 🎯 快速开始

### 0️⃣ 检查配置（强烈推荐）

```bash
cd /home/wsl/projects/verl-enhanced/algorithm_test

# 如果 DP<4，建议修改为：
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=1
```
### 3️⃣ 运行单个实验（调试用）

```bash
# 格式：./benchmark_balance.sh single <method> <seed>
./benchmark_balance.sh single balanced 42

# 可用的算法：
# - karmarkar_karp: Karmarkar-Karp (Baseline)
# - balanced: Stratified Balanced (Enhanced)
# - multiway: Multiway Greedy (LPT)
# - snake_full: Snake Pattern (Full)
# - greedy: Simple Greedy
```

### 1️⃣ 最简单：baseline vs enhanced 对比

```bash
cd /home/wsl/projects/verl-enhanced/algorithm_test

# 运行对比实验（推荐）
./benchmark_balance.sh compare

# 预计耗时：约 40-50 分钟（2个算法 × 3个种子 × 8分钟）
```

**该命令会**：
- 运行 Karmarkar-Karp (baseline) × 3 次
- 运行 Stratified Balanced (enhanced) × 3 次
- 自动切换算法配置
- 导出 CSV 指标
- 生成统计报告




## 📊 结果分析

### 查看实验结果

```bash
# 1. 查看汇总报告
cat /data/wsl/dp_balance_benchmark/results/summary_<timestamp>.txt

# 2. 查看详细指标（CSV）
cat /data/wsl/dp_balance_benchmark/results/metrics_<timestamp>.csv

# 3. 查看统计量（均值±标准差）
cat /data/wsl/dp_balance_benchmark/results/statistics_<timestamp>.csv
```

### 生成可视化图表

```bash
# 使用分析脚本
python analyze_results.py \
    /data/wsl/dp_balance_benchmark/results/metrics_<timestamp>.csv \
    --output-dir ./analysis_results

# 生成的图表：
# - metrics_comparison.png: 四个核心指标的箱线图
# - pipeline_time_trend.png: Pipeline时间趋势对比
# - ranking_heatmap.png: 算法排名热力图
# - statistical_tests.csv: 统计检验结果（t-test, Cohen's d）
# - metrics_table.tex: LaTeX表格（论文用）
```



## ⚙️ 配置说明

### 算法切换机制

**✅ 推荐方式：环境变量**（当前实现）

脚本通过环境变量 `VERL_PARTITION_METHOD` 自动切换算法，无需修改源码：

```bash
# 脚本内部自动设置
export VERL_PARTITION_METHOD="balanced"  # 或 karmarkar_karp, multiway 等

# 训练代码会自动读取该环境变量
python -m verl.trainer.main_ppo ...
```

**源码支持**（已实现）：

`get_seqlen_balanced_partitions` 函数会自动检查环境变量：

```python
def get_seqlen_balanced_partitions(..., method: str = "balanced"):
    import os
    env_method = os.environ.get('VERL_PARTITION_METHOD')
    if env_method:
        method = env_method  # 环境变量优先级最高
    ...
```

**⚠️ 重要：算法配置方式**

实验脚本通过**环境变量**切换不同的负载均衡算法：

```bash
# 在 seqlen_balancing.py 中自动读取环境变量
def get_seqlen_balanced_partitions(..., method: str = "balanced"):
    import os
    env_method = os.environ.get('VERL_PARTITION_METHOD')
    if env_method:
        method = env_method  # 环境变量优先级最高
    ...
```

**可用算法**：

| 算法 | 描述 | 优势 |
|------|------|------|
| **karmarkar_karp** | Karmarkar-Karp (Baseline) | 传统方法，关注总和平衡 |
| **balanced** | Stratified Balanced (Ours) | 分层同分布采样，最佳综合性能 |
| **multiway** | Multiway Greedy (LPT) | 最优工作负载平衡 |
| **snake_full** | Snake Pattern | 交替分配，简单有效 |
| **greedy** | Simple Greedy | 简单贪心分配 |




**✅ Megatron 后端配置**

脚本使用 `--config-name=ppo_megatron_trainer` 配置文件，自动启用 Megatron 后端：
- **配置路径**：`actor_rollout_ref.actor.megatron.tensor_model_parallel_size` 和 `pipeline_model_parallel_size`
- **效果**：Pipeline Parallel (PP) 真实生效，创建实际的流水线阶段
- **关键**：流水线气泡物理存在，V型排序的 25-35% 优化效果可被测量

如果使用默认 `ppo_trainer` 配置（FSDP），PP配置会被忽略！

### 环境变量

```bash
# 自定义数据集路径
export DATA_DIR=/path/to/your/dataset

# 自定义模型路径
export MODEL_PATH=/path/to/your/model

# 自定义输出目录
export OUTPUT_DIR=/path/to/output

# 自定义GPU数量
export N_GPUS=4

# 自定义并行策略（重要！影响DP负载均衡效果）
export TENSOR_PARALLEL_SIZE=1       # TP大小（默认2）
export PIPELINE_PARALLEL_SIZE=1     # PP大小（默认2）
# 注意：DP = N_GPUS / (TP * PP)

# 运行实验
./benchmark_balance.sh compare
```

**并行策略建议**：

| GPU数 | 主要目标 | 推荐配置 | DP大小 | 说明 |
|-------|---------|---------|--------|------|
| 8 | **测试DP负载均衡** | TP=1, PP=1 | **8** | ✅ 最大化DP，最佳测试效果 |
| 8 | 模型并行 + DP平衡 | TP=2, PP=1 | 4 | 平衡方案 |
| 8 | 大模型需要模型并行 | TP=2, PP=2 | 2 | DP=2效果不显著 |
| 4 | 测试DP负载均衡 | TP=1, PP=1 | **4** | ✅ 推荐 |
| 16 | 测试DP负载均衡 | TP=1, PP=1 | **16** | ✅ 最佳 |

**⚠️ 注意**：
- DP负载均衡的效果随DP大小增加而更明显
- **强烈建议在测试时设置 TP=1, PP=1**，以最大化DP大小
- 如果模型太大单卡放不下，才考虑启用TP/PP

**示例**：

```bash
# 推荐配置：最大化DP测试负载均衡效果
export N_GPUS=8
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=1
./benchmark_balance.sh compare
# 结果：DP=8，负载均衡效果最显著

# 如果模型需要模型并行
export TENSOR_PARALLEL_SIZE=2
export PIPELINE_PARALLEL_SIZE=1
./benchmark_balance.sh compare
# 结果：DP=4，负载均衡效果中等
```

### 修改实验参数

编辑 `benchmark_balance.sh` 中的 `COMMON_ARGS` 部分：

```bash
# 关键参数说明
data.train_batch_size=128              # 全局 batch size（DP分配）
data.max_prompt_length=512             # 提示词最大长度
data.max_response_length=512           # 响应最大长度

# DP负载均衡相关
trainer.balance_batch=true             # 必须为true
actor_rollout_ref.actor.use_dynamic_bsz=True  # 必须为true
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000  # MB分割阈值

# 训练规模
trainer.total_epochs=1                 # 训练轮数
trainer.test_freq=5                    # 验证频率

# 硬件配置
trainer.n_gpus_per_node=8              # 每节点GPU数（DP size）
```

### 修改实验种子

编辑 `SEEDS` 数组：

```bash
# 默认：3个种子
SEEDS=(42 2024 9527)

# 改为5个种子（更稳定）
SEEDS=(42 2024 9527 123 456)

# 改为单种子（快速测试）
SEEDS=(42)
```

## 📋 实验检查清单

### 运行前检查

- [ ] 数据集路径正确：`$DATA_DIR/verl_format/train.parquet` 存在
- [ ] 模型路径正确：`$MODEL_PATH` 目录存在
- [ ] GPU可用：`nvidia-smi` 显示足够GPU（≥ N_GPUS）
- [ ] Conda环境激活：`conda activate verl`
- [ ] 磁盘空间充足：至少 50GB（日志 + checkpoint）

### 运行中监控

```bash
# 实时查看日志
tail -f /data/wsl/dp_balance_benchmark/logs/<timestamp>/<method>_seed<N>.log

# 监控GPU使用率
watch -n 1 nvidia-smi

# 检查指标是否正常记录
grep "global_seqlen/compute_imbalance_ratio" <log_file> | wc -l
# 应该 > 5
```

### 运行后验证

- [ ] 所有实验成功完成（检查日志中是否有 "✓ completed successfully"）
- [ ] 指标正常记录（每个日志文件都有足够的指标条目）
- [ ] CSV文件包含所有实验数据
- [ ] TensorBoard日志可正常加载

## 🔧 故障排查

### GPU OOM (Out of Memory)

```bash
# 解决方案1：减小batch size
data.train_batch_size=64  # 从128改为64

# 解决方案2：减小max_token_len
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000  # 从16000减小

# 解决方案3：降低VLLM显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.4  # 从0.6降低
```

### 实验失败自动重试

脚本已内置重试机制（最多重试2次），如果仍然失败：

```bash
# 查看失败日志
cat /data/wsl/dp_balance_benchmark/logs/<timestamp>/<failed_exp>.log

# 手动重新运行失败的实验
./benchmark_balance.sh single <method> <seed>
```

### 指标未记录

检查 `ray_trainer.py` 是否正确：

```bash
# 确认负载均衡代码已启用
grep "balance_batch" ~/projects/verl-enhanced/verl/trainer/ppo/ray_trainer.py

# 确认指标记录代码存在
grep "log_seqlen_unbalance" ~/projects/verl-enhanced/verl/trainer/ppo/ray_trainer.py
```

### 源码修改未生效

```bash
# 检查当前使用的method
grep 'method: str = ' ~/projects/verl-enhanced/verl/utils/seqlen_balancing.py

# 如果修改未生效，手动设置
vim ~/projects/verl-enhanced/verl/utils/seqlen_balancing.py
# 找到 def get_seqlen_balanced_partitions(..., method: str = "XXX"):
# 修改默认值为想要的算法
```

## 📈 论文实验建议

### 最小可发表实验集

```bash
# 1. 运行主对比实验
./benchmark_balance.sh compare

# 2. 生成可视化
python analyze_results.py \
    /data/wsl/dp_balance_benchmark/results/metrics_<timestamp>.csv

# 3. 检查统计显著性
cat analysis_results/statistical_tests.csv
# 确保 p-value < 0.05
```

### 完整消融实验

```bash
# 运行所有算法
./benchmark_balance.sh all

# 对比不同场景
# - 短序列场景：data.max_response_length=256
# - 长序列场景：data.max_response_length=1024
# - 不同DP size：N_GPUS=4, 8, 16
```

## 🎓 预期结果（参考值）

根据之前的测试，预期改进：

| 指标 | Baseline | Enhanced | 改进 |
|------|----------|----------|------|
| Memory Imbalance | 0.04-0.06 | 0.006-0.01 | **~85%** ↓ |
| Compute Imbalance | 0.10-0.15 | 0.07-0.09 | **~30%** ↓ |
| Variance Discrepancy | 0.25-0.35 | 0.15-0.20 | **~40%** ↓ |
| Balance Score | 80-85 | 92-95 | **~15%** ↑ |
| Pipeline Time | 8-10s/step | 6-7s/step | **~25%** ↓ |

## 📞 技术支持

如遇问题，检查以下日志：

1. **训练日志**：`$LOG_DIR/<method>_seed<N>.log`
2. **错误日志**：搜索 "Error" 或 "Exception"
3. **Ray日志**：`/tmp/ray/session_latest/logs/`

常见问题解决：
- GPU不足 → 减少 N_GPUS 或释放其他进程
- OOM → 减小 batch_size 或 max_token_len
- 数据加载慢 → 检查磁盘IO，考虑使用SSD
- 网络错误 → 检查分布式通信（Ray）

## 🔗 相关文件

- 源码：`~/projects/verl-enhanced/verl/utils/seqlen_balancing.py`
- 训练器：`~/projects/verl-enhanced/verl/trainer/ppo/ray_trainer.py`
- 指标定义：`verl/utils/seqlen_balancing.py::log_seqlen_unbalance()`
- 配置示例：`examples/config/ppo_trainer.yaml`
