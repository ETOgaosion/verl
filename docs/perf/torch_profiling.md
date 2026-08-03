# PyTorch Profiling in verl

Last updated: 07/31/2026.

This guide explains how to use the native [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for profiling verl training runs.

## Configuration

Profiling in verl can be configured through parameters in the trainer configuration file (e.g., `ppo_trainer.yaml`).

### Global Profiling Control

In `global_profiler`, you can control when and how profiling occurs globally:

* **`global_profiler.steps`**: List of step numbers to profile. E.g., `[1, 2, 5]` profiles steps 1, 2, and 5. Set to `null` to disable.
* **`global_profiler.save_path`**: Directory to save the profiling results. Default is `outputs/profile`.

### Role Profiling Control

Each RL role (Actor, Critic, etc.) has its own `profiler` configuration:

* **`enable`**: Whether to enable profiling for this role.
* **`all_ranks`**: If `True`, profiles all ranks.
* **`ranks`**: List of specific ranks to profile if `all_ranks` is `False`.
* **`tool_config.torch`**: Configuration specific to the PyTorch Profiler.

#### PyTorch Profiler Options (`tool_config.torch`)

You can customize the PyTorch Profiler behavior using the following fields under `tool_config.torch`:

* **`contents`**: List of contents to profile. An empty list (the default) collects everything.
    *   **`cpu`**: Profile CPU activities. Collected whether or not you list it: operator names and
        verl's per-stage markers are CPU-side events, so a device-only trace would be bare kernels
        with no way to tell which stage they belong to. Listing it is therefore redundant, and the
        rest of `contents` is honored as written.
    *   **`cuda`**: Profile CUDA activities.
    *   **`memory`**: Track tensor memory allocation/free.
    *   **`shapes`**: Record shapes of operator inputs.
    *   **`stack`**: Record source code file and line number.
* **`profile_token_start`**: Effective only for the rollout role; defines the start response-token index for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`profile_token_end`**: Effective only for the rollout role; defines the stop response-token index (exclusive) for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`schedule`**: (Advanced) Enables [`torch.profiler.schedule`](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) so that only part of the Actor update loop is recorded. It only takes effect when `active > 0`; otherwise the profiler collects continuously (the default). verl advances the schedule by calling `profiler.step()` once per mini-batch, so **every field below counts mini-batches**, not RL steps. The fields mirror the official PyTorch API, whose `steps` are these mini-batches:
    *   **`skip_first`**: Number of initial mini-batches to ignore before the first `wait` begins.
    *   **`wait`**: Mini-batches to idle (no collection) before each recording.
    *   **`warmup`**: Mini-batches to trace but discard, letting the profiler stabilize, before each recording.
    *   **`active`**: Mini-batches to actively record, and therefore how many mini-batches each trace file holds. Set `<= 0` (default) to disable scheduling.
    *   **`repeat`**: How many times to run `wait -> warmup -> active`, i.e. how many trace files one profiled RL step produces. `0` (default) keeps repeating until profiling stops.


## Examples

### 1. Whole-Step Collection

Collects one trace file per profiled RL step per process, holding everything that process ran
during the step. Note that this is the whole step *as seen by one worker*, not the whole RL
system: see [What one RL step looks like on disk](#what-one-rl-step-looks-like-on-disk).

```yaml
global_profiler:
  steps: [1, 2, 5]
  save_path: ./outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          contents: [cpu, cuda]
  # rollout & ref follow actor settings
```

### 2. Discrete Mode Collection

Discrete mode saves a separate trace file per stage within each profiled step. This is useful for detailed analysis and is **mandatory** when using Agent Loop.

**Configuration Example**

This configuration supports profiling both Training (Actor) and Inference (Rollout). You can enable/disable them independently.

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True # Set to True to profile training
      all_ranks: False
      ranks: [0] # Global Rank 0
      tool_config:
        torch:
          discrete: True
          contents: [cpu, cuda]
  rollout:
    profiler:
      enable: True # Set to True to profile inference
      all_ranks: False
      ranks: [0] # In Agent Loop, this is the Replica Rank (e.g. 0-th instance)
      tool_config:
        torch:
          discrete: True # REQUIRED 
          # Optional response-token window for rollout engine side collection.
          # If start/stop are not set, the entire rollout stage is collected.
          # Collect tokens in [12, 46), i.e. token index 12~45.
          profile_token_start: 12
          profile_token_end: 46
  # ref follow actor settings
```

**Agent Loop Mode Description**

When Rollout runs in [Agent Loop](../advance/agent_loop.rst) mode, performance data for the Rollout phase **must be collected using discrete mode**. In this case, the Profiler is triggered by the inference engine backend.

1. Rank Definition: ranks in the Rollout configuration refers to Replica Rank (inference instance
   index), not Global Rank. A run has `rollout.nnodes * rollout.n_gpus_per_node /
   rollout.tensor_model_parallel_size` replicas, so with 2 nodes of 8 GPUs at `tp=2` the valid
   values are `0`-`7`. Values outside that range never match a replica and are silently ignored,
   which is easy to hit when copying `ranks` from a training role, where they are global ranks.

2. Inference Engine Support: Currently, vLLM and SGLang engines are supported without additional settings. Specific details are as follows:

   *   **vLLM Engine**: Automatically collects AsyncLLM scheduling stacks and inference process performance data.
   *   **SGLang Engine**: Automatically collects inference process performance data. Does not support the memory option in contents.

3. Collection Window: rollout replicas are profiled for the whole training step. Generation is
   decoupled from the step in the V1 trainer -- prompts are served asynchronously and consumed from
   the replay buffer -- so there is no single generation call to wrap.

4. Trace Location: each replica writes to its own
   `<save_path>/agent_loop_rollout_replica_<n>/` directory on the node that hosts it, and
   `finish_hook_cmd` runs there with `VERL_PROFILE_SAVE_PATH` pointing at that directory. Set the
   hook if you need the traces collected somewhere central, since nothing else moves them off the
   replica's node.

### 3. Scheduled Collection (`wait`/`warmup`/`active`/`repeat`)

For long update loops with many mini-batches (e.g. large gradient accumulation), you usually don't need to trace every mini-batch. A `schedule` records only a few mini-batches at a time, keeping traces small while still capturing steady-state behavior. verl calls `profiler.step()` once per mini-batch so the schedule advances automatically.

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          contents: [cpu, cuda]
          schedule:
            skip_first: 1  # ignore the very first mini-batch
            wait: 1        # then idle 1 mini-batch before recording
            warmup: 1      # warm up 1 mini-batch (traced but discarded)
            active: 3      # record 3 mini-batches
            repeat: 2      # record two such groups, then stop collecting
  # rollout & ref follow actor settings
```

With the configuration above, within each profiled RL step verl skips mini-batch 0, then runs
`wait(1) -> warmup(1) -> active(3)` twice, producing two trace files: one holding mini-batches
3-5 (suffix `_mb3-5`) and one holding mini-batches 8-10 (suffix `_mb8-10`). If the update loop has
fewer mini-batches than the schedule needs, only the mini-batches that were reached are recorded.

`schedule` only applies to the training update loop. It is a no-op for the rollout engine side, which uses `profile_token_start`/`profile_token_end` instead. SFT has no mini-batch loop and advances the schedule once per training step, so there each unit is one `train_batch` call.

## Output file naming

Because profiling runs in every training process, each trace file is named so it can be
attributed to a specific process without opening it. The stem is:

```
[<role>_][<scope>_][step<S>_]rank<r>[-of-<world>][_tp<..>-pp<..>-dp<..>-cp<..>]_pid<pid>_<timestamp>[_mb<A>[-<B>]].json.gz
```

* **`role`**: the worker role (e.g. `actor`, `ref`, `value-model` for the critic), so results
  from different roles at the same rank are distinguishable. A colocated hybrid worker reports
  its combined role, `actor-rollout-ref` (underscores in labels become hyphens, since underscore
  separates the fields).
* **`scope`**: the profiled region passed to `start_profile`/`annotate` -- `train` for a training
  worker's whole-step window, or a stage name such as `actor-update` in discrete mode.
* **`step<S>`**: the RL step (`global_steps`) being profiled, i.e. one of `global_profiler.steps`.
* **`rank`/`world`**: the global `torch.distributed` rank and world size.
* **`tp/pp/dp/cp`**: tensor/pipeline/data/context parallel ranks, included when Megatron's
  parallel state is initialized (plain FSDP data parallelism only reports `rank`).
* **`mb<A>[-<B>]`**: for scheduled runs only, the mini-batches of the update loop this file
  actually contains, counted from the start of the profiled RL step. `_mb3-5` means the trace
  holds mini-batches 3, 4 and 5 of RL step `<S>`. An unscheduled run collects the whole RL step
  into one file and adds no suffix.

### What one RL step looks like on disk

No single trace file covers a whole RL step end to end, because the step does not run in a
single process. A profiled step leaves you with:

* **Training-side traces** (`scope` = `train`), written by the actor/ref/critic workers. These
  hold the work those workers actually run: the log-prob forwards and the actor update's
  forward/backward/optimizer. This is why the scope is called `train` and not `e2e`.
* **Rollout traces**, written by the inference engines themselves into
  `<save_path>/agent_loop_rollout_replica_<n>/`, on the node hosting each replica. Generation
  never appears in a training-side trace: with Agent Loop the engines run in their own
  processes, and in the V1 trainer generation is decoupled from the step entirely (the trainer
  samples already-generated data from the replay buffer), so the actor process is simply idle
  while the rollout happens.

To reason about the full step, read the timings that the trainer logs (`gen`, `old_log_prob`,
`ref`, `update_actor`, ...) and open the per-process traces for whichever part you're drilling
into. Trace timestamps are wall clock, so training and rollout traces from the same step can be
lined up side by side in Perfetto.

### Telling stages apart

Within a training-side trace, `discrete` decides whether the stage shows up in the file name or
inside the trace:

* **`discrete: True`** writes one file per stage, and the stage lands in `scope`:
  `actor-rollout-ref_actor-update_step2_rank0-of-8_pid123_<ts>.json.gz`,
  plus siblings for `actor-compute-log-prob`, `ref-compute-log-prob`, `train-batch` and so on.
  Use this when you want to attribute time to a stage from the file name alone.
* **`discrete: False`** (the default) collects the worker's whole step into one trace, so
  `scope` is `train` and cannot name a single stage. The stages are still separated *inside* the
  trace: each one is wrapped in a `torch.profiler.record_function` carrying its stage label, which
  names the role and the function together -- `actor_compute_log_prob`, `ref_compute_log_prob`,
  `actor_update` -- so searching for it in Perfetto/Chrome tracing gives that stage's window.
  Stages that declare no role, such as the inner engine's `train_batch`, appear under the method
  name.

Note that verl asks the profiler to write exactly `<stem>.json.gz`. If your files carry an
extra segment (e.g. `<stem>.json.1785391501.gz`), it was added after the fact by whatever
post-processes or uploads them, such as a `finish_hook_cmd`.

### Missing roles or stages

Seeing a single `actor...` file per rank, with no separate reference/critic file and no
`compute_log_prob` anywhere, is usually one of the following rather than a lost trace:

* **One file per process, not per role.** The PyTorch profiler is process-global, so a colocated
  hybrid worker records actor *and* reference work into one trace named after the combined role
  (`actor-rollout-ref`). A separate file appears only for a role that runs in its own process,
  e.g. a critic (`value-model`), or a reference model that is not colocated.
* **The hybrid worker follows `actor.profiler`.** It builds its profiler from
  `actor_rollout_ref.actor.profiler`, so `ref.profiler.enable: True` on its own profiles nothing,
  and turning the actor's profiler off also drops the reference stages that share the process.
* **The role may not exist.** There is no critic unless the algorithm uses a value model (GRPO
  and friends do not), and no reference model unless a KL term needs one.
* **Traces collected before CPU activity became unconditional.** A device-only run
  (`contents: [cuda]` on an older verl) has no `record_function` ranges and no operator names, so
  no stage can be located in it even though the kernels of every stage are there, and a log-prob
  forward looks just like the forward half of the update.
* **`compute_log_prob` can legitimately be skipped.** With
  `algorithm.rollout_correction.bypass_mode=True` the trainer reuses the rollout's log probs
  instead of recomputing them, so the actor forward never runs. With LoRA
  (`ref_in_actor`) the reference forward is served by `compute_log_prob` on the actor worker, so
  it appears under that name instead of `compute_ref_log_prob`.

## Traces with no GPU kernels

If a trace only contains CPU operators and no CUDA kernels, the profiler's CUPTI subscription
most likely lost a race. CUPTI accepts a single subscriber per process, and some CUDA images
install a startup hook that points `NVTX_INJECTION64_PATH` at `libcupti.so`. The first NVTX range
in the process then loads libcupti as the NVTX handler and takes that slot, after which Kineto
fails with `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` and drops every CUDA activity. verl
emits NVTX ranges itself, so `NCCL_NVTX_DISABLE=1` does not avoid it.

verl therefore points `NVTX_INJECTION64_PATH` at an unloadable path for all workers when
`global_profiler.tool=torch`, and logs a warning when it does. Set `VERL_KEEP_NVTX_INJECTION=1`
to keep the inherited value, e.g. when you rely on that injection for another tool and accept
traces without CUDA activity.

## Visualization

Collected trace files (usually `.json` or `.json.gz`) are stored flat in the configured
`save_path`: every role, rank and scope writes there directly, since the naming scheme above
already keeps the files unique and self-describing. This also means `finish_hook_cmd`, which
receives `save_path` via `VERL_PROFILE_SAVE_PATH`, sees all of them without recursing.

To ship the traces somewhere after each profiled step, set the hook once on `global_profiler`
(every role inherits it) and single-quote it so your shell does not expand the variable early:

```bash
    global_profiler.finish_hook_cmd='my-upload-tool "$VERL_PROFILE_SAVE_PATH"'
```

The hook prints the command, its output and its exit code to the worker's log on every
`stop_profile`. Rollout replicas run it from their own server actor once the engine has flushed,
with `VERL_PROFILE_SAVE_PATH` set to that replica's `agent_loop_rollout_replica_<n>` directory.
See [Nsight Systems profiling](nsight_profiling.md) for the full description of the hook,
including how to choose which ranks run it.

You can visualize them using:

1.  **Chrome Tracing**: Open `chrome://tracing` in a Chrome browser and load the JSON file.
2.  **Perfetto**: Open [ui.perfetto.dev](https://ui.perfetto.dev/) and load the file (recommended for large traces).
3.  **TensorBoard**: If using the TensorBoard plugin for PyTorch Profiler.
