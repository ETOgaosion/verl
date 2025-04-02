+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=1024 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=qwen2_7b_function_rm trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 00:06:11,843	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=1601310)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1601310)[0m                                                              'hf_model',
[36m(TaskRunner pid=1601310)[0m                                                              'optimizer',
[36m(TaskRunner pid=1601310)[0m                                                              'extra']},
[36m(TaskRunner pid=1601310)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=1601310)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=1601310)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1601310)[0m                                                  'optimizer_offload': True,
[36m(TaskRunner pid=1601310)[0m                                                  'param_offload': True,
[36m(TaskRunner pid=1601310)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1601310)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=1601310)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=1601310)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=1601310)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=1601310)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=1601310)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1601310)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=1601310)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=1601310)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=1601310)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=1601310)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1601310)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m                                  'ppo_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=1601310)[0m                                  'ppo_mini_batch_size': 256,
[36m(TaskRunner pid=1601310)[0m                                  'shuffle': False,
[36m(TaskRunner pid=1601310)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=1601310)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1601310)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=1601310)[0m                                  'use_fused_kernels': True,
[36m(TaskRunner pid=1601310)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=1601310)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=1601310)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=1601310)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=1601310)[0m                                  'external_lib': None,
[36m(TaskRunner pid=1601310)[0m                                  'override_config': {},
[36m(TaskRunner pid=1601310)[0m                                  'path': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=1601310)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=1601310)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=1601310)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1601310)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1601310)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m                                'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=1601310)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=1601310)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=1601310)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=1601310)[0m                                    'do_sample': True,
[36m(TaskRunner pid=1601310)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=1601310)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=1601310)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=1601310)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=1601310)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=1601310)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=1601310)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=1601310)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1601310)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m                                    'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=1601310)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=1601310)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=1601310)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=1601310)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=1601310)[0m                                    'n': 5,
[36m(TaskRunner pid=1601310)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=1601310)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=1601310)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=1601310)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=1601310)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=1601310)[0m                                    'top_k': -1,
[36m(TaskRunner pid=1601310)[0m                                    'top_p': 1,
[36m(TaskRunner pid=1601310)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=1601310)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=1601310)[0m                                                   'n': 1,
[36m(TaskRunner pid=1601310)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=1601310)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=1601310)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=1601310)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=1601310)[0m                'gamma': 1.0,
[36m(TaskRunner pid=1601310)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=1601310)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=1601310)[0m                'lam': 1.0},
[36m(TaskRunner pid=1601310)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1601310)[0m                                         'hf_model',
[36m(TaskRunner pid=1601310)[0m                                         'optimizer',
[36m(TaskRunner pid=1601310)[0m                                         'extra']},
[36m(TaskRunner pid=1601310)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=1601310)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1601310)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=1601310)[0m No module named 'vllm._version'
[36m(TaskRunner pid=1601310)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=1601310)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=1611494)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1611494)[0m No module named 'vllm._version'
[36m(pid=1611494)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=1611781)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1611781)[0m No module named 'vllm._version'
[36m(pid=1611781)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=1601310)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1601310)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=1601310)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=1601310)[0m                       'external_lib': None,
[36m(TaskRunner pid=1601310)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1601310)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=1601310)[0m                                       'param_offload': False,
[36m(TaskRunner pid=1601310)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1601310)[0m                       'override_config': {},
[36m(TaskRunner pid=1601310)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=1601310)[0m                       'tokenizer_path': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=1601310)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=1601310)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=1601310)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1601310)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=1601310)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=1601310)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=1601310)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=1601310)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1601310)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1601310)[0m             'ppo_mini_batch_size': 256,
[36m(TaskRunner pid=1601310)[0m             'shuffle': False,
[36m(TaskRunner pid=1601310)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=1601310)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1601310)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=1601310)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=1601310)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=1601310)[0m           'image_key': 'images',
[36m(TaskRunner pid=1601310)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=1601310)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=1601310)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=1601310)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=1601310)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=1601310)[0m           'shuffle': True,
[36m(TaskRunner pid=1601310)[0m           'tokenizer': None,
[36m(TaskRunner pid=1601310)[0m           'train_batch_size': 1024,
[36m(TaskRunner pid=1601310)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=1601310)[0m           'truncation': 'error',
[36m(TaskRunner pid=1601310)[0m           'val_batch_size': None,
[36m(TaskRunner pid=1601310)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=1601310)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=1601310)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1601310)[0m                   'max_length': None,
[36m(TaskRunner pid=1601310)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=1601310)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1601310)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=1601310)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1601310)[0m                                             'param_offload': False,
[36m(TaskRunner pid=1601310)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1601310)[0m                             'input_tokenizer': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=1601310)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=1601310)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=1601310)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=1601310)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=1601310)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1601310)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=1601310)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=1601310)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=1601310)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=1601310)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm',
[36m(TaskRunner pid=1601310)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=1601310)[0m              'experiment_name': 'qwen2_7b_function_rm',
[36m(TaskRunner pid=1601310)[0m              'logger': ['console'],
[36m(TaskRunner pid=1601310)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=1601310)[0m              'nnodes': 1,
[36m(TaskRunner pid=1601310)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=1601310)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=1601310)[0m              'resume_from_path': False,
[36m(TaskRunner pid=1601310)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=1601310)[0m              'save_freq': -1,
[36m(TaskRunner pid=1601310)[0m              'test_freq': 5,
[36m(TaskRunner pid=1601310)[0m              'total_epochs': 1,
[36m(TaskRunner pid=1601310)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=1601310)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=1601310)[0m WARNING 04-02 00:06:24 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=1601310)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=1601310)[0m dataset len: 7473
[36m(TaskRunner pid=1601310)[0m filter dataset len: 7473
[36m(TaskRunner pid=1601310)[0m dataset len: 1319
[36m(TaskRunner pid=1601310)[0m filter dataset len: 1319
[36m(TaskRunner pid=1601310)[0m Size of train dataloader: 7
[36m(TaskRunner pid=1601310)[0m Total training steps: 1
[36m(pid=1611494)[0m WARNING 04-02 00:06:36 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=1611781)[0m WARNING 04-02 00:06:46 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(WorkerDict pid=1611494)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=1611494)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=1611494)[0m   "architectures": [
[36m(WorkerDict pid=1611494)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1611494)[0m   ],
[36m(WorkerDict pid=1611494)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=1611494)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=1611494)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=1611494)[0m   "hidden_size": 896,
[36m(WorkerDict pid=1611494)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=1611494)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=1611494)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=1611494)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=1611494)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=1611494)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=1611494)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=1611494)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=1611494)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=1611494)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=1611494)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=1611494)[0m   "rope_scaling": null,
[36m(WorkerDict pid=1611494)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=1611494)[0m   "sliding_window": null,
[36m(WorkerDict pid=1611494)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=1611494)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=1611494)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=1611494)[0m   "use_cache": true,
[36m(WorkerDict pid=1611494)[0m   "use_mrope": false,
[36m(WorkerDict pid=1611494)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=1611494)[0m   "vocab_size": 151936
[36m(WorkerDict pid=1611494)[0m }
[36m(WorkerDict pid=1611494)[0m 
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=1024', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=256', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=5', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=qwen2_7b_function_rm', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
Traceback (most recent call last):
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 54, in main
    run_ppo(config)
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 72, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/worker.py", line 2782, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/worker.py", line 929, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ImportError): [36mray::TaskRunner.run()[39m (pid=1601310, ip=127.0.0.1, actor_id=26434996a707a1c601bfe3a001000000, repr=<main_ppo.TaskRunner object at 0x7fbce607c8d0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 170, in run
    trainer.init_workers()
  File "/opt/tiger/verl/verl/trainer/ppo/ray_trainer.py", line 649, in init_workers
    self.ref_policy_wg.init_model()
  File "/opt/tiger/verl/verl/single_controller/ray/base.py", line 42, in func
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ray.exceptions.RayTaskError(ImportError): [36mray::WorkerDict.ref_init_model()[39m (pid=1611494, ip=127.0.0.1, actor_id=041abc59d46266a83829c73401000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7f55155bfc90>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/single_controller/ray/base.py", line 419, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/workers/fsdp_workers.py", line 413, in init_model
    self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/workers/fsdp_workers.py", line 207, in _build_model_optimizer
    apply_monkey_patch(model=actor_module)
  File "/opt/tiger/verl/verl/models/transformers/monkey_patch.py", line 110, in apply_monkey_patch
    from transformers.models.llama.modeling_llama import LlamaFroCausalLM
ImportError: cannot import name 'LlamaFroCausalLM' from 'transformers.models.llama.modeling_llama' (/usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py)

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=1611781)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
