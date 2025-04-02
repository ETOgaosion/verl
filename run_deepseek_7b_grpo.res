+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=deepseek_llm_7b_function_rm trainer.n_gpus_per_node=16 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 16:20:02,117	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=700829)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=700829)[0m                                                              'hf_model',
[36m(TaskRunner pid=700829)[0m                                                              'optimizer',
[36m(TaskRunner pid=700829)[0m                                                              'extra']},
[36m(TaskRunner pid=700829)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=700829)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=700829)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=700829)[0m                                                  'optimizer_offload': False,
[36m(TaskRunner pid=700829)[0m                                                  'param_offload': False,
[36m(TaskRunner pid=700829)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=700829)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=700829)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=700829)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=700829)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=700829)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=700829)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=700829)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=700829)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=700829)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=700829)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=700829)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=700829)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m                                  'ppo_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=700829)[0m                                  'ppo_mini_batch_size': 16,
[36m(TaskRunner pid=700829)[0m                                  'shuffle': False,
[36m(TaskRunner pid=700829)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=700829)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=700829)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=700829)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=700829)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=700829)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=700829)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=700829)[0m                                  'external_lib': None,
[36m(TaskRunner pid=700829)[0m                                  'override_config': {},
[36m(TaskRunner pid=700829)[0m                                  'path': 'deepseek-ai/deepseek-llm-7b-chat',
[36m(TaskRunner pid=700829)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=700829)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=700829)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=700829)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=700829)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m                                'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=700829)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=700829)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=700829)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=700829)[0m                                    'do_sample': True,
[36m(TaskRunner pid=700829)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=700829)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=700829)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=700829)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=700829)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=700829)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=700829)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=700829)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=700829)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m                                    'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=700829)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=700829)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=700829)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=700829)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=700829)[0m                                    'n': 4,
[36m(TaskRunner pid=700829)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=700829)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=700829)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=700829)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=700829)[0m                                    'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=700829)[0m                                    'top_k': -1,
[36m(TaskRunner pid=700829)[0m                                    'top_p': 1,
[36m(TaskRunner pid=700829)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=700829)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=700829)[0m                                                   'n': 1,
[36m(TaskRunner pid=700829)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=700829)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=700829)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=700829)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=700829)[0m                'gamma': 1.0,
[36m(TaskRunner pid=700829)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=700829)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=700829)[0m                'lam': 1.0},
[36m(TaskRunner pid=700829)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=700829)[0m                                         'hf_model',
[36m(TaskRunner pid=700829)[0m                                         'optimizer',
[36m(TaskRunner pid=700829)[0m                                         'extra']},
[36m(TaskRunner pid=700829)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=700829)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=700829)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=700829)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=700829)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=700829)[0m No module named 'vllm._version'
[36m(TaskRunner pid=700829)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=700829)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=711015)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=711015)[0m No module named 'vllm._version'
[36m(pid=711015)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=711315)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=711315)[0m No module named 'vllm._version'
[36m(pid=711315)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=711308)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 6x across cluster][0m
[36m(pid=711308)[0m No module named 'vllm._version'[32m [repeated 6x across cluster][0m
[36m(pid=711308)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 6x across cluster][0m
[36m(TaskRunner pid=700829)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=700829)[0m                       'external_lib': None,
[36m(TaskRunner pid=700829)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=700829)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=700829)[0m                                       'param_offload': False,
[36m(TaskRunner pid=700829)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=700829)[0m                       'override_config': {},
[36m(TaskRunner pid=700829)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=700829)[0m                       'tokenizer_path': 'deepseek-ai/deepseek-llm-7b-chat',
[36m(TaskRunner pid=700829)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=700829)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=700829)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=700829)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=700829)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=700829)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=700829)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=700829)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=700829)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=700829)[0m             'ppo_mini_batch_size': 16,
[36m(TaskRunner pid=700829)[0m             'shuffle': False,
[36m(TaskRunner pid=700829)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=700829)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=700829)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=700829)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=700829)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=700829)[0m           'image_key': 'images',
[36m(TaskRunner pid=700829)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=700829)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=700829)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=700829)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=700829)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=700829)[0m           'shuffle': True,
[36m(TaskRunner pid=700829)[0m           'tokenizer': None,
[36m(TaskRunner pid=700829)[0m           'train_batch_size': 16,
[36m(TaskRunner pid=700829)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=700829)[0m           'truncation': 'error',
[36m(TaskRunner pid=700829)[0m           'val_batch_size': None,
[36m(TaskRunner pid=700829)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=700829)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=700829)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=700829)[0m                   'max_length': None,
[36m(TaskRunner pid=700829)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=700829)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=700829)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=700829)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=700829)[0m                                             'param_offload': False,
[36m(TaskRunner pid=700829)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=700829)[0m                             'input_tokenizer': 'deepseek-ai/deepseek-llm-7b-chat',
[36m(TaskRunner pid=700829)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=700829)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=700829)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=700829)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=700829)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=700829)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=700829)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=700829)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=700829)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=700829)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=700829)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=700829)[0m              'experiment_name': 'deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=700829)[0m              'logger': ['console'],
[36m(TaskRunner pid=700829)[0m              'n_gpus_per_node': 16,
[36m(TaskRunner pid=700829)[0m              'nnodes': 1,
[36m(TaskRunner pid=700829)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=700829)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=700829)[0m              'resume_from_path': False,
[36m(TaskRunner pid=700829)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=700829)[0m              'save_freq': -1,
[36m(TaskRunner pid=700829)[0m              'test_freq': 5,
[36m(TaskRunner pid=700829)[0m              'total_epochs': 1,
[36m(TaskRunner pid=700829)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=700829)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=700829)[0m WARNING 04-02 16:20:15 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=700829)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=700829)[0m dataset len: 7473
[36m(TaskRunner pid=700829)[0m filter dataset len: 7473
[36m(TaskRunner pid=700829)[0m dataset len: 1319
[36m(TaskRunner pid=700829)[0m filter dataset len: 1319
[36m(TaskRunner pid=700829)[0m Size of train dataloader: 467
[36m(TaskRunner pid=700829)[0m Total training steps: 1
[36m(pid=711015)[0m WARNING 04-02 16:20:28 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=711315)[0m WARNING 04-02 16:22:05 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=711308)[0m WARNING 04-02 16:22:13 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 6x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=711015)[0m Model config after override: LlamaConfig {
[36m(WorkerDict pid=711015)[0m   "_name_or_path": "deepseek-ai/deepseek-llm-7b-chat",
[36m(WorkerDict pid=711015)[0m   "architectures": [
[36m(WorkerDict pid=711015)[0m     "LlamaForCausalLM"
[36m(WorkerDict pid=711015)[0m   ],
[36m(WorkerDict pid=711015)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(pid=711318)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 8x across cluster][0m
[36m(pid=711318)[0m No module named 'vllm._version'[32m [repeated 8x across cluster][0m
[36m(pid=711318)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=711311)[0m Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
[36m(WorkerDict pid=711311)[0m Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  3.64it/s]
[36m(WorkerDict pid=711311)[0m Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  3.71it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  3.70it/s]
[36m(WorkerDict pid=711309)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.[32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=711309)[0m Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=711309)[0m Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  3.67it/s][32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=711309)[0m Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  3.84it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  3.81it/s][32m [repeated 10x across cluster][0m
[36m(WorkerDict pid=711318)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in LlamaForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=711322)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.[32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=711322)[0m Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s][32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=711322)[0m Loading checkpoint shards:  50%|█████     | 1/2 [00:00<00:00,  4.04it/s][32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=711322)[0m Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  4.74it/s]Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  4.61it/s][32m [repeated 5x across cluster][0m
[36m(WorkerDict pid=711312)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=711312)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=711309)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in LlamaForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711309)[0m Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=711015)[0m Loading checkpoint shards:  50%|█████     | 1/2 [00:01<00:01,  1.44s/it][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=711015)[0m Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,  1.08s/it]Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,  1.14s/it][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=711312)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=711308)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=711308)[0m   warnings.warn(
[36m(WorkerDict pid=711320)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 31x across cluster][0m
[36m(WorkerDict pid=711320)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711320)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711015)[0m   "attention_bias": false,
[36m(WorkerDict pid=711015)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=711015)[0m   "bos_token_id": 100000,
[36m(WorkerDict pid=711015)[0m   "eos_token_id": 100001,
[36m(WorkerDict pid=711015)[0m   "head_dim": 128,
[36m(WorkerDict pid=711015)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=711015)[0m   "hidden_size": 4096,
[36m(WorkerDict pid=711015)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=711015)[0m   "intermediate_size": 11008,
[36m(WorkerDict pid=711015)[0m   "max_position_embeddings": 4096,
[36m(WorkerDict pid=711015)[0m   "mlp_bias": false,
[36m(WorkerDict pid=711015)[0m   "model_type": "llama",
[36m(WorkerDict pid=711015)[0m   "num_attention_heads": 32,
[36m(WorkerDict pid=711015)[0m   "num_hidden_layers": 30,
[36m(WorkerDict pid=711015)[0m   "num_key_value_heads": 32,
[36m(WorkerDict pid=711015)[0m   "pad_token_id": 100001,
[36m(WorkerDict pid=711015)[0m   "pretraining_tp": 1,
[36m(WorkerDict pid=711015)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=711015)[0m   "rope_scaling": null,
[36m(WorkerDict pid=711015)[0m   "rope_theta": 10000.0,
[36m(WorkerDict pid=711015)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=711015)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=711015)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=711015)[0m   "use_cache": true,
[36m(WorkerDict pid=711015)[0m   "vocab_size": 102400
[36m(WorkerDict pid=711015)[0m }
[36m(WorkerDict pid=711015)[0m 
[36m(pid=711318)[0m WARNING 04-02 16:22:14 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=711311)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=711311)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=711315)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 11x across cluster][0m
[36m(WorkerDict pid=711315)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 11x across cluster][0m
[36m(WorkerDict pid=711015)[0m LlamaForCausalLM contains 6.91B parameters
[36m(WorkerDict pid=711015)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ef31bd45260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ef31bd45120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=711015)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=711015)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=711322)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=711322)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=711320)[0m wrap_policy: functools.partial(<function _or_policy at 0x7fa8c0bc9260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7fa8c0bc9120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711015)[0m Model config after override: LlamaConfig {
[36m(WorkerDict pid=711015)[0m   "_name_or_path": "deepseek-ai/deepseek-llm-7b-chat",
[36m(WorkerDict pid=711015)[0m   "architectures": [
[36m(WorkerDict pid=711015)[0m     "LlamaForCausalLM"
[36m(WorkerDict pid=711015)[0m   ],
[36m(WorkerDict pid=711015)[0m   "attention_bias": false,
[36m(WorkerDict pid=711015)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=711015)[0m   "bos_token_id": 100000,
[36m(WorkerDict pid=711015)[0m   "eos_token_id": 100001,
[36m(WorkerDict pid=711015)[0m   "head_dim": 128,
[36m(WorkerDict pid=711015)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=711015)[0m   "hidden_size": 4096,
[36m(WorkerDict pid=711015)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=711015)[0m   "intermediate_size": 11008,
[36m(WorkerDict pid=711015)[0m   "max_position_embeddings": 4096,
[36m(WorkerDict pid=711015)[0m   "mlp_bias": false,
[36m(WorkerDict pid=711015)[0m   "model_type": "llama",
[36m(WorkerDict pid=711015)[0m   "num_attention_heads": 32,
[36m(WorkerDict pid=711015)[0m   "num_hidden_layers": 30,
[36m(WorkerDict pid=711015)[0m   "num_key_value_heads": 32,
[36m(WorkerDict pid=711015)[0m   "pad_token_id": 100001,
[36m(WorkerDict pid=711015)[0m   "pretraining_tp": 1,
[36m(WorkerDict pid=711015)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=711015)[0m   "rope_scaling": null,
[36m(WorkerDict pid=711015)[0m   "rope_theta": 10000.0,
[36m(WorkerDict pid=711015)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=711015)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=711015)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=711015)[0m   "use_cache": true,
[36m(WorkerDict pid=711015)[0m   "vocab_size": 102400
[36m(WorkerDict pid=711015)[0m }
[36m(WorkerDict pid=711015)[0m 
[36m(WorkerDict pid=711015)[0m LlamaForCausalLM contains 6.91B parameters
[36m(WorkerDict pid=711320)[0m Actor use_remove_padding=True[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711015)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=711015)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=711015)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ef31bd45260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ef31bd45120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=711310)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ede09251260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ede09251120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=711310)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=711310)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=711320)[0m wrap_policy: functools.partial(<function _or_policy at 0x7fa8c0bc9260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7fa8c0bc9120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])[32m [repeated 14x across cluster][0m
[36m(WorkerDict pid=711309)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=711015)[0m Before building vllm rollout, memory allocated (GB): 1.6089458465576172, memory reserved (GB): 13.673828125
[36m(WorkerDict pid=711310)[0m WARNING 04-02 16:22:53 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=711015)[0m Total steps: 1, num_warmup_steps: 0[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711015)[0m Actor use_remove_padding=True[32m [repeated 14x across cluster][0m
[36m(WorkerDict pid=711310)[0m local rank 0
[36m(WorkerDict pid=711311)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=711015)[0m before init cache memory allocated: 5.222139904GB, reserved: 5.314183168GB
[36m(WorkerDict pid=711320)[0m WARNING 04-02 16:22:54 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711320)[0m local rank 0[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711320)[0m NCCL version 2.20.5+cuda12.4[32m [repeated 2x across cluster][0m
[36m(TaskRunner pid=700829)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=711315)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711315)[0m   warnings.warn([32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711015)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=711015)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[36m(WorkerDict pid=711308)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=711015)[0m after init cache memory allocated: 29.318416384GB, reserved: 29.410459648GB
[36m(WorkerDict pid=711015)[0m After building vllm rollout, memory allocated (GB): 24.059267044067383, memory reserved (GB): 27.390625
[36m(WorkerDict pid=711015)[0m After building sharding manager, memory allocated (GB): 24.059267044067383, memory reserved (GB): 27.390625
[36m(TaskRunner pid=700829)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=700829)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/deepseek_llm_7b_function_rm/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=700829)[0m Training from scratch
[36m(TaskRunner pid=700829)[0m test_gen_batch meta info: {'eos_token_id': 100001, 'pad_token_id': 100001, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=700829)[0m validation generation end
[36m(WorkerDict pid=711315)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}[32m [repeated 15x across cluster][0m
[36m(TaskRunner pid=700829)[0m [prompt] User: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Assistant:
[36m(TaskRunner pid=700829)[0m [response] Step 1: Calculate the number of eggs Janet's ducks lay per day.
[36m(TaskRunner pid=700829)[0m Janet's ducks lay 16 eggs per day.
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Step 2: Calculate the number of eggs Janet eats for breakfast.
[36m(TaskRunner pid=700829)[0m Janet eats 3 eggs for breakfast every morning.
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Step 3: Calculate the number of eggs Janet uses for baking muffins.
[36m(TaskRunner pid=700829)[0m Janet bakes muffins with 4 eggs every day.
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Step 4: Calculate the number of eggs Janet has left after breakfast and baking.
[36m(TaskRunner pid=700829)[0m Number of eggs left = Total number of eggs laid - (Eggs eaten for breakfast + Eggs used for baking)
[36m(TaskRunner pid=700829)[0m Number of eggs left = 16 - (3 + 4)
[36m(TaskRunner pid=700829)[0m Number of eggs left = 9
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Step 5: Calculate the number of eggs Janet sells at the farmers' market.
[36m(TaskRunner pid=700829)[0m Number of eggs sold = Number of eggs left
[36m(TaskRunner pid=700829)[0m Number of eggs sold = 9
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Step 6: Calculate the amount of money Janet makes from selling eggs at the farmers' market.
[36m(TaskRunner pid=700829)[0m Price per egg = $2
[36m(TaskRunner pid=700829)[0m Amount of money made = Number of eggs sold × Price per egg
[36m(TaskRunner pid=700829)[0m Amount of money made = 9 × 2
[36m(TaskRunner pid=700829)[0m Amount of money made = 18
[36m(TaskRunner pid=700829)[0m 
[36m(TaskRunner pid=700829)[0m Final answer: Janet makes $18 every day at the farmers' market.
[36m(TaskRunner pid=700829)[0m [ground_truth] 18
[36m(TaskRunner pid=700829)[0m [score] 0.0
[36m(TaskRunner pid=700829)[0m ("Initial validation metrics: {'val/test_score/openai/gsm8k': "
[36m(TaskRunner pid=700829)[0m  '0.3199393479909022}')
[36m(TaskRunner pid=700829)[0m step:0 - val/test_score/openai/gsm8k:0.320
[36m(WorkerDict pid=711015)[0m hidden_states = outputs[0], shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1388, 102400])
[36m(WorkerDict pid=711015)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=711015)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1388]), input_ids_rmpad_rolled.shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m hidden_states = outputs[0], shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1388, 102400])
[36m(WorkerDict pid=711015)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=711015)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1388]), input_ids_rmpad_rolled.shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m hidden_states = outputs[0], shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1388, 4096])
[36m(WorkerDict pid=711015)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1388, 102400])
[36m(WorkerDict pid=711015)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=711015)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1388]), input_ids_rmpad_rolled.shape: torch.Size([1388])
[36m(WorkerDict pid=711015)[0m entropy_loss = verl_F.masked_mean(entropy, response_mask), entropy_loss.shape: torch.Size([]), entropy.shape: torch.Size([4, 1024]), response_mask.shape: torch.Size([4, 1024])
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff2d4d212e431c850dcf83f60801000000 Worker ID: e954d96d346a63ed290230032cf56d0d796d0cf80a9074a1b5307606 Node ID: 0cca822e450292fc9c2e3ae3955db328c8bb6c10ae0e228a87b1f59d Worker IP address: 127.0.0.1 Worker port: 34379 Worker PID: 711321 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=16', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=16', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=4', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=deepseek_llm_7b_function_rm', 'trainer.n_gpus_per_node=16', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=700829, ip=127.0.0.1, actor_id=92d993b1d67e41a590abcbb501000000, repr=<main_ppo.TaskRunner object at 0x7fadf87929d0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 171, in run
    trainer.fit()
  File "/opt/tiger/verl/verl/trainer/ppo/ray_trainer.py", line 913, in fit
    actor_output = self.actor_rollout_wg.update_actor(batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/single_controller/ray/base.py", line 42, in func
    output = ray.get(output)
             ^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
	class_name: create_colocated_worker_cls.<locals>.WorkerDict
	actor_id: 2d4d212e431c850dcf83f60801000000
	pid: 711321
	name: OYP0xKWorkerDict_0:15
	namespace: 876f8a74-b1b2-48ca-b8b5-d6fee98275e7
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=711320)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=711320)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined][32m [repeated 15x across cluster][0m
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff9f11182231c5bc4377c422fb01000000 Worker ID: 7e016a0c6d1ae22f918b3993b876dd060db2e31cd2107f5cd031f885 Node ID: 0cca822e450292fc9c2e3ae3955db328c8bb6c10ae0e228a87b1f59d Worker IP address: 127.0.0.1 Worker port: 39659 Worker PID: 711318 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.[32m [repeated 15x across cluster][0m
