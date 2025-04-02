+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=16 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=Qwen/Qwen2.5-7B actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=qwen2_7b_function_rm trainer.n_gpus_per_node=16 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 16:55:24,184	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=816970)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=816970)[0m                                                              'hf_model',
[36m(TaskRunner pid=816970)[0m                                                              'optimizer',
[36m(TaskRunner pid=816970)[0m                                                              'extra']},
[36m(TaskRunner pid=816970)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=816970)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=816970)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=816970)[0m                                                  'optimizer_offload': True,
[36m(TaskRunner pid=816970)[0m                                                  'param_offload': True,
[36m(TaskRunner pid=816970)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=816970)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=816970)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=816970)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=816970)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=816970)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=816970)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=816970)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=816970)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=816970)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=816970)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=816970)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=816970)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m                                  'ppo_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=816970)[0m                                  'ppo_mini_batch_size': 16,
[36m(TaskRunner pid=816970)[0m                                  'shuffle': False,
[36m(TaskRunner pid=816970)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=816970)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=816970)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=816970)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=816970)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=816970)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=816970)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=816970)[0m                                  'external_lib': None,
[36m(TaskRunner pid=816970)[0m                                  'override_config': {},
[36m(TaskRunner pid=816970)[0m                                  'path': 'Qwen/Qwen2.5-7B',
[36m(TaskRunner pid=816970)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=816970)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=816970)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=816970)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=816970)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m                                'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=816970)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=816970)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=816970)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=816970)[0m                                    'do_sample': True,
[36m(TaskRunner pid=816970)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=816970)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=816970)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=816970)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=816970)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=816970)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=816970)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=816970)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=816970)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m                                    'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=816970)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=816970)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=816970)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=816970)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=816970)[0m                                    'n': 4,
[36m(TaskRunner pid=816970)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=816970)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=816970)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=816970)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=816970)[0m                                    'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=816970)[0m                                    'top_k': -1,
[36m(TaskRunner pid=816970)[0m                                    'top_p': 1,
[36m(TaskRunner pid=816970)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=816970)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=816970)[0m                                                   'n': 1,
[36m(TaskRunner pid=816970)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=816970)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=816970)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=816970)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=816970)[0m                'gamma': 1.0,
[36m(TaskRunner pid=816970)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=816970)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=816970)[0m                'lam': 1.0},
[36m(TaskRunner pid=816970)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=816970)[0m                                         'hf_model',
[36m(TaskRunner pid=816970)[0m                                         'optimizer',
[36m(TaskRunner pid=816970)[0m                                         'extra']},
[36m(TaskRunner pid=816970)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=816970)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=816970)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=816970)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=816970)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=816970)[0m No module named 'vllm._version'
[36m(TaskRunner pid=816970)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=816970)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=827161)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=827161)[0m No module named 'vllm._version'
[36m(pid=827161)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=827471)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=827471)[0m No module named 'vllm._version'
[36m(pid=827471)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=827459)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 6x across cluster][0m
[36m(pid=827459)[0m No module named 'vllm._version'[32m [repeated 6x across cluster][0m
[36m(pid=827459)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 6x across cluster][0m
[36m(WorkerDict pid=827465)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(pid=827462)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 8x across cluster][0m
[36m(pid=827462)[0m No module named 'vllm._version'[32m [repeated 8x across cluster][0m
[36m(pid=827462)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 8x across cluster][0m
[36m(TaskRunner pid=816970)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=816970)[0m                       'external_lib': None,
[36m(TaskRunner pid=816970)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=816970)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=816970)[0m                                       'param_offload': False,
[36m(TaskRunner pid=816970)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=816970)[0m                       'override_config': {},
[36m(TaskRunner pid=816970)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=816970)[0m                       'tokenizer_path': 'Qwen/Qwen2.5-7B',
[36m(TaskRunner pid=816970)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=816970)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=816970)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=816970)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=816970)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=816970)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=816970)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=816970)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=816970)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=816970)[0m             'ppo_mini_batch_size': 16,
[36m(TaskRunner pid=816970)[0m             'shuffle': False,
[36m(TaskRunner pid=816970)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=816970)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=816970)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=816970)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=816970)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=816970)[0m           'image_key': 'images',
[36m(TaskRunner pid=816970)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=816970)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=816970)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=816970)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=816970)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=816970)[0m           'shuffle': True,
[36m(TaskRunner pid=816970)[0m           'tokenizer': None,
[36m(TaskRunner pid=816970)[0m           'train_batch_size': 16,
[36m(TaskRunner pid=816970)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=816970)[0m           'truncation': 'error',
[36m(TaskRunner pid=816970)[0m           'val_batch_size': None,
[36m(TaskRunner pid=816970)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=816970)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=816970)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=816970)[0m                   'max_length': None,
[36m(TaskRunner pid=816970)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=816970)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=816970)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=816970)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=816970)[0m                                             'param_offload': False,
[36m(TaskRunner pid=816970)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=816970)[0m                             'input_tokenizer': 'Qwen/Qwen2.5-7B',
[36m(TaskRunner pid=816970)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=816970)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=816970)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=816970)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=816970)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=816970)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=816970)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=816970)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=816970)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=816970)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm',
[36m(TaskRunner pid=816970)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=816970)[0m              'experiment_name': 'qwen2_7b_function_rm',
[36m(TaskRunner pid=816970)[0m              'logger': ['console'],
[36m(TaskRunner pid=816970)[0m              'n_gpus_per_node': 16,
[36m(TaskRunner pid=816970)[0m              'nnodes': 1,
[36m(TaskRunner pid=816970)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=816970)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=816970)[0m              'resume_from_path': False,
[36m(TaskRunner pid=816970)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=816970)[0m              'save_freq': -1,
[36m(TaskRunner pid=816970)[0m              'test_freq': 5,
[36m(TaskRunner pid=816970)[0m              'total_epochs': 1,
[36m(TaskRunner pid=816970)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=816970)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=816970)[0m WARNING 04-02 16:55:37 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=816970)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=816970)[0m dataset len: 7473
[36m(TaskRunner pid=816970)[0m filter dataset len: 7473
[36m(TaskRunner pid=816970)[0m dataset len: 1319
[36m(TaskRunner pid=816970)[0m filter dataset len: 1319
[36m(TaskRunner pid=816970)[0m Size of train dataloader: 467
[36m(TaskRunner pid=816970)[0m Total training steps: 1
[36m(pid=827161)[0m WARNING 04-02 16:55:49 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=827471)[0m WARNING 04-02 16:57:26 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=827459)[0m WARNING 04-02 16:57:32 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 6x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=827161)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=827161)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",
[36m(WorkerDict pid=827161)[0m   "architectures": [
[36m(WorkerDict pid=827161)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=827161)[0m   ],
[36m(WorkerDict pid=827161)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=827161)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
[36m(WorkerDict pid=827461)[0m Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:00,  4.26it/s]
[36m(WorkerDict pid=827461)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  4.37it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  4.36it/s]
[36m(WorkerDict pid=827460)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=827466)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827460)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=827466)[0m Loading checkpoint shards:  75%|███████▌  | 3/4 [00:00<00:00,  4.38it/s][32m [repeated 47x across cluster][0m
[36m(WorkerDict pid=827466)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  4.81it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  4.51it/s][32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827465)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=827465)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=827470)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827470)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827470)[0m Loading checkpoint shards:  75%|███████▌  | 3/4 [00:00<00:00,  5.01it/s][32m [repeated 48x across cluster][0m
[36m(WorkerDict pid=827470)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00,  5.06it/s][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=827465)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=827457)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=827457)[0m   warnings.warn(
[36m(WorkerDict pid=827464)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 31x across cluster][0m
[36m(WorkerDict pid=827464)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827464)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=827161)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=827161)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=827161)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=827161)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=827161)[0m   "max_position_embeddings": 131072,
[36m(WorkerDict pid=827161)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=827161)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=827161)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=827161)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=827161)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=827161)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=827161)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=827161)[0m   "rope_scaling": null,
[36m(WorkerDict pid=827161)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=827161)[0m   "sliding_window": null,
[36m(WorkerDict pid=827161)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=827161)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=827161)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=827161)[0m   "use_cache": true,
[36m(WorkerDict pid=827161)[0m   "use_mrope": false,
[36m(WorkerDict pid=827161)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=827161)[0m   "vocab_size": 152064
[36m(WorkerDict pid=827161)[0m }
[36m(WorkerDict pid=827161)[0m 
[36m(pid=827462)[0m WARNING 04-02 16:57:34 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 8x across cluster][0m
[36m(WorkerDict pid=827461)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=827461)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=827161)[0m Qwen2ForCausalLM contains 7.62B parameters
[36m(WorkerDict pid=827161)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f0414369260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f0414369120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=827161)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=827161)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=827466)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827466)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827459)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee5bc4ed260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee5bc4ed120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=827161)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",
[36m(WorkerDict pid=827161)[0m   "architectures": [
[36m(WorkerDict pid=827161)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=827161)[0m   ],
[36m(WorkerDict pid=827161)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=827161)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=827161)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=827161)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=827161)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=827161)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=827161)[0m   "max_position_embeddings": 131072,
[36m(WorkerDict pid=827161)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=827161)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=827161)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=827161)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=827161)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=827161)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=827161)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=827161)[0m   "rope_scaling": null,
[36m(WorkerDict pid=827161)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=827161)[0m   "sliding_window": null,
[36m(WorkerDict pid=827161)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=827161)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=827161)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=827161)[0m   "use_cache": true,
[36m(WorkerDict pid=827161)[0m   "use_mrope": false,
[36m(WorkerDict pid=827161)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=827161)[0m   "vocab_size": 152064
[36m(WorkerDict pid=827161)[0m }
[36m(WorkerDict pid=827161)[0m 
[36m(WorkerDict pid=827459)[0m Actor use_remove_padding=True[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m Qwen2ForCausalLM contains 7.62B parameters
[36m(WorkerDict pid=827161)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f0414369260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f0414369120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=827467)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=827467)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=827470)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=827470)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=827459)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee5bc4ed260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee5bc4ed120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827461)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=827161)[0m Before building vllm rollout, memory allocated (GB): 1.773153305053711, memory reserved (GB): 17.6015625
[36m(WorkerDict pid=827470)[0m WARNING 04-02 16:58:10 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=827161)[0m Total steps: 1, num_warmup_steps: 0[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m Actor use_remove_padding=True[32m [repeated 14x across cluster][0m
[36m(WorkerDict pid=827465)[0m local rank 0
[36m(WorkerDict pid=827461)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=827161)[0m before init cache memory allocated: 5.755392GB, reserved: 5.855248384GB
[36m(WorkerDict pid=827462)[0m WARNING 04-02 16:58:10 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827469)[0m local rank 0[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827469)[0m NCCL version 2.20.5+cuda12.4[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=827161)[0m after init cache memory allocated: 29.083848704GB, reserved: 29.225910272GB
[36m(WorkerDict pid=827457)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=827161)[0m After building vllm rollout, memory allocated (GB): 23.538660049438477, memory reserved (GB): 27.21875
[36m(WorkerDict pid=827161)[0m After building sharding manager, memory allocated (GB): 23.538660049438477, memory reserved (GB): 27.21875
[36m(TaskRunner pid=816970)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=816970)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=827469)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827469)[0m   warnings.warn([32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827161)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=827161)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[36m(TaskRunner pid=816970)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=816970)[0m Training from scratch
[36m(TaskRunner pid=816970)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=816970)[0m validation generation end
[36m(WorkerDict pid=827469)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}[32m [repeated 15x across cluster][0m
[36m(TaskRunner pid=816970)[0m [prompt] system
[36m(TaskRunner pid=816970)[0m You are a helpful assistant.
[36m(TaskRunner pid=816970)[0m user
[36m(TaskRunner pid=816970)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=816970)[0m assistant
[36m(TaskRunner pid=816970)[0m 
[36m(TaskRunner pid=816970)[0m [response] Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she has 16 - 3 - 4 = 9 eggs left to sell. She sells each egg for $2, so she makes 9 * $2 = $18 every day at the farmers' market. The final answer is $18.
[36m(TaskRunner pid=816970)[0m [ground_truth] 18
[36m(TaskRunner pid=816970)[0m [score] 0.0
[36m(TaskRunner pid=816970)[0m ("Initial validation metrics: {'val/test_score/openai/gsm8k': "
[36m(TaskRunner pid=816970)[0m  '0.13874147081122062}')
[36m(TaskRunner pid=816970)[0m step:0 - val/test_score/openai/gsm8k:0.139
[36m(WorkerDict pid=827161)[0m hidden_states = outputs[0], shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1937, 152064])
[36m(WorkerDict pid=827161)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=827161)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1937]), input_ids_rmpad_rolled.shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m hidden_states = outputs[0], shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1937, 152064])
[36m(WorkerDict pid=827161)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=827161)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1937]), input_ids_rmpad_rolled.shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m hidden_states = outputs[0], shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1937, 3584])
[36m(WorkerDict pid=827161)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1937, 152064])
[36m(WorkerDict pid=827161)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=827161)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1937]), input_ids_rmpad_rolled.shape: torch.Size([1937])
[36m(WorkerDict pid=827161)[0m entropy_loss = verl_F.masked_mean(entropy, response_mask), entropy_loss.shape: torch.Size([]), entropy.shape: torch.Size([4, 1024]), response_mask.shape: torch.Size([4, 1024])
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff293c3837f623e9c250b15f6b01000000 Worker ID: fc7436e59638fd174340131b12775f5d035aac357a167cc01cd0e156 Node ID: ad099baa840f0800aa6c539151436ee4a3534b58613bf7402a4cc5e3 Worker IP address: 127.0.0.1 Worker port: 39975 Worker PID: 827161 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=16', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-7B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=16', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=4', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=qwen2_7b_function_rm', 'trainer.n_gpus_per_node=16', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=816970, ip=127.0.0.1, actor_id=74b0772920072ea9352e79f901000000, repr=<main_ppo.TaskRunner object at 0x7f450cc6c310>)
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
	actor_id: 293c3837f623e9c250b15f6b01000000
	pid: 827161
	name: AJHLaDWorkerDict_0:0
	namespace: 4fe5d554-791f-4cca-bc48-517a7535b65f
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=827469)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=827469)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined][32m [repeated 15x across cluster][0m
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff0b06c9912d322cd7e911bbc601000000 Worker ID: 1b0d74d82b87104555ea3d78cc2faa49c2fcaa1522959c3c7298e1f4 Node ID: ad099baa840f0800aa6c539151436ee4a3534b58613bf7402a4cc5e3 Worker IP address: 127.0.0.1 Worker port: 47785 Worker PID: 827464 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.[32m [repeated 15x across cluster][0m
