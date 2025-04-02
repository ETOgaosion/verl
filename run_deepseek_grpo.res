+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=1024 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=deepseek-ai/deepseek-coder-1.3b-instruct actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=deepseek_llm_7b_function_rm trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 11:21:15,775	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=1975584)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1975584)[0m                                                              'hf_model',
[36m(TaskRunner pid=1975584)[0m                                                              'optimizer',
[36m(TaskRunner pid=1975584)[0m                                                              'extra']},
[36m(TaskRunner pid=1975584)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=1975584)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=1975584)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1975584)[0m                                                  'optimizer_offload': False,
[36m(TaskRunner pid=1975584)[0m                                                  'param_offload': False,
[36m(TaskRunner pid=1975584)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1975584)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=1975584)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=1975584)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=1975584)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=1975584)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=1975584)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1975584)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=1975584)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=1975584)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=1975584)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=1975584)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1975584)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m                                  'ppo_micro_batch_size_per_gpu': 80,
[36m(TaskRunner pid=1975584)[0m                                  'ppo_mini_batch_size': 256,
[36m(TaskRunner pid=1975584)[0m                                  'shuffle': False,
[36m(TaskRunner pid=1975584)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=1975584)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1975584)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=1975584)[0m                                  'use_fused_kernels': True,
[36m(TaskRunner pid=1975584)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=1975584)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=1975584)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=1975584)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=1975584)[0m                                  'external_lib': None,
[36m(TaskRunner pid=1975584)[0m                                  'override_config': {},
[36m(TaskRunner pid=1975584)[0m                                  'path': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=1975584)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=1975584)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=1975584)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1975584)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1975584)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m                                'log_prob_micro_batch_size_per_gpu': 160,
[36m(TaskRunner pid=1975584)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=1975584)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=1975584)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=1975584)[0m                                    'do_sample': True,
[36m(TaskRunner pid=1975584)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=1975584)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=1975584)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=1975584)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=1975584)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=1975584)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=1975584)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=1975584)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=1975584)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m                                    'log_prob_micro_batch_size_per_gpu': 160,
[36m(TaskRunner pid=1975584)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=1975584)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=1975584)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=1975584)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=1975584)[0m                                    'n': 5,
[36m(TaskRunner pid=1975584)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=1975584)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=1975584)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=1975584)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=1975584)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=1975584)[0m                                    'top_k': -1,
[36m(TaskRunner pid=1975584)[0m                                    'top_p': 1,
[36m(TaskRunner pid=1975584)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=1975584)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=1975584)[0m                                                   'n': 1,
[36m(TaskRunner pid=1975584)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=1975584)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=1975584)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=1975584)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=1975584)[0m                'gamma': 1.0,
[36m(TaskRunner pid=1975584)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=1975584)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=1975584)[0m                'lam': 1.0},
[36m(TaskRunner pid=1975584)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1975584)[0m                                         'hf_model',
[36m(TaskRunner pid=1975584)[0m                                         'optimizer',
[36m(TaskRunner pid=1975584)[0m                                         'extra']},
[36m(TaskRunner pid=1975584)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=1975584)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1975584)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=1975584)[0m No module named 'vllm._version'
[36m(TaskRunner pid=1975584)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=1975584)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=1985855)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1985855)[0m No module named 'vllm._version'
[36m(pid=1985855)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=1986188)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1986188)[0m No module named 'vllm._version'
[36m(pid=1986188)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=1975584)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1975584)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=1975584)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=1975584)[0m                       'external_lib': None,
[36m(TaskRunner pid=1975584)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1975584)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=1975584)[0m                                       'param_offload': False,
[36m(TaskRunner pid=1975584)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1975584)[0m                       'override_config': {},
[36m(TaskRunner pid=1975584)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=1975584)[0m                       'tokenizer_path': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=1975584)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=1975584)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=1975584)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1975584)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=1975584)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=1975584)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=1975584)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=1975584)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1975584)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1975584)[0m             'ppo_mini_batch_size': 256,
[36m(TaskRunner pid=1975584)[0m             'shuffle': False,
[36m(TaskRunner pid=1975584)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=1975584)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1975584)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=1975584)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=1975584)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=1975584)[0m           'image_key': 'images',
[36m(TaskRunner pid=1975584)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=1975584)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=1975584)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=1975584)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=1975584)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=1975584)[0m           'shuffle': True,
[36m(TaskRunner pid=1975584)[0m           'tokenizer': None,
[36m(TaskRunner pid=1975584)[0m           'train_batch_size': 1024,
[36m(TaskRunner pid=1975584)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=1975584)[0m           'truncation': 'error',
[36m(TaskRunner pid=1975584)[0m           'val_batch_size': None,
[36m(TaskRunner pid=1975584)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=1975584)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=1975584)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=1975584)[0m                   'max_length': None,
[36m(TaskRunner pid=1975584)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=1975584)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1975584)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=1975584)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=1975584)[0m                                             'param_offload': False,
[36m(TaskRunner pid=1975584)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=1975584)[0m                             'input_tokenizer': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=1975584)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=1975584)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=1975584)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=1975584)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=1975584)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=1975584)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=1975584)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=1975584)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=1975584)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=1975584)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=1975584)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=1975584)[0m              'experiment_name': 'deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=1975584)[0m              'logger': ['console'],
[36m(TaskRunner pid=1975584)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=1975584)[0m              'nnodes': 1,
[36m(TaskRunner pid=1975584)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=1975584)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=1975584)[0m              'resume_from_path': False,
[36m(TaskRunner pid=1975584)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=1975584)[0m              'save_freq': -1,
[36m(TaskRunner pid=1975584)[0m              'test_freq': 5,
[36m(TaskRunner pid=1975584)[0m              'total_epochs': 1,
[36m(TaskRunner pid=1975584)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=1975584)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=1975584)[0m WARNING 04-02 11:21:28 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=1975584)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=1975584)[0m dataset len: 7473
[36m(TaskRunner pid=1975584)[0m filter dataset len: 7473
[36m(TaskRunner pid=1975584)[0m dataset len: 1319
[36m(TaskRunner pid=1975584)[0m filter dataset len: 1319
[36m(TaskRunner pid=1975584)[0m Size of train dataloader: 7
[36m(TaskRunner pid=1975584)[0m Total training steps: 1
[36m(pid=1985855)[0m WARNING 04-02 11:21:42 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=1986188)[0m WARNING 04-02 11:21:53 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(WorkerDict pid=1985855)[0m Model config after override: LlamaConfig {
[36m(WorkerDict pid=1985855)[0m   "_name_or_path": "deepseek-ai/deepseek-coder-1.3b-instruct",
[36m(WorkerDict pid=1985855)[0m   "architectures": [
[36m(WorkerDict pid=1985855)[0m     "LlamaForCausalLM"
[36m(WorkerDict pid=1985855)[0m   ],
[36m(WorkerDict pid=1985855)[0m   "attention_bias": false,
[36m(WorkerDict pid=1985855)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=1985855)[0m   "bos_token_id": 32013,
[36m(WorkerDict pid=1985855)[0m   "eos_token_id": 32021,
[36m(WorkerDict pid=1985855)[0m   "head_dim": 128,
[36m(WorkerDict pid=1985855)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=1985855)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=1985855)[0m   "hidden_size": 2048,
[36m(WorkerDict pid=1985855)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=1985855)[0m   "intermediate_size": 5504,
[36m(WorkerDict pid=1985855)[0m   "max_position_embeddings": 16384,
[36m(WorkerDict pid=1985855)[0m   "mlp_bias": false,
[36m(WorkerDict pid=1985855)[0m   "model_type": "llama",
[36m(WorkerDict pid=1985855)[0m   "num_attention_heads": 16,
[36m(WorkerDict pid=1985855)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=1985855)[0m   "num_key_value_heads": 16,
[36m(WorkerDict pid=1985855)[0m   "pad_token_id": 32014,
[36m(WorkerDict pid=1985855)[0m   "pretraining_tp": 1,
[36m(WorkerDict pid=1985855)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=1985855)[0m   "rope_scaling": {
[36m(WorkerDict pid=1985855)[0m     "factor": 4.0,
[36m(WorkerDict pid=1985855)[0m     "rope_type": "linear",
[36m(WorkerDict pid=1985855)[0m     "type": "linear"
[36m(WorkerDict pid=1985855)[0m   },
[36m(WorkerDict pid=1985855)[0m   "rope_theta": 100000,
[36m(WorkerDict pid=1985855)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=1985855)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=1985855)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=1985855)[0m   "use_cache": true,
[36m(WorkerDict pid=1985855)[0m   "vocab_size": 32256
[36m(WorkerDict pid=1985855)[0m }
[36m(WorkerDict pid=1985855)[0m 
[36m(WorkerDict pid=1985855)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=1985855)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=1985855)[0m LlamaForCausalLM contains 1.35B parameters
[36m(WorkerDict pid=1985855)[0m wrap_policy: functools.partial(<function _or_policy at 0x7faf00c4d260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7faf00c4d120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=1985855)[0m NCCL version 2.20.5+cuda12.4
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=1024', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=deepseek-ai/deepseek-coder-1.3b-instruct', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=256', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=5', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=deepseek_llm_7b_function_rm', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
[36m(TaskRunner pid=1975584)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): [36mray::WorkerDict.ref_init_model()[39m (pid=1986188, ip=127.0.0.1, actor_id=76ffd5e1082cae2312a152e301000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7fc69108efd0>)
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/opt/tiger/verl/verl/single_controller/ray/base.py", line 419, in func
[36m(TaskRunner pid=1975584)[0m     return getattr(self.worker_dict[key], name)(*args, **kwargs)
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/opt/tiger/verl/verl/single_controller/base/decorator.py", line 404, in inner
[36m(TaskRunner pid=1975584)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/opt/tiger/verl/verl/workers/fsdp_workers.py", line 425, in init_model
[36m(TaskRunner pid=1975584)[0m     self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
[36m(TaskRunner pid=1975584)[0m                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/opt/tiger/verl/verl/workers/actor/dp_actor.py", line 57, in __init__
[36m(TaskRunner pid=1975584)[0m     self.use_fused_kernels = config.use_fused_kernels
[36m(TaskRunner pid=1975584)[0m                              ^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 359, in __getattr__
[36m(TaskRunner pid=1975584)[0m     self._format_and_raise(key=key, value=None, cause=e)
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/base.py", line 231, in _format_and_raise
[36m(TaskRunner pid=1975584)[0m     format_and_raise(
Traceback (most recent call last):
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 819, in format_and_raise
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 54, in main
    run_ppo(config)
[36m(TaskRunner pid=1975584)[0m     _raise(ex, cause)
  File "/opt/tiger/verl/verl/trainer/main_ppo.py", line 72, in run_ppo
    ray.get(runner.run.remote(config))
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 797, in _raise
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m     raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/worker.py", line 2782, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 351, in __getattr__
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/worker.py", line 929, in get_objects
    raise value.as_instanceof_cause()
[36m(TaskRunner pid=1975584)[0m     return self._get_impl(
ray.exceptions.RayTaskError(ConfigAttributeError): [36mray::TaskRunner.run()[39m (pid=1975584, ip=127.0.0.1, actor_id=5e4654dec9bea00cde2d589301000000, repr=<main_ppo.TaskRunner object at 0x7f60def11910>)
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
ray.exceptions.RayTaskError(ConfigAttributeError): [36mray::WorkerDict.ref_init_model()[39m (pid=1985855, ip=127.0.0.1, actor_id=a1d0af0fe160f462af40bbe201000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7fae649b6f90>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/single_controller/ray/base.py", line 419, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/workers/fsdp_workers.py", line 425, in init_model
    self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/tiger/verl/verl/workers/actor/dp_actor.py", line 57, in __init__
    self.use_fused_kernels = config.use_fused_kernels
                             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 359, in __getattr__
    self._format_and_raise(key=key, value=None, cause=e)
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/base.py", line 231, in _format_and_raise
    format_and_raise(
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 819, in format_and_raise
    _raise(ex, cause)
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 797, in _raise
    raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 351, in __getattr__
    return self._get_impl(
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 442, in _get_impl
    node = self._get_child(
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/basecontainer.py", line 73, in _get_child
    child = self._get_node(
            ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 475, in _get_node
    self._validate_get(key)
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 164, in _validate_get
    self._format_and_raise(
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/base.py", line 231, in _format_and_raise
    format_and_raise(
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 899, in format_and_raise
    _raise(ex, cause)
  File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 797, in _raise
    raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
omegaconf.errors.ConfigAttributeError: Key 'use_fused_kernels' is not in struct
    full_key: actor_rollout_ref.ref.use_fused_kernels
    object_type=dict
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 442, in _get_impl
[36m(TaskRunner pid=1975584)[0m     node = self._get_child(
[36m(TaskRunner pid=1975584)[0m            ^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/basecontainer.py", line 73, in _get_child
[36m(TaskRunner pid=1975584)[0m     child = self._get_node(
[36m(TaskRunner pid=1975584)[0m             ^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 475, in _get_node
[36m(TaskRunner pid=1975584)[0m     self._validate_get(key)
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/dictconfig.py", line 164, in _validate_get
[36m(TaskRunner pid=1975584)[0m     self._format_and_raise(
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/base.py", line 231, in _format_and_raise
[36m(TaskRunner pid=1975584)[0m     format_and_raise(
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 899, in format_and_raise
[36m(TaskRunner pid=1975584)[0m     _raise(ex, cause)
[36m(TaskRunner pid=1975584)[0m   File "/usr/local/lib/python3.11/dist-packages/omegaconf/_utils.py", line 797, in _raise
[36m(TaskRunner pid=1975584)[0m     raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
[36m(TaskRunner pid=1975584)[0m     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(TaskRunner pid=1975584)[0m omegaconf.errors.ConfigAttributeError: Key 'use_fused_kernels' is not in struct
[36m(TaskRunner pid=1975584)[0m     full_key: actor_rollout_ref.ref.use_fused_kernels
[36m(TaskRunner pid=1975584)[0m     object_type=dict
[36m(WorkerDict pid=1986188)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=1985855)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=1986188)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=1986188)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=1986188)[0m wrap_policy: functools.partial(<function _or_policy at 0x7fc72c449260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7fc72c449120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=1986188)[0m Actor use_remove_padding=True
