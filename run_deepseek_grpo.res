+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=4 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=deepseek-ai/deepseek-coder-1.3b-instruct actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=deepseek_llm_7b_function_rm trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 14:34:11,190	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=310306)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=310306)[0m                                                              'hf_model',
[36m(TaskRunner pid=310306)[0m                                                              'optimizer',
[36m(TaskRunner pid=310306)[0m                                                              'extra']},
[36m(TaskRunner pid=310306)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=310306)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=310306)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=310306)[0m                                                  'optimizer_offload': False,
[36m(TaskRunner pid=310306)[0m                                                  'param_offload': False,
[36m(TaskRunner pid=310306)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=310306)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=310306)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=310306)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=310306)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=310306)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=310306)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=310306)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=310306)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=310306)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=310306)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=310306)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=310306)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m                                  'ppo_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=310306)[0m                                  'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=310306)[0m                                  'shuffle': False,
[36m(TaskRunner pid=310306)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=310306)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=310306)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=310306)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=310306)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=310306)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=310306)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=310306)[0m                                  'external_lib': None,
[36m(TaskRunner pid=310306)[0m                                  'override_config': {},
[36m(TaskRunner pid=310306)[0m                                  'path': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=310306)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=310306)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=310306)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=310306)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=310306)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m                                'log_prob_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=310306)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=310306)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=310306)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=310306)[0m                                    'do_sample': True,
[36m(TaskRunner pid=310306)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=310306)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=310306)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=310306)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=310306)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=310306)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=310306)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=310306)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=310306)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m                                    'log_prob_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=310306)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=310306)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=310306)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=310306)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=310306)[0m                                    'n': 5,
[36m(TaskRunner pid=310306)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=310306)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=310306)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=310306)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=310306)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=310306)[0m                                    'top_k': -1,
[36m(TaskRunner pid=310306)[0m                                    'top_p': 1,
[36m(TaskRunner pid=310306)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=310306)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=310306)[0m                                                   'n': 1,
[36m(TaskRunner pid=310306)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=310306)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=310306)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=310306)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=310306)[0m                'gamma': 1.0,
[36m(TaskRunner pid=310306)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=310306)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=310306)[0m                'lam': 1.0},
[36m(TaskRunner pid=310306)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=310306)[0m                                         'hf_model',
[36m(TaskRunner pid=310306)[0m                                         'optimizer',
[36m(TaskRunner pid=310306)[0m                                         'extra']},
[36m(TaskRunner pid=310306)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=310306)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=310306)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=310306)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=310306)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=310306)[0m No module named 'vllm._version'
[36m(TaskRunner pid=310306)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=310306)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=320330)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=320330)[0m No module named 'vllm._version'
[36m(pid=320330)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=320623)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=320623)[0m No module named 'vllm._version'
[36m(pid=320623)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(WorkerDict pid=320623)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(TaskRunner pid=310306)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=310306)[0m                       'external_lib': None,
[36m(TaskRunner pid=310306)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=310306)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=310306)[0m                                       'param_offload': False,
[36m(TaskRunner pid=310306)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=310306)[0m                       'override_config': {},
[36m(TaskRunner pid=310306)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=310306)[0m                       'tokenizer_path': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=310306)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=310306)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=310306)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=310306)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=310306)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=310306)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=310306)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=310306)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=310306)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=310306)[0m             'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=310306)[0m             'shuffle': False,
[36m(TaskRunner pid=310306)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=310306)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=310306)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=310306)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=310306)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=310306)[0m           'image_key': 'images',
[36m(TaskRunner pid=310306)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=310306)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=310306)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=310306)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=310306)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=310306)[0m           'shuffle': True,
[36m(TaskRunner pid=310306)[0m           'tokenizer': None,
[36m(TaskRunner pid=310306)[0m           'train_batch_size': 4,
[36m(TaskRunner pid=310306)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=310306)[0m           'truncation': 'error',
[36m(TaskRunner pid=310306)[0m           'val_batch_size': None,
[36m(TaskRunner pid=310306)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=310306)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=310306)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=310306)[0m                   'max_length': None,
[36m(TaskRunner pid=310306)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=310306)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=310306)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=310306)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=310306)[0m                                             'param_offload': False,
[36m(TaskRunner pid=310306)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=310306)[0m                             'input_tokenizer': 'deepseek-ai/deepseek-coder-1.3b-instruct',
[36m(TaskRunner pid=310306)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=310306)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=310306)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=310306)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=310306)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=310306)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=310306)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=310306)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=310306)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=310306)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=310306)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=310306)[0m              'experiment_name': 'deepseek_llm_7b_function_rm',
[36m(TaskRunner pid=310306)[0m              'logger': ['console'],
[36m(TaskRunner pid=310306)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=310306)[0m              'nnodes': 1,
[36m(TaskRunner pid=310306)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=310306)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=310306)[0m              'resume_from_path': False,
[36m(TaskRunner pid=310306)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=310306)[0m              'save_freq': -1,
[36m(TaskRunner pid=310306)[0m              'test_freq': 5,
[36m(TaskRunner pid=310306)[0m              'total_epochs': 1,
[36m(TaskRunner pid=310306)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=310306)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=310306)[0m WARNING 04-02 14:34:23 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=310306)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=310306)[0m dataset len: 7473
[36m(TaskRunner pid=310306)[0m filter dataset len: 7473
[36m(TaskRunner pid=310306)[0m dataset len: 1319
[36m(TaskRunner pid=310306)[0m filter dataset len: 1319
[36m(TaskRunner pid=310306)[0m Size of train dataloader: 1868
[36m(TaskRunner pid=310306)[0m Total training steps: 1
[36m(pid=320330)[0m WARNING 04-02 14:34:37 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=320623)[0m WARNING 04-02 14:34:49 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(WorkerDict pid=320330)[0m Model config after override: LlamaConfig {
[36m(WorkerDict pid=320330)[0m   "_name_or_path": "deepseek-ai/deepseek-coder-1.3b-instruct",
[36m(WorkerDict pid=320330)[0m   "architectures": [
[36m(WorkerDict pid=320330)[0m     "LlamaForCausalLM"
[36m(WorkerDict pid=320330)[0m   ],
[36m(WorkerDict pid=320330)[0m   "attention_bias": false,
[36m(WorkerDict pid=320330)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=320330)[0m   "bos_token_id": 32013,
[36m(WorkerDict pid=320330)[0m   "eos_token_id": 32021,
[36m(WorkerDict pid=320330)[0m   "head_dim": 128,
[36m(WorkerDict pid=320330)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=320330)[0m   "hidden_size": 2048,
[36m(WorkerDict pid=320330)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=320330)[0m   "intermediate_size": 5504,
[36m(WorkerDict pid=320330)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in LlamaForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=320330)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=320330)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=320330)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=320330)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=320330)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=320623)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in LlamaForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=320330)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=320330)[0m   warnings.warn(
[36m(WorkerDict pid=320330)[0m   "max_position_embeddings": 16384,
[36m(WorkerDict pid=320330)[0m   "mlp_bias": false,
[36m(WorkerDict pid=320330)[0m   "model_type": "llama",
[36m(WorkerDict pid=320330)[0m   "num_attention_heads": 16,
[36m(WorkerDict pid=320330)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=320330)[0m   "num_key_value_heads": 16,
[36m(WorkerDict pid=320330)[0m   "pad_token_id": 32014,
[36m(WorkerDict pid=320330)[0m   "pretraining_tp": 1,
[36m(WorkerDict pid=320330)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=320330)[0m   "rope_scaling": {
[36m(WorkerDict pid=320330)[0m     "factor": 4.0,
[36m(WorkerDict pid=320330)[0m     "rope_type": "linear",
[36m(WorkerDict pid=320330)[0m     "type": "linear"
[36m(WorkerDict pid=320330)[0m   },
[36m(WorkerDict pid=320330)[0m   "rope_theta": 100000,
[36m(WorkerDict pid=320330)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=320330)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=320330)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=320330)[0m   "use_cache": true,
[36m(WorkerDict pid=320330)[0m   "vocab_size": 32256
[36m(WorkerDict pid=320330)[0m }
[36m(WorkerDict pid=320330)[0m 
[36m(WorkerDict pid=320623)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=320623)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=320330)[0m LlamaForCausalLM contains 1.35B parameters
[36m(WorkerDict pid=320330)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f08e0f65260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f08e0f65120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=320330)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=320330)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=320330)[0m Model config after override: LlamaConfig {
[36m(WorkerDict pid=320330)[0m   "_name_or_path": "deepseek-ai/deepseek-coder-1.3b-instruct",
[36m(WorkerDict pid=320330)[0m   "architectures": [
[36m(WorkerDict pid=320330)[0m     "LlamaForCausalLM"
[36m(WorkerDict pid=320330)[0m   ],
[36m(WorkerDict pid=320330)[0m   "attention_bias": false,
[36m(WorkerDict pid=320330)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=320330)[0m   "bos_token_id": 32013,
[36m(WorkerDict pid=320330)[0m   "eos_token_id": 32021,
[36m(WorkerDict pid=320330)[0m   "head_dim": 128,
[36m(WorkerDict pid=320330)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=320330)[0m   "hidden_size": 2048,
[36m(WorkerDict pid=320330)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=320330)[0m   "intermediate_size": 5504,
[36m(WorkerDict pid=320330)[0m   "max_position_embeddings": 16384,
[36m(WorkerDict pid=320330)[0m   "mlp_bias": false,
[36m(WorkerDict pid=320330)[0m   "model_type": "llama",
[36m(WorkerDict pid=320330)[0m   "num_attention_heads": 16,
[36m(WorkerDict pid=320330)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=320330)[0m   "num_key_value_heads": 16,
[36m(WorkerDict pid=320330)[0m   "pad_token_id": 32014,
[36m(WorkerDict pid=320330)[0m   "pretraining_tp": 1,
[36m(WorkerDict pid=320330)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=320330)[0m   "rope_scaling": {
[36m(WorkerDict pid=320330)[0m     "factor": 4.0,
[36m(WorkerDict pid=320330)[0m     "rope_type": "linear",
[36m(WorkerDict pid=320330)[0m     "type": "linear"
[36m(WorkerDict pid=320330)[0m   },
[36m(WorkerDict pid=320330)[0m   "rope_theta": 100000,
[36m(WorkerDict pid=320330)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=320330)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=320330)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=320330)[0m   "use_cache": true,
[36m(WorkerDict pid=320330)[0m   "vocab_size": 32256
[36m(WorkerDict pid=320330)[0m }
[36m(WorkerDict pid=320330)[0m 
[36m(WorkerDict pid=320623)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=320623)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=320330)[0m LlamaForCausalLM contains 1.35B parameters
[36m(WorkerDict pid=320330)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f08e0f65260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f08e0f65120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=320623)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=320330)[0m Before building vllm rollout, memory allocated (GB): 2.508000373840332, memory reserved (GB): 4.365234375
[36m(WorkerDict pid=320330)[0m WARNING 04-02 14:35:05 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=320330)[0m Actor use_remove_padding=True[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=320330)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=320330)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=320623)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f3029185260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f3029185120>, transformer_layer_cls={<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>})])
[36m(WorkerDict pid=320330)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=320330)[0m local rank 0
[36m(WorkerDict pid=320330)[0m before init cache memory allocated: 4.079227904GB, reserved: 4.167041024GB
[36m(WorkerDict pid=320330)[0m after init cache memory allocated: 29.547041792GB, reserved: 29.634854912GB
[36m(TaskRunner pid=310306)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=310306)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/deepseek_llm_7b_function_rm/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=310306)[0m Training from scratch
[36m(WorkerDict pid=320330)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=320330)[0m After building vllm rollout, memory allocated (GB): 26.250313758850098, memory reserved (GB): 27.599609375
[36m(WorkerDict pid=320330)[0m After building sharding manager, memory allocated (GB): 26.250313758850098, memory reserved (GB): 27.599609375
[36m(TaskRunner pid=310306)[0m test_gen_batch meta info: {'eos_token_id': 32021, 'pad_token_id': 32014, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(WorkerDict pid=320623)[0m WARNING 04-02 14:35:05 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=320623)[0m local rank 0
[36m(TaskRunner pid=310306)[0m validation generation end
[36m(WorkerDict pid=320623)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=310306)[0m [prompt] You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer
[36m(TaskRunner pid=310306)[0m ### Instruction:
[36m(TaskRunner pid=310306)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=320623)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=320623)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=320623)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=320623)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=320623)[0m   warnings.warn(
[36m(WorkerDict pid=320330)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=320330)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[36m(TaskRunner pid=310306)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=310306)[0m ### Response:
[36m(TaskRunner pid=310306)[0m 
[36m(TaskRunner pid=310306)[0m [response] I'm sorry, but as an AI programming assistant, I'm specialized in answering questions related to computer science and programming. I'm not equipped to provide answers to questions about economics or business calculations. I recommend using a calculator or a business-oriented tool for this kind of question.
[36m(TaskRunner pid=310306)[0m 
[36m(TaskRunner pid=310306)[0m [ground_truth] 18
[36m(TaskRunner pid=310306)[0m [score] 0.0
[36m(TaskRunner pid=310306)[0m "Initial validation metrics: {'val/test_score/openai/gsm8k': 0.0}"
[36m(TaskRunner pid=310306)[0m step:0 - val/test_score/openai/gsm8k:0.000
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1530, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1530]), input_ids_rmpad_rolled.shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 1378, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1378, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1378, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1378])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1378]), input_ids_rmpad_rolled.shape: torch.Size([1378])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 609, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 609, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([609, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([609])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([609]), input_ids_rmpad_rolled.shape: torch.Size([609])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1530, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1530]), input_ids_rmpad_rolled.shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 1378, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1378, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1378, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1378])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1378]), input_ids_rmpad_rolled.shape: torch.Size([1378])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 609, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 609, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([609, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([609])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([609]), input_ids_rmpad_rolled.shape: torch.Size([609])
[36m(WorkerDict pid=320330)[0m hidden_states = outputs[0], shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 1530, 2048])
[36m(WorkerDict pid=320330)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([1530, 32256])
[36m(WorkerDict pid=320330)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=320330)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([1530]), input_ids_rmpad_rolled.shape: torch.Size([1530])
[36m(WorkerDict pid=320330)[0m entropy_loss = verl_F.masked_mean(entropy, response_mask), entropy_loss.shape: torch.Size([]), entropy.shape: torch.Size([4, 1024]), response_mask.shape: torch.Size([4, 1024])
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff07e712cb7b0b521a4322176f01000000 Worker ID: fb1ef65e4764ebcace8a05f272f51560c58cb29cad243ee72c849f9d Node ID: 9557d62af5f975d09decde2d6adbb2a9a0057c45c8cee6a571c44895 Worker IP address: 127.0.0.1 Worker port: 36623 Worker PID: 320330 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=4', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=deepseek-ai/deepseek-coder-1.3b-instruct', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=4', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=5', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=deepseek_llm_7b_function_rm', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=310306, ip=127.0.0.1, actor_id=c605fd4d6321948e8875553501000000, repr=<main_ppo.TaskRunner object at 0x7f314be54950>)
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
	actor_id: 07e712cb7b0b521a4322176f01000000
	pid: 320330
	name: eD07RjWorkerDict_0:0
	namespace: eb9b3611-7e47-4b5b-bec1-7a63e3652042
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=320623)[0m [rank1]:W0402 14:36:14.459000 139842812823232 torch/_inductor/compile_worker/subproc_pool.py:126] SubprocPool unclean exit
[36m(WorkerDict pid=320623)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=320623)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff1e23712b56cd9b037979096a01000000 Worker ID: 809a45bbe273dc2d0570da54a4a18ca11bb66cfb2383214c21c21767 Node ID: 9557d62af5f975d09decde2d6adbb2a9a0057c45c8cee6a571c44895 Worker IP address: 127.0.0.1 Worker port: 32989 Worker PID: 320623 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
