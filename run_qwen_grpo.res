+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=4 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=qwen2_7b_function_rm trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 13:42:32,586	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=139996)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=139996)[0m                                                              'hf_model',
[36m(TaskRunner pid=139996)[0m                                                              'optimizer',
[36m(TaskRunner pid=139996)[0m                                                              'extra']},
[36m(TaskRunner pid=139996)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=139996)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=139996)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=139996)[0m                                                  'optimizer_offload': True,
[36m(TaskRunner pid=139996)[0m                                                  'param_offload': True,
[36m(TaskRunner pid=139996)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=139996)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=139996)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=139996)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=139996)[0m                                  'optim': {'lr': 1e-06,
[36m(TaskRunner pid=139996)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=139996)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=139996)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=139996)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=139996)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=139996)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=139996)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=139996)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m                                  'ppo_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=139996)[0m                                  'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=139996)[0m                                  'shuffle': False,
[36m(TaskRunner pid=139996)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=139996)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=139996)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=139996)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=139996)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=139996)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=139996)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=139996)[0m                                  'external_lib': None,
[36m(TaskRunner pid=139996)[0m                                  'override_config': {},
[36m(TaskRunner pid=139996)[0m                                  'path': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=139996)[0m                                  'use_remove_padding': True},
[36m(TaskRunner pid=139996)[0m                        'ref': {'fsdp_config': {'param_offload': True,
[36m(TaskRunner pid=139996)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=139996)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=139996)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m                                'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=139996)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=139996)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=139996)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=139996)[0m                                    'do_sample': True,
[36m(TaskRunner pid=139996)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=139996)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=139996)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=139996)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=139996)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=139996)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=139996)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=139996)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=139996)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m                                    'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=139996)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=139996)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=139996)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=139996)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=139996)[0m                                    'n': 5,
[36m(TaskRunner pid=139996)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=139996)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=139996)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=139996)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=139996)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=139996)[0m                                    'top_k': -1,
[36m(TaskRunner pid=139996)[0m                                    'top_p': 1,
[36m(TaskRunner pid=139996)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=139996)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=139996)[0m                                                   'n': 1,
[36m(TaskRunner pid=139996)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=139996)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=139996)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=139996)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=139996)[0m                'gamma': 1.0,
[36m(TaskRunner pid=139996)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=139996)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=139996)[0m                'lam': 1.0},
[36m(TaskRunner pid=139996)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=139996)[0m                                         'hf_model',
[36m(TaskRunner pid=139996)[0m                                         'optimizer',
[36m(TaskRunner pid=139996)[0m                                         'extra']},
[36m(TaskRunner pid=139996)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=139996)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=139996)[0m             'forward_micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=139996)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=139996)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=139996)[0m No module named 'vllm._version'
[36m(TaskRunner pid=139996)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=139996)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=150008)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=150008)[0m No module named 'vllm._version'
[36m(pid=150008)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=150298)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=150298)[0m No module named 'vllm._version'
[36m(pid=150298)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(WorkerDict pid=150008)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(TaskRunner pid=139996)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=139996)[0m                       'external_lib': None,
[36m(TaskRunner pid=139996)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=139996)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=139996)[0m                                       'param_offload': False,
[36m(TaskRunner pid=139996)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=139996)[0m                       'override_config': {},
[36m(TaskRunner pid=139996)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=139996)[0m                       'tokenizer_path': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=139996)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=139996)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=139996)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=139996)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=139996)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=139996)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=139996)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=139996)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=139996)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=139996)[0m             'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=139996)[0m             'shuffle': False,
[36m(TaskRunner pid=139996)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=139996)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=139996)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=139996)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=139996)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=139996)[0m           'image_key': 'images',
[36m(TaskRunner pid=139996)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=139996)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=139996)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=139996)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=139996)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=139996)[0m           'shuffle': True,
[36m(TaskRunner pid=139996)[0m           'tokenizer': None,
[36m(TaskRunner pid=139996)[0m           'train_batch_size': 4,
[36m(TaskRunner pid=139996)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=139996)[0m           'truncation': 'error',
[36m(TaskRunner pid=139996)[0m           'val_batch_size': None,
[36m(TaskRunner pid=139996)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=139996)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=139996)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=139996)[0m                   'max_length': None,
[36m(TaskRunner pid=139996)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=139996)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=139996)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=139996)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=139996)[0m                                             'param_offload': False,
[36m(TaskRunner pid=139996)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=139996)[0m                             'input_tokenizer': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=139996)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1',
[36m(TaskRunner pid=139996)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=139996)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=139996)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=139996)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=139996)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=139996)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=139996)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=139996)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=139996)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm',
[36m(TaskRunner pid=139996)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=139996)[0m              'experiment_name': 'qwen2_7b_function_rm',
[36m(TaskRunner pid=139996)[0m              'logger': ['console'],
[36m(TaskRunner pid=139996)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=139996)[0m              'nnodes': 1,
[36m(TaskRunner pid=139996)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=139996)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=139996)[0m              'resume_from_path': False,
[36m(TaskRunner pid=139996)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=139996)[0m              'save_freq': -1,
[36m(TaskRunner pid=139996)[0m              'test_freq': 5,
[36m(TaskRunner pid=139996)[0m              'total_epochs': 1,
[36m(TaskRunner pid=139996)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=139996)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=139996)[0m WARNING 04-02 13:42:45 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=139996)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=139996)[0m dataset len: 7473
[36m(TaskRunner pid=139996)[0m filter dataset len: 7473
[36m(TaskRunner pid=139996)[0m dataset len: 1319
[36m(TaskRunner pid=139996)[0m filter dataset len: 1319
[36m(TaskRunner pid=139996)[0m Size of train dataloader: 1868
[36m(TaskRunner pid=139996)[0m Total training steps: 1
[36m(pid=150008)[0m WARNING 04-02 13:42:57 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=150298)[0m WARNING 04-02 13:43:07 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(WorkerDict pid=150008)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=150008)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=150008)[0m   "architectures": [
[36m(WorkerDict pid=150008)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=150008)[0m   ],
[36m(WorkerDict pid=150008)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=150008)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=150008)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=150008)[0m   "hidden_size": 896,
[36m(WorkerDict pid=150008)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=150008)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=150008)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=150008)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=150008)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=150008)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=150008)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=150298)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=150298)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=150298)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=150298)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=150298)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=150298)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=150298)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=150298)[0m   warnings.warn(
[36m(WorkerDict pid=150008)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=150008)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=150008)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=150008)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=150008)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=150008)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=150008)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=150008)[0m   "rope_scaling": null,
[36m(WorkerDict pid=150008)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=150008)[0m   "sliding_window": null,
[36m(WorkerDict pid=150008)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=150008)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=150008)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=150008)[0m   "use_cache": true,
[36m(WorkerDict pid=150008)[0m   "use_mrope": false,
[36m(WorkerDict pid=150008)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=150008)[0m   "vocab_size": 151936
[36m(WorkerDict pid=150008)[0m }
[36m(WorkerDict pid=150008)[0m 
[36m(WorkerDict pid=150008)[0m Monkey patch forward in Qwen2 and Llama
[36m(WorkerDict pid=150008)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention
[36m(WorkerDict pid=150008)[0m Qwen2ForCausalLM contains 494.03M parameters
[36m(WorkerDict pid=150008)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee95dbd9260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee95dbd9120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=150008)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=150008)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=150008)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=150008)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=150008)[0m   "architectures": [
[36m(WorkerDict pid=150008)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=150008)[0m   ],
[36m(WorkerDict pid=150008)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=150008)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=150008)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=150008)[0m   "hidden_size": 896,
[36m(WorkerDict pid=150008)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=150008)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=150008)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=150008)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=150008)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=150008)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=150008)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=150008)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=150008)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=150008)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=150008)[0m   "rope_scaling": null,
[36m(WorkerDict pid=150008)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=150008)[0m   "sliding_window": null,
[36m(WorkerDict pid=150008)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=150008)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=150008)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=150008)[0m   "use_cache": true,
[36m(WorkerDict pid=150008)[0m   "use_mrope": false,
[36m(WorkerDict pid=150008)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=150008)[0m   "vocab_size": 151936
[36m(WorkerDict pid=150008)[0m }
[36m(WorkerDict pid=150008)[0m 
[36m(WorkerDict pid=150008)[0m Qwen2ForCausalLM contains 494.03M parameters
[36m(WorkerDict pid=150008)[0m Monkey patch forward in Qwen2 and Llama[32m [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=150008)[0m Monkey patch _flash_attention_forward in transformers.integrations.flash_attention[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150008)[0m wrap_policy: functools.partial(<function _or_policy at 0x7ee95dbd9260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7ee95dbd9120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=150298)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=150008)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=150008)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=150008)[0m Before building vllm rollout, memory allocated (GB): 0.9205484390258789, memory reserved (GB): 2.78125
[36m(WorkerDict pid=150298)[0m WARNING 04-02 13:43:24 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=150298)[0m wrap_policy: functools.partial(<function _or_policy at 0x7f1014f69260>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7f1014f69120>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=150298)[0m Total steps: 1, num_warmup_steps: 0
[36m(WorkerDict pid=150298)[0m Actor use_remove_padding=True
[36m(WorkerDict pid=150298)[0m local rank 0
[36m(WorkerDict pid=150008)[0m before init cache memory allocated: 1.495428096GB, reserved: 1.549795328GB
[36m(WorkerDict pid=150008)[0m after init cache memory allocated: 27.81888GB, reserved: 27.873247232GB
[36m(WorkerDict pid=150298)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=150008)[0m WARNING 04-02 13:43:25 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=150008)[0m local rank 0
[36m(TaskRunner pid=139996)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=139996)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=139996)[0m Training from scratch
[36m(WorkerDict pid=150008)[0m After building vllm rollout, memory allocated (GB): 25.448018074035645, memory reserved (GB): 25.958984375
[36m(WorkerDict pid=150008)[0m After building sharding manager, memory allocated (GB): 25.448018074035645, memory reserved (GB): 25.958984375
[36m(TaskRunner pid=139996)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=139996)[0m validation generation end
[36m(WorkerDict pid=150008)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=139996)[0m [prompt] system
[36m(TaskRunner pid=139996)[0m You are a helpful assistant.
[36m(TaskRunner pid=139996)[0m user
[36m(TaskRunner pid=139996)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=139996)[0m assistant
[36m(TaskRunner pid=139996)[0m 
[36m(TaskRunner pid=139996)[0m [response] Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=150008)[0m /usr/local/lib/python3.11/dist-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .
[36m(WorkerDict pid=150008)[0m   warnings.warn(
[36m(WorkerDict pid=150008)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=150008)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=139996)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg
[36m(TaskRunner pid=139996)[0m [ground_truth] 18
[36m(TaskRunner pid=139996)[0m [score] 0.0
[36m(TaskRunner pid=139996)[0m ("Initial validation metrics: {'val/test_score/openai/gsm8k': "
[36m(TaskRunner pid=139996)[0m  '0.009097801364670205}')
[36m(TaskRunner pid=139996)[0m step:0 - val/test_score/openai/gsm8k:0.009
[36m(WorkerDict pid=150008)[0m hidden_states = outputs[0], shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([6235, 151936])
[36m(WorkerDict pid=150008)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=150008)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([6235])
[36m(WorkerDict pid=150008)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([6235]), input_ids_rmpad_rolled.shape: torch.Size([6235])
[36m(WorkerDict pid=150008)[0m hidden_states = outputs[0], shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([6235, 151936])
[36m(WorkerDict pid=150008)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=150008)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([6235])
[36m(WorkerDict pid=150008)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([6235]), input_ids_rmpad_rolled.shape: torch.Size([6235])
[36m(WorkerDict pid=150008)[0m hidden_states = outputs[0], shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits = self.lm_head(hidden_states), shape: torch.Size([1, 6235, 896])
[36m(WorkerDict pid=150008)[0m logits_rmpad = output.logits.squeeze(0), shape: torch.Size([6235, 151936])
[36m(WorkerDict pid=150008)[0m logits_rmpad.div_(temperature), temperature: 1.0
[36m(WorkerDict pid=150008)[0m entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad), shape: torch.Size([6235])
[36m(WorkerDict pid=150008)[0m log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled), shape: torch.Size([6235]), input_ids_rmpad_rolled.shape: torch.Size([6235])
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff8c22c88e5dbb3e566fc0518001000000 Worker ID: 1efd167431990baae8361c1abf2e5773e42430b6a530fd0bf81be063 Node ID: 0d733c08c95487a8952a8a6c8f44ccc8c8503b2422b7f8a40f7846ce Worker IP address: 127.0.0.1 Worker port: 43489 Worker PID: 150008 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=4', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=4', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=5', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=qwen2_7b_function_rm', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
ray.exceptions.RayTaskError(ActorDiedError): [36mray::TaskRunner.run()[39m (pid=139996, ip=127.0.0.1, actor_id=2847ac2fe8e1ec720f06f39d01000000, repr=<main_ppo.TaskRunner object at 0x7f9780dbea50>)
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
	actor_id: 8c22c88e5dbb3e566fc0518001000000
	pid: 150008
	name: Y1y0SuWorkerDict_0:0
	namespace: 880e7111-2116-49d2-8ed6-1fa9e419f149
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
[36m(WorkerDict pid=150298)[0m /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
[36m(WorkerDict pid=150298)[0m   with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff7bed129e63c3e7a11405cf5801000000 Worker ID: 283e15bb856e22a89b3dd201631fa2b095766731457ea06b235b4b2c Node ID: 0d733c08c95487a8952a8a6c8f44ccc8c8503b2422b7f8a40f7846ce Worker IP address: 127.0.0.1 Worker port: 39969 Worker PID: 150298 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
