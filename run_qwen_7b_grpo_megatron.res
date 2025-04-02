+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo --config-path=config --config-name=ppo_megatron_trainer.yaml algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=32 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=Qwen/Qwen2.5-7B actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=4 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=qwen2_7b_function_rm_megatron trainer.n_gpus_per_node=16 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 18:46:09,153	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=1075016)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1075016)[0m                                                              'hf_model',
[36m(TaskRunner pid=1075016)[0m                                                              'optimizer',
[36m(TaskRunner pid=1075016)[0m                                                              'extra']},
[36m(TaskRunner pid=1075016)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=1075016)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=1075016)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=1075016)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=1075016)[0m                                  'load_weight': True,
[36m(TaskRunner pid=1075016)[0m                                  'megatron': {'pipeline_model_parallel_size': 2,
[36m(TaskRunner pid=1075016)[0m                                               'seed': 1,
[36m(TaskRunner pid=1075016)[0m                                               'sequence_parallel': True,
[36m(TaskRunner pid=1075016)[0m                                               'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=1075016)[0m                                               'use_distributed_optimizer': True,
[36m(TaskRunner pid=1075016)[0m                                               'virtual_pipeline_model_parallel_size': 2},
[36m(TaskRunner pid=1075016)[0m                                  'optim': {'clip_grad': 1.0,
[36m(TaskRunner pid=1075016)[0m                                            'lr': 1e-06,
[36m(TaskRunner pid=1075016)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=1075016)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1075016)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=1075016)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=1075016)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=1075016)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=1075016)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1075016)[0m                                  'ppo_micro_batch_size_per_gpu': 8,
[36m(TaskRunner pid=1075016)[0m                                  'ppo_mini_batch_size': 32,
[36m(TaskRunner pid=1075016)[0m                                  'shuffle': True,
[36m(TaskRunner pid=1075016)[0m                                  'strategy': 'megatron',
[36m(TaskRunner pid=1075016)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=1075016)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=1075016)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=1075016)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=1075016)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=1075016)[0m                                  'external_lib': None,
[36m(TaskRunner pid=1075016)[0m                                  'override_config': {},
[36m(TaskRunner pid=1075016)[0m                                  'path': 'Qwen/Qwen2.5-7B'},
[36m(TaskRunner pid=1075016)[0m                        'ref': {'load_weight': True,
[36m(TaskRunner pid=1075016)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1075016)[0m                                'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=1075016)[0m                                'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=1075016)[0m                                             'seed': 1,
[36m(TaskRunner pid=1075016)[0m                                             'sequence_parallel': True,
[36m(TaskRunner pid=1075016)[0m                                             'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=1075016)[0m                                             'use_distributed_optimizer': True,
[36m(TaskRunner pid=1075016)[0m                                             'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=1075016)[0m                                'param_offload': False},
[36m(TaskRunner pid=1075016)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=1075016)[0m                                    'do_sample': True,
[36m(TaskRunner pid=1075016)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=1075016)[0m                                    'enable_chunked_prefill': False,
[36m(TaskRunner pid=1075016)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=1075016)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=1075016)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=1075016)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=1075016)[0m                                    'layer_name_map': {'gate_proj_layer_name': 'gate_up',
[36m(TaskRunner pid=1075016)[0m                                                       'qkv_layer_name': 'qkv'},
[36m(TaskRunner pid=1075016)[0m                                    'load_format': 'dummy_megatron',
[36m(TaskRunner pid=1075016)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=1075016)[0m                                    'log_prob_micro_batch_size_per_gpu': 16,
[36m(TaskRunner pid=1075016)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=1075016)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=1075016)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=1075016)[0m                                    'n': 4,
[36m(TaskRunner pid=1075016)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=1075016)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=1075016)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=1075016)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=1075016)[0m                                    'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=1075016)[0m                                    'top_k': -1,
[36m(TaskRunner pid=1075016)[0m                                    'top_p': 1,
[36m(TaskRunner pid=1075016)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=1075016)[0m                                                   'n': 1,
[36m(TaskRunner pid=1075016)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=1075016)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=1075016)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=1075016)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=1075016)[0m                'gamma': 1.0,
[36m(TaskRunner pid=1075016)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=1075016)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=1075016)[0m                'lam': 1.0},
[36m(TaskRunner pid=1075016)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=1075016)[0m                                         'hf_model',
[36m(TaskRunner pid=1075016)[0m                                         'optimizer',
[36m(TaskRunner pid=1075016)[0m                                         'extra']},
[36m(TaskRunner pid=1075016)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=1075016)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=1075016)[0m No module named 'vllm._version'
[36m(TaskRunner pid=1075016)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=1075016)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=1085031)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1085031)[0m No module named 'vllm._version'
[36m(pid=1085031)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=1085324)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=1085324)[0m No module named 'vllm._version'
[36m(pid=1085324)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=1085329)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 8x across cluster][0m
[36m(pid=1085329)[0m No module named 'vllm._version'[32m [repeated 8x across cluster][0m
[36m(pid=1085329)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 8x across cluster][0m
[36m(TaskRunner pid=1075016)[0m             'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=1075016)[0m             'load_weight': True,
[36m(TaskRunner pid=1075016)[0m             'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=1075016)[0m                          'seed': 1,
[36m(TaskRunner pid=1075016)[0m                          'sequence_parallel': True,
[36m(TaskRunner pid=1075016)[0m                          'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=1075016)[0m                          'use_distributed_optimizer': True,
[36m(TaskRunner pid=1075016)[0m                          'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=1075016)[0m             'model': {'enable_gradient_checkpointing': False,
[36m(TaskRunner pid=1075016)[0m                       'external_lib': None,
[36m(TaskRunner pid=1075016)[0m                       'override_config': {},
[36m(TaskRunner pid=1075016)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=1075016)[0m                       'tokenizer_path': 'Qwen/Qwen2.5-7B'},
[36m(TaskRunner pid=1075016)[0m             'optim': {'clip_grad': 1.0,
[36m(TaskRunner pid=1075016)[0m                       'lr': 1e-05,
[36m(TaskRunner pid=1075016)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=1075016)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=1075016)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=1075016)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=1075016)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=1075016)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=1075016)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1075016)[0m             'ppo_mini_batch_size': 32,
[36m(TaskRunner pid=1075016)[0m             'shuffle': True,
[36m(TaskRunner pid=1075016)[0m             'strategy': 'megatron',
[36m(TaskRunner pid=1075016)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=1075016)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=1075016)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=1075016)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=1075016)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=1075016)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=1075016)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=1075016)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=1075016)[0m           'shuffle': True,
[36m(TaskRunner pid=1075016)[0m           'tokenizer': None,
[36m(TaskRunner pid=1075016)[0m           'train_batch_size': 32,
[36m(TaskRunner pid=1075016)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=1075016)[0m           'truncation': 'error',
[36m(TaskRunner pid=1075016)[0m           'val_batch_size': None,
[36m(TaskRunner pid=1075016)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=1075016)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=1075016)[0m                   'load_weight': True,
[36m(TaskRunner pid=1075016)[0m                   'max_length': None,
[36m(TaskRunner pid=1075016)[0m                   'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=1075016)[0m                                'seed': 1,
[36m(TaskRunner pid=1075016)[0m                                'sequence_parallel': True,
[36m(TaskRunner pid=1075016)[0m                                'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=1075016)[0m                                'use_distributed_optimizer': True,
[36m(TaskRunner pid=1075016)[0m                                'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=1075016)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=1075016)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=1075016)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=1075016)[0m                             'input_tokenizer': 'Qwen/Qwen2.5-7B',
[36m(TaskRunner pid=1075016)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1'},
[36m(TaskRunner pid=1075016)[0m                   'param_offload': False,
[36m(TaskRunner pid=1075016)[0m                   'strategy': 'megatron',
[36m(TaskRunner pid=1075016)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=1075016)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=1075016)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=1075016)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=1075016)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm_megatron',
[36m(TaskRunner pid=1075016)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=1075016)[0m              'experiment_name': 'qwen2_7b_function_rm_megatron',
[36m(TaskRunner pid=1075016)[0m              'logger': ['console'],
[36m(TaskRunner pid=1075016)[0m              'n_gpus_per_node': 16,
[36m(TaskRunner pid=1075016)[0m              'nnodes': 1,
[36m(TaskRunner pid=1075016)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=1075016)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=1075016)[0m              'resume_from_path': False,
[36m(TaskRunner pid=1075016)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=1075016)[0m              'save_freq': -1,
[36m(TaskRunner pid=1075016)[0m              'test_freq': 5,
[36m(TaskRunner pid=1075016)[0m              'total_epochs': 1,
[36m(TaskRunner pid=1075016)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=1075016)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=1075016)[0m WARNING 04-02 18:46:22 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=1075016)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=1075016)[0m dataset len: 7473
[36m(TaskRunner pid=1075016)[0m filter dataset len: 7473
[36m(TaskRunner pid=1075016)[0m dataset len: 1319
[36m(TaskRunner pid=1075016)[0m filter dataset len: 1319
[36m(TaskRunner pid=1075016)[0m Size of train dataloader: 233
[36m(TaskRunner pid=1075016)[0m Total training steps: 1
[36m(pid=1085031)[0m WARNING 04-02 18:46:33 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=1085326)[0m WARNING 04-02 18:48:14 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=1085329)[0m WARNING 04-02 18:48:20 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 8x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=1085031)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=1085031)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",
[36m(WorkerDict pid=1085320)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
[36m(pid=1085331)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:[32m [repeated 6x across cluster][0m
[36m(pid=1085331)[0m No module named 'vllm._version'[32m [repeated 6x across cluster][0m
[36m(pid=1085331)[0m   from vllm.version import __version__ as VLLM_VERSION[32m [repeated 6x across cluster][0m
[36m(WorkerDict pid=1085328)[0m Loading checkpoint shards:  25%|██▌       | 1/4 [00:00<00:00,  3.59it/s]
[36m(WorkerDict pid=1085328)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  3.55it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:01<00:00,  3.56it/s]
[36m(WorkerDict pid=1085031)[0m   "architectures": [
[36m(WorkerDict pid=1085031)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085031)[0m   ],
[36m(WorkerDict pid=1085031)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=1085031)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=1085031)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=1085031)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=1085031)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=1085031)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=1085031)[0m   "max_position_embeddings": 131072,
[36m(WorkerDict pid=1085031)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=1085031)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=1085031)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=1085031)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=1085031)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=1085031)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=1085031)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=1085031)[0m   "rope_scaling": null,
[36m(WorkerDict pid=1085031)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=1085031)[0m   "sliding_window": null,
[36m(WorkerDict pid=1085031)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=1085031)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=1085031)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=1085031)[0m   "use_cache": true,
[36m(WorkerDict pid=1085031)[0m   "use_mrope": false,
[36m(WorkerDict pid=1085031)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=1085031)[0m   "vocab_size": 152064
[36m(WorkerDict pid=1085031)[0m }
[36m(WorkerDict pid=1085031)[0m 
[36m(WorkerDict pid=1085031)[0m self.config.ref.load_weight: True
[36m(WorkerDict pid=1085031)[0m after load model cls
[36m(WorkerDict pid=1085031)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=1085031)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=1085031)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=1085031)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(pid=1085331)[0m WARNING 04-02 18:48:20 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.[32m [repeated 6x across cluster][0m
[36m(WorkerDict pid=1085328)[0m  > number of parameters on (tensor, pipeline) model parallel rank (3, 1): 952030464
[36m(WorkerDict pid=1085328)[0m load ref weight start
[36m(WorkerDict pid=1085328)[0m load from local dir Qwen/Qwen2.5-7B
[36m(WorkerDict pid=1085328)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...
[36m(WorkerDict pid=1085328)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {
[36m(WorkerDict pid=1085328)[0m   "_attn_implementation_autoset": true,
[36m(WorkerDict pid=1085328)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085328)[0m   "bos_token_id": 151643,
[36m(WorkerDict pid=1085328)[0m 
[36m(WorkerDict pid=1085320)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085320)[0m 
[36m(WorkerDict pid=1085331)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085331)[0m 
[36m(WorkerDict pid=1085329)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085329)[0m 
[36m(WorkerDict pid=1085332)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085332)[0m 
[36m(WorkerDict pid=1085319)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085319)[0m 
[36m(WorkerDict pid=1085318)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085318)[0m 
[36m(WorkerDict pid=1085326)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085326)[0m 
[36m(WorkerDict pid=1085323)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085323)[0m 
[36m(WorkerDict pid=1085327)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085327)[0m 
[36m(WorkerDict pid=1085321)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085321)[0m 
[36m(WorkerDict pid=1085325)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085325)[0m 
[36m(WorkerDict pid=1085330)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085330)[0m 
[36m(WorkerDict pid=1085324)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085324)[0m 
[36m(WorkerDict pid=1085322)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085322)[0m 
[36m(WorkerDict pid=1085031)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085031)[0m 
[36m(WorkerDict pid=1085031)[0m loading embeddings...
[36m(WorkerDict pid=1085031)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=1085031)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "architectures": [[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   ],[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "attention_dropout": 0.0,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "eos_token_id": 151643,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "hidden_act": "silu",[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "hidden_size": 3584,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "initializer_range": 0.02,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "intermediate_size": 18944,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "max_position_embeddings": 131072,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "max_window_layers": 28,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "model_type": "qwen2",[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_attention_heads": 28,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_hidden_layers": 28,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_key_value_heads": 4,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rms_norm_eps": 1e-06,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rope_scaling": null,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rope_theta": 1000000.0,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "sliding_window": null,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "tie_word_embeddings": false,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "torch_dtype": "bfloat16",[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "transformers_version": "4.48.3",[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_cache": true,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_mrope": false,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_sliding_window": false,[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "vocab_size": 152064[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m }[32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085324)[0m self.config.ref.load_weight: True[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085324)[0m after load model cls[32m [repeated 31x across cluster][0m
[36m(WorkerDict pid=1085324)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)[32m [repeated 286x across cluster][0m
[36m(WorkerDict pid=1085324)[0m pipeline_dtype=megatron_config torch.bfloat16[32m [repeated 286x across cluster][0m
[36m(WorkerDict pid=1085031)[0m loading layer #0...
[36m(WorkerDict pid=1085327)[0m  > number of parameters on (tensor, pipeline) model parallel rank (2, 1): 952030464[32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=1085324)[0m load ref weight start[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085324)[0m load from local dir Qwen/Qwen2.5-7B[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m loading layer #1...
[36m(WorkerDict pid=1085031)[0m loading layer #2...
[36m(WorkerDict pid=1085031)[0m loading layer #3...
[36m(WorkerDict pid=1085031)[0m loading layer #4...
[36m(WorkerDict pid=1085031)[0m loading layer #5...
[36m(WorkerDict pid=1085031)[0m loading layer #6...
[36m(WorkerDict pid=1085031)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "_attn_implementation_autoset": true,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "bos_token_id": 151643,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m loading layer #7...
[36m(WorkerDict pid=1085031)[0m loading layer #8...
[36m(WorkerDict pid=1085031)[0m loading layer #9...
[36m(WorkerDict pid=1085031)[0m loading layer #10...
[36m(WorkerDict pid=1085031)[0m loading layer #11...
[36m(WorkerDict pid=1085031)[0m loading layer #12...
[36m(WorkerDict pid=1085031)[0m loading layer #13...
[36m(WorkerDict pid=1085031)[0m loading layer #14...
[36m(WorkerDict pid=1085031)[0m loading layer #15...
[36m(WorkerDict pid=1085031)[0m loading layer #16...
[36m(WorkerDict pid=1085031)[0m loading layer #17...
[36m(WorkerDict pid=1085031)[0m loading layer #18...
[36m(WorkerDict pid=1085031)[0m loading layer #19...
[36m(WorkerDict pid=1085031)[0m loading layer #20...
[36m(WorkerDict pid=1085031)[0m loading layer #21...
[36m(WorkerDict pid=1085031)[0m loading layer #22...
[36m(WorkerDict pid=1085031)[0m loading layer #23...
[36m(WorkerDict pid=1085031)[0m loading layer #24...
[36m(WorkerDict pid=1085031)[0m loading layer #25...
[36m(WorkerDict pid=1085031)[0m loading layer #26...
[36m(WorkerDict pid=1085031)[0m loading layer #27...
[36m(WorkerDict pid=1085031)[0m loading final layernorm...
[36m(WorkerDict pid=1085031)[0m loading lm_head...
[36m(WorkerDict pid=1085318)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=1085318)[0m TransformerConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=True, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=28, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=3584, num_attention_heads=28, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=4, ffn_hidden_size=18944, kv_channels=128, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7f4348334220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7f410233f240>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7f410233d1c0>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=18944, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)
[36m(WorkerDict pid=1085031)[0m loading megatron ckpt done, time elapsed 11.888616800308228s
[36m(WorkerDict pid=1085332)[0m create pp model torch allocated 2.8437 GB, reserved 3.0094 GB
[36m(WorkerDict pid=1085332)[0m after load model cls
[36m(WorkerDict pid=1085332)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=1085332)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=1085332)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=1085332)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=1085332)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=1085332)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=1085031)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=1085031)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",
[36m(WorkerDict pid=1085031)[0m   "architectures": [
[36m(WorkerDict pid=1085031)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085031)[0m   ],
[36m(WorkerDict pid=1085031)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=1085031)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=1085031)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=1085031)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=1085031)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=1085031)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=1085031)[0m   "max_position_embeddings": 131072,
[36m(WorkerDict pid=1085031)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=1085031)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=1085031)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=1085031)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=1085031)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=1085031)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=1085031)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=1085031)[0m   "rope_scaling": null,
[36m(WorkerDict pid=1085031)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=1085031)[0m   "sliding_window": null,
[36m(WorkerDict pid=1085031)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=1085031)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=1085031)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=1085031)[0m   "use_cache": true,
[36m(WorkerDict pid=1085031)[0m   "use_mrope": false,
[36m(WorkerDict pid=1085031)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=1085031)[0m   "vocab_size": 152064
[36m(WorkerDict pid=1085031)[0m }
[36m(WorkerDict pid=1085031)[0m 
[36m(WorkerDict pid=1085328)[0m  > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 952026880
[36m(WorkerDict pid=1085319)[0m NCCL version 2.20.5+cuda12.4[32m [repeated 6x across cluster][0m
[36m(WorkerDict pid=1085031)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085031)[0m Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.12it/s][32m [repeated 47x across cluster][0m
[36m(WorkerDict pid=1085031)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.17it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.15it/s][32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m TransformerConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=True, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=28, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=3584, num_attention_heads=28, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=4, ffn_hidden_size=18944, kv_channels=128, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7efefe93c220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7efe131f8180>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7efe131f8220>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=18944, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m actor_module: 2
[36m(WorkerDict pid=1085031)[0m load from local dir Qwen/Qwen2.5-7B
[36m(WorkerDict pid=1085324)[0m create pp model torch allocated 3.7833 GB, reserved 8.8185 GB[32m [repeated 31x across cluster][0m
[36m(WorkerDict pid=1085327)[0m after load model cls[32m [repeated 63x across cluster][0m
[36m(WorkerDict pid=1085324)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)[32m [repeated 573x across cluster][0m
[36m(WorkerDict pid=1085324)[0m pipeline_dtype=megatron_config torch.bfloat16[32m [repeated 573x across cluster][0m
[36m(WorkerDict pid=1085318)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...
[36m(WorkerDict pid=1085318)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {
[36m(WorkerDict pid=1085318)[0m   "_attn_implementation_autoset": true,
[36m(WorkerDict pid=1085318)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",
[36m(WorkerDict pid=1085318)[0m   "architectures": [
[36m(WorkerDict pid=1085318)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085318)[0m   ],
[36m(WorkerDict pid=1085318)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=1085318)[0m   "bos_token_id": 151643,
[36m(WorkerDict pid=1085318)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=1085318)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=1085318)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=1085318)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=1085318)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=1085318)[0m   "max_position_embeddings": 131072,
[36m(WorkerDict pid=1085318)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=1085318)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=1085318)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=1085318)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=1085318)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=1085318)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=1085318)[0m   "rope_scaling": null,
[36m(WorkerDict pid=1085318)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=1085318)[0m   "sliding_window": null,
[36m(WorkerDict pid=1085318)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=1085318)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=1085318)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=1085318)[0m   "use_cache": true,
[36m(WorkerDict pid=1085318)[0m   "use_mrope": false,
[36m(WorkerDict pid=1085318)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=1085318)[0m   "vocab_size": 152064
[36m(WorkerDict pid=1085318)[0m }
[36m(WorkerDict pid=1085318)[0m 
[36m(WorkerDict pid=1085326)[0m  > number of parameters on (tensor, pipeline) model parallel rank (1, 1): 952030464[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085321)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085321)[0m 
[36m(WorkerDict pid=1085322)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085322)[0m 
[36m(WorkerDict pid=1085323)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085323)[0m 
[36m(WorkerDict pid=1085329)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085329)[0m 
[36m(WorkerDict pid=1085332)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085332)[0m 
[36m(WorkerDict pid=1085328)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085328)[0m 
[36m(WorkerDict pid=1085325)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085325)[0m 
[36m(WorkerDict pid=1085326)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085326)[0m 
[36m(WorkerDict pid=1085319)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085319)[0m 
[36m(WorkerDict pid=1085331)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085331)[0m 
[36m(WorkerDict pid=1085330)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085330)[0m 
[36m(WorkerDict pid=1085324)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085324)[0m 
[36m(WorkerDict pid=1085320)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085320)[0m 
[36m(WorkerDict pid=1085327)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085327)[0m 
[36m(WorkerDict pid=1085031)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=1085031)[0m 
[36m(WorkerDict pid=1085031)[0m loading embeddings...
[36m(WorkerDict pid=1085031)[0m loading layer #0...
[36m(WorkerDict pid=1085031)[0m loading layer #1...
[36m(WorkerDict pid=1085031)[0m loading layer #2...
[36m(WorkerDict pid=1085031)[0m loading layer #3...
[36m(WorkerDict pid=1085324)[0m actor_module: 2[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085324)[0m load from local dir Qwen/Qwen2.5-7B[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m loading layer #4...
[36m(WorkerDict pid=1085031)[0m loading layer #5...
[36m(WorkerDict pid=1085031)[0m loading layer #6...
[36m(WorkerDict pid=1085031)[0m loading layer #7...
[36m(WorkerDict pid=1085031)[0m loading layer #8...
[36m(WorkerDict pid=1085031)[0m loading layer #9...
[36m(WorkerDict pid=1085031)[0m loading layer #10...
[36m(WorkerDict pid=1085031)[0m loading layer #11...
[36m(WorkerDict pid=1085031)[0m loading layer #12...
[36m(WorkerDict pid=1085031)[0m loading layer #13...
[36m(WorkerDict pid=1085031)[0m loading layer #14...
[36m(WorkerDict pid=1085031)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "_attn_implementation_autoset": true,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "_name_or_path": "Qwen/Qwen2.5-7B",[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "architectures": [[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   ],[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "attention_dropout": 0.0,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "bos_token_id": 151643,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "eos_token_id": 151643,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "hidden_act": "silu",[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "hidden_size": 3584,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "initializer_range": 0.02,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "intermediate_size": 18944,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "max_position_embeddings": 131072,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "max_window_layers": 28,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "model_type": "qwen2",[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_attention_heads": 28,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_hidden_layers": 28,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "num_key_value_heads": 4,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rms_norm_eps": 1e-06,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rope_scaling": null,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "rope_theta": 1000000.0,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "sliding_window": null,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "tie_word_embeddings": false,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "torch_dtype": "bfloat16",[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "transformers_version": "4.48.3",[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_cache": true,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_mrope": false,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "use_sliding_window": false,[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m   "vocab_size": 152064[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m }[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m loading layer #15...
[36m(WorkerDict pid=1085031)[0m loading layer #16...
[36m(WorkerDict pid=1085031)[0m loading layer #17...
[36m(WorkerDict pid=1085031)[0m loading layer #18...
[36m(WorkerDict pid=1085031)[0m loading layer #19...
[36m(WorkerDict pid=1085031)[0m loading layer #20...
[36m(WorkerDict pid=1085031)[0m loading layer #21...
[36m(WorkerDict pid=1085031)[0m loading layer #22...
[36m(WorkerDict pid=1085031)[0m loading layer #23...
[36m(WorkerDict pid=1085031)[0m loading layer #24...
[36m(WorkerDict pid=1085031)[0m loading layer #25...
[36m(WorkerDict pid=1085031)[0m loading layer #26...
[36m(WorkerDict pid=1085031)[0m loading layer #27...
[36m(WorkerDict pid=1085031)[0m loading final layernorm...
[36m(WorkerDict pid=1085031)[0m loading lm_head...
[36m(WorkerDict pid=1085031)[0m loading megatron ckpt done, time elapsed 5.953923940658569s
[36m(WorkerDict pid=1085318)[0m TransformerConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=True, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=28, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=3584, num_attention_heads=28, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=4, ffn_hidden_size=18944, kv_channels=128, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7f4348334220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7f40f05ccf40>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7f40f05ccfe0>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=18944, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)
[36m(WorkerDict pid=1085321)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=1085031)[0m DistributedDataParallel contains 544.14M parameters
[36m(WorkerDict pid=1085318)[0m WARNING 04-02 18:49:07 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=1085318)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=1085318)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=1085329)[0m Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m Loading checkpoint shards:  75%|███████▌  | 3/4 [00:02<00:00,  1.11it/s][32m [repeated 48x across cluster][0m
[36m(WorkerDict pid=1085031)[0m Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.15it/s]Loading checkpoint shards: 100%|██████████| 4/4 [00:03<00:00,  1.13it/s][32m [repeated 16x across cluster][0m
[36m(WorkerDict pid=1085318)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=1085318)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(TaskRunner pid=1075016)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=1085321)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 30x across cluster][0m
[36m(WorkerDict pid=1085321)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085321)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:653: UserWarning: When using sequence parallelism it is recommended to set the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for maximum speedup
[36m(WorkerDict pid=1085031)[0m   warnings.warn(
[36m(WorkerDict pid=1085325)[0m /usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py:768: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
[36m(WorkerDict pid=1085325)[0m   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[36m(WorkerDict pid=1085326)[0m /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:653: UserWarning: When using sequence parallelism it is recommended to set the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for maximum speedup[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085326)[0m   warnings.warn([32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m TransformerConfig(tensor_model_parallel_size=4, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=4, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=True, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=2, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=28, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=3584, num_attention_heads=28, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=4, ffn_hidden_size=18944, kv_channels=128, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7efefe93c220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7efdf83ddbc0>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7efdf83ddc60>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=18944, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085330)[0m NCCL version 2.20.5+cuda12.4[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=1085318)[0m local rank 0
[36m(WorkerDict pid=1085031)[0m before init cache memory allocated: 18.09707008GB, reserved: 18.503172096GB
[36m(WorkerDict pid=1085330)[0m WARNING 04-02 18:49:09 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085321)[0m local rank 0[32m [repeated 15x across cluster][0m
[36m(WorkerDict pid=1085031)[0m after init cache memory allocated: 33.599217664GB, reserved: 34.00531968GB
[36m(WorkerDict pid=1085320)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=1075016)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=1075016)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm_megatron/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=1075016)[0m Training from scratch
[36m(TaskRunner pid=1075016)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=1075016)[0m validation generation end
[36m(WorkerDict pid=1085330)[0m kwargs: {'n': 4, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}[32m [repeated 15x across cluster][0m
[36m(TaskRunner pid=1075016)[0m [prompt] system
[36m(TaskRunner pid=1075016)[0m You are a helpful assistant.
[36m(TaskRunner pid=1075016)[0m user
[36m(TaskRunner pid=1075016)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=1075016)[0m assistant
[36m(TaskRunner pid=1075016)[0m 
[36m(TaskRunner pid=1075016)[0m [response] Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she has 16 - 3 - 4 = 9 eggs left to sell. She sells each egg for $2, so she makes 9 * $2 = $18 every day at the farmers' market. The final answer is $18.
[36m(TaskRunner pid=1075016)[0m [ground_truth] 18
[36m(TaskRunner pid=1075016)[0m [score] 0.0
[36m(TaskRunner pid=1075016)[0m ("Initial validation metrics: {'val/test_score/openai/gsm8k': "
[36m(TaskRunner pid=1075016)[0m  '0.1326762699014405}')
[36m(TaskRunner pid=1075016)[0m step:0 - val/test_score/openai/gsm8k:0.133
[36m(TaskRunner pid=1075016)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=1075016)[0m validation generation end
[36m(TaskRunner pid=1075016)[0m [prompt] system
[36m(TaskRunner pid=1075016)[0m You are a helpful assistant.
[36m(TaskRunner pid=1075016)[0m user
[36m(TaskRunner pid=1075016)[0m Training Progress:   0%|          | 0/1 [00:53<?, ?it/s]
[36m(WorkerDict pid=1085319)[0m /usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py:768: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)[32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=1085319)[0m   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass[32m [repeated 7x across cluster][0m
[36m(TaskRunner pid=1075016)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=1075016)[0m assistant
[36m(TaskRunner pid=1075016)[0m 
[36m(TaskRunner pid=1075016)[0m [response] Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she has 16 - 3 - 4 = 9 eggs left to sell. She sells each egg for $2, so she makes 9 * $2 = $18 every day at the farmers' market. The final answer is $18.
[36m(TaskRunner pid=1075016)[0m [ground_truth] 18
[36m(TaskRunner pid=1075016)[0m [score] 0.0
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff814f12c74fc0ab7ce00658d901000000 Worker ID: 8c489e97fb04fb8073b7e6ad3748c4f143ac54fd7bae4475ce23bc6d Node ID: 2b4598e19bbbae3705111f616c84508861100638e4e1f7856c77461b Worker IP address: 127.0.0.1 Worker port: 47591 Worker PID: 1075016 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=32', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-7B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.ppo_mini_batch_size=32', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8', 'actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2', 'actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2', 'actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=4', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=4', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=qwen2_7b_function_rm_megatron', 'trainer.n_gpus_per_node=16', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
  File "/home/tiger/.local/lib/python3.11/site-packages/ray/_private/worker.py", line 931, in get_objects
    raise value
ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
	class_name: TaskRunner
	actor_id: 814f12c74fc0ab7ce00658d901000000
	pid: 1075016
	namespace: 3ecb758f-3579-43d8-832a-5e772917a5e7
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
