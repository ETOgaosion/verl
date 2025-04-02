+ export VLLM_ATTENTION_BACKEND=XFORMERS
+ VLLM_ATTENTION_BACKEND=XFORMERS
+ python3 -m verl.trainer.main_ppo --config-path=config --config-name=ppo_megatron_trainer.yaml algorithm.adv_estimator=grpo data.train_files=/home/tiger/data/gsm8k/train.parquet data.val_files=/home/tiger/data/gsm8k/test.parquet data.train_batch_size=4 data.max_prompt_length=512 data.max_response_length=1024 data.filter_overlong_prompts=True data.truncation=error actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.n=5 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 algorithm.kl_ctrl.kl_coef=0.001 trainer.critic_warmup=0 'trainer.logger=[console]' trainer.project_name=verl_grpo_example_gsm8k trainer.experiment_name=qwen2_7b_function_rm_megatron trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=5 trainer.total_epochs=1 trainer.total_training_steps=1
2025-04-02 14:36:22,978	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
[36m(TaskRunner pid=334103)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=334103)[0m                                                              'hf_model',
[36m(TaskRunner pid=334103)[0m                                                              'optimizer',
[36m(TaskRunner pid=334103)[0m                                                              'extra']},
[36m(TaskRunner pid=334103)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=334103)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=334103)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=334103)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=334103)[0m                                  'load_weight': True,
[36m(TaskRunner pid=334103)[0m                                  'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=334103)[0m                                               'seed': 1,
[36m(TaskRunner pid=334103)[0m                                               'sequence_parallel': True,
[36m(TaskRunner pid=334103)[0m                                               'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=334103)[0m                                               'use_distributed_optimizer': True,
[36m(TaskRunner pid=334103)[0m                                               'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=334103)[0m                                  'optim': {'clip_grad': 1.0,
[36m(TaskRunner pid=334103)[0m                                            'lr': 1e-06,
[36m(TaskRunner pid=334103)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=334103)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=334103)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=334103)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=334103)[0m                                            'warmup_style': 'constant'},
[36m(TaskRunner pid=334103)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=334103)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=334103)[0m                                  'ppo_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=334103)[0m                                  'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=334103)[0m                                  'shuffle': True,
[36m(TaskRunner pid=334103)[0m                                  'strategy': 'megatron',
[36m(TaskRunner pid=334103)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=334103)[0m                                  'use_kl_loss': True,
[36m(TaskRunner pid=334103)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=334103)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=334103)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=334103)[0m                                  'external_lib': None,
[36m(TaskRunner pid=334103)[0m                                  'override_config': {},
[36m(TaskRunner pid=334103)[0m                                  'path': 'Qwen/Qwen2.5-0.5B'},
[36m(TaskRunner pid=334103)[0m                        'ref': {'load_weight': True,
[36m(TaskRunner pid=334103)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=334103)[0m                                'log_prob_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=334103)[0m                                'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=334103)[0m                                             'seed': 1,
[36m(TaskRunner pid=334103)[0m                                             'sequence_parallel': True,
[36m(TaskRunner pid=334103)[0m                                             'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=334103)[0m                                             'use_distributed_optimizer': True,
[36m(TaskRunner pid=334103)[0m                                             'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=334103)[0m                                'param_offload': False},
[36m(TaskRunner pid=334103)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=334103)[0m                                    'do_sample': True,
[36m(TaskRunner pid=334103)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=334103)[0m                                    'enable_chunked_prefill': False,
[36m(TaskRunner pid=334103)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=334103)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=334103)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=334103)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=334103)[0m                                    'layer_name_map': {'gate_proj_layer_name': 'gate_up',
[36m(TaskRunner pid=334103)[0m                                                       'qkv_layer_name': 'qkv'},
[36m(TaskRunner pid=334103)[0m                                    'load_format': 'dummy_megatron',
[36m(TaskRunner pid=334103)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=334103)[0m                                    'log_prob_micro_batch_size_per_gpu': 4,
[36m(TaskRunner pid=334103)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=334103)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=334103)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=334103)[0m                                    'n': 5,
[36m(TaskRunner pid=334103)[0m                                    'name': 'vllm',
[36m(TaskRunner pid=334103)[0m                                    'prompt_length': 512,
[36m(TaskRunner pid=334103)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=334103)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=334103)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=334103)[0m                                    'top_k': -1,
[36m(TaskRunner pid=334103)[0m                                    'top_p': 1,
[36m(TaskRunner pid=334103)[0m                                    'val_kwargs': {'do_sample': False,
[36m(TaskRunner pid=334103)[0m                                                   'n': 1,
[36m(TaskRunner pid=334103)[0m                                                   'temperature': 0,
[36m(TaskRunner pid=334103)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=334103)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=334103)[0m  'algorithm': {'adv_estimator': 'grpo',
[36m(TaskRunner pid=334103)[0m                'gamma': 1.0,
[36m(TaskRunner pid=334103)[0m                'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=334103)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=334103)[0m                'lam': 1.0},
[36m(TaskRunner pid=334103)[0m  'critic': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=334103)[0m                                         'hf_model',
[36m(TaskRunner pid=334103)[0m                                         'optimizer',
[36m(TaskRunner pid=334103)[0m                                         'extra']},
[36m(TaskRunner pid=334103)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=334103)[0m             'kl_ctrl': {'kl_coef': 0.001, 'type': 'fixed'},
[36m(TaskRunner pid=334103)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(TaskRunner pid=334103)[0m No module named 'vllm._version'
[36m(TaskRunner pid=334103)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=334103)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(pid=344114)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=344114)[0m No module named 'vllm._version'
[36m(pid=344114)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(pid=344574)[0m /usr/local/lib/python3.11/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
[36m(pid=344574)[0m No module named 'vllm._version'
[36m(pid=344574)[0m   from vllm.version import __version__ as VLLM_VERSION
[36m(TaskRunner pid=334103)[0m             'load_weight': True,
[36m(TaskRunner pid=334103)[0m             'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=334103)[0m                          'seed': 1,
[36m(TaskRunner pid=334103)[0m                          'sequence_parallel': True,
[36m(TaskRunner pid=334103)[0m                          'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=334103)[0m                          'use_distributed_optimizer': True,
[36m(TaskRunner pid=334103)[0m                          'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=334103)[0m             'model': {'enable_gradient_checkpointing': False,
[36m(TaskRunner pid=334103)[0m                       'external_lib': None,
[36m(TaskRunner pid=334103)[0m                       'override_config': {},
[36m(TaskRunner pid=334103)[0m                       'path': '~/models/deepseek-llm-7b-chat',
[36m(TaskRunner pid=334103)[0m                       'tokenizer_path': 'Qwen/Qwen2.5-0.5B'},
[36m(TaskRunner pid=334103)[0m             'optim': {'clip_grad': 1.0,
[36m(TaskRunner pid=334103)[0m                       'lr': 1e-05,
[36m(TaskRunner pid=334103)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=334103)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=334103)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=334103)[0m                       'warmup_style': 'constant'},
[36m(TaskRunner pid=334103)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=334103)[0m             'ppo_micro_batch_size': None,
[36m(TaskRunner pid=334103)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=334103)[0m             'ppo_mini_batch_size': 4,
[36m(TaskRunner pid=334103)[0m             'shuffle': True,
[36m(TaskRunner pid=334103)[0m             'strategy': 'megatron',
[36m(TaskRunner pid=334103)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=334103)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=334103)[0m  'data': {'filter_overlong_prompts': True,
[36m(TaskRunner pid=334103)[0m           'max_prompt_length': 512,
[36m(TaskRunner pid=334103)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=334103)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=334103)[0m           'return_raw_chat': False,
[36m(TaskRunner pid=334103)[0m           'return_raw_input_ids': False,
[36m(TaskRunner pid=334103)[0m           'shuffle': True,
[36m(TaskRunner pid=334103)[0m           'tokenizer': None,
[36m(TaskRunner pid=334103)[0m           'train_batch_size': 4,
[36m(TaskRunner pid=334103)[0m           'train_files': '/home/tiger/data/gsm8k/train.parquet',
[36m(TaskRunner pid=334103)[0m           'truncation': 'error',
[36m(TaskRunner pid=334103)[0m           'val_batch_size': None,
[36m(TaskRunner pid=334103)[0m           'val_files': '/home/tiger/data/gsm8k/test.parquet'},
[36m(TaskRunner pid=334103)[0m  'reward_model': {'enable': False,
[36m(TaskRunner pid=334103)[0m                   'load_weight': True,
[36m(TaskRunner pid=334103)[0m                   'max_length': None,
[36m(TaskRunner pid=334103)[0m                   'megatron': {'pipeline_model_parallel_size': 1,
[36m(TaskRunner pid=334103)[0m                                'seed': 1,
[36m(TaskRunner pid=334103)[0m                                'sequence_parallel': True,
[36m(TaskRunner pid=334103)[0m                                'tensor_model_parallel_size': 4,
[36m(TaskRunner pid=334103)[0m                                'use_distributed_optimizer': True,
[36m(TaskRunner pid=334103)[0m                                'virtual_pipeline_model_parallel_size': None},
[36m(TaskRunner pid=334103)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=334103)[0m                   'micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=334103)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=334103)[0m                             'input_tokenizer': 'Qwen/Qwen2.5-0.5B',
[36m(TaskRunner pid=334103)[0m                             'path': '~/models/FsfairX-LLaMA3-RM-v0.1'},
[36m(TaskRunner pid=334103)[0m                   'param_offload': False,
[36m(TaskRunner pid=334103)[0m                   'strategy': 'megatron',
[36m(TaskRunner pid=334103)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=334103)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=334103)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=334103)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=334103)[0m              'default_local_dir': 'checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm_megatron',
[36m(TaskRunner pid=334103)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=334103)[0m              'experiment_name': 'qwen2_7b_function_rm_megatron',
[36m(TaskRunner pid=334103)[0m              'logger': ['console'],
[36m(TaskRunner pid=334103)[0m              'n_gpus_per_node': 2,
[36m(TaskRunner pid=334103)[0m              'nnodes': 1,
[36m(TaskRunner pid=334103)[0m              'project_name': 'verl_grpo_example_gsm8k',
[36m(TaskRunner pid=334103)[0m              'remove_previous_ckpt_in_save': False,
[36m(TaskRunner pid=334103)[0m              'resume_from_path': False,
[36m(TaskRunner pid=334103)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=334103)[0m              'save_freq': -1,
[36m(TaskRunner pid=334103)[0m              'test_freq': 5,
[36m(TaskRunner pid=334103)[0m              'total_epochs': 1,
[36m(TaskRunner pid=334103)[0m              'total_training_steps': 1,
[36m(TaskRunner pid=334103)[0m              'val_generations_to_log_to_wandb': 0}}
[36m(TaskRunner pid=334103)[0m WARNING 04-02 14:36:35 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(TaskRunner pid=334103)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=334103)[0m dataset len: 7473
[36m(TaskRunner pid=334103)[0m filter dataset len: 7473
[36m(TaskRunner pid=334103)[0m dataset len: 1319
[36m(TaskRunner pid=334103)[0m filter dataset len: 1319
[36m(TaskRunner pid=334103)[0m Size of train dataloader: 1868
[36m(TaskRunner pid=334103)[0m Total training steps: 1
[36m(pid=344114)[0m WARNING 04-02 14:36:47 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(pid=344574)[0m WARNING 04-02 14:36:59 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
[36m(WorkerDict pid=344114)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=344114)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=344114)[0m   "architectures": [
[36m(WorkerDict pid=344114)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344114)[0m   ],
[36m(WorkerDict pid=344114)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=344114)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=344114)[0m   "hidden_size": 896,
[36m(WorkerDict pid=344114)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=344114)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=344114)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=344114)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=344114)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=344114)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=344114)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=344114)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=344114)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=344114)[0m   "rope_scaling": null,
[36m(WorkerDict pid=344114)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=344114)[0m   "sliding_window": null,
[36m(WorkerDict pid=344114)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=344114)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=344114)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=344114)[0m   "use_cache": true,
[36m(WorkerDict pid=344114)[0m   "use_mrope": false,
[36m(WorkerDict pid=344114)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=344114)[0m   "vocab_size": 151936
[36m(WorkerDict pid=344114)[0m }
[36m(WorkerDict pid=344114)[0m 
[36m(WorkerDict pid=344114)[0m self.config.ref.load_weight: True
[36m(WorkerDict pid=344114)[0m after load model cls
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16
[36m(WorkerDict pid=344114)[0m  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 247038336
[36m(WorkerDict pid=344114)[0m load ref weight start
[36m(WorkerDict pid=344114)[0m load from local dir Qwen/Qwen2.5-0.5B
[36m(WorkerDict pid=344114)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...
[36m(WorkerDict pid=344114)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {
[36m(WorkerDict pid=344114)[0m   "_attn_implementation_autoset": true,
[36m(WorkerDict pid=344114)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=344114)[0m   "architectures": [
[36m(WorkerDict pid=344114)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344114)[0m   ],
[36m(WorkerDict pid=344114)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=344114)[0m   "bos_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=344114)[0m   "hidden_size": 896,
[36m(WorkerDict pid=344114)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=344114)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=344114)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=344114)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=344114)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=344114)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=344114)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=344114)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=344114)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=344114)[0m   "rope_scaling": null,
[36m(WorkerDict pid=344114)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=344114)[0m   "sliding_window": null,
[36m(WorkerDict pid=344114)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=344114)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=344114)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=344114)[0m   "use_cache": true,
[36m(WorkerDict pid=344114)[0m   "use_mrope": false,
[36m(WorkerDict pid=344114)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=344114)[0m   "vocab_size": 151936
[36m(WorkerDict pid=344114)[0m }
[36m(WorkerDict pid=344114)[0m 
[36m(WorkerDict pid=344114)[0m loading embeddings...
[36m(WorkerDict pid=344114)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=344574)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344574)[0m 
[36m(WorkerDict pid=344114)[0m loading layer #0...
[36m(WorkerDict pid=344114)[0m loading layer #1...
[36m(WorkerDict pid=344114)[0m loading layer #2...
[36m(WorkerDict pid=344114)[0m loading layer #3...
[36m(WorkerDict pid=344114)[0m loading layer #4...
[36m(WorkerDict pid=344114)[0m loading layer #5...
[36m(WorkerDict pid=344114)[0m loading layer #6...
[36m(WorkerDict pid=344114)[0m loading layer #7...
[36m(WorkerDict pid=344114)[0m loading layer #8...
[36m(WorkerDict pid=344114)[0m loading layer #9...
[36m(WorkerDict pid=344114)[0m loading layer #10...
[36m(WorkerDict pid=344114)[0m loading layer #11...
[36m(WorkerDict pid=344114)[0m loading layer #12...
[36m(WorkerDict pid=344114)[0m loading layer #13...
[36m(WorkerDict pid=344114)[0m loading layer #14...
[36m(WorkerDict pid=344114)[0m loading layer #15...
[36m(WorkerDict pid=344114)[0m loading layer #16...
[36m(WorkerDict pid=344114)[0m loading layer #17...
[36m(WorkerDict pid=344114)[0m loading layer #18...
[36m(WorkerDict pid=344114)[0m loading layer #19...
[36m(WorkerDict pid=344114)[0m loading layer #20...
[36m(WorkerDict pid=344114)[0m loading layer #21...
[36m(WorkerDict pid=344114)[0m loading layer #22...
[36m(WorkerDict pid=344114)[0m loading layer #23...
[36m(WorkerDict pid=344114)[0m loading final layernorm...
[36m(WorkerDict pid=344114)[0m tie_word_embeddings skip load lm_head
[36m(WorkerDict pid=344114)[0m loading megatron ckpt done, time elapsed 2.2618789672851562s
[36m(WorkerDict pid=344114)[0m TransformerConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=False, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=24, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=896, num_attention_heads=14, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=2, ffn_hidden_size=4864, kv_channels=64, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7f63599c4220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7f62a53844a0>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7f62a5384540>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=4864, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)
[36m(WorkerDict pid=344574)[0m create pp model torch allocated 0.6956 GB, reserved 0.9206 GB
[36m(WorkerDict pid=344574)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=344574)[0m   "architectures": [
[36m(WorkerDict pid=344574)[0m   ],
[36m(WorkerDict pid=344574)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=344574)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=344574)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=344574)[0m   "hidden_size": 896,
[36m(WorkerDict pid=344574)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=344574)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=344574)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=344574)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=344574)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=344574)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=344574)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=344574)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=344574)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=344574)[0m   "rope_scaling": null,
[36m(WorkerDict pid=344574)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=344574)[0m   "sliding_window": null,
[36m(WorkerDict pid=344574)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=344574)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=344574)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=344574)[0m   "use_cache": true,
[36m(WorkerDict pid=344574)[0m   "use_mrope": false,
[36m(WorkerDict pid=344574)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=344574)[0m   "vocab_size": 151936
[36m(WorkerDict pid=344574)[0m }
[36m(WorkerDict pid=344574)[0m self.config.ref.load_weight: True
[36m(WorkerDict pid=344574)[0m after load model cls[32m [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=344574)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)[32m [repeated 48x across cluster][0m
[36m(WorkerDict pid=344574)[0m pipeline_dtype=megatron_config torch.bfloat16[32m [repeated 48x across cluster][0m
[36m(WorkerDict pid=344574)[0m  > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 247038336
[36m(WorkerDict pid=344574)[0m load ref weight start
[36m(WorkerDict pid=344574)[0m load from local dir Qwen/Qwen2.5-0.5B
[36m(WorkerDict pid=344114)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=344114)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",
[36m(WorkerDict pid=344114)[0m   "architectures": [
[36m(WorkerDict pid=344114)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344114)[0m   ],
[36m(WorkerDict pid=344114)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=344114)[0m   "eos_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=344114)[0m   "hidden_size": 896,
[36m(WorkerDict pid=344114)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=344114)[0m   "intermediate_size": 4864,
[36m(WorkerDict pid=344114)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=344114)[0m   "max_window_layers": 24,
[36m(WorkerDict pid=344114)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=344114)[0m   "num_attention_heads": 14,
[36m(WorkerDict pid=344114)[0m   "num_hidden_layers": 24,
[36m(WorkerDict pid=344114)[0m   "num_key_value_heads": 2,
[36m(WorkerDict pid=344114)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=344114)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=344114)[0m   "rope_scaling": null,
[36m(WorkerDict pid=344114)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=344114)[0m   "sliding_window": null,
[36m(WorkerDict pid=344114)[0m   "tie_word_embeddings": true,
[36m(WorkerDict pid=344114)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=344114)[0m   "transformers_version": "4.48.3",
[36m(WorkerDict pid=344114)[0m   "use_cache": true,
[36m(WorkerDict pid=344114)[0m   "use_mrope": false,
[36m(WorkerDict pid=344114)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=344114)[0m   "vocab_size": 151936
[36m(WorkerDict pid=344114)[0m }
[36m(WorkerDict pid=344114)[0m 
[36m(WorkerDict pid=344574)[0m  > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 247038336
[36m(WorkerDict pid=344574)[0m actor_module: 1
[36m(WorkerDict pid=344574)[0m load from local dir Qwen/Qwen2.5-0.5B
[36m(WorkerDict pid=344574)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344574)[0m 
[36m(WorkerDict pid=344114)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=344114)[0m 
[36m(WorkerDict pid=344114)[0m loading embeddings...
[36m(WorkerDict pid=344114)[0m loading layer #0...
[36m(WorkerDict pid=344114)[0m loading layer #1...
[36m(WorkerDict pid=344114)[0m loading layer #2...
[36m(WorkerDict pid=344114)[0m loading layer #3...
[36m(WorkerDict pid=344114)[0m before weight loader: architectures = ['Qwen2ForCausalLM']...[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=344114)[0m call weight loader arch = Qwen2ForCausalLM, model config = Qwen2Config {[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "_attn_implementation_autoset": true,[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "bos_token_id": 151643,[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=344574)[0m NCCL version 2.20.5+cuda12.4
[36m(WorkerDict pid=344114)[0m loading layer #4...
[36m(WorkerDict pid=344114)[0m loading layer #5...
[36m(WorkerDict pid=344114)[0m loading layer #6...
[36m(WorkerDict pid=344114)[0m loading layer #7...
[36m(WorkerDict pid=344114)[0m loading layer #8...
[36m(WorkerDict pid=344114)[0m loading layer #9...
[36m(WorkerDict pid=344114)[0m loading layer #10...
[36m(WorkerDict pid=344114)[0m loading layer #11...
[36m(WorkerDict pid=344114)[0m loading layer #12...
[36m(WorkerDict pid=344114)[0m loading layer #13...
[36m(WorkerDict pid=344114)[0m loading layer #14...
[36m(WorkerDict pid=344114)[0m loading layer #15...
[36m(WorkerDict pid=344114)[0m loading layer #16...
[36m(WorkerDict pid=344114)[0m loading layer #17...
[36m(WorkerDict pid=344114)[0m loading layer #18...
[36m(WorkerDict pid=344114)[0m loading layer #19...
[36m(WorkerDict pid=344114)[0m loading layer #20...
[36m(WorkerDict pid=344114)[0m loading layer #21...
[36m(WorkerDict pid=344114)[0m loading layer #22...
[36m(WorkerDict pid=344114)[0m loading layer #23...
[36m(WorkerDict pid=344114)[0m loading final layernorm...
[36m(WorkerDict pid=344114)[0m tie_word_embeddings skip load lm_head
[36m(WorkerDict pid=344114)[0m loading megatron ckpt done, time elapsed 0.3187425136566162s
[36m(WorkerDict pid=344114)[0m DistributedDataParallel contains 247.04M parameters
[36m(WorkerDict pid=344114)[0m WARNING 04-02 14:37:14 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=344574)[0m TransformerConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=True, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=True, overlap_p2p_comm=False, batch_p2p_comm=False, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=24, num_layers_in_first_pipeline_stage=None, num_layers_in_last_pipeline_stage=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, hidden_size=896, num_attention_heads=14, attention_backend=<AttnBackend.auto: 5>, softmax_scale=None, num_query_groups=2, ffn_hidden_size=4864, kv_channels=64, hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False, apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05, layernorm_zero_centered_gamma=False, add_bias_linear=False, add_qkv_bias=False, gated_linear_unit=True, activation_func=<function silu at 0x7f29f45e0220>, activation_func_fp8_input_store=False, num_moe_experts=None, rotary_interleaved=False, window_size=None, normalization='RMSNorm', qk_layernorm=False, test_mode=False, calculate_per_token_loss=False, multi_latent_attention=False, init_method=<function init_method_normal.<locals>.init_ at 0x7f286018cd60>, output_layer_init_method=<function scaled_init_method_normal.<locals>.init_ at 0x7f286018ce00>, init_method_std=0.02, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, bias_activation_fusion=False, masked_softmax_fusion=True, persist_layer_norm=False, memory_efficient_layer_norm=False, bias_dropout_fusion=False, apply_rope_fusion=False, recompute_granularity=None, recompute_method=None, recompute_num_layers=None, distribute_saved_activations=None, fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, fp8_dot_product_attention=False, fp8_multi_head_attention=False, tp_only_amax_red=False, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_layer_freq=1, moe_ffn_hidden_size=4864, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_router_topk_limited_devices=None, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_pre_softmax=False, moe_router_topk_scaling_factor=None, moe_router_score_function='softmax', moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dropping=False, moe_token_dispatcher_type='alltoall', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_permute_fusion=False, cp_comm_type=None, enable_cuda_graph=False, cuda_graph_use_single_mempool=False, cuda_graph_retain_backward_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, clone_scatter_output_in_embedding=True, disable_parameter_transpose_cache=False, config_logger_dir='', flash_decode=False, inference_rng_tracker=False)[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=344114)[0m create pp model torch allocated 0.6956 GB, reserved 0.9206 GB
[36m(WorkerDict pid=344114)[0m after load model cls
[36m(WorkerDict pid=344114)[0m megatron config ModelParallelConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, sequence_parallel=True, context_parallel_size=1, hierarchical_context_parallel_sizes=None, expert_model_parallel_size=1, expert_tensor_parallel_size=2, moe_extended_tp=False, perform_initialization=True, use_cpu_initialization=False, fp16=False, bf16=True, params_dtype=torch.bfloat16, timers=None, finalize_model_grads_func=None, grad_scale_func=None, no_sync_func=None, grad_sync_func=None, param_sync_func=None, deterministic_mode=False, enable_autocast=False, autocast_dtype=torch.bfloat16, num_microbatches_with_partial_activation_checkpoints=None, gradient_accumulation_fusion=False, async_tensor_model_parallel_allreduce=False, use_te_rng_tracker=False, tp_comm_overlap=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True, tp_comm_atomic_rs=False, cross_entropy_loss_fusion=False, tp_comm_overlap_disable_qkv=False, tp_comm_overlap_disable_fc1=False, tp_comm_bootstrap_backend='nccl', pipeline_dtype=torch.bfloat16, variable_seq_lengths=False, overlap_p2p_comm=False, batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False, deallocate_pipeline_outputs=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, pipeline_model_parallel_split_rank=None, overlap_p2p_comm_warmup_flush=False, microbatch_group_size_per_vp_stage=1, cpu_offloading=False, cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True, cpu_offloading_weights=True, barrier_with_L1_time=True)[32m [repeated 30x across cluster][0m
[36m(WorkerDict pid=344114)[0m pipeline_dtype=megatron_config torch.bfloat16[32m [repeated 30x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "_name_or_path": "Qwen/Qwen2.5-0.5B",[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "architectures": [[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   ],[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "attention_dropout": 0.0,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "eos_token_id": 151643,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "hidden_act": "silu",[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "hidden_size": 896,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "initializer_range": 0.02,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "intermediate_size": 4864,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "max_position_embeddings": 32768,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "max_window_layers": 24,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "model_type": "qwen2",[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "num_attention_heads": 14,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "num_hidden_layers": 24,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "num_key_value_heads": 2,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "rms_norm_eps": 1e-06,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=344114)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=344114)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
[36m(WorkerDict pid=344114)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=344114)[0m   "rope_scaling": null,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "rope_theta": 1000000.0,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "sliding_window": null,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "tie_word_embeddings": true,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "torch_dtype": "bfloat16",[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "transformers_version": "4.48.3",[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "use_cache": true,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "use_mrope": false,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "use_sliding_window": false,[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m   "vocab_size": 151936[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m }[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344114)[0m  > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 247038336
[36m(WorkerDict pid=344114)[0m actor_module: 1
[36m(WorkerDict pid=344114)[0m load from local dir Qwen/Qwen2.5-0.5B
[36m(WorkerDict pid=344114)[0m local rank 0
[36m(WorkerDict pid=344114)[0m before init cache memory allocated: 3.880043008GB, reserved: 4.072669184GB
[36m(WorkerDict pid=344114)[0m after init cache memory allocated: 28.139897344GB, reserved: 28.33252352GB
[36m(WorkerDict pid=344574)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(WorkerDict pid=344574)[0m WARNING 04-02 14:37:14 config.py:380] To see benefits of async output processing, enable CUDA graph. Since, enforce-eager is enabled, async output processor cannot be used
[36m(WorkerDict pid=344574)[0m local rank 0
[36m(TaskRunner pid=334103)[0m Using LocalLogger is deprecated. The constructor API will change 
[36m(TaskRunner pid=334103)[0m Checkpoint tracker file does not exist: %s /opt/tiger/verl/checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm_megatron/latest_checkpointed_iteration.txt
[36m(TaskRunner pid=334103)[0m Training from scratch
[36m(TaskRunner pid=334103)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=334103)[0m validation generation end
[36m(WorkerDict pid=344114)[0m kwargs: {'n': 5, 'logprobs': 0, 'max_tokens': 1024, 'detokenize': False, 'temperature': 1.0, 'top_k': -1, 'top_p': 1, 'ignore_eos': False}
[36m(TaskRunner pid=334103)[0m [prompt] system
[36m(TaskRunner pid=334103)[0m You are a helpful assistant.
[36m(TaskRunner pid=334103)[0m user
[36m(TaskRunner pid=334103)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=334103)[0m assistant
[36m(TaskRunner pid=334103)[0m 
[36m(TaskRunner pid=334103)[0m [response] Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Training Progress:   0%|          | 0/1 [00:00<?, ?it/s]
[36m(WorkerDict pid=344574)[0m /usr/local/lib/python3.11/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=344574)[0m   @torch.library.impl_abstract("xformers_flash::flash_fwd")
[36m(WorkerDict pid=344574)[0m   @torch.library.impl_abstract("xformers_flash::flash_bwd")
[36m(WorkerDict pid=344114)[0m /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:653: UserWarning: When using sequence parallelism it is recommended to set the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for maximum speedup
[36m(WorkerDict pid=344114)[0m   warnings.warn(
[36m(WorkerDict pid=344114)[0m /usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py:768: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
[36m(WorkerDict pid=344114)[0m   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
[36m(TaskRunner pid=334103)[0m ceği
[36m(TaskRunner pid=334103)[0m Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
[36m(TaskRunner pid=334103)[0m [ground_truth] 18
[36m(TaskRunner pid=334103)[0m [score] 0.0
[36m(TaskRunner pid=334103)[0m ("Initial validation metrics: {'val/test_score/openai/gsm8k': "
[36m(TaskRunner pid=334103)[0m  '0.006065200909780136}')
[36m(TaskRunner pid=334103)[0m step:0 - val/test_score/openai/gsm8k:0.006
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1204, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([2408, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([2408, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([2407, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([2090, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([4180, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([4180, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([4180, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([680, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([1360, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([1360, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([1360, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1678, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([3356, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([3356, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([3356, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1745, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([3490, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([3490, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([3489, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1204, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([2408, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([2408, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([2407, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([2090, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([4180, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([4180, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([4180, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([680, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([1360, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([1360, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([1360, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1678, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([3356, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([3356, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([3356, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1745, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([3490, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([3490, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([3489, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1198, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([2396, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([2396, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([2396, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = output.logits, shape: torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:, -response_length - 1:-1].contiguous(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m logits_back = logits.clone(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m log_prob = vocab_parallel_log_probs_from_logits(logits, responses), shape: torch.Size([4, 1024]), responses shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m entropy_loss, entropy = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask), entropy_loss shape: torch.Size([]), entropy shape: torch.Size([4, 1024]), response_mask shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1771, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([3542, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([3542, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([3541, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = output.logits, shape: torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:, -response_length - 1:-1].contiguous(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m logits_back = logits.clone(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m log_prob = vocab_parallel_log_probs_from_logits(logits, responses), shape: torch.Size([4, 1024]), responses shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m entropy_loss, entropy = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask), entropy_loss shape: torch.Size([]), entropy shape: torch.Size([4, 1024]), response_mask shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([1247, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([2494, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([2494, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([2493, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = output.logits, shape: torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:, -response_length - 1:-1].contiguous(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m logits_back = logits.clone(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m log_prob = vocab_parallel_log_probs_from_logits(logits, responses), shape: torch.Size([4, 1024]), responses shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m entropy_loss, entropy = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask), entropy_loss shape: torch.Size([]), entropy shape: torch.Size([4, 1024]), response_mask shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([2218, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([4436, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([4436, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([4436, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = output.logits, shape: torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:, -response_length - 1:-1].contiguous(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m logits_back = logits.clone(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m log_prob = vocab_parallel_log_probs_from_logits(logits, responses), shape: torch.Size([4, 1024]), responses shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m entropy_loss, entropy = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask), entropy_loss shape: torch.Size([]), entropy shape: torch.Size([4, 1024]), response_mask shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m hidden_states = outputs, shape: torch.Size([963, 1, 896])
[36m(WorkerDict pid=344114)[0m logits = self._forward_head(hidden_states): torch.Size([1926, 1, 75968])
[36m(WorkerDict pid=344114)[0m logits = torch.squeeze(logits, dim=1): torch.Size([1926, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:totol_nnz]: torch.Size([1926, 75968])
[36m(WorkerDict pid=344114)[0m logits = pad_input(logits, indices, batch_size, seqlen=sequence_length): torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = output.logits, shape: torch.Size([4, 1536, 75968])
[36m(WorkerDict pid=344114)[0m logits = logits[:, -response_length - 1:-1].contiguous(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m logits_back = logits.clone(), shape: torch.Size([4, 1024, 75968])
[36m(WorkerDict pid=344114)[0m log_prob = vocab_parallel_log_probs_from_logits(logits, responses), shape: torch.Size([4, 1024]), responses shape: torch.Size([4, 1024])
[36m(WorkerDict pid=344114)[0m entropy_loss, entropy = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask), entropy_loss shape: torch.Size([]), entropy shape: torch.Size([4, 1024]), response_mask shape: torch.Size([4, 1024])
[36m(TaskRunner pid=334103)[0m test_gen_batch meta info: {'eos_token_id': 151643, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
[36m(TaskRunner pid=334103)[0m validation generation end
[36m(TaskRunner pid=334103)[0m [prompt] system
[36m(TaskRunner pid=334103)[0m You are a helpful assistant.
[36m(TaskRunner pid=334103)[0m user
[36m(TaskRunner pid=334103)[0m Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
[36m(TaskRunner pid=334103)[0m assistant
[36m(TaskRunner pid=334103)[0m 
[36m(TaskRunner pid=334103)[0m [response] Janet makes $2 per fresh egg, so she makes $2 x 16 = $32 every day. She eats 3 eggs every morning, so she makes $3 x 16 = $48 every day. She also bakes muffins for her friends, so she makes $48 x 7 = $336 every day. She sells the remaining eggs at the farmers' market, so she makes $336 - $32 = $304 every day.####
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m  cougar
[36m(TaskRunner pid=334103)[0m Training Progress:   0%|          | 0/1 [02:02<?, ?it/s]
[36m(WorkerDict pid=344574)[0m /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:653: UserWarning: When using sequence parallelism it is recommended to set the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for maximum speedup
[36m(WorkerDict pid=344574)[0m   warnings.warn(
[36m(WorkerDict pid=344574)[0m /usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py:768: UserWarning: c10d::broadcast_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at ../torch/csrc/autograd/autograd_not_implemented_fallback.cpp:63.)
[36m(WorkerDict pid=344574)[0m   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[36m(TaskRunner pid=334103)[0m [ground_truth] 18
[36m(TaskRunner pid=334103)[0m [score] 0.0
[33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffffb1ccdc83520b693454dd678201000000 Worker ID: 72a2fb8e1355633dc9fdc0511502095e64303d54e13a7bb80d9206a3 Node ID: 95d27c249c1e6a341cdc8b8568043f38d017fdfc06c0317a27c2282d Worker IP address: 127.0.0.1 Worker port: 37747 Worker PID: 334103 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.
Error executing job with overrides: ['algorithm.adv_estimator=grpo', 'data.train_files=/home/tiger/data/gsm8k/train.parquet', 'data.val_files=/home/tiger/data/gsm8k/test.parquet', 'data.train_batch_size=4', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.actor.ppo_mini_batch_size=4', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4', 'actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=vllm', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=5', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4', 'algorithm.kl_ctrl.kl_coef=0.001', 'trainer.critic_warmup=0', 'trainer.logger=[console]', 'trainer.project_name=verl_grpo_example_gsm8k', 'trainer.experiment_name=qwen2_7b_function_rm_megatron', 'trainer.n_gpus_per_node=2', 'trainer.nnodes=1', 'trainer.save_freq=-1', 'trainer.test_freq=5', 'trainer.total_epochs=1', 'trainer.total_training_steps=1']
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
	actor_id: b1ccdc83520b693454dd678201000000
	pid: 334103
	namespace: 03ac15ba-9ad7-4900-b742-89820b196826
	ip: 127.0.0.1
The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly. Worker exits with an exit code None. The worker may have exceeded K8s pod memory limits.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
