import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NVTE_DEBUG'] = '1'
os.environ['NVTE_DEBUG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import ModelType, GPTModel
import torch
import torch.distributed
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig, MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams

import random
import numpy as np
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core import tensor_parallel
import torch.distributed as dist
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.use_deterministic_algorithms(True)

def test():

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=1,
                                virtual_pipeline_model_parallel_size=None,
                                context_parallel_size=1,
                                expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(seed)

    tfconfig = TransformerConfig(
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation_func=F.silu,
        normalization="RMSNorm",
        gated_linear_unit=True,
        use_cpu_initialization=False,
        add_bias_linear=False,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        attention_backend=AttnBackend.fused,
        bf16=True,
        layernorm_epsilon=1e-5,
        ffn_hidden_size=5632,
        qk_layernorm=True,
        moe_token_dispatcher_type="alltoall",
    )
    transformer_layer_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=True)
    model = GPTModel(config=tfconfig,
                              transformer_layer_spec=transformer_layer_spec,
                              vocab_size=32000,
                              max_sequence_length=2048,
                              pre_process=True,
                              post_process=False,
                              share_embeddings_and_output_weights=False,
                              position_embedding_type='rope',
                              rotary_base=10000.0)
    model.cuda()
    device = 'cuda'
    input_ids = torch.randint(0, 32000, (1, 10)).to(device)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    output = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,)
    output.sum().backward()
    print("------------------okokokok---------------------")
    return
    


if __name__ == "__main__":
    test()

    