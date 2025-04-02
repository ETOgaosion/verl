import torch
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

@dataclass
class FusedCausalLMOutputWithPast(CausalLMOutputWithPast):
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None

@dataclass
class FusedQwen2VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    log_probs: torch.Tensor = None
    entropy: torch.Tensor = None