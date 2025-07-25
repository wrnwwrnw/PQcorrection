
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from transformers.cache_utils import Cache, DynamicCache

# perform qk calculation and get indices
# this version will not update in inference mode
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 800 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 800 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # if q_len < self.max_capacity_prompt:
        return key_states, value_states
        # else:
        #     attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        #     mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        #     mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        #     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        #     mask = mask.to(attn_weights.device)
        #     attention_mask = mask[None, None, :, :]

        #     attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #     attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
        #     if self.pooling == 'avgpool':
        #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     elif self.pooling == 'maxpool':
        #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     else:
        #         raise ValueError('Pooling method not supported')
        #     indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
        #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        #     k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        #     v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
        #     k_cur = key_states[:, :, -self.window_size:, :]
        #     v_cur = value_states[:, :, -self.window_size:, :]
        #     key_states = torch.cat([k_past_compress, k_cur], dim = 2)
        #     value_states = torch.cat([v_past_compress, v_cur], dim = 2)
        #     return key_states, value_states
    def update_kv_decode(self, key_states, value_states, sheet, layer_idx, cache, count, kv_seq_len, cache_kwargs, ifindex, cacheidx):
        #count是开始位置，index是相对位置
        head_dim = key_states.shape[3]
        top_k = int((sheet.shape[3] / 100) * 60)
        #实例化单层的注意力张量
        layer_attn = sheet[layer_idx]
        #头间平衡得分
        avg_attn_per_head = layer_attn.mean(dim=0)
        avg_attn_per_head = F.avg_pool1d(avg_attn_per_head, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #拿目录
        topk_indices = torch.topk(avg_attn_per_head, k=top_k, dim=1).indices
        topk_indices_sorted, _ = topk_indices.sort(dim=1, descending=False)
        num_heads, num_tokens = avg_attn_per_head.shape  
        all_indices = torch.arange(num_tokens, device=topk_indices.device).unsqueeze(0).expand(num_heads, -1)
        topk_exp = topk_indices.unsqueeze(-1) 
        all_exp = all_indices.unsqueeze(1)
        topk_mask = (all_exp == topk_exp).any(dim=1)
        non_topk_mask = ~topk_mask
        non_topk_indices = all_indices[non_topk_mask].view(num_heads, num_tokens - top_k)
        #topk构造
        topk_indices_sorted = topk_indices_sorted + count -1
        topk_indices_expanded = topk_indices_sorted.unsqueeze(0).unsqueeze(-1)
        topk_indices_expanded = topk_indices_expanded.expand(-1, -1, -1, head_dim)
        #反topk构造,存入容器
        non_topk_indices = non_topk_indices + count - 1
        non_topk_indices_expanded = non_topk_indices.unsqueeze(0).unsqueeze(-1)
        non_topk_indices_expanded = non_topk_indices_expanded.expand(-1, -1, -1, head_dim).to(key_states.device)
        k_drop = key_states[:, :, :, :].gather(dim = 2, index = non_topk_indices_expanded).cpu()
        v_drop = value_states[:, :, :, :].gather(dim = 2, index = non_topk_indices_expanded).cpu()
        if non_topk_indices_expanded.shape[2]<=144 :
            cache.update(k_drop, v_drop, layer_idx, cache_kwargs)
            # if layer_idx == 31:
            #     ifindex[0] = 0
            #完成首次填充，不可构建
        else :
            cache.key_cache[layer_idx] = torch.cat([cache.key_cache[layer_idx], k_drop], dim=2).to(torch.float32)
            cache.value_cache[layer_idx] = torch.cat([cache.value_cache[layer_idx], v_drop], dim=2)
            if layer_idx == 31:
                ifindex[0] = 0
                #完成普通填充，可以构建
        #gather
        left_k = key_states[:, :, :count, :]
        left_v = value_states[:, :, :count, :]
        topk_indices_expanded = topk_indices_expanded.to(key_states.device)
        k_past_compress = key_states[:, :, :, :].gather(dim = 2, index = topk_indices_expanded)
        v_past_compress = value_states[:, :, :, :].gather(dim = 2, index = topk_indices_expanded)
        right_k = key_states[:, :, -40:, :]
        right_v = value_states[:, :, -40:, :]
        if len(cacheidx[layer_idx]) > 0:
            search_indices_expaned = torch.tensor(cacheidx[layer_idx], device=key_states.device).unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, head_dim)
            search_key = cache.key_cache[layer_idx].to(key_states.device).gather(dim = 2, index = search_indices_expaned).to(key_states.dtype)
            search_value = cache.value_cache[layer_idx].to(key_states.device).gather(dim = 2, index = search_indices_expaned)
            # mask = torch.ones(1, 32, seq_len, dtype=torch.bool, device=cache_key.device)
            # mask.scatter_(2, search_indices, False)
            k_past_compress = torch.cat([search_key,k_past_compress], dim=2)
            v_past_compress = torch.cat([search_value,v_past_compress], dim=2) 
        key_states = torch.cat([left_k, k_past_compress, right_k], dim=2)
        value_states = torch.cat([left_v, v_past_compress, right_v], dim=2)
        return key_states, value_states
        #test
        # if layer_idx == 0:
        #    print(topk_indices_sorted.shape)
        #    print(topk_indices_sorted[0][:])
        #    print(non_topk_indices.shape)
        #    print(non_topk_indices[0][:])
        #    print(key_states.device)
        #    print(non_topk_indices_expanded.shape)
        #    print(topk_indices_expanded.shape)
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 256
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )