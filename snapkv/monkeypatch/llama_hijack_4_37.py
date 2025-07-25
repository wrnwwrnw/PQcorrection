import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)
from snapkv.monkeypatch.snapkv_utils import init_snapkv
import math
from snapkv.monkeypatch.index import build, search
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logger = logging.get_logger(__name__)

# https://github.com/huggingface/transformers/blob/v4.37-release/src/transformers/models/llama/modeling_llama.py
def llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_snapkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False
    # if not hasattr(llama_flash_attn2_forward, "count"):
    #                 llama_flash_attn2_forward.count = 0
    # cpu容器、索引容器[层][头]、构建索引状态码,-1表示静止，0表示可以构建，1表示正在构造，2表示构造完成
    if not hasattr(llama_flash_attn2_forward, "cache"):
       llama_flash_attn2_forward.cache = DynamicCache()
    if not hasattr(llama_flash_attn2_forward, "cacheidx"):
       llama_flash_attn2_forward.cacheidx = [[] for _ in range(32)]
    if not hasattr(llama_flash_attn2_forward, "index_grid"):
       manager = multiprocessing.Manager()
       llama_flash_attn2_forward.index_grid = manager.list(
           [manager.list([None for _ in range(32)]) for _ in range(32)]
       )
    if not hasattr(llama_flash_attn2_forward, "ifindex"):
       llama_flash_attn2_forward.ifindex = manager.list([-1])
    # 条件判断未完成，满足一定数值再开始
    if llama_flash_attn2_forward.ifindex[0]==0 :
       executor = ProcessPoolExecutor()
       llama_flash_attn2_forward.ifindex[0] = 1 
    #    print("调用开始")
    #    print(llama_flash_attn2_forward.cache.key_cache[0].dtype)
    #    pool = Pool(10)
    #    result = pool.apply_async(build, (llama_flash_attn2_forward.cache, llama_flash_attn2_forward.index_grid, llama_flash_attn2_forward.ifindex))
    #    pool.close()
       future = executor.submit(build ,llama_flash_attn2_forward.cache ,llama_flash_attn2_forward.index_grid ,llama_flash_attn2_forward.ifindex)
       executor.shutdown(wait=False)
    #    build(llama_flash_attn2_forward.cache ,llama_flash_attn2_forward.index_grid ,llama_flash_attn2_forward.ifindex)
    #    print("异步开始")
    #    print(llama_flash_attn2_forward.ifindex,"内部循环")
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # 无关计数
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # if kv_seq_len==2558:
    #     print(past_key_value.value_cache[self.layer_idx].shape)
    #  打印一下最终 算压缩率
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # 看当前长度
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
            # 写入
            #     with open(num_file_path, "r") as f:
            #         num = int(f.read().strip())
            #     new_num = kv_seq_len
            #     with open(num_file_path, "w") as f:
            #         f.write(str(new_num))
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            if self.layer_idx ==0:
                llama_flash_attn2_forward.count1 = kv_seq_len
                llama_flash_attn2_forward.count2 = 0
                llama_flash_attn2_forward.conut3 = key_states_compress.shape[2]
        # 窗口末期分支
        # elif llama_flash_attn2_forward.count2 > llama_flash_attn2_forward.count1 and \
        #         (llama_flash_attn2_forward.count2 - llama_flash_attn2_forward.count1) % 10 == 0:
        #     sin = cos 
        #     if self.layer_idx ==31:     
        #           llama_flash_attn2_forward.count2 += 1
        else:
            if self.layer_idx ==0:     
                llama_flash_attn2_forward.count2 += 1
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # 核心代码
            if (llama_flash_attn2_forward.count2%200==0)and(llama_flash_attn2_forward.count2%400!=0)and(llama_flash_attn2_forward.ifindex[0]==2):
               search(query_states ,self.layer_idx ,0 ,int(self.kv_seq_len / 100) ,llama_flash_attn2_forward.cacheidx ,llama_flash_attn2_forward.index_grid)
            # 查询
            if llama_flash_attn2_forward.count2>360:
               # 上个轮回已经定义，删掉重新定义，第一个轮回直接定义
               if hasattr(llama_flash_attn2_forward, "sheet") and llama_flash_attn2_forward.count2 == 361 and self.layer_idx ==0:
                  del llama_flash_attn2_forward.sheet
               if not hasattr(llama_flash_attn2_forward, "sheet"):
                  llama_flash_attn2_forward.sheet = torch.zeros(32, 40, 32, key_states.shape[2]-llama_flash_attn2_forward.conut3-1)
               head_dim = 128
               attn_weight = torch.matmul(query_states, key_states[:, :, llama_flash_attn2_forward.conut3:llama_flash_attn2_forward.conut3+llama_flash_attn2_forward.sheet.shape[3], :].transpose(2, 3)) / math.sqrt(head_dim)
               attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
            #  按token、层插入注意力得分
               llama_flash_attn2_forward.sheet[self.layer_idx, llama_flash_attn2_forward.count2-361, :, :] = attn_weight.squeeze(0).squeeze(1)
            # 这段有点问题，如果满了400个进剪枝逻辑
               if llama_flash_attn2_forward.count2==400:
            #     打印测试
            #     key_states_compress, value_states_compress = self.kv_cluster.update_kv_decode(key_states, value_states, llama_flash_attn2_forward.sheet, self.layer_idx)
                  key_states_compress, value_states_compress = self.kv_cluster.update_kv_decode(key_states, value_states, llama_flash_attn2_forward.sheet, self.layer_idx, llama_flash_attn2_forward.cache, llama_flash_attn2_forward.conut3, self.kv_seq_len, cache_kwargs,llama_flash_attn2_forward.ifindex, llama_flash_attn2_forward.cacheidx)
                  past_key_value.key_cache[self.layer_idx] = key_states_compress
                  past_key_value.value_cache[self.layer_idx] = value_states_compress
                  #这次修剪结束，重置计数单位
                  if self.layer_idx == 31:
                    llama_flash_attn2_forward.count2 = 0
            # 连接到cache
            # 观察
            # if llama_flash_attn2_forward.count2==500 and self.layer_idx ==0:
            #     print(llama_flash_attn2_forward.sheet.shape[3]-1)
            #     print(key_states.shape[2])
            #     print(query_states.shape)
            # if llama_flash_attn2_forward.count2==501 and self.layer_idx ==0:
            #     print(llama_flash_attn2_forward.sheet.shape[3]-1)
            #     print(key_states.shape[2])
            #     print(query_states.shape)
            # 观察长度,注意力手动计算测试
            # if(llama_flash_attn2_forward.count2>1000):
            #     current_dir = os.path.dirname(os.path.abspath(__file__))
            #     num_file_path = os.path.join(current_dir, "num.txt")
            # head_dim = 128
            # attn_weight = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            # attn_weight = F.softmax(attn_weight, dim=-1)
            #     with open(num_file_path, "w") as f:
            #         f.write(str(attn_weight))
            # print(attn_weight.shape)
    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )
    # if (self.kv_seq_len==8000):
    #     print(key_states.shape)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None

    # 观察长度
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # num_file_path = os.path.join(current_dir, "num.txt")
    # if (1<llama_flash_attn2_forward.count2-llama_flash_attn2_forward.count1<42) and (self.layer_idx == 1):
    #     with open(num_file_path, "a") as f:
    #       top_indices = torch.topk(attn_output.view(-1), k=50).indices
    #       f.write(" ".join([str(i.item()) for i in top_indices]) + "\n")
    #       print(attn_output[0,0,4095])
    #       print(attn_output.shape)


    return attn_output, attn_weights, past_key_value

def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None: # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
