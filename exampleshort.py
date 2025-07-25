import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from snapkv.monkeypatch.monkeypatch import replace_llama

# 应用 SnapKV 的 Llama patch
replace_llama()

# 设置使用 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型路径
model_path = "/home/yangzf/Llama-2-7b-chat-hf"

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建生成管道（不指定 device）
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 单个问题
question = """
give me a list form 1890 to 2020 who win the nobel prices.
"""

# 计时并生成答案
start_time = time.time()
response = qa_pipeline(question, max_length=2560, num_return_sequences=1, do_sample=False)
end_time = time.time()

# 提取信息
generated_text = response[0]["generated_text"]
num_tokens = len(tokenizer.encode(generated_text))
generation_time = end_time - start_time

# 输出结果
print("--------------------------------------------------------------------------------")
print(f"Question: {question}")
print(f"Answer: {generated_text}")
print(f"Tokens generated: {num_tokens}")
print(f"Time taken: {generation_time:.2f} seconds")
print("--------------------------------------------------------------------------------")

