import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地模型路径
model_path = "/home/yangzf/Llama-2-7b-chat-hf"

# 你可以修改这里的参数，选择想看的层和头
layer_index = 1  # 取最后一层，0表示第一层，-1表示最后一层
head_index = 0    # 取第几个头，从0开始计数

# 加载 tokenizer 和模型（注意开启 output_attentions）
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    output_attentions=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

prompt = """
The neon hum of the cryo-chamber faded as Dr. Elara Voss pried open the frost-rimmed hatch. Her breath crystallized in the air—a bad sign. Protocol demanded the lab maintain -18°C, but the readout showed -23.4°C and dropping. The anomaly detector blipped erratically, casting jagged red shadows across walls lined with containment units holding specimens from Kepler-452b.
She almost missed the faint pattern in the chaos. Buried beneath the thermal alerts was a rhythmic pulse in the quantum entanglement logs—exactly 11.7-second intervals matching no known celestial phenomenon. Her glove hovered over the emergency purge button when the specimen wall lit up. Bioluminescent tendrils inside Pod #7 writhed like living circuitry, their glow synchronizing with the pulses. The Kepler moss they'd deemed inert three years ago was communicating. Or calculating.
The lab AI's voice crackled unnaturally: "Thermal core breach in Sector—" before dissolving into static. Elara's retinal display flickered with corrupted data streams. Through the distortion, she glimpsed security footage from ten minutes prior—her own silhouette still working at the terminal, though she'd been in the cafeteria then. The moss tendrils now formed perfect Fibonacci spirals.
Her hand froze mid-reach for the manual override. The cryo-chamber's frost patterns had changed. What she'd mistaken for random crystallization now clearly depicted the Cassiopeia constellation—mirroring the scar on her collarbone from childhood radiation treatment. A treatment administered after
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

generated = input_ids
max_new_tokens = 300

topK = 10  # 你想取多少个最高分token

for step in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(generated, output_attentions=True)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    generated = torch.cat([generated, next_token], dim=1)

    attentions = outputs.attentions[layer_index]  # 选择层
    # attentions shape: [batch, num_heads, tgt_len, src_len]
    last_token_attn = attentions[0, head_index, -1, :].cpu()  # shape: [src_len]

    decoded_token = tokenizer.decode(next_token[0])

    # topk筛选逻辑（只取<=72的token索引）
    attn_scores = last_token_attn.numpy()
    # argsort降序排列所有token索引
    sorted_indices = attn_scores.argsort()[::-1]

    selected_tokens = []
    for idx in sorted_indices:
        if idx <= 72:
            selected_tokens.append(idx)
        if len(selected_tokens) >= topK:
            break

    print(f"Step {step+1} generated token: {decoded_token}")
    print(f"Top {topK} attention token indices (<=72): {selected_tokens}\n")

    # 如果生成了结束符，停止循环
    if next_token.item() == tokenizer.eos_token_id:
        print("遇到结束符，停止生成。")
        break