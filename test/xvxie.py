import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM

# ======= 模型路径（本地） =======
model_path = "/home/yangzf/Llama-2-7b-chat-hf"

# ======= 加载 tokenizer 和模型（注意用 ForCausalLM 才能生成）=======
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# ======= 输入文本 =======
text = """
This article explains and discusses the immense complexity of the psychoanalytic process as it is becoming increasingly understood at the millennium, and offers the possibility that it can be viewed from at least five channels of psychoanalytic listening. The careful ongoing examination of the transference-countertransference interactions or enactments, and their 'analytic third' location in the transitional space is extremely important in psychoanalytic practice. We must be careful in our interpretations of the clinical data not to stray any farther from the fundamental concepts of Freud than is necessary, lest we end up with a set of conflicting speculative metaphysical systems and become a marginalized esoteric cult. Freud's work remains our basic paradigm, the core of psychoanalysis, even though his papers on technique and his emphasis on the curative power of interpretation are from a one-person psychology standpoint and his view of psychoanalysis as just another empirical 19th-century science requires proper understanding and emendation in the light of accumulated clinical experience since his time. The above quotation is from a letter written by St. Augustine about 395 A.D. in his effort to reach some kind of reconciliation with the Donatists. Both the Donatists and the Catholics shared a great many basic principles at the time but the Donatists, named after Donatus, a martyred hero of the resistance to any compromise, were known as stubbornly refusing to negotiate or yield even on the smallest details of their theological doctrines. As a result, Augustine's efforts failed and he ended up resorting to traditional Christian solutions—violence, restriction of civil liberties, and deportation—in an effort to stamp out this heresy (which nevertheless lasted until the 7th century A.D.). So it is that plus ça change, plus c’est la même chose, the more things change, the more they remain the same: When I was a resident in psychiatry we had a series of seminars by one of my most influential and revered teachers, Franz Alexander, who could be counted on to discover in each and every case presentation, regardless of the material or diagnosis, that the nucleus of the disorder was an Oedipus complex. With all this high-powered psychoanalytic training, information, and experience gathered in my residency, after spending two years doing military service in the United States Public Health Service working with drug addicts in a federal prison, I entered into the full time private practice of psychoanalytic psychotherapy in 1960. At this point, I encountered a fascinating schizophrenic patient who was not so schizophrenic that I could not stand her, and who, after several years of intensive psychotherapy, made a very noticeable recovery and adaptation that enabled her to live a reasonably decent life in our lunatic culture. Bursting with pride near the end of the treatment as I had watched this woman evolve from a dilapidated human wreck into a very presentable and now married young lady, I could not refrain from asking her, near the end of the treatment, which of my interpretations had had the most significant impact on her improvement and development. Her response was, 'You have kind eyes.' This took me down a considerable distance and set me thinking about what it is that actually brings about a cure in psychoanalytic treatment and about what sort of theoretical orientation is most suitable for what I believe to be first and foremost a clinical medical discipline. I gradually began to think of my patients and their narratives as sort of Rorschach cards on which a variety of theories may be imposed. Kohut tried to distinguish between experience-distant and experience-near theories, but this proved to be simply more narcissism on the part of psychoanalysts, because now we know that all theories are experience-distant, and it is in the area of theories that there is the most argument. So the philosopher Epictetus, in his classic treatise Encheiridion ('Manual,' 100 A.D.), gave us an aphorism that Laurence Sterne thought was so worthwhile that he used it as a motto in his famous novel Tristram Shandy, published around 1760: Tarassei tous Anthropous ou ta Pragmata, Alla ta peri ton Pragmaton Dogmata, which I roughly translate as meaning 'people are not disturbed by things, they are disturbed by theories about things.' Over a period of 25 years, from about the 1970s to the present, I gradually evolved what I have called the five-channel theory of psychoanalytic listening. I have published a book and a number of papers on this topic and so I will only very briefly review here the five standpoints or channels or models or perspectives or frameworks from which I suggest we tune in to the transmission from the patient. The first was presented by Freud. It involves the Oedipus complex and the emergence in a properly conducted psychoanalysis of the pressure for drive gratification in the transference. This enables us to study the patient's conflict in terms of defenses against the instinctual drives and the resulting compromise formations produced by the ego in dealing with its three harsh masters—the superego, the id, and external reality. Freud's structural theory, placing the Oedipus complex at the focus of all psychoneuroses, was developed in order to best depict this one-person standpoint; the analyst is thought to be simply the observer of it all. The second channel utilizes the perspective of object relations theory for its model. It is based on the work of Klein and her analysand Bion, and focuses on the earliest projective and introjective fantasies of the patient as they appear in the object relatedness manifest in the transference, and on the process of projective identification, which is defined differently by every author. Understanding of these processes through a conceptualization of the patient's earliest internalized object relations yields data about how the patient as an infant organized these relations into self and object representations and then projected and reintrojected various aspects of these images. This helps to clarify the patient's relationships in the present, because all such current relationships are perceived and reacted to through the spectacles of these early organized self and object representations
"""
# ===== 编码输入文本 =====
inputs = tokenizer(text, return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[1]

# ===== 生成新 token 并输出注意力 =====
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        return_dict_in_generate=True,
        output_attentions=True
    )

# ===== 提取生成的 token =====
generated_ids = outputs.sequences[0][prompt_len:]
generated_token = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\n🧾 原始 Prompt 长度: {prompt_len} 个 token")
print(f"🆕 生成的 token ID: {generated_ids.tolist()}")
print(f"🔤 生成的 token 文本: {generated_token}")

# ===== 提取注意力并搬到 CPU =====
attentions_raw = outputs.attentions[0]  # tuple of (num_layers,) tensors
attentions = [layer_attn.to("cpu") for layer_attn in attentions_raw]

# ===== 设置分析参数 =====
layer = 1
head = 5
topk = 200
prefix_len = 1400
start = 1401
end = 1402

# ===== 提取该层该头的注意力矩阵 =====
# attention shape: (1, num_heads, total_len, total_len)
attention = attentions[layer][0, head].detach().numpy()  # shape: (total_len, total_len)

# ===== 截取目标注意力子矩阵 =====
attn_sub = attention[start:end, :prefix_len]  # shape: (500, 4000)

# ===== 归一化：对每一行做 sum=1 归一化（或 softmax）=====
attn_sub_normalized = attn_sub / attn_sub.sum(axis=1, keepdims=True)

# ===== 聚合得分：每列累加 =====
scores = attn_sub_normalized.sum(axis=0)  # shape: (4000,)

# ===== topk 得分最高的 token 编号 =====
topk_indices = np.argsort(scores)[-topk:][::-1]  # 从高到低排列

# ===== 编号矩阵输出 =====
topk_matrix = np.array(topk_indices).reshape(1, -1)

print(f"\n🎯 Top-{topk} 前缀 token 编号（注意力得分最高）:")
print(topk_matrix)