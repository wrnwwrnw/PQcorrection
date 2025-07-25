import torch

x = torch.randn(4, 5, 6)  # 举例输入 (L=3, H=3, P=3)
avg_prob_per_head = x.mean(dim=0)  # 平均层维度 → 结果 shape: (3, 3)

print(avg_prob_per_head.shape)  # 输出: torch.Size([3, 3])
print(avg_prob_per_head)
