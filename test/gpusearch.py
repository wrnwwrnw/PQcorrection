import torch
import numpy as np
import time

# -------------------------------
# 参数设置
d = 128        # 向量维度
nb = 3000000  # 库中向量数量
nq = 5         # 查询向量数量
topk = 5       # top-k

np.random.seed(1234)

# -------------------------------
# 构造数据库向量 + 归一化
xb = np.random.random((nb, d)).astype('float32')
xb /= np.linalg.norm(xb, axis=1, keepdims=True)

xq = np.random.random((nq, d)).astype('float32')
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

# -------------------------------
# ✅ PyTorch GPU 版本 Top-k 内积搜索
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转为 torch tensor 并搬到 GPU
xb_tensor = torch.from_numpy(xb).to(device)
xq_tensor = torch.from_numpy(xq).to(device)

start_time = time.time()

# 内积矩阵 (nq, nb)
scores = torch.matmul(xq_tensor, xb_tensor.T)  # (5, 3_000_000)

# Top-k 搜索（dim=1 表示每一行找 topk）
D_gpu, I_gpu = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)

gpu_time = time.time() - start_time

# 搬回 CPU（可选）
D_gpu = D_gpu.cpu().numpy()
I_gpu = I_gpu.cpu().numpy()

# -------------------------------
# ✅ 输出部分结果
for i in range(nq):
    # print(f"\nQuery {i+1}")
    # print("Torch-GPU Top-k indices:", I_gpu[i])
    # print("Torch-GPU Top-k scores :", np.round(D_gpu[i], 5))
    break
print("\n========== 运行时间统计 ==========")
print(f"PyTorch GPU Search Time : {gpu_time:.4f} seconds")
