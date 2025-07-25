import faiss
import numpy as np
import time

# -------------------------------
# 参数设置
d = 128        # 向量维度
nb = 7000000  # 库中向量数量
nq = 5         # 查询向量数量
topk = 5       # top-k

np.random.seed(1234)

# -------------------------------
# 构造数据库向量 + 归一化（单位向量）
xb = np.random.random((nb, d)).astype('float32')
xb /= np.linalg.norm(xb, axis=1, keepdims=True)

# 构造查询向量 + 归一化（单位向量）
xq = np.random.random((nq, d)).astype('float32')
xq /= np.linalg.norm(xq, axis=1, keepdims=True)

# -------------------------------
# ✅ Faiss 构建索引并搜索
start_time = time.time()
index = faiss.IndexFlatIP(d)
index.add(xb)
faiss_time0 = time.time() - start_time
start_time = time.time()
D_faiss, I_faiss = index.search(xq, topk)
faiss_time = time.time() - start_time

# -------------------------------
# ✅ NumPy 暴力搜索
start_time = time.time()
scores = np.dot(xq, xb.T)
I_numpy = np.argsort(-scores, axis=1)[:, :topk]
D_numpy = np.take_along_axis(scores, I_numpy, axis=1)
numpy_time = time.time() - start_time

# -------------------------------
# ✅ 输出结果对比和时间
for i in range(nq):
    # print(f"\nQuery {i+1}")
    # print("FAISS Top-k indices:", I_faiss[i])
    # print("NumPy Top-k indices:", I_numpy[i])
    # print("FAISS Top-k scores :", np.round(D_faiss[i], 5))
    # print("NumPy Top-k scores :", np.round(D_numpy[i], 5))
    break
print("\n========== 运行时间统计 ==========")
print(f"Faiss Index Time : {faiss_time0:.4f} seconds")
print(f"Faiss Search Time : {faiss_time:.4f} seconds")
print(f"NumPy Brute-force : {numpy_time:.4f} seconds")