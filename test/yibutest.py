import faiss
import numpy as np
import time

# -------------------------------
# 参数设置
d = 128       # 向量维度
nb = 1000  # 库中向量数量
nq = 1        # 查询向量数量
topk = 5      # top-k
runs = 1000   # 任务重复次数

# -------------------------------
# 计时累加器
total_faiss_index_time = 0.0
total_faiss_search_time = 0.0
total_numpy_time = 0.0

for run in range(runs):
    np.random.seed(1234 + run)  # 每次不同种子

    # 构造数据库向量 + 归一化（单位向量）
    xb = np.random.random((nb, d)).astype('float32')
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)

    # 构造查询向量 + 归一化（单位向量）
    xq = np.random.random((nq, d)).astype('float32')
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)

    # ✅ Faiss 构建索引
    start_time = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    total_faiss_index_time += time.time() - start_time

    # ✅ Faiss 搜索
    start_time = time.time()
    D_faiss, I_faiss = index.search(xq, topk)
    total_faiss_search_time += time.time() - start_time

    # # ✅ NumPy 暴力搜索
    # start_time = time.time()
    # scores = np.dot(xq, xb.T)
    # I_numpy = np.argsort(-scores, axis=1)[:, :topk]
    # D_numpy = np.take_along_axis(scores, I_numpy, axis=1)
    # total_numpy_time += time.time() - start_time

# -------------------------------
# ✅ 输出累计和平均运行时间
print("\n========== 总运行时间统计 ==========")
print(f"Faiss Index Total Time : {total_faiss_index_time:.4f} seconds")
print(f"Faiss Search Total Time: {total_faiss_search_time:.4f} seconds")
print(f"NumPy Brute-force Time : {total_numpy_time:.4f} seconds")

print("\n========== 平均运行时间 (每次) ==========")
print(f"Avg Faiss Index Time   : {total_faiss_index_time / runs:.4f} seconds")
print(f"Avg Faiss Search Time  : {total_faiss_search_time / runs:.4f} seconds")
print(f"Avg NumPy Brute-force  : {total_numpy_time / runs:.4f} seconds")