import numpy as np
import faiss
import time

# 向量维度
d = 128
nb = 100000
np.random.seed(123)
xb = np.random.random((nb, d)).astype('float32')

# 如果使用内积，为了正确度量，建议先将所有向量归一化
faiss.normalize_L2(xb)  # 归一化后，内积≈余弦相似度
start_time = time.time()
# 构建 HNSW 索引（使用内积作为距离度量）
M = 32
index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)

# 设置构建和搜索参数
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64

# 添加向量
index.add(xb)
faiss_time0 = time.time() - start_time
print(faiss_time0)
# 构造查询向量
xq = xb[:5]
faiss.normalize_L2(xq)  # 查询向量也需要归一化
start_time = time.time()
# 执行查询
k = 4
D, I = index.search(xq, k)
faiss_time = time.time() - start_time
print(faiss_time)
# print("内积相似度 (越大越相似):")
# print("Scores:\n", D)
# print("Indices:\n", I)