# todo：
# 写一个异步索引构建
# 遍历每层每头
# 把索引存在attn函数的绑定字典上
# index = 2

# 写一个查询操作，参数是头、层、q、阈值
# 查询出来若干个索引，把他们的id拿出来放到hashset中

# util新增hashset参数，把这部分的实际向量值拿出来连接到头部或者尾部，trans中去掉这部分向量
import numpy as np
import faiss
import time
import torch

def build(cache,index_grid,ifindex):
    
    # with open("read.txt", "a", encoding="utf-8", buffering=1) as f:
    #     f.write(f"{ifindex}\n")
    #     ifindex[0] = 3
    #     f.write(f"{ifindex}\n")
    #     f.write(f"{ifindex}\n")
    time0 = time.time()
    layer_count = len(cache.value_cache)
    for i in range(layer_count):
        # i是层，头在嵌套内部循环
        build_layer(cache.key_cache[i],i,index_grid)
    ifindex[0] = 2
    print("构造完成")
    time1 = time.time()
    print(time1-time0)
    pass
def build_layer(xb,layer_idx,index_grid):
    # with open("read.txt", "a", encoding="utf-8", buffering=1) as f:
    # f.write(f"xb: {xb}\n")
    key_np = xb.squeeze(0).detach().numpy().copy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    for head in range(key_np.shape[0]):
        index = faiss.IndexFlatIP(key_np.shape[2])
        index.add(key_np[head])
        index_grid[layer_idx][head] = index
    # if layer_idx == 31:
    #     query = np.random.random(( 32, 1, 128)).astype('float32')
    #     query /= np.linalg.norm(query, axis=2, keepdims=True)
    #     topk = 5
    #     threshold = 0.6
    #     for i in range(32):
    #        xq = query[i]
    #        D_faiss, I_faiss = index_grid[10][i].search(xq, topk) 
    #        print("FAISS Top-k indices:", I_faiss)

def search(query,layer_idx,threshold,topk,cacheidx,index_grid):
    xq = query.squeeze(0).squeeze(1).to(torch.float32).cpu().numpy()
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    for head in range(xq.shape[0]):
        if head==0:
            cacheidx[layer_idx], _ = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            cacheidx[layer_idx] = cacheidx[layer_idx].astype(np.int64)
        else:
            result, _ = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            result = result.astype(np.int64)
            cacheidx[layer_idx] = np.concatenate((cacheidx[layer_idx], result), axis=0)