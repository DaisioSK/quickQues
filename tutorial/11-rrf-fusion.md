# 11 · RRF 融合

> 前置章节：[10 BM25 + jieba](10-bm25-keyword.md)
> 下一章 → [12 Cross-encoder](12-cross-encoder-reranker.md)

---

## 一句话

**RRF（Reciprocal Rank Fusion，倒数排名融合）—— 把多路检索结果"按排名"而非"按分数"合并，免校准、跨场景稳健。**

---

## 类比：两个评委打分

想象 10 个选手参加才艺比赛，两位评委独立打分：

| 选手 | 评委A 分数 | 评委A 排名 | 评委B 分数 | 评委B 排名 |
|---|---|---|---|---|
| 小明 | 95 | 1 | 60 | 6 |
| 小红 | 85 | 3 | 90 | 1 |
| 小张 | 80 | 5 | 85 | 2 |
| 小李 | 88 | 2 | 50 | 9 |
| 小王 | 70 | 8 | 78 | 4 |

怎么合并出总冠军？

### 蠢办法：分数加权和

`小明 = 95 + 60 = 155`，`小红 = 85 + 90 = 175`，小红赢。

**致命问题**：

- 如果评委A 是严师（满分 100，平均 70），评委B 是水货（满分 100，平均 85）—— A 给的 80 比 B 给的 90 更可贵，**但加权和看不到这个差异**
- 同样道理：cosine similarity（[-1, 1]）和 BM25（[0, +∞]）**量纲完全不同**，硬加是数值垃圾

### 聪明办法（RRF）：按排名

公式：每个选手的总分 = Σ over 评委 `1 / (k + rank)`，k=60 是常数。

```
小明  = 1/(60+1) + 1/(60+6) = 0.0164 + 0.0152 = 0.0316
小红  = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
小张  = 1/(60+5) + 1/(60+2) = 0.0154 + 0.0161 = 0.0315
小李  = 1/(60+2) + 1/(60+9) = 0.0161 + 0.0145 = 0.0306
小王  = 1/(60+8) + 1/(60+4) = 0.0147 + 0.0156 = 0.0303
```

排序：**小红 > 小明 > 小张 > 小李 > 小王**。

直觉：
- 排得越前 → 加分越多（分母小）
- 出现在多个评委的榜上 → 多次累加 → 浮到最上面
- **不需要知道评委打分的具体量纲** → 跨评委公平

---

## 在我们项目里

[src/jcontract/retrieve/hybrid.py:30](../src/jcontract/retrieve/hybrid.py#L30)：

```python
RRF_K = 60

def rrf_fuse(rankings: list[list[SearchResult]], k_constant: int = RRF_K) -> list[SearchResult]:
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):  # rank 从 1 开始
            cid = result.chunk.id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)
            chunks_by_id[cid] = result.chunk

    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [SearchResult(chunk=chunks_by_id[cid], score=score) for cid, score in ordered]
```

**20 行不到**。

输入：N 个 ranked list（我们 N=2：向量 + BM25）
输出：1 个融合后的 ranked list

---

## 一个具体例子

问题："桥梁防水谁负责？"

### Vector 检索返回（top-5）

```
1. chunk-A: "Trackwork Contractor shall be responsible for waterproofing..." (cosine 0.85)
2. chunk-B: "桥梁结构防水层的施工..."                                       (cosine 0.81)
3. chunk-C: "Bridge deck waterproofing requirements..."                     (cosine 0.79)
4. chunk-D: "工程承包商责任范围..."                                          (cosine 0.76)
5. chunk-E: "...防水材料标准..."                                             (cosine 0.74)
```

### BM25 检索返回（top-5）

```
1. chunk-A: "Trackwork Contractor shall be responsible for waterproofing..." (BM25 12.3)
2. chunk-F: "...责任方为业主方..."                                            (BM25 8.1)
3. chunk-B: "桥梁结构防水层的施工..."                                         (BM25 7.5)
4. chunk-G: "...防水工程总责任..."                                           (BM25 6.9)
5. chunk-H: "...防水措施详见..."                                             (BM25 5.2)
```

### RRF 融合（k=60）

```
chunk-A: 1/(60+1) + 1/(60+1) = 0.0164 + 0.0164 = 0.0328  ⭐ 两路都第一
chunk-B: 1/(60+2) + 1/(60+3) = 0.0161 + 0.0159 = 0.0320  ⭐ 两路都靠前
chunk-C: 1/(60+3)            = 0.0159           = 0.0159
chunk-D: 1/(60+4)            = 0.0156           = 0.0156
chunk-E: 1/(60+5)            = 0.0154           = 0.0154
chunk-F:            1/(60+2) = 0.0161           = 0.0161
chunk-G:            1/(60+4) = 0.0156           = 0.0156
chunk-H:            1/(60+5) = 0.0154           = 0.0154
```

排序：
```
1. chunk-A  0.0328  ← 两路冠军，毫无争议
2. chunk-B  0.0320  ← 两路都进前 3
3. chunk-F  0.0161  ← 只在 BM25 中靠前，但仍有位置
4. chunk-C  0.0159  ← 只在 vector 中
5. chunk-D  0.0156  ← 只在 vector 中
6. chunk-G  0.0156  ← 只在 BM25 中
7. chunk-E  0.0154
8. chunk-H  0.0154
```

**关键观察**：
- chunk-A 在两路都第一 → 融合后稳居榜首
- chunk-B 两路都靠前 → 第二
- 只在单路出现的也保留位置（vector 强项语义、BM25 强项关键词，互补）

---

## 为什么 k=60

公式里的 k=60 是 Cormack et al. 2009 论文（"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"）的实证推荐。

直觉：

- **k 越小**（k=1, 5）：第一名权重比第二名大很多 → 头部强项主导
- **k 越大**（k=200, 1000）：差距摊平 → 头部和中段差异变小
- k=60：在多个公开 benchmark（TREC）上经验最优

我们没改这个值。`RRF_K = 60` 是常数。

---

## RRF 的好处（为什么这么受欢迎）

1. **免校准** — 不需要知道每路评分的量纲
2. **稳健** — 一路完全爆掉（比如全部 0 分），另一路仍能产生有意义结果
3. **简单** — 20 行代码，没有超参数（除了 k，且 k=60 通用）
4. **可扩展** — 想加第三路（比如 RefGraph 召回）？append 进 rankings list 就行
5. **业界标准** — Elasticsearch 8.10+、OpenSearch、Pinecone Hybrid Search 全都用 RRF

---

## RRF 的局限

诚实讲一下它不擅长的：

1. **不能体现"压倒性优势"** — 如果 vector 检索打 0.99，BM25 打 0.99，跟两边都打 0.5 一样进 top-1。RRF 只看排名。
2. **稀疏召回会被埋没** — 如果某个 chunk 只在 vector 第 100 名出现，RRF 给它 1/(60+100) = 0.006 几乎可忽略。
3. **不学习** — 比起 learned-to-rank（用 ML 模型学融合权重），RRF 是无监督的。

为了弥补 #1 和 #2，**RRF 之后我们再上 cross-encoder reranker**（[12](12-cross-encoder-reranker.md)）——精读后能"翻盘"被埋没的强候选。

---

## 在 HybridRetriever 里的位置

```python
class HybridRetriever:
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query_vector = self.embedder.embed([query])[0]

        # 1️⃣ 并发跑两路（per_backend_k=20）
        with ThreadPoolExecutor(max_workers=2) as pool:
            vec_future = pool.submit(self.vector_store.search, query_vector, 20)
            kw_future = pool.submit(self.keyword_index.search, query, 20)
            vec_results = vec_future.result()
            kw_results = kw_future.result()

        # 2️⃣ RRF 融合
        fused = rrf_fuse([vec_results, kw_results])

        # 3️⃣ 可选 cross-encoder 精排
        if self.reranker is None:
            return fused[:k]
        candidates = fused[:self.rerank_top_n]  # top 30
        reranked = self.reranker.rerank(query, candidates)
        return reranked[:k]
```

`per_backend_k=20` 意思每路各取 20 个候选，融合后总共最多 40 个（重叠的算一个）。

---

## per_backend_k 怎么选

- **太小**（k=5）：候选不够，融合空间小
- **太大**（k=100）：噪声大，融合后排在前面的可能是"两路都中位"而非"两路都靠前"
- **20**：经验值，对几千 chunk 规模适合

未来可以根据 reranker 是否启用动态调整：
- 有 reranker：`per_backend_k=30, rerank_top_n=50`（让 reranker 精排更多）
- 无 reranker：`per_backend_k=20, k=5`（直接出 top-5）

---

## 自己跑一遍体验

```bash
# 先 ingest 数据
uv run jcontract ingest some.pdf --parser claude-cli-vision

# 看 hybrid 检索结果
uv run jcontract search "桥梁防水谁负责" --k 5
```

输出会显示每个 hit 的 chunk + 它的 RRF 分数。把 chunk_type、page、score 列出来，你能直观感受到向量召回（语义）和 BM25 召回（关键词）的差异。

---

## 关键决策（DECISION）

### 为什么不用 weighted sum

如前述：cosine 和 BM25 量纲不同，加权和需要 per-corpus 校准。DEMO 上调好的权重，换 J108 完全失效。RRF 跨场景稳定。

### 为什么并发跑两路

向量检索是 I/O bound（要去 Qdrant HTTP），BM25 是 CPU bound（内存计算）。并发可以**重叠两者耗时**，节省总延迟。

ThreadPoolExecutor(max_workers=2) 而非 asyncio：BM25 不是 async API，混 sync+async 太麻烦。线程池简单粗暴。

### 为什么不缓存 fused 结果

每个 query 都是用户输入，唯一性强，缓存命中率低。**embedding 缓存**会更有价值（同一个 query 多次问），但目前没做。

---

## 下一步阅读

- 融合后怎么精排 → [12 Cross-encoder](12-cross-encoder-reranker.md)
- 整体检索 + 答案怎么连 → [14 Answerer](14-answerer-and-citations.md)
