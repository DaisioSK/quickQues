# 09 · Vector Store / Qdrant

> 前置章节：[08 Embedding](08-embedding.md)
> 下一章 → [10 BM25 + jieba](10-bm25-keyword.md)

---

## 一句话

**向量库 = 专门为"找最近邻"优化的数据库。Qdrant 是当前默认实现。**

---

## 类比：3D 城市快递站

想象一个 3D 仓库（其实是 768 维），每个包裹（chunk 向量）都有自己的坐标。

新订单来了，问"离 (116, 40, ...) 最近的 20 个包裹在哪？"

- **暴力解**：算订单坐标和**每个包裹**的距离，排序，取前 20。10 万包裹要算 10 万次距离 → 慢
- **聪明解**：仓库预先把包裹按区域分层索引（HNSW、IVF 等算法）→ 几毫秒返回

向量库就是这种"聪明仓库"。

---

## Qdrant 是什么

Qdrant 是 Rust 写的**开源向量数据库**，特性：

- 单机/集群都行
- 持久化（数据存磁盘，重启不丢）
- HTTP + gRPC 双 API
- Docker 一键起
- 支持 metadata filter（`WHERE chunk_type = 'qa_pair'` 这种）

**我们用 Docker 起单机版**，数据在 `data/qdrant/` 卷。

### 为什么不用 pgvector / Chroma / Milvus

| 选项 | 优点 | 缺点 |
|---|---|---|
| **Qdrant** ✅ | 单文件、零依赖、HTTP API、metadata 强 | 多一个服务 |
| pgvector | Postgres 扩展，结合传统 SQL | 要装 PG，索引调优麻烦 |
| Chroma | 纯 Python 库，零部署 | 性能差、规模上不去 |
| Milvus | 大规模专家级 | 部署复杂，杀鸡用牛刀 |

DEMO 这种 1 万-10 万 chunk 规模，Qdrant 是甜点。

---

## 三个核心概念

### Collection（集合）

= 一张表。我们的 collection 名是 `contract`。

每个 collection 有固定的：
- **向量维度**（768 for mpnet）
- **距离度量**（cosine / euclidean / dot product）— 我们用 cosine

### Point（点）

= 一行记录。每个 Point 含：
- `id`：唯一 ID（用 chunk.id）
- `vector`：768 维向量
- `payload`：任意 JSON 元数据（这里塞整个 Chunk 的字段）

### Search

`search(vector, limit=20)` → 返回 top-20 最近邻 Points。

---

## 我们的 QdrantStore 怎么用

[src/jcontract/impls/qdrant_store.py](../src/jcontract/impls/qdrant_store.py) 实现 VectorStore Protocol。

### 初始化

```python
store = QdrantStore(collection_name="contract")
```

构造时**不**连接 Qdrant 服务。**懒连接**：第一次 add/search 时才连。

### 添加 chunks + 向量

```python
chunks = [Chunk(...), Chunk(...), ...]  # 来自 chunker
vectors = [[0.1, 0.2, ...], [0.3, ...]] # 来自 embedder

store.add(chunks, vectors)
```

内部：
1. 第一次 add 时，按 `len(vectors[0])` 推断维度，创建 collection
2. 把每个 (chunk, vector) 转成 Qdrant Point
3. 批量 upsert（重复 id 会替换，不会重复）

### 查询

```python
hits = store.search(query_vector, k=20)
# hits 是 list[SearchResult]
for hit in hits:
    print(hit.chunk.id, hit.score)  # score 是 cosine similarity, [0, 1]
```

返回的是 `SearchResult`（含 Chunk 完整对象 + score），不是 Qdrant 的原生 Point —— **接口边界把 vendor 类型挡在外面**。

---

## HNSW：Qdrant 用什么算法找最近邻

HNSW = **Hierarchical Navigable Small World**（分层导航小世界图）。

直觉：
- 把所有向量构建成一张"小世界图"（社交网络那种结构）
- 每个向量是图的一个节点
- 节点之间按距离连边
- 分层：上层是"高速公路"（节点稀疏，长跳），下层是"乡间小道"（节点密集，短跳）

找最近邻：
1. 从上层入口开始
2. 在当前层贪心走（每步选离 query 更近的邻居）
3. 走不动了下一层
4. 直到最底层，找到真正的最近邻

**速度**：O(log N)，10 万节点也只要几毫秒。

**代价**：是近似算法，理论上可能漏 0.1%-1% 的真正最近邻。但对 RAG 检索完全无所谓。

Qdrant 默认参数已经调好，我们不动 HNSW 参数。

---

## 距离度量：为什么用 cosine

三种常见度量：

| 度量 | 公式 | 直觉 |
|---|---|---|
| **cosine** | `A·B / (|A|×|B|)` | 看夹角，不看长度 |
| euclidean | `√Σ(A_i - B_i)²` | 直线距离 |
| dot product | `A·B` | 夹角 + 长度 |

**embedding 用 cosine**：
- embedding 向量的"长度"没物理意义（不同模型尺度差很多）
- 我们只关心**方向**（语义方向相同 = cosine 接近 1）
- cosine 自带归一化，跨语料稳定

mpnet 输出已经是 normalized 向量（长度=1），cosine 和 dot product 在这种情况下等价。但配置成 cosine 是惯例（明确意图）。

---

## Payload：每个 Point 存了什么

每个 Qdrant Point 的 payload 是完整的 Chunk 字段：

```json
{
  "id": "ContractDEMO(1of9)TQA:42:3",
  "text": "Question No.: ACME/TRACKWORK/16\n...",
  "file": "Contract DEMO(1of9) TQA.pdf",
  "page": 42,
  "chunk_type": "qa_pair",
  "section_path": "Section 7 > Clause 7.3",
  "revision": "Rev A",
  "drawing_refs": ["T/PRJ/CWD/WS/2101A"],
  "clause_refs": ["7.3.2"],
  "question_no": "ACME/TRACKWORK/16"
}
```

**好处**：搜索时一次性拿到完整 Chunk，不需要再去别处查 text/metadata。**冗余换简单**。

**潜在升级**：Phase 2 量大时可考虑只存 ID，text/payload 走另一个 KV store。当前 1 万 chunk 全文存进 Qdrant 也才 ~50MB，无所谓。

---

## metadata filter（暂未启用，但接口支持）

Qdrant 支持在向量搜索的同时过滤 payload：

```python
# 假设我们启用了 filter
store.search(
    query_vector,
    k=20,
    filter=Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="qa_pair"))]),
)
# 只返回 qa_pair 类型的 chunk
```

这是 Qdrant 比 Chroma 强的一大特性。**Phase 3 检索调优时可能启用**——比如"只搜 Rev A 的 chunks"、"只搜某份 PDF"。

当前简单起见没用，所有查询都全库搜。

---

## 持久化：数据在哪

`docker-compose.yml` 里：

```yaml
qdrant:
  image: qdrant/qdrant:v1.12.1
  volumes:
    - ./data/qdrant:/qdrant/storage
  ports:
    - "6333:6333"  # HTTP
    - "6334:6334"  # gRPC
```

数据落在 `data/qdrant/` 目录，**git 不跟踪**（在 `.gitignore` 里）。

重启 docker container → 数据还在。删 `data/qdrant/` → 数据没了，得重新 ingest。

---

## 怎么手动看 Qdrant 里有啥

启动 Qdrant 后浏览器打开：
```
http://localhost:6333/dashboard
```

Qdrant 自带 Web UI，能看 collections、point 数、跑临时查询。开发调试很方便。

或者命令行：
```bash
curl http://localhost:6333/collections/contract
# {"result": {"status": "green", "vectors_count": 1234, ...}}
```

---

## 性能数据

DEMO 单份 1of9 TQA（109 页 → ~600 chunks）：

| 操作 | 耗时 |
|---|---|
| 全部入库（含 embedding） | ~3-5 min |
| 单次查询（k=20） | ~10-20ms |
| Qdrant 启动 | ~2-3s |

未来 4100 页 → 估计 ~20k chunks → 查询仍 < 50ms（HNSW 是 log scale）。

---

## 关键决策（DECISION）

### 为什么不分离向量和 payload

Qdrant 默认是"payload 嵌入 vector entry 一起存"。理论上可以分离（节省内存），但需要额外存储层。DEMO 规模下不必要。

### 为什么 lazy connect

测试时（pytest）经常构造 QdrantStore 但不真调用——如果 `__init__` 就连，每个测试都要 mock。Lazy connect 让"只构造不用"零成本。

### 为什么用 collection name 而非项目隔离

未来如果多项目（J108、J109），用 collection name 区分（`contract`、`j108`），不用搞多个 Qdrant 服务。

---

## 切换实现的可能路径

如果某天 Qdrant 不够用，VectorStore 接口允许这些替代：

| 替代 | 触发场景 |
|---|---|
| pgvector | 已经有 Postgres、想合并管理 |
| Pinecone / Weaviate Cloud | 不想自己运维 |
| FAISS 纯内存 | 嵌入式部署 |

**业务代码不动**，写一个 `PgvectorStore` 实现 VectorStore Protocol 就行。

---

## 下一步阅读

- 关键词检索那条路 → [10 BM25](10-bm25-keyword.md)
- 两路结果合一 → [11 RRF](11-rrf-fusion.md)
