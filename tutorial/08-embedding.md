# 08 · Embedding（嵌入）

> 前置章节：[07 切块](07-chunking.md)
> 下一章 → [09 Vector Store](09-vector-store.md)

---

## 一句话

**Embedding 是把"一段文字"翻译成"几百个数字组成的坐标"——坐标相近就意味着语义相近。**

---

## 类比：城市坐标

把所有城市放到地图上，**北京在 (116°, 40°)**、**上海在 (121°, 31°)**、**纽约在 (-74°, 41°)**。

- 北京和上海坐标接近 → 都在中国，距离近
- 北京和纽约坐标差很多 → 在地球两边

embedding 就是把**文字**放到一张"语义地图"上：

| 句子 | 坐标（简化） |
|---|---|
| "桥梁防水谁负责？" | (0.2, 0.5, -0.1, ...) |
| "Who handles bridge waterproofing?" | (0.21, 0.49, -0.09, ...) ← 跟上面几乎一样！|
| "今天吃什么？" | (0.8, -0.3, 0.6, ...) ← 跟上面差很远 |

中英文意思一样 → 坐标接近。意思不同 → 坐标远。这是 embedding 的核心魔法。

不同之处：embedding 不是 2 维坐标，而是 **几百维**（典型 384、768、1024 维）。维度越高，能表达的"语义维度"越丰富。

---

## 为什么需要 embedding

回到检索问题：用户问"桥梁防水谁负责"，怎么在几万个 chunks 里找到答案？

### 老办法（BM25）

逐字匹配："桥梁防水责任"这几个词在哪个 chunk 出现？
- ✅ 文字层面精准
- ❌ 不懂语义：用户问"防水责任"，文档写"waterproofing accountability"——BM25 找不到

### 向量办法（Embedding）

- 把所有 chunks 预先 embed 成向量，存进数据库
- 用户问问题时，把问题也 embed 成向量
- 找**距离最近的几个 chunk 向量**

距离公式叫 **cosine similarity**（余弦相似度）：
```
similarity = (向量A · 向量B) / (|向量A| × |向量B|)
```
范围 [-1, 1]，1 = 一模一样，0 = 无关，-1 = 相反。

---

## 我们用什么 embedding 模型

代码在 [src/jcontract/impls/fastembed_embedder.py](../src/jcontract/impls/fastembed_embedder.py)。

**当前默认**：`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

| 维度 | 模型 | 大小 | 维度 | 速度 |
|---|---|---|---|---|
| 我们用的 | paraphrase-multilingual-mpnet-base-v2 | ~1GB | 768 | 中等 |
| 备选 | paraphrase-multilingual-MiniLM-L12-v2 | ~220MB | 384 | 快 |
| 备选 | intfloat/multilingual-e5-large | ~2.24GB | 1024 | 慢 |
| 计划升级 | BAAI/bge-m3 | ~2GB | 1024 | 慢但功能强 |

### 为什么不直接用 bge-m3

bge-m3 是 2024 年 BAAI 出的"三合一"嵌入模型，三个 Multi：

- **Multi-Lingual** — 100+ 种语言
- **Multi-Functionality** — 同一个模型同时输出 dense 向量 / sparse 向量 / ColBERT 多向量
- **Multi-Granularity** — 最长 8192 tokens（mpnet 只能 512）

**理论上比 mpnet 强**。

**为什么没用**：

我们用 `qdrant-client[fastembed]==1.12.1`，它 pin 死 `fastembed==0.3.6`。0.3.6 的支持模型列表里**没有 bge-m3**。要升级 fastembed 到 0.4+ 就得连带升 qdrant-client，连锁反应风险大。

**计划**：Phase 2 升级路径已经写好——切到 `FlagEmbedding` 库直接跑 bge-m3。Embedder Protocol 不变，只换 impls/。

---

## bi-encoder vs cross-encoder（重要预热）

**这个项目同时用了两种 encoder**，先理清概念：

### bi-encoder（双编码器）— mpnet / bge-m3

- 问题和文档**分开**编码
- 每段文字编完得到独立向量
- 可以**预先把所有 chunks 编码好存起来**
- 查询时只编码问题，然后查最近邻
- **优点**：快、可扩展。10 万 chunks 也能毫秒返回
- **缺点**：质量天花板低，因为模型从没"同时看过"问题和文档

### cross-encoder（交叉编码器）— bge-reranker-v2-m3

- 问题和文档**拼成一对**喂给模型
- 模型可以**自由 attention**两者所有词的关系
- 每对要跑一次模型，**不能预先索引**
- **优点**：质量天花板高
- **缺点**：慢，O(N) per query

详细对比和 cross-encoder 怎么用见 [12 Cross-encoder](12-cross-encoder-reranker.md)。

**本章只讲 bi-encoder（用于初次召回）**。

---

## fastembed 是什么

[fastembed](https://github.com/qdrant/fastembed) 是 Qdrant 公司开发的轻量 embedding 库。特点：

- **不依赖 PyTorch**（省 ~2GB）
- 用 **ONNX runtime**（C++ 写的高速推理库）
- 在 CPU 上跑得动（不强求 GPU）

这是个**工程选择**：本项目要打包成 Docker 给老王部署，少 2GB 的镜像大小很关键。

代价：fastembed 只支持一小撮预导出的 ONNX 模型（mpnet、MiniLM、e5），换模型不像 transformers 那么自由。

---

## 第一次跑会发生什么

```bash
uv run jcontract ingest some.pdf
```

第一次执行 `embedder.embed(...)` 时：

```
[fastembed] Downloading paraphrase-multilingual-mpnet-base-v2...
Downloading model.onnx: 100% [================] 1024MB
Downloading tokenizer.json: 100% [================] 17MB
Saved to ~/.cache/fastembed/
```

下载 **~1GB ONNX 模型** + **~17MB tokenizer**。下载一次永久缓存，后续都是 cache hit。

---

## 实际怎么用

[impls/fastembed_embedder.py](../src/jcontract/impls/fastembed_embedder.py) 核心代码：

```python
class FastEmbedEmbedder:
    def __init__(self, model_name=DEFAULT_MODEL):
        self._model_name = model_name
        self._dim = _MODEL_DIMS[model_name]  # 硬编码维度，不用加载模型查
        self._model = None  # 懒加载

    @property
    def dim(self) -> int:
        return self._dim  # 768 for mpnet

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()  # 第一次调用时加载
        return [vec.tolist() for vec in model.embed(texts)]
```

注意几个工程细节：

### 维度硬编码

```python
_MODEL_DIMS = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    ...
}
```

为什么不动态查？因为查需要先加载模型（30 秒）。但 `QdrantStore.dim` 要在 collection 创建时**立刻**知道维度。硬编码让 `dim` 属性 0 秒返回。

### 懒加载

```python
self._model = None
def _ensure_model(self):
    if self._model is None:
        self._model = TextEmbedding(model_name=self._model_name)
    return self._model
```

构造 `FastEmbedEmbedder` 时不加载模型——避免"只想看看 dim"的调用也付下载代价。第一次 `embed()` 才真正下载/加载。

### 空 list 短路

```python
if not texts:
    return []
```

如果有人传空列表（探测 .dim），不要为此触发下载。这是**契约层的"不付意外代价"原则**。

---

## 在 pipeline 里的位置

```
PDFParser → ParsedPage
              ↓
Chunker → list[Chunk]
              ↓
Embedder ← (这里)
              ↓
embed([c.text for c in chunks])  →  list[list[float]]
              ↓
VectorStore.add(chunks, vectors)
```

每个 chunk 的 `text` 字段被独立 embed → 一个 768 维向量 → 跟 chunk 一起灌进 Qdrant。

查询时**同一个 embedder 实例 embed 问题**：

```
"桥梁防水谁负责？" → embedder.embed([q]) → [0.21, 0.49, -0.09, ...]
              ↓
QdrantStore.search(向量, k=20) → top-20 最近邻 chunks
```

**关键**：索引和查询必须用**同一个模型**，否则两套向量在不同空间，距离没意义。

---

## 多语言怎么工作

mpnet 是个**多语言模型**，训练数据里同时有中英对照句对。所以：

- "桥梁防水" 的向量 ≈ "bridge waterproofing" 的向量
- 用户用中文问 → embed 成"中英共享语义空间"的向量 → 能匹配到英文 chunk

这是为什么本项目能**中文提问、英文 PDF 检索**——靠的就是多语言 embedding。

如果换成英文-only 模型（比如 `all-mpnet-base-v2`），中文查询就废了。

---

## 性能数据（实测）

| 操作 | 耗时 |
|---|---|
| 首次加载模型 | ~3-5s |
| 单条 chunk embed | ~5-10ms |
| 批量 100 chunks | ~200-300ms |
| 批量 1000 chunks | ~2s |

**批量比单条快**——因为 ONNX 内部能并行处理多条。

ingest 时 chunker 输出几百 chunks 一次性 embed，**不要循环 embed(single)**。

---

## 关键决策（DECISION）

### 为什么用 mpnet 而不是 MiniLM

- MiniLM 只有 384 维，质量略弱
- mpnet 768 维，质量稳，4 倍于 MiniLM 但仍是 CPU 友好
- 中英平衡上 mpnet 经过更多多语言训练

### 为什么不本地装 PyTorch + sentence-transformers

- sentence-transformers 装上来 ~2GB（torch + nvidia wheels）
- fastembed 用 ONNX 走纯 CPU 路径，~280MB
- **但** reranker（cross-encoder）后来还是装了 sentence-transformers（fastembed 不支持 cross-encoder）—— 现在 torch 已经在依赖里，未来可以直接跑 bge-m3。

### 为什么不调云端 embedding API（如 OpenAI text-embedding-3）

- 数据隐私：合同不上传第三方
- 成本：每次 ingest 都过 API，4100 页 × ~$0.0001 = 不便宜
- 离线可用：客户机房没网也能跑

---

## 下一步阅读

- 向量存哪里、怎么查 → [09 Vector Store](09-vector-store.md)
- 关键词检索那条路 → [10 BM25](10-bm25-keyword.md)
- 两路结果怎么合 → [11 RRF](11-rrf-fusion.md)
- cross-encoder 怎么用 → [12 Reranker](12-cross-encoder-reranker.md)
