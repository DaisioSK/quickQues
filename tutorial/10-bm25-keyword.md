# 10 · BM25 + jieba 关键词检索

> 前置章节：[09 Vector Store](09-vector-store.md)
> 下一章 → [11 RRF 融合](11-rrf-fusion.md)

---

## 一句话

**BM25 是 1994 年发明的"按关键词找文档"打分公式，至今仍是行业标配——配上 jieba 分词，能同时搞定中英文。**

---

## 类比：图书馆老馆员

向量检索（embedding）像**听口语的助理**：你说"防水责任"她就懂"waterproofing accountability"也是这个意思，**懂语义**。

BM25 像**老派图书馆员**：你给她一个词单 `["桥梁", "防水", "责任"]`，她翻卡片柜，**精确**告诉你哪些卡片含这些词，含得越多越独特越靠前。

| 维度 | 向量检索 | BM25 |
|---|---|---|
| 懂语义 | ✅ | ❌ |
| 懂同义词 | ✅ | ❌（"防水" ≠ "waterproofing"）|
| 找专有名词 | 容易混淆 | ✅ 完美 |
| 找编号（T/PRJ/CWD/WS/2101A）| 容易找错 | ✅ 一字不差 |
| 速度 | 快 | 极快 |

**两者互补**，所以我们**同时跑两路**（[11 RRF](11-rrf-fusion.md)）。

---

## BM25 公式（不用怕，直觉就好）

```
score(D, Q) = Σ over q in Q [ IDF(q) × (tf(q, D) × (k+1)) / (tf(q, D) + k × (1 - b + b × |D|/avgdl)) ]
```

吓人，但直觉只有 3 条：

1. **词在文档里出现得越多** → 分越高
2. **词越罕见**（在全语料里只出现几次）→ 命中时加分越多
3. **文档越短** → 同样命中，短文档得分更高（信息密度大）

举例。语料里有 10000 个 chunk，"桥梁"在 200 个 chunk 里出现，"防水"在 50 个 chunk 里出现：

| Chunk 内容 | 桥梁次数 | 防水次数 | BM25 |
|---|---|---|---|
| "...桥梁的防水做法..." (50 字) | 1 | 1 | 中 |
| "...桥梁..."（500 字长文档但只提 1 次）| 1 | 0 | 低 |
| "...防水防水防水..." (30 字小段) | 0 | 3 | 高（防水比桥梁罕见）|
| "...桥梁防水防水责任..." (40 字) | 1 | 2 | **最高** |

**罕见词权重大**（IDF = inverse document frequency 的影响），**文档短信息密**（归一化 |D|/avgdl 的影响）。

---

## rank-bm25 库

[rank-bm25](https://pypi.org/project/rank-bm25/) 是个**纯 Python 实现**，3MB，没有 C 扩展。

我们用 `BM25Okapi`（Okapi 是 BM25 的标准变体）：

```python
from rank_bm25 import BM25Okapi

# 索引阶段
corpus = [
    ["桥梁", "防水", "责任"],     # chunk 1 的 tokens
    ["材料", "检验", "标准"],     # chunk 2 的 tokens
    ...
]
bm25 = BM25Okapi(corpus)

# 查询阶段
query_tokens = ["桥梁", "防水"]
scores = bm25.get_scores(query_tokens)
# scores 是数组，每个元素是对应 chunk 的分数
top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
```

**核心**：rank-bm25 只接受 **list[list[str]]**——已经分好词的 tokens。它不做分词。

**所以我们需要 jieba。**

---

## jieba：中文分词

中文没空格，**"桥梁防水责任"**字面上就是一长串。要让 BM25 工作，先得切：

```python
import jieba

list(jieba.cut("桥梁防水的责任方是谁"))
# ['桥梁', '防水', '的', '责任', '方', '是', '谁']
```

jieba 是 Python 中文 NLP 的事实标准。3MB，纯 Python，MIT 许可，活跃维护，无依赖。

### 三种分词模式

```python
jieba.cut("桥梁防水的责任方", cut_all=False)  # 精确模式（默认）
# → 桥梁 / 防水 / 的 / 责任 / 方

jieba.cut("桥梁防水的责任方", cut_all=True)   # 全模式
# → 桥梁 / 防水 / 水的 / 责任 / 责任方

jieba.cut_for_search("桥梁防水的责任方")       # 搜索引擎模式
# → 桥梁 / 防水 / 的 / 责任 / 责任方
```

我们用**精确模式**（cut_all=False）—— 召回精度平衡好。

---

## 中英混杂怎么办（关键技巧）

合同里经常出现：
> 桥梁 waterproofing 责任 by Trackwork Contractor

如果只对中文用 jieba、对英文用 split，需要**语言检测**——加一个 langdetect 依赖，还容易在混杂句失效。

**我们的简化做法**（[bm25_index.py:84](../src/jcontract/impls/bm25_index.py#L84)）：

```python
def _tokenize(text):
    tokens = []
    for raw in jieba.cut(text, cut_all=False):
        tok = raw.strip().lower()
        if not tok:
            continue
        if all(ch in _PUNCT for ch in tok):  # 跳过标点
            continue
        tokens.append(tok)
    return tokens
```

**关键发现**：jieba 对英文行为合理——遇到空格分隔的 ASCII，会把每个英文单词当独立 token：

```python
list(jieba.cut("waterproofing at pier"))
# → ['waterproofing', ' ', 'at', ' ', 'pier']
```

我们过滤掉空格 token（`tok.strip()` 后变空），剩下 `['waterproofing', 'at', 'pier']`——完美。

而且 `.lower()` 让 "Waterproofing" 和 "waterproofing" 视为同一 token。

---

## "in-memory" 是什么意思

[bm25_index.py:103](../src/jcontract/impls/bm25_index.py#L103) 的 `Bm25Index` 类核心：

```python
class Bm25Index:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._id_index: dict[str, int] = {}
```

整个索引就是 **3 个 Python list + 1 个 dict**。**没有磁盘文件**，全在内存里。

进程一退出，索引就没了。

### 为什么不持久化

1. 重建很快——几千 chunk 几秒重新算 BM25
2. BM25 实现没有"增量更新"——每次 add 都要重算 IDF（全局统计），写磁盘没意义
3. Phase 1 量小，省力为先

### 那 chunks 哪里存

**JSONL snapshot**（下一节）。chunks 本身存盘，BM25 内存索引每次从 JSONL 重建。

---

## JSONL snapshot 持久化

[src/jcontract/ingest/pipeline.py](../src/jcontract/ingest/pipeline.py) 的设计：

### Ingest 时

```python
def ingest(self, pdf_path):
    chunks = self.chunker.chunk(pages, pdf_path.name)
    ...
    self._append_snapshot(chunks)  # ← 追加到 JSONL

def _append_snapshot(self, chunks):
    with self.chunks_snapshot_path.open("a", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
```

每个 chunk 序列化成一行 JSON，**append** 模式写。

文件长这样（`data/chunks_snapshot.jsonl`）：
```jsonl
{"id":"...:42:1","text":"...","file":"...pdf","page":42,"chunk_type":"paragraph",...}
{"id":"...:42:2","text":"Question No.: ...","file":"...pdf","page":42,"chunk_type":"qa_pair",...}
{"id":"...:43:1","text":"...","file":"...pdf","page":43,"chunk_type":"table",...}
```

### CLI 启动时

`cli.py` 的 `_build_stack()`：

```python
keyword_index = Bm25Index()
cached = load_chunks_snapshot(CHUNKS_SNAPSHOT)
if cached:
    keyword_index.add(cached)
    logger.info("cli.bm25_rehydrated", chunks=len(cached))
```

每次 CLI 启动：
1. 创建空 Bm25Index
2. 读 `data/chunks_snapshot.jsonl` 还原所有 chunks
3. 喂给 Bm25Index → 重新建索引
4. 几秒后 ready

**只有 BM25 需要这个 snapshot**。Qdrant 自己持久化（Docker volume），RefGraph 在 SQLite（自动持久化）。

### 为什么 JSONL 而非 JSON

| | JSONL | JSON |
|---|---|---|
| 增量追加 | ✅ 直接 append 一行 | ❌ 要读出来 + 改 + 写回 |
| 流式读 | ✅ 逐行 | ❌ 要全部 load 才能解析 |
| 部分损坏 | ✅ 只丢一行 | ❌ 整个文件废 |
| 可读性 | 类似 | 略好（带缩进时） |

ingest 是**逐份 PDF 追加**的，JSONL 完美匹配这种模式。

---

## 一些工程细节

### 空 token 处理

```python
safe_tokens = [t if t else ["__empty__"] for t in self._tokens]
self._bm25 = BM25Okapi(safe_tokens)
```

某些 chunk 文字可能全是空格或标点（OCR 出错时），分词后 tokens = []。BM25Okapi 喂空 list 会**除零崩溃**。我们用 `["__empty__"]` 哨兵替代——一个真实查询永远不会匹配它。

### idempotent add

```python
if chunk.id in self._id_index:
    pos = self._id_index[chunk.id]
    self._chunks[pos] = chunk    # 替换
    self._tokens[pos] = toks
else:
    self._id_index[chunk.id] = len(self._chunks)
    self._chunks.append(chunk)   # 新增
    self._tokens.append(toks)
```

同一个 id 反复 add 不会重复——后来的覆盖前面的。**契约要求**。

### 全量重建

```python
self._bm25 = BM25Okapi(safe_tokens)  # 每次 add 都重建
```

BM25Okapi 不支持增量。每次 add 后必须从头算。**几千 chunk 约 < 1s**，可接受。

---

## 性能数据

- 1000 chunk 建索引：~200ms
- 10000 chunk 建索引：~2s
- 单次查询：~10-30ms（取决于 query tokens 数）

对比向量检索：
- BM25 完全 CPU，无下载，启动快
- 向量检索 cold start 慢（要加载 1GB 模型）

所以 BM25 的"重建快"在我们场景里**比"省下重建"更值**。

---

## 关键决策（DECISION）

### 为什么不用 Qdrant 的内置 BM25 / sparse vector

Qdrant 1.10+ 支持 sparse vector + BM25 native。**理论上**可以省一个组件。

不选的原因：
1. Qdrant BM25 配置复杂（要算 IDF、配 tokenizer）
2. **不支持中文分词** —— 必须先 jieba 切再喂
3. 两个独立组件让调试更清晰：查 BM25 出问题就只看 Bm25Index

**未来 Phase 2 考虑迁移**，但目前外置 BM25 + Qdrant 的组合实证可靠。

### 为什么 jieba 不开启自定义词典

jieba 支持加自定义词（"T/PRJ/CWD/WS/2101A" 当成一个 token）。但：
- 维护词典是负担
- 实测默认精确模式对 Drawing No. 处理已经合理（切成 ['t', '/', 'contract', ...] 后 BM25 仍能命中）
- Phase 1 不追求极致，保持简单

未来如果发现 Drawing No. 检索召回低，可以加自定义词典。

### 为什么不归一化 BM25 score

BM25 分数无上限（语料统计相关），最高可能几十。

**不归一化**的原因：[hybrid.py](../src/jcontract/retrieve/hybrid.py) 用 RRF 融合——只看排名不看分数。归一化纯属画蛇添足。**调试时**保留原始分数能看清是不是真的强命中（高 BM25 = 词真重叠多）。

---

## 下一步阅读

- 向量 + BM25 怎么合 → [11 RRF 融合](11-rrf-fusion.md)
- 融合后还要怎么精排 → [12 Cross-encoder](12-cross-encoder-reranker.md)
