# 02 · 整体数据流

> 前置章节：[01 这个项目是什么](01-what-is-this-project.md)
> 下一章 → [03 目录结构](03-directory-structure.md)

---

## 一句话

**两条主流水线 + 一条旁路：入库（一次性，慢）/ 查询（高频，快）/ 引用图谱（精确查编号）。**

---

## 类比：图书馆

把整个系统想象成一个**懂中文的智能图书馆**：

### 📦 入库阶段（建馆）— 一次性投入

```
新书运来（PDF）
   ↓
图书管理员（PDFParser）   把每页扫描件变成文字
   ↓
分类员（Chunker）          按段落/Q&A/表格/图说切成小卡片
   ↓
打标签员（Embedder）       给每张卡片生成一个"语义指纹"（向量）
   ↓
三类索引同时建立：
  📕 卡片柜A（VectorStore） 按"语义指纹"找最相似的卡片
  📘 卡片柜B（BM25）        按关键词找包含该词的卡片
  📗 卡片柜C（RefGraph）    按 Drawing No./Clause/Q&A 编号精确查
```

### 🔍 查询阶段（读者借书）— 每次问问题

```
读者问："桥梁防水谁负责？"
   ↓
图书管理员（HybridRetriever）同时跑去三处：
  → 卡片柜A：找语义相近的卡片
  → 卡片柜B：找含"桥梁/防水/责任"的卡片
   ↓
（用 RRF 公式合并两路结果，排出 top-30）
   ↓
精读员（Cross-encoder reranker）逐张精读，重排得 top-8
   ↓
讲解员（Answerer / Claude）读完 top-8，写中文答案 + 标页码
   ↓
质检员（Postprocess）检查每个 [文件 p.X] 引用真实存在
   ↓
返回答案给读者
```

### 🎯 引用图谱旁路（特殊查询）

如果读者问的是 **"图纸 T/PRJ/CWD/WS/2101A 被哪些条款引用？"**——
这种问题**完全不走两路检索**，直接查卡片柜C（SQLite 倒排表），毫秒级精确返回。

---

## 详细数据流（按代码）

### 🔵 入库流程（Ingest）

```
input-docs/Contract DEMO(1of9) TQA.pdf
            │
            ▼
┌───────────────────────────────────────────┐
│ ClaudeCliVisionParser  (PDFParser 接口)   │
│  1. pypdfium2 渲染每页 → JPEG (DPI 150)   │
│  2. SHA-256 算缓存 key                    │
│  3. 命中缓存 → 跳过                       │
│  4. 缺失 → 调 `claude -p` 子进程 OCR      │
│  5. 输出 list[ParsedPage(page_num, text)] │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ QaAwareChunker  (Chunker 接口)            │
│  - 扫描"Question No."边界 → qa_pair       │
│  - 识别 Drawing No. / Clause / Section    │
│  - 段落切 400-800 字符                    │
│  - 输出 list[Chunk] (含完整 metadata)     │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ FastEmbedEmbedder  (Embedder 接口)        │
│  - mpnet 模型 → 768d 向量                 │
│  - 首次自动下载 ~1GB 模型                 │
│  - 输出 list[list[float]]                 │
└───────────────────────────────────────────┘
            │
            ▼
       分流到 4 个目的地：
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ QdrantStore │ Bm25Index   │ RefGraph    │ JSONL       │
│ (向量库)    │ (BM25 内存) │ (SQLite)    │ snapshot    │
│             │             │             │             │
│ HNSW 索引   │ jieba 分词  │ 5 种 entity │ 每行一个    │
│ docker      │ 进程内      │ 倒排表      │ Chunk dict  │
│ volume      │ (重启就没) │ data/ref_   │ 用于重建    │
│ 持久化      │             │  graph.db   │ BM25        │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 🟢 查询流程（Query）

```
用户："桥梁防水谁负责？"
            │
            ▼
┌───────────────────────────────────────────┐
│ HybridRetriever.search(query, k=5)        │
│                                           │
│  1. embedder.embed(query) → 768d 向量     │
│                                           │
│  2. 并发 (ThreadPoolExecutor max=2):      │
│     ├─ qdrant.search(向量, 20)   ──┐      │
│     └─ bm25.search(query, 20)    ──┤      │
│                                    ▼      │
│  3. rrf_fuse([vec, bm25])                 │
│     按 1/(60+rank) 累加各 chunk           │
│     得到融合后排名                        │
│                                           │
│  4. (可选) reranker.rerank(top-30)        │
│     cross-encoder 精排 → top-8            │
│                                           │
│  5. 返回 list[SearchResult] (top-k)       │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ ClaudeCliAnswerer  (Answerer 接口)        │
│                                           │
│  1. build_prompt(query, chunks)           │
│     系统指令: "只能引用给定段落..."       │
│     <context_chunk file=".." page="">     │
│       Trackwork Contractor shall ...      │
│     </context_chunk>                      │
│     <question>桥梁防水谁负责？</question>  │
│                                           │
│  2. subprocess `claude -p` (订阅模式)     │
│                                           │
│  3. 输出原始答案 text                     │
└───────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────┐
│ postprocess (引用守约)                    │
│  - 正则提取所有 [filename p.X]            │
│  - 校验 (filename, page) 真实在 raw_      │
│    context 里                             │
│  - 删除伪造引用                           │
│  - 计算 confidence: high/medium/low       │
└───────────────────────────────────────────┘
            │
            ▼
Answer(
  text="桥梁防水由 Trackwork Contractor 负责...",
  citations=[("Contract DEMO(1of9) TQA.pdf", 42)],
  confidence="high",
  raw_context=[Chunk1, Chunk2, ...]
)
```

### 🟡 引用图谱旁路（RefGraph）

```
用户："图纸 T/PRJ/CWD/WS/2101A 被引用在哪？"
            │
            ▼
┌───────────────────────────────────────────┐
│ SqliteRefGraph.mentions_of(               │
│   "drawing", "T/PRJ/CWD/WS/2101A")       │
│                                           │
│  SELECT m.chunk_id, c.file, c.page        │
│  FROM mentions m JOIN chunks c            │
│  WHERE m.entity_type='drawing'            │
│    AND m.entity_value='T/PRJ/CWD/WS/2101A'│
│                                           │
│  → 毫秒级返回所有提到这张图的 chunks      │
└───────────────────────────────────────────┘
```

---

## 关键观察

### 1. 入库是慢的，查询是快的

- **入库**：4100 页 OCR ≈ 几小时（订阅模式 ~10s/页 + 并发）
- **查询**：单次 < 5 秒（不含 LLM 答题）

这是 RAG 的标准 trade-off：**预处理换查询速度**。

### 2. 每个箭头都是一个 Protocol 接口

任何一段流水线的具体实现都可以换。明天换 GPT-4 答题、换 pgvector 存向量、换 BGE-M3 embedding——业务代码不动。这是 [04 接口和实现](04-interface-and-impls.md) 要展开的事。

### 3. 三种持久化媒介

| 媒介 | 存什么 | 跨进程吗 |
|---|---|---|
| Qdrant Docker volume | 向量 + payload | ✅ |
| SQLite (`data/ref_graph.db`) | 实体倒排 | ✅ |
| JSONL (`data/chunks_snapshot.jsonl`) | 所有 Chunk 的副本 | ✅（用于重建 BM25）|
| BM25 内存 | tokens + 索引 | ❌（每次 CLI 启动重建）|

### 4. OCR 缓存让重跑无成本

OCR 是按页 SHA-256 缓存的（`data/ocr_cache/<hash>.text.txt`）。同一份 PDF 重跑 ingest，**OCR 阶段 100% cache hit，不再花钱/花时间**。

---

## 下一步阅读

- 想看代码长啥样 → [03 目录结构](03-directory-structure.md)
- 想理解"接口"是什么意思 → [04 接口和实现](04-interface-and-impls.md)
- 想细看某个环节 → 查 [00 目录](00-index.md) 跳到对应章节
