# 17 · 术语表

> 前置章节：无 — 查字典用
> 回到 → [00 目录](00-index.md)

按字母 / 拼音排序。点链接跳到详细章节。

---

## A

### Answerer
LLM 答题层的接口。输入"问题 + chunks"，输出"中文答案 + 引用"。当前默认是 `ClaudeCliAnswerer`（订阅模式）。详见 [14](14-answerer-and-citations.md)。

### asyncio
Python 标准库的异步编程框架。`batch-ingest` 用它做多份 PDF 并发处理。

---

## B

### batch-ingest
CLI 子命令，批量入库多份 PDF。带并发、断点续跑、预算守门。详见 [15](15-cli-walkthrough.md)。

### bbox
Bounding box，矩形边界框 `(x0, y0, x1, y1)`。Chunk 的可选字段，Phase 5 UI 高亮原文用。当前为 None。

### BGE
Beijing General Embedding，北京智源研究院（BAAI）的嵌入模型系列。
- **bge-m3**：多语言、多功能（dense + sparse + ColBERT）、多粒度。我们的 embedding **计划升级目标**。
- **bge-reranker-v2-m3**：多语言 cross-encoder reranker。**已在用**。

详见 [08](08-embedding.md) / [12](12-cross-encoder-reranker.md)。

### bi-encoder
双编码器。问题和文档**分开**编码成向量，靠 cosine 距离匹配。可以预先索引文档，速度快但质量不如 cross-encoder。mpnet 是 bi-encoder。详见 [12](12-cross-encoder-reranker.md)。

### BM25
1994 年的关键词检索打分公式。基于 TF-IDF 改进，按"词频 + 逆文档频率 + 文档长度"算分。我们用 `rank-bm25` 库 + jieba 分词。详见 [10](10-bm25-keyword.md)。

---

## C

### chunk
全系统的"原子单位"。一段文字 + 完整溯源元数据（file、page、chunk_type、各种 refs）。详见 [05](05-chunk-anatomy.md)。

### chunk_type
枚举字段，4 种：`qa_pair` / `table` / `paragraph` / `drawing`。决定 chunk 的"长相"。详见 [05](05-chunk-anatomy.md)。

### Chunker
切块层的接口。输入 ParsedPage，输出 Chunk。默认实现是 `QaAwareChunker`（结构感知）。详见 [07](07-chunking.md)。

### claude-cli-vision
PDFParser 的一种实现，用 `claude -p` subprocess 调订阅模式 OCR。**当前默认**。详见 [06](06-pdf-parsing-and-ocr.md)。

### Claude Code
Anthropic 出的 CLI 工具，可以通过订阅（Max/Pro）使用 Claude 模型，不需要 API key。本项目所有 LLM 调用都通过它。

### Clause
建筑合同的条款编号，如 `7.3.2`。是 RefGraph 的 5 种 entity 之一。

### confidence
Answer 的字段，`"high"` / `"medium"` / `"low"`。按 citation 数量 + 是否 fallback 算出。

### cosine similarity
余弦相似度。两个向量夹角的余弦值，[-1, 1]，**1 = 一模一样**。embedding 检索用它衡量"语义距离"。详见 [09](09-vector-store.md)。

### cross-encoder
交叉编码器。问题和文档**拼一起**编码，一次模型前向出"匹配分"。不能预先索引，但质量比 bi-encoder 高。我们用 bge-reranker-v2-m3。详见 [12](12-cross-encoder-reranker.md)。

---

## D

### dev-contract
本项目的"开发宪法"，所有 sub-sprint 必须遵守。位于 `dev-contract/` 目录。

### DI（Dependency Injection）
依赖注入。业务代码不 `new` 具体类，而是接受外部传入的接口对象。让"换实现 = 改一行装配"。详见 [04](04-interface-and-impls.md)。

### Drawing No.
建筑图纸编号，如 `T/PRJ/CWD/WS/2101A`。是 RefGraph 的 5 种 entity 之一。

---

## E

### Embedder
嵌入模型层的接口。输入 list[str]，输出 list[list[float]]（向量列表）。默认 `FastEmbedEmbedder`（mpnet）。详见 [08](08-embedding.md)。

### embedding
把文字翻译成"语义坐标"（向量）的过程。语义相近 → 向量相近。详见 [08](08-embedding.md)。

### Entity（RefGraph 语境）
RefGraph 的"被引用对象"。5 种类型：drawing、clause、question_no、section、revision。详见 [13](13-ref-graph.md)。

### EvalCase
评测集里的一道题。含 question、expected_sources、expected_keywords、category。详见 [16](16-evaluation.md)。

---

## F

### fastembed
Qdrant 公司的轻量 embedding 库，ONNX runtime 跑模型，不依赖 PyTorch。我们用它跑 mpnet。详见 [08](08-embedding.md)。

### FlagEmbedding
BAAI 出的 embedding 库，bge-m3 / bge-reranker 的官方推荐运行环境。Phase 2 升级 bge-m3 时可能引入。

### frozen=True / False
Python `@dataclass` 装饰器的参数。`frozen=True` = immutable，`frozen=False` = 可修改。Chunk 是 frozen=False（chunker 需要逐步填字段）。详见 [05](05-chunk-anatomy.md)。

---

## G

### golden_cases
评测集文件 `src/jcontract/eval/golden_cases.jsonl`，27 个测试 case。详见 [16](16-evaluation.md)。

---

## H

### Hybrid Retrieval / HybridRetriever
混合检索。同时跑向量检索 + BM25 + （可选）reranker，用 RRF 融合。**本项目的核心检索类**。详见 [11](11-rrf-fusion.md)。

### HNSW
Hierarchical Navigable Small World，Qdrant 默认的向量索引算法。O(log N) 找近邻。详见 [09](09-vector-store.md)。

---

## I

### IDF（Inverse Document Frequency）
BM25 的"逆文档频率"部分。词越罕见，IDF 越高，命中加分越多。详见 [10](10-bm25-keyword.md)。

### ingest
入库。CLI 子命令 `ingest` / `batch-ingest`，把 PDF 处理后存进各种数据库。

### in-memory（内存中）
不持久化到磁盘，全在进程内存里。我们的 BM25 索引是 in-memory（每次 CLI 启动从 JSONL snapshot 重建）。详见 [10](10-bm25-keyword.md)。

### Interface（接口）
本项目特指 `src/jcontract/interfaces/` 下的 Protocol。10 个核心抽象。详见 [04](04-interface-and-impls.md)。

---

## J

### DEMO
本项目的试点合同——新加坡 MRT Trackwork 工程，编号 DEMO。共 9 份 PDF，~4100 页。

### jieba
Python 中文分词库。我们用它给 BM25 切词。**也用它处理英文**（jieba 对 ASCII 友好）。详见 [10](10-bm25-keyword.md)。

### JSONL（JSON Lines）
每行一个 JSON 对象的文件格式。我们用它存 chunks snapshot（用于重建 BM25）和 ingest checkpoint。可追加、流式读，比 JSON 更适合日志式数据。

---

## K

### k（in RRF）
RRF 公式 `1/(k+rank)` 里的常数。我们用 60（Cormack 2009 推荐值）。

### k（top-k）
检索返回的数量。`search --k 5` 返回前 5 个 hits。

### KeywordIndex
BM25 关键词索引层的接口。默认 `Bm25Index`。详见 [10](10-bm25-keyword.md)。

---

## L

### lazy load
延迟加载。**对象构造时不付重资源代价，第一次真用时才加载**。fastembed、bge_reranker 都用这个模式。详见 [12](12-cross-encoder-reranker.md)。

### LLM
Large Language Model。本项目里主要指 Claude（Sonnet / Haiku）。

---

## M

### mention
RefGraph 里 (chunk, entity) 的连接关系。表示"这个 chunk 提到了这个 entity"。详见 [13](13-ref-graph.md)。

### metadata
Chunk 上除了 text 之外的所有字段：file、page、chunk_type、refs、question_no 等。**检索系统的弹药库**。

### mpnet
我们用的多语言 embedding 模型：`paraphrase-multilingual-mpnet-base-v2`。768 维。详见 [08](08-embedding.md)。

### mypy
Python 静态类型检查器。本项目"三件套"门之一（必须通过才能合流）。

---

## O

### OCR
Optical Character Recognition，光学字符识别。把图片里的文字识别出来。本项目用 Claude Vision 做 OCR。详见 [06](06-pdf-parsing-and-ocr.md)。

### ONNX
Open Neural Network Exchange，跨框架的神经网络模型格式。fastembed 用 ONNX runtime 跑 embedding，不需要 PyTorch。

---

## P

### ParsedPage
PDFParser 输出的每页结构。含 `page_num`、`text`、`tables`。详见 [05](05-chunk-anatomy.md)。

### payload（Qdrant 语境）
Qdrant Point 上挂的元数据 JSON。我们塞了整个 Chunk 的字段。详见 [09](09-vector-store.md)。

### PDFParser
PDF 解析层的接口。三种实现：`PyPdfParser`（文本）、`ClaudeVisionParser`（API OCR）、`ClaudeCliVisionParser`（订阅 OCR）。详见 [06](06-pdf-parsing-and-ocr.md)。

### postprocess
答案后处理。剥假引用、算 confidence。详见 [14](14-answerer-and-citations.md)。

### Protocol
Python typing.Protocol，结构化类型匹配。不需要继承，方法签名匹配就行。本项目所有接口都用 Protocol。详见 [04](04-interface-and-impls.md)。

### prompt injection
提示词注入攻击。恶意输入"忽略以上指令"试图劫持 LLM。我们用 XML 标签 + 字符转义防御。详见 [14](14-answerer-and-citations.md)。

### pyproject.toml
Python 项目的总配置文件（TOML 格式）。详见 [03](03-directory-structure.md)。

### pypdf
纯 Python PDF 库，文本提取。对扫描件无效。

### pypdfium2
基于 pdfium（Chrome 用的 PDF 库）的 Python 包。**把 PDF 渲染成图片**。OCR 流程的第一步。

---

## Q

### Qdrant
Rust 写的开源向量数据库。我们用它存 chunk 向量 + payload。详见 [09](09-vector-store.md)。

### qa_pair
Chunk 的一种类型，建筑合同的"问答对"。详见 [05](05-chunk-anatomy.md)。

### Question No.
建筑合同的澄清问题编号，如 `ACME/TRACKWORK/16`。是 RefGraph 的 5 种 entity 之一。

---

## R

### RAG（Retrieval-Augmented Generation）
检索增强生成。LLM 答题前先去外部库检索相关内容当上下文。**本项目就是 RAG 系统**。

### rank-bm25
Python BM25 库。我们的 BM25 实现基础。详见 [10](10-bm25-keyword.md)。

### Recall@K
评测指标："top-K 检索结果里命中正确文档的比例"。详见 [16](16-evaluation.md)。

### RefGraph
引用图谱。SQLite 倒排表，按 entity 精确查 mentions。详见 [13](13-ref-graph.md)。

### Reranker
精排层的接口。对召回的候选用 cross-encoder 重新打分。默认 `BgeReranker`。详见 [12](12-cross-encoder-reranker.md)。

### Revision
版次，如 `Rev A` / `Revision 0`。是 RefGraph 的 5 种 entity 之一。

### RRF（Reciprocal Rank Fusion）
倒数排名融合。把多路检索按"1/(60+rank)"累加合并。详见 [11](11-rrf-fusion.md)。

### ruff
Python linter + formatter。本项目"三件套"门之一。

---

## S

### Section
合同的章节编号，如 `Section 7`。是 RefGraph 的 5 种 entity 之一。

### sentence-transformers
HuggingFace 生态的 bi-encoder / cross-encoder 库。我们用它跑 bge-reranker-v2-m3。带 PyTorch ~2GB。详见 [12](12-cross-encoder-reranker.md)。

### SearchResult
检索结果的数据类。含 `chunk: Chunk` + `score: float`。详见 [05](05-chunk-anatomy.md)。

### SHA-256
密码学哈希函数。我们用它做 OCR 缓存 key（按渲染 JPEG 字节 hash）。详见 [06](06-pdf-parsing-and-ocr.md)。

### snapshot
JSONL 快照。`data/chunks_snapshot.jsonl` 存所有 chunks，用于跨 CLI 启动重建 BM25 索引。详见 [10](10-bm25-keyword.md)。

### sparse vector
稀疏向量，大多数元素为 0 的高维向量（典型词袋表示）。bge-m3 能同时输出稀疏向量做 BM25-like 检索。

### Sprint / Sub-sprint
本项目契约的开发单位。Phase（方向）→ Sprint（功能）→ Sub-sprint（可观测落地单元）。

### SQLite
单文件嵌入式数据库，Python 标准库自带。我们用它存 RefGraph。详见 [13](13-ref-graph.md)。

### subprocess
Python 启动外部进程的模块。我们用它调 `claude -p` 命令行。

---

## T

### table
Chunk 的一种类型，表格内容。不切，按行整段保留。详见 [05](05-chunk-anatomy.md)。

### TF（Term Frequency）
词频，BM25 公式的核心成分之一。详见 [10](10-bm25-keyword.md)。

### TOML
Tom's Obvious, Minimal Language。配置文件格式，比 YAML 不易出错。`pyproject.toml` 用这个格式。详见 [03](03-directory-structure.md)。

### TQA
Technical Query & Answer，建筑合同的"澄清问答"文档。DEMO 的 `Contract DEMO(1of9) TQA.pdf` 就是这类。

### Trackwork
DEMO 项目的具体工程类型——MRT 铁路轨道工程。

---

## U

### uv
新一代 Python 包管理器（Rust 写的，比 pip 快 10-100 倍）。本项目唯一指定的包管理器。

### uv.lock
uv 自动生成的精确依赖版本快照。**必须 commit 进 git**。

---

## V

### VectorStore
向量库层的接口。默认 `QdrantStore`。详见 [09](09-vector-store.md)。

### vision parser
本项目里特指 `ClaudeVisionParser` / `ClaudeCliVisionParser` 这两个用 Claude Vision 做 OCR 的 PDFParser 实现。详见 [06](06-pdf-parsing-and-ocr.md)。

---

## X

### XML（in prompt）
我们的 prompt 用 XML 标签包裹 context 和 question。Anthropic 推荐这种结构，比 Markdown 更稳定。详见 [14](14-answerer-and-citations.md)。

---

## 缩写表

| 缩写 | 全称 | 中文 |
|---|---|---|
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| OCR | Optical Character Recognition | 光学字符识别 |
| BM25 | Best Matching 25 | (无中文名) |
| BGE | BAAI General Embedding | 北京智源通用嵌入 |
| IDF | Inverse Document Frequency | 逆文档频率 |
| TF | Term Frequency | 词频 |
| LLM | Large Language Model | 大语言模型 |
| CLI | Command Line Interface | 命令行界面 |
| API | Application Programming Interface | 应用程序接口 |
| HNSW | Hierarchical Navigable Small World | 分层导航小世界图 |
| ONNX | Open Neural Network Exchange | 开放神经网络交换 |
| DI | Dependency Injection | 依赖注入 |
| MVP | Minimum Viable Product | 最小可行产品 |
| TQA | Technical Query & Answer | 技术澄清问答 |
| TSA | Temporary Staging Area | 临时存放区（DEMO 术语） |

---

回到 [00 目录](00-index.md)。
