# 05 · Chunk 解剖

> 前置章节：[04 接口和实现](04-interface-and-impls.md)
> 下一章 → [06 PDF 解析与 OCR](06-pdf-parsing-and-ocr.md)

---

## 一句话

**Chunk 是这个系统的"原子单位"——存进数据库的是 Chunk，搜出来的是 Chunk，答案引用的是 Chunk。**

---

## 类比：图书馆的卡片柜

老式图书馆里每本书都有一张**目录卡片**，写着：
- 书名、作者
- 馆藏位置（A 区 3 排 2 列）
- 主题、关键词
- 内容摘要

读者**不直接面对几万本书**，先翻卡片柜，找到卡片，按位置取书。

**Chunk 就是这种卡片**，但更细——一本书拆成很多卡片，每张卡片对应一段内容：

| 现实图书馆 | j-contract |
|---|---|
| 一本书 | 一份 PDF |
| 一张卡片 | 一个 Chunk |
| 卡片上的内容摘要 | `chunk.text` |
| 馆藏位置 | `chunk.file` + `chunk.page` |
| 卡片主题 | `chunk.chunk_type` |
| 卡片关键词 | `chunk.drawing_refs` / `chunk.clause_refs` 等 |

---

## Chunk 的完整字段

定义在 [src/jcontract/interfaces/schema.py:38](../src/jcontract/interfaces/schema.py#L38)：

```python
@dataclass
class Chunk:
    id: str                                          # 唯一 ID
    text: str                                        # 实际内容
    file: str                                        # PDF 文件名
    page: int                                        # 第几页（从 1 算）
    chunk_type: ChunkType                            # 4 种之一
    section_path: str | None = None                  # "Section 7 > Clause 7.3"
    revision: str | None = None                      # "Rev A" / "Revision 0"
    drawing_refs: list[str] = field(default_factory=list)
    clause_refs: list[str] = field(default_factory=list)
    question_no: str | None = None                   # "ACME/TRACKWORK/16"
    bbox: tuple[float, float, float, float] | None = None  # UI 高亮用
```

逐字段讲：

### `id` — 身份证

格式约定：`<file_stem>:<page>:<idx>`，例如：
```
ContractDEMO(1of9)TQA:42:7
```

意思：DEMO(1of9) TQA 这份 PDF、第 42 页、第 7 个 chunk。

**唯一性保证**：重新 ingest 同一份 PDF 不会产生重复——Bm25Index 和 QdrantStore 都按 id 去重（idempotent）。

### `text` — 内容本体

实际的文字。**长度控制在 400-800 字符**（中文：~150-300 字；英文：~70-140 词）。

为什么这个长度？两头权衡：
- **太短**（< 200 字符）：上下文信息不够，embedding 抓不到语义
- **太长**（> 1500 字符）：稀释主题，多个事实混在一起；并且**embedding 模型有 token 上限**（mpnet 是 512 tokens）

### `file` + `page` — 溯源

**全系统的引用基础**。所有 Answer 引用都得能落回到 `(file, page)`。

页码是 1-indexed（跟 PDF 阅读器对齐），不是 0-indexed。

### `chunk_type` — 4 种长相

这是**枚举字段**，固定 4 个值：

| 值 | 中文 | 长啥样 |
|---|---|---|
| `qa_pair` | 问答对 | `Question No.: TQA-001\nAnswer: ...` |
| `table` | 表格 | `\| col1 \| col2 \|`（Markdown 表格） |
| `paragraph` | 普通段落 | 多行连贯正文 |
| `drawing` | 工程图说明 | OCR 出来的工程图文字描述 |

为什么要分？因为**不同类型应该不同处理**：
- qa_pair 检索时要保持完整（拆开问和答没用）
- table 渲染到 UI 时是表格组件不是段落
- drawing 通常文字稀疏但有重要 metadata（Drawing No.）

加新类型要走独立 sub-sprint（接口变更属于"宪法"修改）。

### `section_path` — 结构定位

格式：`"Section 7 > Clause 7.3"`

由 chunker 扫到上文 Section / Clause 标题时填入。**让答案能说"根据 Section 7 第 3 条 ..."**——比纯页码更有意义。

### `revision` — 版次

`"Rev A"`、`"Revision 0"` 等。建筑合同的 Q&A 经常被修订；同一个 Question No. 可能有多个版次，**revision 让检索能区分**。

### `drawing_refs` / `clause_refs` — 交叉引用

这两个是**列表**，可能为空也可能多个。

例子：某 Chunk 文字里提到 "...detailed in Drawing No. T/PRJ/CWD/WS/2101A and Clause 7.3..."

那么 chunker 会填：
```python
drawing_refs = ["T/PRJ/CWD/WS/2101A"]
clause_refs = ["7.3"]
```

这些**喂给 RefGraph 建倒排表**，让"图纸 X 被谁引用"这种查询毫秒级返回。

### `question_no` — Q&A 编号

只有 `chunk_type == "qa_pair"` 时填。

格式因合同而异：`ACME/TRACKWORK/16`、`TQA-001`、`Q12` 都见过。

### `bbox` — UI 高亮（Phase 5）

PDF 页面上的边界框 `(x0, y0, x1, y1)`，给前端 PDF.js 高亮原文用。**Phase 5 才用，目前为 None**。

---

## 一个真实的 Chunk 长什么样

```python
Chunk(
    id="ContractDEMO(1of9)TQA:42:3",
    text=(
        "Question No.: ACME/TRACKWORK/16\n"
        "Query: Who is responsible for the waterproofing of the bridge deck?\n"
        "Answer: The Trackwork Contractor shall be responsible for the supply "
        "and installation of waterproofing system as specified in Clause 7.3.2 "
        "and Drawing No. T/PRJ/CWD/WS/2101A."
    ),
    file="Contract DEMO(1of9) TQA.pdf",
    page=42,
    chunk_type="qa_pair",
    section_path=None,
    revision="Rev A",
    drawing_refs=["T/PRJ/CWD/WS/2101A"],
    clause_refs=["7.3.2"],
    question_no="ACME/TRACKWORK/16",
    bbox=None,
)
```

注意几件事：
- `text` 完整保留了 Question/Answer 结构
- 三个 ref 字段都填了（drawing / clause / question_no）
- `revision="Rev A"` —— 这个 Q&A 是修订版
- 这个 Chunk 进了 4 个地方：Qdrant + BM25 + RefGraph + JSONL snapshot

---

## 为什么 frozen=False（可变）

注意 dataclass 装饰器是 `@dataclass`，不是 `@dataclass(frozen=True)`。

[schema.py 注释](../src/jcontract/interfaces/schema.py#L46) 说原因：

> Mutable (frozen=False) so impls/qa_chunker.py can compose chunks incrementally during a single chunk() call before yielding. Treat as immutable after the chunker returns.

翻译：**chunker 在构造 chunk 时需要逐步填充字段**（先 text 再扫 drawing_refs 再扫 clause_refs），如果一开始就 frozen，就得反复 `replace()` 创建新对象，性能差且代码丑。

**约定**：chunker 返回之后，下游所有代码**当它是 immutable 的**，不允许修改。

---

## SearchResult vs Chunk vs Answer

很多地方会混淆这三个类型。串一遍：

```python
# 检索回来：Chunk + score
@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk        # ← 卡片本体
    score: float        # ← 这次检索的得分（vector cosine / BM25 / RRF / reranker）

# 答完题：text + 引用 + raw chunks
@dataclass(frozen=True)
class Answer:
    text: str                                # 中文答案
    citations: list[tuple[str, int]]         # [("file.pdf", 42), ...]
    confidence: Confidence                   # "high" / "medium" / "low"
    raw_context: list[Chunk]                 # 喂给 LLM 的 chunks（审计用）
```

**串起来**：
```
HybridRetriever.search()  → list[SearchResult]   （每个含 Chunk + score）
                          ↓
                  取 .chunk 部分
                          ↓
Answerer.answer()         → Answer              （含 list[Chunk] 在 raw_context）
                          ↓
                  postprocess 校验 citations 对应到 raw_context 里的 (file, page)
```

---

## Chunk 设计的核心理念

1. **自带溯源**：每个 Chunk 都能独立回答"我从哪来"（file/page），不依赖外部状态。
2. **结构化元数据**：除了 text，还有 chunk_type / refs / question_no——**让检索可以走 metadata filter**（"只搜 qa_pair 类型"、"只搜 Rev A 的"）。
3. **冗余设计**：drawing_refs 既在 Chunk 里又在 RefGraph 里——前者用于答案生成，后者用于精确查询。**不追求规范化数据库范式**，追求各种查询场景都好用。

---

## 下一步阅读

- 想看 Chunk 是怎么"生"出来的 → [06 PDF 解析与 OCR](06-pdf-parsing-and-ocr.md) + [07 切块](07-chunking.md)
- 想看 Chunk 怎么变向量 → [08 Embedding](08-embedding.md)
- 想看 Chunk 怎么被搜出来 → [09 Vector Store](09-vector-store.md)
