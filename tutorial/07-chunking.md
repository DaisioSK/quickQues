# 07 · 切块（Chunking）

> 前置章节：[06 PDF 解析与 OCR](06-pdf-parsing-and-ocr.md)
> 下一章 → [08 Embedding](08-embedding.md)

---

## 一句话

**Parser 给你每页几千字，Chunker 把它切成几百字一块的"知识卡片"——切得好坏直接决定检索质量。**

---

## 类比：包饺子

Parser 给你的是**一大块面**（一整页的文字）。Chunker 要把它切成饺子（chunks）——大小合适、馅料完整。

切法关键：
- **太小**：饺子皮包不住馅 → chunk 文字太少，embedding 抓不到完整意思
- **太大**：饺子破皮 → chunk 太长稀释主题，多个事实糊在一起
- **乱切**：把 Q 和 A 切开了 → 检索时只能拿到半边，答非所问
- **整齐切**：按自然结构（Q&A 边界、段落空行、表格行）切 → 完美

---

## 为什么不能用"滑动窗口"傻切

很多 RAG 教程示例代码长这样：

```python
# ❌ naive sliding window
def chunk(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks
```

每 500 字切一刀，相邻 50 字重叠。

**对小说、博客行，对合同**：

```
...前面是上一个 Q&A...
Question No.: ACME/TRACKWORK/15
Query: What is the warranty period?

←这里被切了一刀

Answer: 12 months from the date of...
```

问题：
- 用户问 "TRACKWORK/15 的保修期是多久？"
- 检索引擎会把 **包含 "TRACKWORK/15" 但没有 Answer 的上半截** 检出来
- 也会把 **下半截 Answer 但没有 Question No.** 检出来
- LLM 拿到这两个碎片，**没法组合出正确答案**

---

## 我们的做法：结构感知切块（QaAwareChunker）

[src/jcontract/impls/qa_chunker.py](../src/jcontract/impls/qa_chunker.py) 的核心算法：

### Step 1 — 拼页 + 记位置

```python
# 把所有页的 text 拼成一长串
full_text = "\n".join(p.text for p in pages)
# 同时维护 char_offset → page_num 的映射
# 让任何 chunk 都能知道它"起源于第几页"
```

### Step 2 — 找 Q&A 边界

正则扫描：

```python
_QUESTION_NO_RE = re.compile(
    r"^\s*Question\s*No[.:]?\s*[:.]?\s*([\w/\-]+)",
    re.IGNORECASE | re.MULTILINE,
)
```

匹配各种写法：
- `Question No.: ACME/TRACKWORK/16`
- `Question No.:ACME/TRACKWORK/16`
- `Question No ACME/TRACKWORK/16`
- `question no: 12`

**两个 "Question No." 之间的全部内容 = 一个 `qa_pair` chunk**。完整保留 Question + Query + Answer，不切开。

### Step 3 — Q&A 之间的内容按段落切

如果某段不是 Q&A（比如合同正文、定义段、表格）：

- 按**空行**分段
- 每段目标 **400-800 字符**
- 太短的段落（< 200 字符）跟下一段合并
- 太长的段落（> 1000 字符）从就近的句号/中文句号切

### Step 4 — 表格识别

如果一段含有 `|` 符号（Markdown 表格分隔符），标记为 `chunk_type="table"`，**不切**（表格切了就废）。

### Step 5 — 提取交叉引用

每个 chunk 扫一遍：

```python
_DRAWING_REF_RE = re.compile(r"(?:Drawing\s*No\.?\s*|Dwg\.?\s*)([\w/\-]+/\d+[A-Z]?)", ...)
_CLAUSE_REF_RE = re.compile(r"(?:Clause|Cl\.?)\s+(\d+(?:\.\d+)*)", ...)
```

匹配出来的填到 `chunk.drawing_refs` / `chunk.clause_refs`。

### Step 6 — Section / Clause 路径

扫到 `Section 7` / `Clause 7.3` 标题行时：
- 记下"当前所在的 section"
- 后续 chunk 的 `section_path` = "Section 7 > Clause 7.3"
- 遇到下一个 Section 标题，更新当前值

---

## 一个真实例子

输入（parser 输出的某页文字）：

```
SECTION 7 - PERMANENT WAY WORKS

Clause 7.3 Waterproofing

The waterproofing system shall comply with...

Question No.: ACME/TRACKWORK/16
Query: Who is responsible for the waterproofing of the bridge deck?
Rev A
Answer: The Trackwork Contractor shall be responsible for the supply and
installation of waterproofing system as specified in Clause 7.3.2 and
Drawing No. T/PRJ/CWD/WS/2101A.

Question No.: ACME/TRACKWORK/17
Query: ...
```

输出（chunker 切出的 chunks）：

```python
[
    Chunk(
        id="...:42:1",
        text="SECTION 7 - PERMANENT WAY WORKS\n\nClause 7.3 Waterproofing\n\nThe waterproofing system shall comply with...",
        chunk_type="paragraph",
        section_path="Section 7 > Clause 7.3",
        ...
    ),
    Chunk(
        id="...:42:2",
        text=(
            "Question No.: ACME/TRACKWORK/16\n"
            "Query: Who is responsible for the waterproofing of the bridge deck?\n"
            "Rev A\n"
            "Answer: The Trackwork Contractor shall be responsible for the supply and\n"
            "installation of waterproofing system as specified in Clause 7.3.2 and\n"
            "Drawing No. T/PRJ/CWD/WS/2101A."
        ),
        chunk_type="qa_pair",
        section_path="Section 7 > Clause 7.3",
        revision="Rev A",
        question_no="ACME/TRACKWORK/16",
        drawing_refs=["T/PRJ/CWD/WS/2101A"],
        clause_refs=["7.3.2"],
        ...
    ),
    Chunk(
        id="...:42:3",
        text="Question No.: ACME/TRACKWORK/17\nQuery: ...",
        chunk_type="qa_pair",
        question_no="ACME/TRACKWORK/17",
        ...
    ),
]
```

**注意每个 chunk 的元数据多么丰富**——这些是后续检索精确度的弹药。

---

## 切块的 4 个关键决策

### DECISION-1：Q&A 完整不切

哪怕一个 Q&A 块超过 2000 字符也不在中间切。理由：**Q 和 A 拆开就没用**。

但若 Q&A 长达 5000+ 字符（极少见），会被切成多个 chunk，`question_no` 字段**在每个子 chunk 里都填**，让"按 question_no 查"仍然能召回所有片段。

### DECISION-2：段落目标 400-800 字符

实证经验：
- mpnet embedding 模型在 200-1000 字符范围最稳
- 上下文质量 vs 检索精度的甜点是 ~500

### DECISION-3：chunk 的 page = 起始页

一个 chunk 可能跨页（比如某段从 p.42 末尾跨到 p.43 开头）。我们用**起始页**作为 chunk.page。

理由：**用户跳到 p.42 能看到这一段开头**，向下滚就能读完。反过来用结束页会让用户跳到一半，要往上翻。

### DECISION-4：正则容忍各种写法

合同写得乱七八糟：`Question No.:` / `Question No:` / `Question No`（无冒号）/ `QUESTION NO:`——正则全部接受。

代价：偶尔会误命中。比如某段文字里写 "we may have a question. no answer yet" 可能误判。**但实证比保守正则漏检的代价小**。

---

## 切块质量怎么验证

切块质量没有完美指标，但有间接信号：

1. **chunk 数量是否合理**：4100 页大概产出 10k-20k chunks。如果只有 1000 个说明切太大；如果 100k 说明切太碎。
2. **qa_pair 占比**：DEMO 这种 TQA 文档，qa_pair 应该是主导类型。如果都被切成 paragraph 说明正则失效。
3. **metadata 召回**：跑 `jcontract refs drawing T/PRJ/CWD/WS/2101A` 看能不能查到——查不到说明 drawing_refs 提取漏了。
4. **下游评测**：`jcontract evaluate` 跑 27 个 golden case，Recall@5 < 0.5 时切块大概率有问题。

---

## 未来可能的改进（FORESHADOW）

- **学习型 chunker**：用 LLM 判断"这段应不应该和下段合并"，而不是规则正则
- **结构感知 token 截断**：换更大上下文的 embedding（bge-m3 8k tokens），允许更长 chunk
- **多粒度索引**：同一份文档同时存"细 chunk"和"粗 chunk"，按查询类型分发

但**当前规则版已经够 DEMO Phase 1**，先用着。

---

## 下一步阅读

- chunk 完后变向量 → [08 Embedding](08-embedding.md)
- 向量存哪里 → [09 Vector Store](09-vector-store.md)
- chunk 怎么被检索 → [10 BM25](10-bm25-keyword.md)
