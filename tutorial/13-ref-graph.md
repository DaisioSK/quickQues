# 13 · 引用图谱 RefGraph

> 前置章节：[12 Cross-encoder](12-cross-encoder-reranker.md)
> 下一章 → [14 Answerer 与引用守约](14-answerer-and-citations.md)

---

## 一句话

**专治"编号查询"——图纸 No.、Clause 号、Question No. 这种唯一标识，用 SQLite 倒排表，毫秒级精确返回。**

---

## 类比：电话黄页

向量检索 + BM25 像**"找一家好吃的川菜馆"**——根据"川菜"语义找相关餐厅，再用关键词排序。

但如果用户问"**电话 010-1234-5678 是哪家店？**"——黄页才是对的工具。电话号唯一、精确、有就有，没就没。

建筑合同里有大量这种"电话号"：
- **Drawing No.**：`T/PRJ/CWD/WS/2101A`（图纸编号）
- **Clause**：`7.3.2`（条款号）
- **Question No.**：`ACME/TRACKWORK/16`（澄清问题号）
- **Section**：`Section 7`
- **Revision**：`Rev A`

这些编号查询用向量检索会出错（"2101A" 和 "2102A" 向量相近，会混淆），用 BM25 也不可靠（分词可能把 `T/PRJ/...` 切成乱七八糟）。

→ **必须精确查**。RefGraph 就是这张"黄页"。

---

## 为什么叫"图谱"

它本质上是一个**二部图**（bipartite graph）：

```
                    mentions
       ┌──────────┐    ┌────────────┐
       │   Chunk  │ ───►│   Entity   │
       │  (一段卡片) │     │ (一个编号)   │
       └──────────┘    └────────────┘
       chunk-A ────► T/PRJ/CWD/WS/2101A
       chunk-A ────► Clause 7.3.2
       chunk-A ────► ACME/TRACKWORK/16
       chunk-B ────► T/PRJ/CWD/WS/2101A    ← 不同 chunk 提到同一图纸
       chunk-B ────► Section 7
       ...
```

**"谁提到谁"** 是图的边。可以正向查（这个 chunk 提了哪些编号？）也可以反向查（这个编号被哪些 chunk 提及？）。

**反向查最有用**：用户提"图纸 T/PRJ/CWD/WS/2101A 涉及哪些章节？"——RefGraph 直接给你**所有提及它的 chunk + 在哪一页**。

---

## 5 种 Entity 类型

[src/jcontract/impls/sqlite_ref_graph.py:67](../src/jcontract/impls/sqlite_ref_graph.py#L67)：

```python
ENTITY_DRAWING = "drawing"          # T/PRJ/CWD/WS/2101A
ENTITY_CLAUSE = "clause"            # 7.3.2
ENTITY_QUESTION_NO = "question_no"  # ACME/TRACKWORK/16
ENTITY_SECTION = "section"          # Section 7
ENTITY_REVISION = "revision"        # Rev A
```

这 5 种是建筑合同 DEMO 实测出来的"高价值编号"。其他领域可能不同（医学合同可能要加 ICD code，金融可能要加合同流水号）。

加新 entity 类型只要：
1. 加常量
2. 让 chunker 在元数据里产出对应字段
3. RefGraph 的 `index()` 把字段索引进去

不需要改 schema（mention 表是泛型 `(entity_type, entity_value)`）。

---

## SQLite Schema

为什么 SQLite：

| 优点 | 说明 |
|---|---|
| Python 标准库 | `import sqlite3`，**0 新依赖** |
| 单文件持久化 | `data/ref_graph.db`，可备份、可移动 |
| ACID | 索引时崩溃也不会半截损坏 |
| 索引能力强 | `(type, value)` 复合索引 → sub-ms 查 |
| 易替换 | 量大了切 Neo4j / DuckDB，Protocol 抽掉细节 |

Schema（简化）：

```sql
-- Entity 表：所有不同编号的去重列表
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,     -- 'drawing' / 'clause' / ...
    value TEXT NOT NULL,    -- 'T/PRJ/CWD/WS/2101A' / '7.3.2' / ...
    UNIQUE(type, value)
);

-- Chunk 投影表：从 chunk 提取 (id, file, page, chunk_type)，让查询不用回 Qdrant
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    file TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk_type TEXT NOT NULL
);

-- Mention 表：(chunk, entity) 多对多关系
CREATE TABLE mentions (
    chunk_id TEXT,
    entity_id INTEGER,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE INDEX idx_mentions_entity ON mentions(entity_id);
CREATE INDEX idx_entities_type_value ON entities(type, value);
```

---

## RefGraph 怎么填

### 从 Chunker 来的元数据已经够了

[Chunker](07-chunking.md) 早在切块时就用正则提取了 drawing_refs / clause_refs / question_no 等。**RefGraph 不重新提取**，直接消费这些字段：

```python
def index(self, chunks: list[Chunk]) -> None:
    for chunk in chunks:
        # 写 chunks 投影表
        cur.execute("INSERT OR REPLACE INTO chunks ...",
                    (chunk.id, chunk.file, chunk.page, chunk.chunk_type))

        # 写 entities + mentions
        for dn in chunk.drawing_refs:
            self._link(chunk.id, ENTITY_DRAWING, dn)
        for cl in chunk.clause_refs:
            self._link(chunk.id, ENTITY_CLAUSE, cl)
        if chunk.question_no:
            self._link(chunk.id, ENTITY_QUESTION_NO, chunk.question_no)
        # ... section, revision
```

**单一真理源原则**：Chunker 是正则的唯一来源。如果未来 chunker 提取改进了，RefGraph 自动受益。

---

## 怎么查（API）

[src/jcontract/interfaces/ref_graph.py](../src/jcontract/interfaces/ref_graph.py)：

```python
class RefGraph(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...
    def mentions_of(self, entity_type: str, entity_value: str) -> list[Mention]: ...
    def entities_in(self, chunk_id: str) -> list[Entity]: ...
    def stats(self) -> dict[str, int]: ...
```

### `mentions_of` — 反向查（最常用）

```python
graph.mentions_of("drawing", "T/PRJ/CWD/WS/2101A")
# → [
#     Mention(chunk_id="...:42:3", file="Contract DEMO(1of9) TQA.pdf", page=42, chunk_type="qa_pair"),
#     Mention(chunk_id="...:88:1", file="Contract DEMO(4of9)...pdf", page=88, chunk_type="paragraph"),
#     ...
#   ]
```

回答 **"这张图被哪些段落引用？"**。SQL：
```sql
SELECT c.chunk_id, c.file, c.page, c.chunk_type
FROM mentions m
JOIN entities e ON m.entity_id = e.id
JOIN chunks c ON m.chunk_id = c.chunk_id
WHERE e.type='drawing' AND e.value='T/PRJ/CWD/WS/2101A';
```

sub-ms 返回（索引精准）。

### `entities_in` — 正向查

```python
graph.entities_in("ContractDEMO(1of9)TQA:42:3")
# → [
#     Entity(type="drawing", value="T/PRJ/CWD/WS/2101A"),
#     Entity(type="clause", value="7.3.2"),
#     Entity(type="question_no", value="ACME/TRACKWORK/16"),
#   ]
```

回答 **"这个 chunk 提了哪些编号？"**。用于"图谱跳一跳"——拿到一个相关 chunk 后，自动把它引用的图纸/条款的其他 chunks 也拉进上下文。

### `stats` — 摘要

```python
graph.stats()
# → {
#     "total_entities": 234,
#     "total_mentions": 1820,
#     "drawing_count": 89,
#     "clause_count": 67,
#     "question_no_count": 41,
#     "section_count": 12,
#     "revision_count": 25,
#   }
```

入库后看一眼就能发现 chunker 提取是否合理（drawing_count == 0 一定是 regex 出问题）。

---

## CLI 入口：`jcontract refs`

```bash
# 查图纸被谁引用
uv run jcontract refs drawing T/PRJ/CWD/WS/2101A

# 查条款 7.3 被谁引用
uv run jcontract refs clause 7.3

# 查 Question No. 是哪段
uv run jcontract refs question_no ACME/TRACKWORK/16

# 查统计
uv run jcontract refs stats
```

输出列出所有 mention 的 chunks（含 file/page/type）。

---

## 跨文档查询（RefGraph 的杀手锏）

最有价值的场景：**同一个图纸在多份合同里出现**。

```bash
$ uv run jcontract refs drawing T/PRJ/CWD/WS/2101A

Mentions of drawing 'T/PRJ/CWD/WS/2101A':
  - Contract DEMO(1of9) TQA.pdf p.42  [qa_pair]
  - Contract DEMO(1of9) TQA.pdf p.88  [qa_pair]
  - Contract DEMO(1of9)Consol.pdf p.103 [paragraph]
  - Contract DEMO(4of9)2of2Part1of2Consol.pdf p.512 [drawing]
  - Contract DEMO(4of9)2of2Part1of2Consol.pdf p.513 [drawing]
```

这种"哪份文件、哪几页提到这张图"用 BM25 也能勉强做，但：
- **BM25 会漏命中**（jieba 对 `T/PRJ/CWD/WS/2101A` 切碎，匹配率不稳）
- **BM25 不能区分 mention 类型**（qa_pair 提到 vs paragraph 提到的语义不同）

RefGraph 是**结构化的、确定性的、精确的**。

---

## 与 Hybrid 检索的关系

RefGraph **不在** HybridRetriever 的主流程里。它是个**旁路**：

```
普通问题: "桥梁防水谁负责"
  → HybridRetriever (vector + BM25 + RRF + 可选 rerank)
  → Answerer

编号问题: "图纸 X 涉及哪些条款"
  → RefGraph.mentions_of  → 直接返回 chunks
  → 可选传给 Answerer 用上下文写答案
```

未来可能合并到主流程：检测到问题里有像 Drawing No. 的字符串 → 自动把 RefGraph 的 mentions 加进 candidates → 再 RRF 融合。**Phase 3 调优时再说**。

---

## 关键决策（DECISION）

### 为什么不用 Neo4j

Neo4j 是图数据库，做"跳几跳"的图遍历强。但：
- 当前用例都是"一跳"（A 提到 X → X 被谁提）
- Neo4j 是独立服务，运维多一个组件
- 单项目 10^4 mention 规模，SQLite 完全够

未来如果做"图纸 X → Clause Y → Section Z"这种多跳推理，再切 Neo4j。

### 为什么投影 chunks 表（denormalised）

`mentions_of` 返回的 Mention 含 file/page/chunk_type——这些原本在 Qdrant payload 里。如果不做投影，查询要回 Qdrant 取 → 增加网络延迟。

投影 4 个字段约 30 字节/chunk × 20k chunks ≈ 600KB。空间换时间。

### 为什么 INSERT OR IGNORE / INSERT OR REPLACE

`index()` 可能被重复调用（相同 chunks 重 ingest）。SQLite 的：
- `INSERT OR IGNORE`（entities、mentions）：已有就跳过，幂等
- `INSERT OR REPLACE`（chunks）：可能 chunk 重切了，需要更新 metadata

---

## 调试技巧

直接打开 SQLite 看：

```bash
sqlite3 data/ref_graph.db
sqlite> .schema
sqlite> SELECT type, COUNT(*) FROM entities GROUP BY type;
sqlite> SELECT * FROM mentions LIMIT 10;
sqlite> SELECT e.value, COUNT(m.chunk_id) AS cnt
        FROM entities e LEFT JOIN mentions m ON e.id = m.entity_id
        WHERE e.type='drawing'
        GROUP BY e.value ORDER BY cnt DESC LIMIT 10;
```

最后一条会告诉你"哪张图纸被引用最多"——往往是合同的核心图。

---

## 下一步阅读

- 检索回来 chunks 怎么生成答案 → [14 Answerer](14-answerer-and-citations.md)
- 全套命令怎么用 → [15 CLI 命令](15-cli-walkthrough.md)
