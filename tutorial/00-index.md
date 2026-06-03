# j-contract 小白教程目录

> 写给：不熟悉 RAG / 向量检索 / Python 工程化的人，从零建立这个项目的认知。
> 风格：**先打比方，再讲技术**。每章独立可读，不要求顺序。
> 推荐阅读路径见每章开头的"前置章节"提示。

---

## 🧭 阅读路线建议

**只想知道这个项目是什么** → 读 [01](01-what-is-this-project.md) + [02](02-overall-flow.md) + [17](17-glossary.md)

**想读懂代码** → 加读 [03](03-directory-structure.md) + [04](04-interface-and-impls.md) + [05](05-chunk-anatomy.md)

**想深入某个模块** → 直接查对应章节

---

## 📚 章节列表

### 第一部分：建立认知

| # | 章节 | 一句话 |
|---|---|---|
| [01](01-what-is-this-project.md) | 这个项目是什么 | 领域无关的"PDF 文档智能问答"（建筑 DEMO 为首个示例） |
| [02](02-overall-flow.md) | 整体数据流 | 从 PDF 到答案要走的两条主流水线 |
| [03](03-directory-structure.md) | 目录结构 | 每个文件夹都装什么、为什么这么分 |
| [04](04-interface-and-impls.md) | 接口和实现 | "插座"和"电器"的核心架构思想 |

### 第二部分：数据的"长相"

| # | 章节 | 一句话 |
|---|---|---|
| [05](05-chunk-anatomy.md) | Chunk 解剖 | 全系统的"原子单位"叫 Chunk |
| [06](06-pdf-parsing-and-ocr.md) | PDF 解析与 OCR | 怎么把扫描 PDF 变成可搜索的文字 |
| [07](07-chunking.md) | 切块（Chunking） | 一页几百字怎么切才不破坏检索 |

### 第三部分：检索三路 + 融合

| # | 章节 | 一句话 |
|---|---|---|
| [08](08-embedding.md) | Embedding | 把文字翻译成"语义坐标"的过程 |
| [09](09-vector-store.md) | Vector Store / Qdrant | 千万个向量里找最近邻的数据库 |
| [10](10-bm25-keyword.md) | BM25 + jieba | 老派但仍然好用的关键词检索 |
| [11](11-rrf-fusion.md) | RRF 融合 | 两路检索结果怎么合成一路 |
| [12](12-cross-encoder-reranker.md) | Cross-encoder 精排 | 检索回来的最后一道"精读关" |
| [13](13-ref-graph.md) | 引用图谱 RefGraph | 专治建筑合同的"图引条款"难题 |

### 第四部分：生成答案 + 工具

| # | 章节 | 一句话 |
|---|---|---|
| [14](14-answerer-and-citations.md) | Answerer 与引用守约 | LLM 怎么被"逼着"老实回答 |
| [15](15-cli-walkthrough.md) | CLI 命令一览 | 每个命令干嘛、怎么连起来用 |
| [16](16-evaluation.md) | 评测体系 | 怎么知道改了代码有没有变好 |
| [17](17-glossary.md) | 术语表 | 按字母查的技术词典 |

---

## 🎯 5 分钟极速版

如果只有 5 分钟，记住这三件事：

1. **Chunk 是宇宙基本粒子** — 所有数据都被切成 Chunk 存进数据库，所有搜索结果都是 Chunk 列表，所有答案都基于 Chunk。看 [05](05-chunk-anatomy.md)。

2. **检索是混合的（Hybrid）** — 同一个问题同时跑两路检索：
   - 向量检索（懂语义但模糊）
   - BM25 关键词检索（精确但不懂语义）
   - 用 **RRF** 算法把两路结果融合。看 [11](11-rrf-fusion.md)。

3. **接口和实现分离** — 业务代码只调"插座"（Protocol），具体实现可换。今天用 Claude，明天换 DeepSeek，改一行代码。看 [04](04-interface-and-impls.md)。

---

## 📖 字典查询

不知道某个词什么意思？先查 [17 术语表](17-glossary.md)。
