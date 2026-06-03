# 03 · 目录结构

> 前置章节：[02 整体数据流](02-overall-flow.md)
> 下一章 → [04 接口和实现](04-interface-and-impls.md)

---

## 一句话

**项目根目录像一栋公寓楼：每层楼有明确职责，串门不能乱串。**

---

## 类比：公寓楼

```
🏢 j-contract/                  ← 整栋楼
├── 📋 1 楼 公共区域             ← 治理 / 文档 / 契约
│   ├── dev-contract/            楼规（开发规则）
│   ├── docs/                    楼史（开发日志）
│   ├── reference/               图书馆（外部技术资料缓存）
│   ├── plan/                    会议纪要
│   ├── tutorial/                <- 你正在读的
│   └── README.md                门牌
│
├── 🔧 2 楼 后勤                 ← 工程脚手架
│   ├── pyproject.toml           水电规格
│   ├── uv.lock                  水电封口（具体型号锁死）
│   ├── Dockerfile               搬家说明
│   ├── docker-compose.yml       几栋楼怎么连起来
│   ├── scripts/                 维修工的工具
│   └── .env.example             门禁卡模板
│
├── 📦 3 楼 仓库                 ← 数据
│   ├── input-docs/              原料（PDF 进货）
│   ├── data/                    成品（chunks / DB / 缓存）
│   └── eval/                    考卷
│
├── 🧠 4 楼 大脑                 ← 业务代码（重点）
│   └── src/jcontract/
│       ├── interfaces/          插座规格
│       ├── impls/               具体电器
│       ├── ingest/              入库流水线
│       ├── retrieve/            检索逻辑
│       ├── answer/              答案逻辑
│       ├── eval/                评测逻辑
│       ├── cli.py               总控制面板
│       └── config.py            配置中心
│
└── 🧪 5 楼 实验室               ← 测试
    └── tests/
```

---

## 1 楼 · 治理与文档

### `dev-contract/` — 楼规

整栋楼开发要遵守的规则。比如：
- 怎么提交代码（commit message 格式）
- 测试不允许统一后补
- 用 uv 不用 pip
- 一定要写 dev_log

**这不是你写的代码**，是 AI 协作者（也就是我）和你共同遵守的"宪法"。

> 不需要每次都读完。具体规则触发时再翻——就像你不会每天读小区物业条例，但出问题时知道去哪查。

### `docs/` — 楼史

- `project_guideline.md` — 长期定位（系统边界、分层原则、接口清单）
- `sprint-template.md` — 每个 sprint 怎么写
- `dev-sprint.md` — 每个 sprint 的计划
- `dev_log.md` — 每次开发完写的总结（含 DECISION、UNCERTAIN、FORESHADOW）

### `reference/` — 图书馆

外部技术资料的本地缓存（API 文档、模型卡片、技术调研）。**避免每次问问题都上网搜**，离线就能查。

### `plan/` — 会议纪要

plan-mode 输出的实施计划（你看到的 `ai-ai-mb-pdf-baseline-expected-rag-grap-vectorized-fairy.md`）。

### `tutorial/` — 你正在读的

新人入门文档。

---

## 2 楼 · 工程脚手架

### `pyproject.toml` — 项目身份证

详见 [00 目录](00-index.md)，TOML 是配置格式。这文件管：
- 项目元信息（名字、版本）
- 依赖列表（运行时 + dev）
- ruff / mypy / pytest 的配置
- CLI 命令注册（`jcontract` 这个命令怎么来）

### `uv.lock` — 依赖封口

uv 自动生成的"精确版本快照"。你机器上装的版本和我机器上一字不差。**必须 commit 进 git**。

### `Dockerfile` + `docker-compose.yml` — 搬家说明

把整个项目打包成容器，朋友老王可以一键起服务。当前 `docker-compose.yml` 主要管 Qdrant 服务。

### `scripts/`

shell 工具脚本。最常用的是 `check.sh`：跑三件套（ruff + mypy + pytest）的合流体检。

### `.env.example`

API key 模板。真正的 `.env` 在 `.gitignore` 里，不会进 git。

---

## 3 楼 · 数据

### `input-docs/` — PDF 进货

9 份 DEMO 项目的原始 PDF 放这里。**不进 git**（太大），用 `.gitignore` 屏蔽。

### `data/` — 运行时产物

| 子目录 / 文件 | 装啥 |
|---|---|
| `data/qdrant/` | Qdrant Docker volume（向量库） |
| `data/ocr_cache/<sha256>.text.txt` | OCR 缓存，按页 hash 命名 |
| `data/_render_tmp/` | OCR 临时 JPEG（用完即删） |
| `data/chunks_snapshot.jsonl` | 所有 Chunk 的 JSONL 快照（用于重建 BM25） |
| `data/ref_graph.db` | SQLite 引用图谱 |
| `data/ingest_checkpoint.jsonl` | 批量入库的断点续跑记录 |
| `data/eval-results/` | 评测跑分结果 |

### `eval/` (顶层)

预留位置，目前评测数据放在 `src/jcontract/eval/golden_cases.jsonl`。

---

## 4 楼 · 业务代码（重点）

```
src/jcontract/
├── interfaces/          [核心抽象层]
├── impls/               [具体实现层]
├── ingest/              [入库编排]
├── retrieve/            [检索编排]
├── answer/              [答案编排]
├── eval/                [评测编排]
├── cli.py               [总入口]
├── config.py            [配置中心]
└── __init__.py          [包导出]
```

### `interfaces/` — 10 个抽象接口

每个文件定义一个 Protocol。这层**不能 import 任何第三方库**，纯定义"角色契约"。

| 文件 | 接口 | 干什么 |
|---|---|---|
| `parser.py` | PDFParser | PDF → ParsedPage 列表 |
| `chunker.py` | Chunker | ParsedPage → Chunk 列表 |
| `embedding.py` | Embedder | str 列表 → 向量列表 |
| `vector_store.py` | VectorStore | 存向量 / 按向量查近邻 |
| `keyword.py` | KeywordIndex | 存文本 / 按关键词查 |
| `reranker.py` | Reranker | 重排候选 |
| `answerer.py` | Answerer | 给问题+context → 答案 |
| `ref_graph.py` | RefGraph | 实体倒排表 |
| `vision.py` | VisionCaptioner | （暂未启用） |
| `ocr.py` | OCREngine | （暂未启用，OCR 直接走 vision parser） |
| `schema.py` | （非接口）| 共享 dataclass：Chunk / ParsedPage / SearchResult / Answer / EvalCase |

详见 [04 接口和实现](04-interface-and-impls.md)。

### `impls/` — 各 Protocol 的具体实现

| 文件 | 实现哪个接口 | 用啥技术 |
|---|---|---|
| `pypdf_parser.py` | PDFParser | pypdf 库，**只对文字版 PDF 有用** |
| `claude_vision_parser.py` | PDFParser | Claude Vision API（要 API key） |
| **`claude_cli_vision_parser.py`** | PDFParser | `claude -p` 命令行（订阅模式）← 当前默认 |
| `qa_chunker.py` | Chunker | 正则识别 Q&A + 段落切 |
| `fastembed_embedder.py` | Embedder | mpnet 多语言模型 |
| `qdrant_store.py` | VectorStore | Qdrant 向量库 |
| `bm25_index.py` | KeywordIndex | rank-bm25 + jieba |
| `bge_reranker.py` | Reranker | bge-reranker-v2-m3 cross-encoder |
| `claude_answerer.py` | Answerer | Claude API |
| **`claude_cli_answerer.py`** | Answerer | `claude -p` 命令行（订阅模式）← 当前默认 |
| `codex_cli_answerer.py` | Answerer | Codex CLI（备用） |
| `sqlite_ref_graph.py` | RefGraph | SQLite 倒排表 |

### `ingest/` — 入库编排

- `pipeline.py` — **IngestPipeline 单文档流水**：把 parser → chunker → embedder → 三处存储串起来。还有 `load_chunks_snapshot()` 用于 BM25 重建。
- `batch.py` — **多文档批处理**：asyncio + Semaphore 并发跑多个 PDF，带 checkpoint。

### `retrieve/` — 检索编排

- `hybrid.py` — **HybridRetriever**：跑向量 + BM25 + RRF + 可选 reranker。20 行的 `rrf_fuse()` 在这。

### `answer/` — 答案编排

- `prompt.py` — Prompt 模板组装（系统指令 + XML context + question）
- `postprocess.py` — 引用守约：剥假引用、提 confidence

### `eval/` — 评测编排

- `golden_cases.jsonl` — 27 个测试题
- `runner.py` — 跑评测
- `metrics.py` — Recall@K、citation accuracy 等指标

### `cli.py` — 总控制面板

定义所有 `jcontract <subcommand>` 子命令。是用户和系统的唯一入口（Phase 5 之前）。

### `config.py` — 配置中心

集中读 `.env`、提供 settings。**所有 secret 走这里**，业务代码不直接读环境变量。

---

## 5 楼 · 测试

```
tests/
├── interfaces/        Protocol 边界测试
├── impls/             各 impl 单元测试
├── ingest/            pipeline 集成测试
├── retrieve/          检索集成测试
├── answer/            prompt + postprocess 测试
└── eval/              评测 runner 测试
```

**契约要求**：每个 sub-sprint 交付时必须带测试。不允许"统一后补"。

---

## 一个小练习

看 `cli.py` 的 `_build_stack()` 函数：

```python
def _build_stack(collection: str, *, use_reranker: bool = False) -> Stack:
    embedder = FastEmbedEmbedder()
    vector_store = QdrantStore(collection_name=collection)
    keyword_index = Bm25Index()
    ...
    retriever = HybridRetriever(embedder, vector_store, keyword_index, ...)
```

**问题**：上面用到了哪些 4 楼的子目录？

**答**：
- `impls/fastembed_embedder.py` → `FastEmbedEmbedder`
- `impls/qdrant_store.py` → `QdrantStore`
- `impls/bm25_index.py` → `Bm25Index`
- `retrieve/hybrid.py` → `HybridRetriever`

它们都实现了 `interfaces/` 里的某个 Protocol，被 cli.py 拼装起来。**业务代码（cli.py）只知道"我要一个 Embedder"，不在意是 mpnet 还是 bge-m3**。

这就是下一章 [04 接口和实现](04-interface-and-impls.md) 要展开的核心思想。
