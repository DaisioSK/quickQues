# 15 · CLI 命令一览

> 前置章节：[14 Answerer 与引用守约](14-answerer-and-citations.md)
> 下一章 → [16 评测体系](16-evaluation.md)

---

## 一句话

**`jcontract <subcommand>` 是用户的总入口——所有功能都从这里调起。**

---

## 命令列表

```
jcontract
├── ingest           入库一份 PDF
├── batch-ingest     批量入库多份 PDF（带并发 + 断点续跑）
├── search           检索（不出答案，看 retrieval）
├── refs             查 RefGraph
├── show-chunks      看入库后的 chunks
└── evaluate         跑评测
```

启动方式（必须在项目根目录）：

```bash
uv run jcontract <subcommand> [options]
```

`uv run` 会自动激活虚拟环境。**直接 `jcontract` 在不装 uv 的环境不可用**。

---

## 1. `ingest` — 单文档入库

```bash
uv run jcontract ingest "input-docs/Contract DEMO(1of9) TQA.pdf" \
  --parser claude-cli-vision \
  --max-pages 5
```

参数：

| 参数 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `<pdf-path>` | ✅ | — | PDF 文件路径 |
| `--parser` |  | `pypdf` | `pypdf` / `claude-vision` / `claude-cli-vision` |
| `--max-pages` |  | 全部 | 限制处理页数（试跑用）|
| `--collection` |  | `contract` | Qdrant collection 名 |

流程：parser → chunker → embedder → Qdrant + BM25 + RefGraph + JSONL snapshot。

**典型用法**：

```bash
# 试跑 5 页
uv run jcontract ingest "input-docs/Contract DEMO(1of9) TQA.pdf" \
  --parser claude-cli-vision --max-pages 5

# 全量单份
uv run jcontract ingest "input-docs/Contract DEMO(1of9) TQA.pdf" \
  --parser claude-cli-vision
```

---

## 2. `batch-ingest` — 批量入库

```bash
uv run jcontract batch-ingest input-docs/*.pdf \
  --parser claude-cli-vision \
  --max-concurrent 4 \
  --resume
```

参数：

| 参数 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `<pdf-paths>` | ✅ | — | 多个 PDF（shell glob 展开） |
| `--parser` |  | `pypdf` | 同上 |
| `--max-concurrent` |  | `2` | 同时跑几份 PDF |
| `--resume` |  | true | 断点续跑（跳过已 ingest 的）|
| `--estimated-cost-per-page` |  | `0` | 预算守门：超过就停 |

特色：

### 断点续跑

每入库完一份 PDF，写一行到 `data/ingest_checkpoint.jsonl`：

```jsonl
{"pdf":"Contract DEMO(1of9) TQA.pdf","status":"done","chunks":612,"timestamp":"2026-05-29T10:23:15"}
```

下次跑同样命令带 `--resume` → 自动跳过已 done 的。

### 并发控制

`asyncio.Semaphore(max_concurrent)` 限制同时跑的 PDF 数量。

订阅模式建议 `--max-concurrent 4`，看到大量 `cli_vision.error` 就降到 2。

### 预算守门

```bash
--estimated-cost-per-page 0.025
```

跑之前先估算总成本（`page_count * cost`），超过预算自动 abort。**保险栓**，防止意外烧光钱。

订阅模式可以设 `0`（永不触发）。

---

## 3. `search` — 检索

```bash
uv run jcontract search "桥梁防水谁负责" --k 5
```

参数：

| 参数 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `<query>` | ✅ | — | 中英文查询 |
| `--k` |  | `5` | 返回多少个 hits |
| `--rerank` |  | false | 启用 cross-encoder 精排 |
| `--collection` |  | `contract` | 同上 |

输出（示例）：

```
[rank 1] score=0.0328
  file: Contract DEMO(1of9) TQA.pdf | page: 42 | type: qa_pair
  text: Question No.: ACME/TRACKWORK/16
        Answer: The Trackwork Contractor shall be responsible for waterproofing
        as specified in Clause 7.3.2 and Drawing No. T/PRJ/CWD/WS/2101A.

[rank 2] score=0.0320
  file: Contract DEMO(1of9)Consol.pdf | page: 103 | type: paragraph
  text: ...
```

**只看 retrieval，不出 LLM 答案**——用来调检索质量、debug 召回。

---

## 4. `refs` — 引用图谱查询

```bash
# 查图纸被谁引用
uv run jcontract refs drawing T/PRJ/CWD/WS/2101A

# 查条款 7.3 被谁引用
uv run jcontract refs clause 7.3

# 查 Question No.
uv run jcontract refs question_no ACME/TRACKWORK/16

# 查 Section
uv run jcontract refs section "Section 7"

# 查 Revision
uv run jcontract refs revision "Rev A"

# 看全局统计
uv run jcontract refs stats
```

**毫秒级精确返回**。详见 [13 RefGraph](13-ref-graph.md)。

---

## 5. `show-chunks` — 看入库结果

```bash
uv run jcontract show-chunks --n 10
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n` | `5` | 显示几个 |
| `--type` | 全部 | 只看某类型 (`qa_pair`/`paragraph`/`table`/`drawing`) |
| `--file` | 全部 | 只看某文件 |

用途：
- ingest 完检查 chunker 切得合不合理
- 看某类型的 chunks 长啥样
- 调试时定位"这段为什么没被切出来"

---

## 6. `evaluate` — 跑评测

```bash
uv run jcontract evaluate \
  --answerer claude-cli \
  --rerank
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--answerer` | `none` | `none` / `claude-api` / `claude-cli` / `codex-cli` |
| `--enable-answer` | true | 是否跑 LLM 答题（false 只跑 retrieval 指标） |
| `--rerank` | false | 启用 reranker |
| `--cases-path` | `src/jcontract/eval/golden_cases.jsonl` | 评测集 |
| `--k` | `5` | top-k |

输出：

```
[eval] cases: 27 | answerer: claude-cli | rerank: yes
[eval] q001 ... Recall@5=1.0 cite=true
[eval] q002 ... Recall@5=1.0 cite=true
...
[eval] q903 ... Recall@5=0.0 cite=false

=== Summary ===
Recall@5:           0.78
Citation accuracy:  0.85
No-answer rate:     0.15
Avg chunks/answer:  3.2

Results saved to: data/eval-results/2026-05-29_142315.json
```

详见 [16 评测](16-evaluation.md)。

---

## 典型工作流

### 场景 A：第一次试一份 PDF

```bash
# 1. 起 Qdrant
docker-compose up -d qdrant

# 2. 试跑 5 页看 OCR 效果
uv run jcontract ingest "input-docs/Contract DEMO(1of9) TQA.pdf" \
  --parser claude-cli-vision --max-pages 5

# 3. 看入库结果
uv run jcontract show-chunks --n 10

# 4. 试几个查询
uv run jcontract search "Trackwork Contractor" --k 5
uv run jcontract refs drawing T/PRJ/CWD/WS/2101A
```

### 场景 B：单 PDF 全量评测

```bash
# 1. 清旧数据
rm -f data/chunks_snapshot.jsonl data/ref_graph.db data/ingest_checkpoint.jsonl

# 2. 全量入库（~10-20 min on 订阅）
uv run jcontract batch-ingest "input-docs/Contract DEMO(1of9) TQA.pdf" \
  --parser claude-cli-vision --max-concurrent 4

# 3. 评测
uv run jcontract evaluate --answerer claude-cli
```

### 场景 C：全量 9 份 PDF（几小时）

```bash
# 1. 全量
uv run jcontract batch-ingest input-docs/*.pdf \
  --parser claude-cli-vision --max-concurrent 4

# 2. 评测带 reranker
uv run jcontract evaluate --answerer claude-cli --rerank
```

---

## 数据清理命令

```bash
# 清 OCR 缓存（强制重 OCR）
rm -rf data/ocr_cache/

# 清 chunks 快照（强制重建 BM25）
rm -f data/chunks_snapshot.jsonl

# 清 RefGraph
rm -f data/ref_graph.db

# 清 checkpoint（强制重新 ingest 所有 PDF）
rm -f data/ingest_checkpoint.jsonl

# 清 Qdrant（小心！）
docker-compose down
rm -rf data/qdrant/
docker-compose up -d qdrant
```

---

## 常见问题

### "Qdrant connection refused"

Qdrant 没启动。`docker-compose up -d qdrant`。

### "claude CLI not found in PATH"

没装 Claude Code 或没加进 PATH。`claude --version` 验证。

### "Recall@5 = 0 但我以为入库成功了"

可能原因：
1. evaluate 跑的是 27 个 golden case，其中 6 个对应 synthetic PDF、21 个对应真 PDF。如果你只 ingest 了 1 份真 PDF，那 ~5 个 case 对应这份，其他 case Recall 必然为 0
2. BM25 没重建：`cat data/chunks_snapshot.jsonl | wc -l` 看是否有内容
3. Qdrant 集合没数据：`curl http://localhost:6333/collections/contract`

### "ingest 卡住不动了"

订阅 OCR 单页 ~10-15s，正常。看日志有 `cli_vision.ocr_complete` 说明在跑。如果 1 分钟没新日志可能是 rate limit，等等再试或降 `--max-concurrent`。

### "--help 报错"

某些子命令的 `--help` 在 typer/click 上游兼容性有 bug。具体子命令 help 用：
```bash
uv run jcontract <cmd> --help
```

---

## 下一步阅读

- 评测指标和怎么解读 → [16 评测](16-evaluation.md)
- 不懂的术语 → [17 术语表](17-glossary.md)
