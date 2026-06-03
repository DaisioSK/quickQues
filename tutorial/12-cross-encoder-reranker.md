# 12 · Cross-encoder 精排

> 前置章节：[11 RRF 融合](11-rrf-fusion.md)
> 下一章 → [13 引用图谱](13-ref-graph.md)

---

## 一句话

**RRF 拿到 top-30 候选后，让一个"精读模型"逐一仔细评分重排——速度慢但质量高。**

---

## 类比：HR 招聘流程

招聘一个岗位：

1. **简历海选**（recall）— HR 看简历关键词，从 1000 份筛 100 份
   - 类比：vector + BM25 + RRF 从 10k chunk 召回 30 个
2. **HR 一面**（rerank）— HR 看每份简历 + 问候选人具体问题，深入评估
   - 类比：cross-encoder 逐一精读 (question, chunk) 对
3. **录用决定**（answer）— LLM 基于精排 top-5 写答案

第一步快但粗，第二步慢但精。**不能省第一步**（不可能给 1000 人都一面），**不能省第二步**（关键词命中 ≠ 真的合适）。

---

## bi-encoder vs cross-encoder（再讲一遍）

| | bi-encoder | cross-encoder |
|---|---|---|
| **代表模型** | mpnet, bge-m3, e5 | bge-reranker-v2-m3, cohere-rerank, jina-rerank |
| **怎么编码** | 问题、文档**分开**编码 | 问题+文档**拼一起**编码 |
| **能否预索引文档** | ✅ 可以（编完存数据库）| ❌ 不行（每次查询都要重跑）|
| **复杂度** | O(1) per query（向量近邻）| O(N) per query（N=候选数）|
| **典型延迟** | ~10ms | ~30-50ms × N |
| **质量** | 中等 | **高** |
| **核心机制** | 各自编码 → cosine 距离 | 联合编码 → 直接出"匹配分" |

为什么 cross-encoder 质量高：

```
[bi-encoder]
  问题 "桥梁防水谁负责" → 向量 q   ┐
                                  ├── cosine(q, d)
  文档 "Trackwork 负责..."  → 向量 d ┘
  → 问题和文档在编码时不"知道"对方存在

[cross-encoder]
  "桥梁防水谁负责 [SEP] Trackwork 负责..." → 一次模型前向 → 单个匹配分
  → 模型每个 attention head 都能在两者之间穿梭，理解上下文关联
```

cross-encoder 能捕获细微差别，比如：
- 问题问 "X 谁负责"，文档说 "X is Y's responsibility" → 高分
- 问题问 "X 谁负责"，文档说 "X exists" → 低分
- bi-encoder 容易在这种细微差异上失手

---

## 我们用的 bge-reranker-v2-m3

代码：[src/jcontract/impls/bge_reranker.py](../src/jcontract/impls/bge_reranker.py)

**模型**：`BAAI/bge-reranker-v2-m3`

- **来源**：BAAI（智源），bge 家族的多语言 reranker
- **大小**：~568MB（torch + 模型权重）
- **支持语言**：100+（含中文、英文）
- **延迟**：~50ms/pair（CPU）

为什么这个而不是更小的：

| 模型 | 大小 | 速度 | 质量 | 语言 |
|---|---|---|---|---|
| bge-reranker-base | ~300MB | 快 | 中等 | EN+CN |
| bge-reranker-large | ~560MB | 中 | 好 | EN 偏向 |
| **bge-reranker-v2-m3** ⭐ | ~568MB | 中 | 好 | **多语言**（我们要中英）|
| bge-reranker-v2-gemma | ~2.5GB | 慢 | 最好 | EN 偏向 |

我们语料中英混杂 → 必须多语言 → v2-m3 是甜点。

---

## sentence-transformers 库

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [
    ("桥梁防水谁负责？", "Trackwork Contractor shall be responsible..."),
    ("桥梁防水谁负责？", "今天天气真好"),
]
scores = model.predict(pairs)
# [3.45, -2.81]  ← 高分相关，低分（甚至负）不相关
```

**sentence-transformers** 是 HuggingFace 生态里的 cross-encoder 标准库。带 torch + transformers，约 2GB。

### 为什么不用 fastembed

[09 Vector Store](09-vector-store.md) 提到 fastembed 跑 bi-encoder。但 fastembed 0.3.6 **不支持 cross-encoder**。

要么升级 fastembed（连带升 qdrant-client，风险大），要么加 sentence-transformers（多 2GB 依赖）。

我们选了后者。8 字依赖检查（[dev-contract/24-domain-deps-env.md](../dev-contract/24-domain-deps-env.md)）的 8 个问题都过：

1. 标准库够吗？不够，要 transformer 推理
2. 现有依赖够吗？不够（fastembed 0.3.6 没 cross-encoder）
3. 活跃吗？是（UKPLab + HF，15k+ stars）
4. License？Apache 2.0 ✅
5. 大小？2GB 但**反正 Phase 2 要装 torch**（bge-m3 升级）
6. 来源可信吗？UKPLab + HuggingFace ✅
7. CVE？无 ✅
8. 替代方案？ONNX 社区导出（typosquatting 风险）、FlagEmbedding（ST 超集）、自己写 ONNX runner（超范围）

**Decision**：值得。

---

## 实际怎么用

[bge_reranker.py:129](../src/jcontract/impls/bge_reranker.py#L129)：

```python
class BgeReranker:
    backend: ClassVar[str] = "bge-cross-encoder"

    def __init__(self, *, model=DEFAULT_MODEL, batch_size=16):
        self._model_name = model
        self._batch_size = batch_size
        self._model = None  # 懒加载

    def rerank(self, question: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(question, c.chunk.text) for c in candidates]
        scores = model.predict(pairs, batch_size=self._batch_size, ...)
        rescored = [
            SearchResult(chunk=cand.chunk, score=float(new_score))
            for cand, new_score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
```

输入：`(question, candidates)`
输出：**候选数量不变**，但**重新按 cross-encoder 分数排序**

注意：
1. **长度不变**：不过滤，policy 留给上层（业务代码可以加 threshold）
2. **score 替换**：返回的 SearchResult.score 是 cross-encoder 新分数，不再是 RRF 分
3. **懒加载**：第一次 rerank 才下载模型 + 加载

---

## 在 HybridRetriever 里的位置

[retrieve/hybrid.py:107](../src/jcontract/retrieve/hybrid.py#L107)：

```python
def search(self, query: str, k: int = 5) -> list[SearchResult]:
    # ... vector + BM25 + RRF ...
    fused = rrf_fuse([vec_results, kw_results])

    if self.reranker is None:
        return fused[:k]

    # 取 top-30 给 reranker
    candidates_for_rerank = fused[:self.rerank_top_n]  # rerank_top_n=30
    reranked = self.reranker.rerank(query, candidates_for_rerank)
    return reranked[:k]
```

**为什么只精排 top-30**：

- cross-encoder 是 O(N)。精排 30 个 ≈ 30 × 50ms = 1.5 秒
- 精排 100 个 ≈ 5 秒（用户感知明显变慢）
- 精排 1000 个 ≈ 50 秒（不可接受）

RRF 已经把大概率好的候选浮到前面，**剩下的我们假设没机会**。30 是经验值。

---

## 何时该开启 reranker

代码默认 **off**（`cli.py` 里 `use_reranker=False`）。开关用法：

```bash
# 不开 reranker（默认，快）
uv run jcontract search "..." --k 5

# 开 reranker（慢但准）
uv run jcontract search "..." --k 5 --rerank
uv run jcontract evaluate --rerank
```

何时该开：

| 场景 | 开 reranker？ |
|---|---|
| < 1000 chunks 总规模 | ❌ 不必，RRF 够 |
| 4000+ chunks 总规模 | ✅ 召回噪声大，精排有用 |
| 跑 evaluate 想看上限 | ✅ |
| 实时问答（追求 < 2s） | ❌ 开了 ~3-4s |
| 离线批量处理 | ✅ 无所谓速度 |

DEMO 全量入库后 ~20k chunks，**强烈建议开**。

---

## 第一次开启会发生什么

```bash
$ uv run jcontract search "桥梁防水" --rerank

[ 2026-05-29 ] downloading BAAI/bge-reranker-v2-m3 ...
config.json:        100% [================]  500 B
tokenizer.json:     100% [================] 18 MB
pytorch_model.bin:  100% [================] 568 MB
... (~30s 下载) ...

[ 2026-05-29 ] cli.reranker_enabled model=BAAI/bge-reranker-v2-m3
[ 2026-05-29 ] loading model into memory ... (~5s)

[results]
...
```

下载 + 加载只发生一次。下次启动是 cache hit。

模型存在 `~/.cache/huggingface/` 目录。

---

## 量化效果（理论 + 经验）

公开 benchmark（BEIR、LoTTE）实证：

- 单纯 vector retrieval Recall@5 ≈ 0.55-0.70
- + BM25 + RRF → Recall@5 ≈ 0.70-0.80（+10-15%）
- + cross-encoder rerank → Recall@5 ≈ 0.80-0.90（再 +10%）

DEMO 实测要等用户跑完评测才知道。**FORESHADOW**：评测后会知道在 27 个 golden case 上的 lift。

---

## 关键决策（DECISION）

### 为什么 batch_size=16

- 32 是常见默认，但 8GB 内存机器可能 OOM
- 8 太小，CPU 利用率低
- 16 是稳健中间值，跨机器兼容

### 为什么 lazy load

模型 568MB 在硬盘，启动 5s 加载到内存。如果 CLI 启动每次都加载（即使不 rerank），太浪费。lazy load 让"没开 --rerank 的命令零成本"。

### 为什么不算 confidence

cross-encoder 分数是原始 logit，可能在 -10 到 +10 之间。**没有"分数阈值 = 相关性阈值"的可移植性**（每个模型尺度不同）。

未来 Phase 3 调优时可以根据评测数据学一个 threshold。当前**不丢弃任何候选**，policy 留给上层。

---

## 一个误解：cross-encoder 不能"召回"

新人常误以为"cross-encoder 又强又准，干嘛不直接用它代替整个检索？"

答：**因为它不能预先索引文档**。

```
[场景] 10000 chunks，回答 1 个问题
[bi-encoder] 索引一次 chunks（一次性 ~30s），每个 query 算 1 次 + Qdrant 查 → ~20ms
[cross-encoder] 不能索引，每个 query 要算 10000 次 (q, chunk) 对 → 500 秒
```

cross-encoder **只能用于精排小批候选**。这就是为什么 RAG 经典架构是"bi-encoder 召回 + cross-encoder 精排"两阶段。

---

## 下一步阅读

- 完全不同的"精确查"另一种路 → [13 引用图谱](13-ref-graph.md)
- 精排后怎么生成答案 → [14 Answerer](14-answerer-and-citations.md)
