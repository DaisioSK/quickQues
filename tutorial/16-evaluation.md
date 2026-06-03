# 16 · 评测体系

> 前置章节：[15 CLI 命令一览](15-cli-walkthrough.md)
> 下一章 → [17 术语表](17-glossary.md)

---

## 一句话

**评测就是"自动批改作业"——给系统一份带标准答案的考卷，看它考多少分。**

---

## 为什么需要评测

如果没有评测，每次改代码（换模型、调 prompt、加 reranker）都只能**人肉感觉**"好像变好了？"——不可靠、不可比、不可回归。

评测让"变好"**变成数字**：

- "Recall@5 从 0.65 涨到 0.78" → 信号清晰
- "Citation accuracy 从 0.7 跌到 0.5" → 立刻知道改坏了

---

## 评测三件事

1. **评测集**（考卷）= 一堆 (问题, 标准答案/出处) 对
2. **指标**（评分标准）= 怎么打分
3. **Runner**（批改员）= 跑评测的代码

---

## 评测集：golden_cases.jsonl

文件：`src/jcontract/eval/golden_cases.jsonl`，27 个测试 case。

每行一个 JSON：

```json
{
  "id": "q001",
  "question": "桥梁防水的责任方是谁？依据是哪一条款？",
  "expected_sources": [
    {"file": "synthetic_contract_tqa.pdf", "page_min": 2, "page_max": 4}
  ],
  "expected_keywords": ["waterproofing", "Trackwork Contractor", "防水"],
  "category": "responsibility"
}
```

字段：

| 字段 | 用途 |
|---|---|
| `id` | case 唯一标识（q001、q101...）|
| `question` | 中文问题 |
| `expected_sources` | 标准答案应该来自哪份文件、哪页范围 |
| `expected_keywords` | 答案文字里**应该包含的词**（中英都行）|
| `category` | 分组（responsibility / definition / quantity ...）|

### 27 个 case 的来源分布

- **q001-q006**：对应合成的小 PDF（`synthetic_contract_tqa.pdf`，4 页测试用）
- **q101-q104**：对应 `Contract DEMO(1of9) TQA.pdf`
- **q201-q202**：对应 `Contract DEMO(1of9)Consol.pdf`
- **q301-q302**：对应 `Contract DEMO(3of9)1of2Consol.pdf`
- ... 以此类推
- **qX01-qX03**：跨文档 case

**为什么页码范围 (page_min, page_max)**：chunker 切 chunk 时可能选起始页或结束页，给一个范围让评测对边界不敏感。

### 8 个 category（题型）

```
responsibility   责任主体
definition       定义
quantity         数量/规模
procedure        流程/工序
revision         版次/差异
reference        引用/跳转
list             清单/枚举
cross_doc        跨文档
```

可以按 category 看哪类问题答得差。

---

## 指标：Recall@K + Citation Accuracy

### Recall@K — 召回率

**问**：top-K 检索结果里，有几个落在 expected_sources 的范围内？

公式：
```
Recall@K = (top-K 中位于 expected sources 页码范围的 chunk 数) / (expected sources 总数)
```

例子：

```
Expected sources: [
    {"file": "...TQA.pdf", "page_min": 1, "page_max": 5},
    {"file": "...Consol.pdf", "page_min": 100, "page_max": 110},
]

Top-5 hits:
  1. file=TQA.pdf page=3       ✅ 落在第一个范围
  2. file=TQA.pdf page=4       ✅ 落在第一个范围（重复算同一个 source 一次）
  3. file=Consol.pdf page=105  ✅ 落在第二个范围
  4. file=Other.pdf page=20    ❌
  5. file=TQA.pdf page=50      ❌

Recall@5 = 2/2 = 1.0  （两个 expected sources 都被覆盖到）
```

**Recall 衡量"该找的有没有找到"**。0.0 = 完全没找到，1.0 = 全找到。

### Citation Accuracy — 引用准确率

**问**：LLM 答案给出的每个引用 [file p.X]，是否真的在召回的 chunks 里？

公式：
```
Citation Accuracy = (有效引用数) / (LLM 给出的总引用数)
```

例子：

```
Top-5 chunks 提供的 (file, page) 白名单:
  - (TQA.pdf, 3), (TQA.pdf, 4), (Consol.pdf, 105), (Other.pdf, 20), (TQA.pdf, 50)

LLM 答案:
  "桥梁防水由 Trackwork 负责 [TQA.pdf p.3]，
  依据 Clause 7.3 [Consol.pdf p.105]，
  另见图纸说明 [TQA.pdf p.99]。"  ← p.99 是假引用！

引用列表: [(TQA.pdf, 3), (Consol.pdf, 105), (TQA.pdf, 99)]
有效引用: [(TQA.pdf, 3), (Consol.pdf, 105)]  (前两个白名单里有)

Citation Accuracy = 2/3 = 0.667
```

**Citation Accuracy 衡量"LLM 老不老实"**。1.0 = 不编造，<1.0 = 在编。

注意：postprocess **会自动剥假引用**。但 citation accuracy 算**剥之前**的，看 LLM 原始诚实度。

### No-answer Rate — 弃答率

```
No-answer Rate = (答 "文档中未明确说明" 的 case 数) / 总 case 数
```

理想：和数据匹配的 case 都答出来，超出范围的 case 老实说不知道。

如果一份 PDF 都没入库就跑评测，所有 case 都应该 no-answer = 0%（这才对）。

---

## Runner：怎么跑评测

代码：[src/jcontract/eval/runner.py](../src/jcontract/eval/runner.py)。

伪代码：

```python
def run_eval(cases, retriever, answerer=None, k=5):
    metrics = MetricsAccumulator()

    for case in cases:
        # 1. 检索
        hits = retriever.search(case.question, k=k)

        # 2. 算 Recall@K
        recall = compute_recall(hits, case.expected_sources)
        metrics.add_recall(recall)

        # 3. 如果有 answerer，跑答题
        if answerer:
            answer = answerer.answer(case.question, [h.chunk for h in hits])

            # 4. 算 Citation Accuracy
            cite_acc = compute_citation_accuracy(answer)
            metrics.add_citation(cite_acc)

            # 5. 算 keyword coverage（答案是否含 expected_keywords）
            kw_cov = compute_keyword_coverage(answer, case.expected_keywords)
            metrics.add_keyword(kw_cov)

    return metrics.summarize()
```

跑：

```bash
uv run jcontract evaluate --answerer claude-cli
```

输出：

```
[eval] cases: 27 | answerer: claude-cli | rerank: false
[eval] q001 ✅ Recall@5=1.0 cite=1.0 keywords=0.67
[eval] q002 ✅ Recall@5=1.0 cite=1.0 keywords=1.0
[eval] q003 ❌ Recall@5=0.0 cite=N/A keywords=0.0  (no-answer)
...

=== Summary ===
Recall@5:              0.78
Citation accuracy:     0.91
Keyword coverage:      0.65
No-answer rate:        0.22
Cases per category:
  responsibility:      Recall=0.83  N=3
  definition:          Recall=0.75  N=4
  ...

Results saved to: data/eval-results/2026-05-29_142315.json
```

---

## 怎么解读分数

### Recall@5

| 分数 | 含义 |
|---|---|
| > 0.85 | 优秀，检索基本可靠 |
| 0.70 - 0.85 | 良好，可用 |
| 0.50 - 0.70 | 一般，有改进空间 |
| < 0.50 | 差，检索经常 miss |

**对策**（如果低）：
1. chunker 切坏了 → `show-chunks` 检查
2. embedding 模型不够 → 升级到 bge-m3
3. 没开 reranker → `--rerank`
4. 入库还不全 → `batch-ingest`

### Citation Accuracy

| 分数 | 含义 |
|---|---|
| > 0.95 | LLM 老实 |
| 0.85 - 0.95 | 偶尔小编造 |
| < 0.85 | 编造严重，prompt 需要加强 |

**对策**（如果低）：
1. prompt 不够明确 → 重写 system_prompt
2. 模型本身倾向编造 → 换更稳的（Claude 比 GPT 更稳）
3. context 太长 → 砍 k

### No-answer Rate

| 数据状态 | 期望 rate |
|---|---|
| 部分入库（仅 1 份 PDF） | ~70-80%（多数 case 没数据匹配） |
| 全量入库（9 份 PDF） | ~10-20%（应该多数能答）|
| 入库完整还高 → 检索/chunker 问题 |

---

## 评测结果存哪

`data/eval-results/2026-05-29_142315.json`：

```json
{
  "timestamp": "2026-05-29T14:23:15",
  "config": {
    "answerer": "claude-cli",
    "rerank": false,
    "k": 5
  },
  "summary": {
    "recall_at_5": 0.78,
    "citation_accuracy": 0.91,
    "no_answer_rate": 0.22
  },
  "per_case": [
    {"id": "q001", "recall": 1.0, "citation": 1.0, "answer": "桥梁防水由..."},
    ...
  ]
}
```

可以：
- 跨次比较（"改 prompt 前 vs 改 prompt 后"）
- 找 bad case（recall=0 的 case 详细分析）
- 给老王看进度

---

## 怎么扩充评测集

随着真实使用积累，golden_cases.jsonl 会增长：

```bash
# 编辑文件，加一行
echo '{"id":"q904","question":"...","expected_sources":[...],...}' \
  >> src/jcontract/eval/golden_cases.jsonl
```

**好 case 的特征**：
- 问题真实（老王问过的）
- 答案有共识（不是开放性问题）
- 出处明确（能定位到具体页）
- 覆盖不同 category

未来 Phase 6 会专门做"评测集建设" sprint，让老王列 30-50 真实问题。

---

## Bad case 分析

跑评测发现 recall=0 的 case → 是金矿。分析步骤：

```bash
# 1. 手动跑 search 看检索结果
uv run jcontract search "原问题" --k 10

# 2. 看为什么 expected source 没召回
#    - chunker 是否切了这段？
uv run jcontract show-chunks --file "源文件" | grep "关键词"

#    - BM25 是否命中关键词？(chunks 在 snapshot.jsonl 找)
grep "关键词" data/chunks_snapshot.jsonl

# 3. 形成 hypothesis 改进
#    - 调 chunker 正则
#    - 加 metadata filter
#    - 换 embedding 模型
```

每周一份 bad case 复盘记到 `docs/dev_log.md`，长期是项目质量的护城河。

---

## 关键决策（DECISION）

### 为什么用 page 范围而非精确 page

chunker 偶尔会把跨页的内容归到起始页或结束页，二选一不可避免。范围让评测稳定。

代价：评测略宽松——精确 page 验证更严格但太敏感。DEMO 这种合同的 page_min/page_max 通常 5-10 页范围，足够区分"对的章节"和"错的章节"。

### 为什么 keyword coverage 是软指标

LLM 用同义词答（"Trackwork 承包商" vs "Trackwork Contractor"）也算对。所以 keyword 是"加分项"不是"决定项"。

实际指标权重：Citation > Recall > Keyword。

### 为什么不算 ROUGE / BLEU

NLG 评估指标对 RAG 答案不合适——答案可以有多种正确写法，ROUGE 看不出对错。Citation Accuracy + 人工抽检更靠谱。

未来 Phase 6 可能加 "LLM as judge"（用另一个模型评分）。

---

## 下一步阅读

- 不懂术语 → [17 术语表](17-glossary.md)
- 回到首页 → [00 目录](00-index.md)
