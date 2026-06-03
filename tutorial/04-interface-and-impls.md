# 04 · 接口和实现

> 前置章节：[03 目录结构](03-directory-structure.md)
> 下一章 → [05 Chunk 解剖](05-chunk-anatomy.md)

---

## 一句话

**业务代码只对"插座规格"编程，不在意插的是哪家的"电器"——这样换电器不用砸墙。**

---

## 类比：插座 vs 电器

你家墙上是 **220V Type-G 三孔插座**。你买了：

- 飞利浦的台灯（用 Type-G 插头）✅
- 戴森的吸尘器（用 Type-G 插头）✅
- 苹果的充电器（用 Type-G 插头）✅

它们都能插上工作。墙没变，电器随便换。

**反过来想**：如果墙上是台灯专用插孔（圆形 5mm 三孔），那你只能用飞利浦那一款台灯，**换品牌就要砸墙重新布线**。这就是"没有接口抽象"的代价。

---

## 这个项目的 10 个"插座"

`src/jcontract/interfaces/` 目录就是一栋楼的"墙上所有插座规格汇总"。每个文件定义一个插座。

```python
# src/jcontract/interfaces/embedding.py
from typing import Protocol

class Embedder(Protocol):
    """规格：能把文本列表变成向量列表，要有 dim 属性。"""

    @property
    def dim(self) -> int:
        """向量维度。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """文本 → 向量。顺序保持。"""
```

这就是 `Embedder` "插座"。**任何提供 `dim` 和 `embed()` 方法的类**都能插进这个槽。

### 然后是各种"电器"

```python
# src/jcontract/impls/fastembed_embedder.py
class FastEmbedEmbedder:
    """mpnet 多语言模型，ONNX 推理"""
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

# 假设未来要换 BGE-M3
# src/jcontract/impls/bgem3_embedder.py
class Bgem3Embedder:
    """BGE-M3 多语言模型，FlagEmbedding 库"""
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

业务代码（HybridRetriever）只看到 "我有一个 Embedder"：

```python
# src/jcontract/retrieve/hybrid.py
class HybridRetriever:
    def __init__(self, embedder: Embedder, ...):  # ← 只声明插座类型
        self.embedder = embedder

    def search(self, query):
        vec = self.embedder.embed([query])[0]  # ← 不关心是哪家的
        ...
```

**切换实现 = 改 cli.py 里一行**：

```python
# 现在
embedder = FastEmbedEmbedder()

# 想换 BGE-M3 ↓
embedder = Bgem3Embedder()
```

HybridRetriever、IngestPipeline、所有测试——一行都不用动。

---

## 为什么用 Protocol 而不是 ABC？

Python 有两种"接口"机制：

### `abc.ABC`（继承）

```python
class Embedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts): ...

class FastEmbedEmbedder(Embedder):  # ← 必须显式继承
    def embed(self, texts): ...
```

### `typing.Protocol`（结构化匹配，duck typing）

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class FastEmbedEmbedder:  # ← 不需要继承！
    def embed(self, texts): ...
```

只要 `FastEmbedEmbedder` **方法签名匹配** Protocol，就被认为实现了它。mypy 会在编译期检查。

**为什么选 Protocol**：
- 第三方库的类（比如 `qdrant_client.QdrantClient`）你**不可能让它继承你的 ABC**——它早就写好了。但 Protocol 不要求继承，**已存在的兼容类自动适配**。
- 实现层和接口层**零耦合**：impls/ 里的代码不需要 `import interfaces`，也不依赖接口。改接口不会影响实现，反之亦然。

---

## 完整插座列表（10 个）

| Protocol | 定义在 | 干什么 | 默认实现 |
|---|---|---|---|
| `PDFParser` | [interfaces/parser.py](../src/jcontract/interfaces/parser.py) | PDF → ParsedPage | claude-cli-vision |
| `Chunker` | [interfaces/chunker.py](../src/jcontract/interfaces/chunker.py) | ParsedPage → Chunk | qa-aware |
| `Embedder` | [interfaces/embedding.py](../src/jcontract/interfaces/embedding.py) | str → 向量 | fastembed mpnet |
| `VectorStore` | [interfaces/vector_store.py](../src/jcontract/interfaces/vector_store.py) | 存/查向量 | Qdrant |
| `KeywordIndex` | [interfaces/keyword.py](../src/jcontract/interfaces/keyword.py) | BM25 检索 | rank-bm25 + jieba |
| `Reranker` | [interfaces/reranker.py](../src/jcontract/interfaces/reranker.py) | 重排候选 | bge-reranker-v2-m3 |
| `Answerer` | [interfaces/answerer.py](../src/jcontract/interfaces/answerer.py) | 生成答案 | claude-cli |
| `RefGraph` | [interfaces/ref_graph.py](../src/jcontract/interfaces/ref_graph.py) | 实体倒排表 | SQLite |
| `OCREngine` | [interfaces/ocr.py](../src/jcontract/interfaces/ocr.py) | OCR | （未启用） |
| `VisionCaptioner` | [interfaces/vision.py](../src/jcontract/interfaces/vision.py) | 图说 | （未启用） |

加上 `interfaces/schema.py`（共享 dataclass，不是 Protocol）。

---

## DI（依赖注入）：谁负责装电器

业务代码不 `new` 具体实现，而是**接收已 new 好的实例**：

```python
# ❌ 坏例子（强耦合）
class HybridRetriever:
    def __init__(self):
        self.embedder = FastEmbedEmbedder()  # 写死了
```

```python
# ✅ 好例子（依赖注入）
class HybridRetriever:
    def __init__(self, embedder: Embedder, ...):  # 外部传进来
        self.embedder = embedder
```

**装配工作在 `cli.py` 集中做**（搜 `_build_stack`）：

```python
def _build_stack(collection: str, *, use_reranker: bool = False) -> Stack:
    embedder = FastEmbedEmbedder()           # 1. new 各个电器
    vector_store = QdrantStore(collection)
    keyword_index = Bm25Index()
    reranker = BgeReranker() if use_reranker else None

    retriever = HybridRetriever(             # 2. 拼装
        embedder, vector_store, keyword_index, reranker=reranker
    )
    return Stack(embedder, vector_store, keyword_index, retriever)
```

这种"装配工厂"模式好处：
- 测试可以传 mock：`HybridRetriever(MockEmbedder(), MockStore(), ...)`
- 切换实现只改这一个函数
- 一眼能看出整个系统由哪些组件组成

---

## 实战：怎么加一个新实现

假设你想加一个 **DeepSeekAnswerer**（用 DeepSeek API 替代 Claude）：

### Step 1 看接口契约

```python
# src/jcontract/interfaces/answerer.py
class Answerer(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...
```

只需要实现一个 `answer()` 方法。

### Step 2 写实现

```python
# src/jcontract/impls/deepseek_answerer.py
from typing import ClassVar
from jcontract.interfaces import Answer, Chunk

class DeepseekAnswerer:
    backend: ClassVar[str] = "deepseek"

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self._api_key = api_key
        self._model = model
        # ... 初始化 HTTP client

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        # 1. 用 jcontract.answer.prompt.build_prompt() 复用现有 prompt
        # 2. 调 DeepSeek HTTP API
        # 3. 跑 postprocess 守约
        # 4. 返回 Answer 对象
        ...
```

### Step 3 注册到 CLI

```python
# src/jcontract/cli.py
def _maybe_build_answerer(backend: str = "claude-cli") -> Answerer | None:
    if backend == "deepseek":
        from jcontract.impls.deepseek_answerer import DeepseekAnswerer
        return DeepseekAnswerer(api_key=os.environ["DEEPSEEK_API_KEY"])
    ...
```

### Step 4 加测试

```python
# tests/impls/test_deepseek_answerer.py
def test_answer_returns_answer_object(): ...
def test_no_context_returns_fallback(): ...
def test_citations_are_validated(): ...
```

**其他地方一行不用改**。检索层、ingest 层、UI 层、评测层全部自动兼容。

---

## "N=2 即升级"规则

项目契约里有一条规则（[dev-contract/01-project-seasee.md](../dev-contract/01-project-seasee.md) §5）：

> 当第二个实现出现时，必须升级接口以兼容新旧，禁止造平行轮子。

**翻译成人话**：

- 第一次写 `ClaudeAnswerer` 时，接口可能不那么"通用"——为 Claude 特定参数留了字段
- **要加 `DeepseekAnswerer` 时**，必须先重审接口：通用部分提取到 Protocol，DeepSeek 特定参数走 `__init__()`
- 不允许 `ClaudeAnswerer.answer_with_streaming()` 和 `DeepseekAnswerer.stream_answer()` 这种**接口分叉**

这条规则保护了"切换实现 = 改一行配置"的承诺。

---

## 哲学：把"易变"和"稳定"分开

| 层 | 稳定性 | 变化原因 |
|---|---|---|
| `interfaces/` | 极稳定 | 业务需求变（半年改一次） |
| `impls/` | 不稳定 | 厂商涨价、模型更新（每月可能换） |
| `retrieve/` / `answer/` / `ingest/` | 稳定 | 业务逻辑变（季度级） |
| `cli.py` 装配段 | 不稳定 | 跟随 impls 变 |

接口层就像"宪法"——基本不动；impls 像"日常法规"——经常调整。把它们解耦，业务代码享受"宪法稳定"的红利。

---

## 下一步阅读

- 想看核心数据类型 → [05 Chunk 解剖](05-chunk-anatomy.md)
- 想看接口实例 → 直接读 [src/jcontract/interfaces/](../src/jcontract/interfaces/) 的文件，每个都很短
