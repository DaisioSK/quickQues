# 06 · PDF 解析与 OCR

> 前置章节：[05 Chunk 解剖](05-chunk-anatomy.md)
> 下一章 → [07 切块](07-chunking.md)

---

## 一句话

**把 PDF 变成"可搜索文字"——根据 PDF 是文字版还是扫描版，走不同的"翻译"路线。**

---

## 类比：翻译外文书

你拿到一本日语合同想看，有 3 种翻译路径：

| 路径 | 适用 | 速度 | 成本 |
|---|---|---|---|
| **复制文字直接查字典** | 电子书有文字层 | 秒级 | 0 |
| **找翻译公司（按字收费）** | 纸质书要拍照 | 慢 | 贵 |
| **包月翻译会员，请翻译员手抄** | 纸质书要拍照 | 慢 | 月费包干 |

j-contract 的 3 个 parser 就对应这三种：

| Parser | 适用 PDF | 速度 | 成本 |
|---|---|---|---|
| `PyPdfParser` | 文字版（电子签发的） | 毫秒 | 0 |
| `ClaudeVisionParser` | 扫描版（按 token 收费） | ~5s/页 | $0.015/页 (Sonnet) |
| `ClaudeCliVisionParser` ⭐ | 扫描版（订阅模式） | ~10s/页 | 包月不限量 |

---

## 三个 Parser 的代码长啥样

它们都实现同一个 Protocol：

```python
# src/jcontract/interfaces/parser.py
class PDFParser(Protocol):
    def parse(self, pdf_path: Path) -> list[ParsedPage]: ...
```

**输入**：PDF 路径
**输出**：每页一个 `ParsedPage(page_num, text, tables)`

下游（chunker）只看 `ParsedPage`，**不在乎你用哪个 parser**。

---

## ① PyPdfParser（纯文本提取）

**怎么工作**：pypdf 库直接读 PDF 的"文字层"。所有现代电子文档（Word 导出的 PDF、电子签发的合同）都有这一层。

```python
import pypdf

reader = pypdf.PdfReader(pdf_path)
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    yield ParsedPage(page_num=i + 1, text=text)
```

**毫秒级**，零成本。

**致命弱点**：DEMO 的 9 份 PDF **全是扫描件**——纸质合同扫描成 PDF，每页就是张大图片，**没有文字层可读**。pypdf 提取出来全是空字符串。

→ 所以这个 parser 在本项目里**没法用**。它存在是为了：
1. 测试时跑合成 PDF（我们用 fpdf2 生成的小 PDF 是文字版）
2. 给未来电子化的合同留接口

---

## ② ClaudeVisionParser（API OCR）

**怎么工作**：把每页 PDF **渲染成图片**，发给 Claude Vision API，让 Claude "读图说话"。

```python
# 简化版伪代码
pdf = pdfium.PdfDocument(pdf_path)
for page_idx in range(len(pdf)):
    pil_image = pdf[page_idx].render(scale=150/72).to_pil()
    jpeg_bytes = encode_jpeg(pil_image)

    # 调 Anthropic API
    response = anthropic.messages.create(
        model="claude-sonnet-4-6",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"data": base64(jpeg_bytes)}},
                {"type": "text", "text": "Extract all text..."},
            ],
        }],
    )
    text = response.content[0].text
```

**优点**：
- 处理扫描件、手写、印章、混排都行
- Vision 模型理解版面、表格、Q&A 结构
- 中英混杂没问题

**缺点**：
- **要 ANTHROPIC_API_KEY**（按 token 计费）
- Sonnet ~$0.015/页，Haiku ~$0.0025/页
- 4100 页全跑 Haiku ≈ $10，Sonnet ≈ $60

---

## ③ ClaudeCliVisionParser（订阅模式）⭐ 当前默认

这是 2026-05-29 新加的，**专门解决"我不想买 API key，我有 Claude Code 订阅"的需求**。

### 核心 trick

不直接调 SDK，而是**用 subprocess 调 `claude -p` 命令行**：

```python
# src/jcontract/impls/claude_cli_vision_parser.py
def _call_claude_cli(self, image_path, page_num, pdf_name):
    cmd = [
        "claude", "-p", prompt,
        "--model", "haiku",
        "--output-format", "json",
        "--allowedTools", "Read",       # 允许 Claude 用 Read 工具读图片
        "--add-dir", str(render_dir),   # 让 Claude 能访问图片目录
        "--permission-mode", "bypassPermissions",  # 自动允许
    ]
    result = subprocess.run(cmd, capture_output=True, ...)
    data = json.loads(result.stdout)
    return data["result"]
```

Claude CLI 用**你已经登录的 OAuth**（`claude login` 那次的），调用计入你的 Max/Pro 订阅 quota。**业务代码不接触任何 API key**。

### 工作流详解

```
1. pypdfium2 渲染 PDF 第 42 页 → JPEG 字节流
2. 算 SHA-256(JPEG) = "ab12cd..."
3. 检查 data/ocr_cache/ab12cd....text.txt 是否存在
   - 存在 → 直接返回缓存
   - 不存在 ↓
4. 把 JPEG 写到 data/_render_tmp/ab12cd....jpg
5. subprocess.run(["claude", "-p", "Use Read tool to open <jpg path>, then extract text...", ...])
6. Claude:
   - 收到 prompt 后调它的 Read 工具读那张图
   - 然后用 Vision 能力 OCR
   - 返回纯文本
7. 写缓存 data/ocr_cache/ab12cd....text.txt
8. 删 data/_render_tmp/ab12cd....jpg
```

### 性能 / 成本对比（实测）

| 维度 | claude-vision（API） | claude-cli-vision（订阅）|
|---|---|---|
| 单页延迟 | ~3-5s | ~10-15s |
| 单页成本 | $0.0025 (Haiku) | $0 marginal |
| Token overhead | 极小 | ~40k cache_creation tokens/call（Claude Code 系统上下文） |
| 鉴权 | API key | OAuth (`claude login`) |
| 4100 页总成本 | ~$10 | $0 |

**为什么订阅版慢 2-3 倍**：每次 `claude -p` 调用都要重启 Claude Code、加载系统 prompt、Read 工具用一次——这套"启动成本"在每页都重复一次。订阅免费但有隐形时间税。

---

## 内容相同的缓存设计（关键）

**两种 vision parser 使用同一个缓存目录**（`data/ocr_cache/`）：

- 缓存 key = SHA-256(JPEG 字节)
- JPEG 渲染参数相同（DPI 150 + quality 85）→ 同一页 PDF 算出同一个 hash

意味着：
1. 你今天用 `claude-vision`（API）跑了 50 页，缓存 50 个文件
2. 明天切到 `claude-cli-vision`（订阅）跑同一份 PDF
3. **直接命中缓存 50 次，0 调用**——两个 parser 缓存互通

**清缓存**：`rm -rf data/ocr_cache/`，下次跑会重 OCR。

---

## OCR Prompt 设计

[claude_cli_vision_parser.py:65](../src/jcontract/impls/claude_cli_vision_parser.py#L65) 的 prompt：

```
Use the Read tool to open the image at: {image_path}

Then extract ALL text from that image. Return ONLY the extracted text, preserving:
- Paragraph breaks (blank line between paragraphs).
- Section / Clause headers on their own lines.
- "Question No.:" and "Answer:" markers exactly as printed.
- Drawing No. references (e.g. T/PRJ/CWD/WS/2101A) verbatim.
- Revision markers (Rev A, Revision 0, etc.).
- Tables: render as plain text with " | " column separators.

Do NOT:
- Add commentary, summaries, or analysis.
- Translate any text (keep English as English).
- Skip handwritten annotations or stamps — transcribe inline with a [handwritten: ...] marker.

If the page is blank or contains no useful text, return exactly: <empty page>
```

**设计要点**：
1. **告诉它用 Read 工具**（订阅 parser 特有）—— Claude 不直接看 stdin，得通过 Read tool 主动读
2. **保留结构标记**（Q&A、Drawing No.、Revision、表格 `|`）—— 让下游 chunker 的正则能命中
3. **禁止翻译** —— 否则英文条款会被翻成中文，无法回溯原文
4. **手写用 `[handwritten:...]` 标记** —— 区分印刷文字和章戳/手批
5. **空页返回 `<empty page>`** —— 系统检测到这个字符串就跳过该页

---

## 关键决策（DECISION）

### 为什么 DPI 150 而不是 300

更高 DPI = 更清楚 = OCR 更准。但：
- 150 DPI 的 A4 页面 JPEG ≈ 100KB；300 DPI ≈ 400KB
- Vision 模型按图片大小算 token——大 4 倍意味着每页 4 倍 token
- 实测 150 已经够清楚识别建筑合同的小字

### 为什么 JPEG quality 85

- 90 以上视觉无差，但文件大很多
- 70 以下开始 OCR 错字
- 85 是行业老经验的"足够清楚 + 文件小"甜点

### 为什么 SHA-256 缓存 key（而不是 PDF path）

- 同一份 PDF 改名不影响 → 不会重 OCR
- 不同 PDF 但某页一字不差的话也共享缓存（罕见但发生过：模板页）
- hash 长 64 字符，路径冲突概率 0

---

## 错误处理

OCR 单页失败**不能整个批处理崩**。[claude_cli_vision_parser.py:188](../src/jcontract/impls/claude_cli_vision_parser.py#L188)：

```python
try:
    text = self._call_claude_cli(image_path, page_num, pdf_name)
except Exception as exc:
    logger.warning("cli_vision.error", pdf=pdf_name, page=page_num, ...)
    image_path.unlink(missing_ok=True)
    return ""  # ← 返回空字符串，chunker 会跳过空页
```

**契约**：parser 永远返回 `list[ParsedPage]`，单页失败 → text="" → chunker 自然忽略。**永不抛异常给上层**。

---

## 下一步阅读

- 文字进来之后怎么切 → [07 切块](07-chunking.md)
- 文字怎么变向量 → [08 Embedding](08-embedding.md)
