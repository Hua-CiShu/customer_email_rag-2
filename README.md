# Customer Email Automation (Python + Streamlit 版)

这是对 Dify 工作流 **Customer Email Automation** 的完整重建：

```
User Input → Question Classifier → Knowledge Retrieval → LLM → Output
```

- **分类**：LLM 把客户邮件分为「物流/配送」或「退款/退货/换货」两类
- **检索**：按分类用 embedding 在对应知识库中检索 Top-K 相关段落（纯 NumPy 实现，无需向量数据库）
- **生成**：LLM 基于检索到的知识撰写客服回复邮件
- **缓存**：embedding 结果写入 `.embed_cache.npz`，重启后无需重复计算

## 快速开始

```bash
cd customer_email_rag
pip install -r requirements.txt
streamlit run app.py
```

打开页面后，在**左侧边栏粘贴你的 API Key** 即可使用 —— 网关地址
（HKUST GenAI）和模型名已内置为默认值，无需在终端配置任何环境变量。
每个使用者输入自己的 key；key 只保存在当前浏览器会话里，不会写入磁盘。

（高级用法：`.env` 环境变量仍可覆盖默认值，参考 `.env.example`。)

## 文件结构

| 文件 | 说明 |
|---|---|
| `app.py` | Streamlit 界面（输入表单、流程状态、回复展示、检索结果查看） |
| `rag.py` | 核心逻辑：分类 / 分块 / embedding / 检索 / 生成 |
| `knowledge/shipping_policy.md` | 物流政策知识库（Class 1）— 替换为你自己的文档 |
| `knowledge/refund_policy.md` | 退款退货知识库（Class 2）— 替换为你自己的文档 |
| `.env.example` | 环境变量模板 |

## 更换知识库

直接把你在 Dify 里上传的两份文档内容替换进 `knowledge/` 下的两个 markdown
文件（或新增文件并修改 `rag.py` 里的 `CLASS_INFO`），然后在侧边栏点
「🔄 Rebuild embedding index」重建索引即可。

## 兼容其他模型

设置 `OPENAI_BASE_URL` 即可接入任何 OpenAI 兼容接口（DeepSeek、Moonshot、
Azure、本地 vLLM 等）；embedding 模型同理，可用 `EMBED_MODEL` 覆盖。
