# 🦜 LangChain 1.0 个人学习总结

[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0-orange.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 系统学习 LangChain 1.0 过程中的个人总结，包含自己归纳的 4 套工程化调用模板、完整 RAG 检索链路、MCP 服务端/客户端、多 Agent A2A 协作等实战项目，整理为可复用的参考框架。

---

## 📝 关于本项目

以下内容均为本人在学习 LangChain 1.0 过程中独立总结整理，非搬运官方文档。每个代码示例都经过本地运行验证，涵盖了从基础调用到企业级多 Agent 协作的完整链路，可直接作为项目参考框架使用。

---

## 📖 内容

| 文件 | 说明 |
|------|------|
| [langchain1.0框架：.md](./langchain1.0框架：.md) | 核心总结，13 章：模型接入 → Prompt → 输出解析 → LCEL → 记忆 → 工具 → RAG → MCP → Agent |
| [claude code.md](./claude%20code.md) | Claude Code 辅助工具资源汇总 |

---

## 🧠 知识体系

### 模型接入与调用
- DeepSeek / 通义千问 / Ollama 本地模型
- 从基础 Demo 到企业级调用模板（日志 + 异常处理）
- `invoke` / `stream` / `batch`，同步 + 异步全覆盖

### Prompt 工程
- `PromptTemplate` / `ChatPromptTemplate` / `MessagesPlaceholder`
- 联合拼接、预设变量、外部 JSON 文件加载

### 结构化输出
- `JsonOutputParser` / `PydanticOutputParser` / `TypedDict`
- Pydantic Field 约束 + 自定义 validator 校验

### LCEL 链式调用
- 顺序链、分支链、并行链、函数链、子链嵌套

### 多轮对话记忆
- `RedisChatMessageHistory` + `RunnableWithMessageHistory` 持久化

### Tool 工具调用
- `@tool` 装饰器 + `bind_tools` + 工具链构建

### RAG 检索增强生成
```
文档加载 → 文本分割 → Embedding 向量化 → Redis 向量检索 → LLM 问答
```
- TXT / PDF / Markdown / JSON / Word / CSV 六种文档格式
- 阿里云 DashScope Embedding + Redis 向量存储

### MCP 模型上下文协议
- FastMCP 服务端搭建 + MultiServerMCPClient 客户端 + SSE / STDIO

### Agent 智能体
- 单 Agent 工具调用 + 多 Agent A2A 协作（完整跨平台案例）

### 工程规范
- `.env` 安全管理 / Loguru 日志 / 异常分类捕获

---

## 🛠 技能覆盖

| 方向 | 能力 |
|------|------|
| LLM 应用开发 | LangChain 1.0 全链路开发 |
| Prompt 设计 | 多模板方案，结构化输出约束与校验 |
| RAG 系统 | 文档处理 → 向量化 → 检索问答，端到端搭建 |
| Agent 开发 | 单/多智能体工具调用与协作 |
| MCP 协议 | 服务端开发与客户端集成 |
| 多模型集成 | DeepSeek / Qwen / Ollama 本地部署 |

---

## 🚀 环境

```bash
pip install langchain langgraph python-dotenv loguru
```

- Python 3.10+ / LangChain 1.0+
- Redis（记忆缓存与向量存储场景，可选）

---

## ⚠️ 注意

文档中 API Key 均为示例占位符，实际使用通过 `.env` 配置，**切勿提交真实 Key**。

---

## 📄 License

MIT License — 欢迎 Star ⭐、Fork 和 PR！
