


# Agentic RAG Pipeline with LangGraph

这是一个基于 **LangGraph** 构建的有状态多智能体 RAG 系统。系统通过“自我反思”机制（Self-Reflective Loop），在检索质量不达标时自动触发查询重写，从而确保复杂技术咨询场景下的回答准确性。

---

## 系统架构 (Architecture)

本系统采用有状态图结构编排不同职能的智能体节点：

graph TD
    User([用户提问]) --> Researcher[Researcher: 向量检索]
    Researcher --> Analyst{Analyst: 质量评估}
    Analyst -- "内容不相关/不足" --> Rewriter[Rewriter: 查询优化]
    Rewriter -- "重试次数 +1" --> Researcher
    Analyst -- "满足回答要求" --> Generator[Generator: 生成报告]
    Generator --> End([最终输出])
    Analyst -- "达到重试上限" --> Fail[提示并终止]
    Fail --> End



---

##  核心特性 (Key Features)

* **多智能体协同机制**：
* **Researcher**: 负责从 FAISS 向量库中检索相关技术文档。
* **Analyst**: 作为逻辑大脑，评估检索内容的相关性，决定进入回答阶段或触发重写。
* **Rewriter**: 利用 LLM 将模糊或口语化的提问转化为专业的搜索关键词，提升检索召回率。


* **状态管理与持久化**: 使用 `TypedDict` 维护全局状态（State），确保 `context`、`steps` 和 `retry_count` 在节点间稳定传递。
* **防御性工程设计**:
* **路径鲁棒性**: 针对 Windows 环境下 FAISS 底层 C++ 接口对中文路径支持不佳的问题，实施了绝对路径标准化处理。
* **循环保护**: 设定 `max_retries` 机制，防止智能体在无效检索路径上产生无限 Token 消耗。


* **实时可视化**: 基于 **Streamlit** 开发监控界面，侧边栏实时流式输出智能体的“思考路径”（Thought Log）。

---

##  技术栈 (Tech Stack)

* **编排框架**: LangGraph (Stateful Framework)
* **大语言模型**: GPT-4o / GPT-4o-mini (via LangChain LCEL)
* **向量数据库**: FAISS
* **嵌入模型**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **交互界面**: Streamlit

---

##  快速启动 (Quick Start)

### 1. 安装环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. 环境变量配置

在根目录新建 `.env` 文件并填入：

```env
OPENAI_API_KEY=your_openai_api_key

```

### 3. 文档处理 (Ingestion)

将 PDF 放置于 `data/raw/` 目录，执行索引构建：

```bash
python src/ingest.py

```

### 4. 启动咨询平台

```bash
streamlit run src/app.py

```

```

---

需要我为你准备一个专门适配此项目的 `.gitignore` 文件，以防止将庞大的索引文件或虚拟环境上传到 GitHub 吗？

```
