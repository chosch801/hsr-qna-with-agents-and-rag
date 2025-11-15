# hsr-qna-with-agents-and-rag

Demonstrating Agentic RAG's power with Honkai: Star Rail, this project automates complex QA through through intelligent agents that dynamically retrieve and synthesize information.

# 崩坏：星穹铁道智能问答系统

基于 RAG + Agent 的智能问答系统，支持本地知识库检索和网络搜索。

## 功能特点

- 🤖 **智能Agent模式**：自动选择最优回答策略
- 📚 **本地RAG检索**：基于游戏文档的精准问答
- 🌐 **联网搜索**：获取最新资讯和更新信息
- 🔄 **混合模式**：整合多源信息提供全面回答
- 🎨 **图形界面**：简易友好的 Tkinter GUI

## 系统架构

- IntelligentAgent：智能决策和策略选择
- RAGTool：本地知识库检索
- GoogleSearchTool：网络信息搜索
- IntegrationAgent：多源信息整合

## 回答模式

1. **智能Agent**：AI自动选择最优策略
2. **仅本地RAG**：仅使用本地知识库
3. **仅联网搜索**：仅使用Google搜索
4. **两者结合**：整合两种信息源

## 项目结构

```
├── final_code.py           # 主程序
├── star_rail_stories/      # 文档目录
├── faiss_index/            # 向量数据库
├── requirements.txt        # 依赖包
└── README.md              # 说明文档
```

## 环境要求

- Python 3.8+
- Ollama (本地LLM服务)，需要一定的版本才能使用qwen3，可替换为其他模型
- 必需的Python包

## 注意事项

- Google搜索需要你自己的API密钥
- 首次运行需要时间创建向量数据库，如有文档更新需要及时更新向量数据库
- 确保Ollama服务在 localhost:11434 运行
