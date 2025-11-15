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

## 环境要求

- Python 3.8+
- Ollama (本地LLM服务)，需要一定的版本才能使用qwen3，可替换为其他模型
- 必需的Python包
