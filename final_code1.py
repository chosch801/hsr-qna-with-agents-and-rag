import os
import requests
import time
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import threading
import json
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import re
from enum import Enum

class AnswerMode(Enum):
    INTELLIGENT = "intelligent"  # 智能代理
    RAG_ONLY = "rag_only"       # 仅本地RAG
    GOOGLE_ONLY = "google_only" # 仅联网搜索
    BOTH = "both"               # 两者结合

class GoogleSearchTool:
    """Google搜索工具 - 供Agent调用"""
    def __init__(self, api_key=None, search_engine_id=None):
        # 从环境变量或配置文件读取
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    def search(self, query, num_results=10):
        """执行Google搜索"""
        try:
            # 为崩坏星穹铁道相关搜索添加游戏名称，提高搜索精度
            enhanced_query = f"崩坏星穹铁道 {query}"
            
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': enhanced_query,
                'num': num_results,
                'lr': 'lang_zh'
            }
            
            response = requests.get(self.base_url, params=params, timeout=300)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Google搜索'
                    })
            
            return results
        except Exception as e:
            print(f"Google搜索失败: {str(e)}")
            return self._mock_search_results(query)
    
    def _mock_search_results(self, query):
        """模拟搜索结果"""
        return [
            {
                'title': f'崩坏星穹铁道 - {query} 官方介绍',
                'link': 'https://sr.mihoyo.com/news',
                'snippet': f'崩坏星穹铁道中关于{query}的详细介绍。{query}作为游戏的重要元素，拥有独特的设定和丰富的背景故事。在游戏世界中，{query}扮演着重要的角色，影响着主线剧情的发展...',
                'source': 'Google搜索(模拟)'
            },
            {
                'title': f'{query} - 星穹铁道角色攻略与解析', 
                'link': 'https://wiki.biligame.com/sr',
                'snippet': f'{query}的技能机制、配队建议和培养攻略。作为崩坏星穹铁道的重要角色，{query}具有独特的战斗风格和强大的技能组合，在团队中承担重要职责...',
                'source': 'Google搜索(模拟)'
            },
            {
                'title': f'崩坏星穹铁道{query}深度解析 - 世界观探究',
                'link': 'https://bbs.mihoyo.com/sr',
                'snippet': f'深度分析{query}在崩坏星穹铁道中的地位和作用。从世界观设定到实际应用，全面解读{query}的各个方面，包括背景故事、能力设定等...',
                'source': 'Google搜索(模拟)'
            }
        ]

class RAGTool:
    """RAG检索工具 - 供Agent调用"""
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
    
    def search(self, query):
        """执行RAG检索"""
        try:
            result = self.rag_chain({"query": query})
            return {
                'success': True,
                'answer': result['result'],
                'source_documents': result.get('source_documents', [])
            }
        except Exception as e:
            return {
                'success': False,
                'answer': f"RAG检索失败: {str(e)}",
                'source_documents': []
            }

class SearchAnswerGenerator:
    """搜索结果回答生成器"""
    def __init__(self, llm):
        self.llm = llm
        self.answer_prompt = """
        你是崩坏星穹铁道的专业解答员。请基于以下搜索结果回答用户问题。

        用户问题：{question}

        搜索结果：
        {search_results}

        回答要求：
        1. 基于搜索结果提供准确完整的回答
        2. 保持游戏世界观的一致性
        3. 使用游戏术语和官方表达
        4. 整合多个搜索结果的信息
        5. 回答要有逻辑性和条理性
        6. 如果搜索结果不足，请明确说明

        请提供详细的回答：
        """
    
    def generate_answer(self, question, search_results):
        """基于搜索结果生成回答"""
        try:
            # 格式化搜索结果
            results_text = ""
            for i, result in enumerate(search_results, 1):
                results_text += f"{i}. 标题：{result['title']}\n"
                results_text += f"   内容：{result['snippet']}\n"
                results_text += f"   来源：{result['link']}\n\n"
            
            prompt = self.answer_prompt.format(
                question=question,
                search_results=results_text
            )
            
            answer = self.llm.invoke(prompt)
            return answer
        except Exception as e:
            return f"回答生成失败: {str(e)}"

class IntelligentAgent:
    """智能Agent - 使用工具进行推理和回答"""
    def __init__(self, llm, google_tool, rag_tool):
        self.llm = llm
        self.google_tool = google_tool
        self.rag_tool = rag_tool
        self.search_generator = SearchAnswerGenerator(llm)
        
        # 改进的Agent系统提示词
        self.system_prompt = """你是崩坏星穹铁道的智能助手Agent。

你需要根据用户问题智能选择信息源来回答：

1. 游戏内容问题（角色、剧情、设定等）：使用本地知识库
2. 最新资讯问题（版本更新、活动等）：使用网络搜索
3. 如果本地知识库没有足够信息，自动切换到网络搜索

请按以下步骤回答：
1. 分析问题类型
2. 选择合适的信息源
3. 基于获得的信息提供完整回答

用户问题：{question}"""
    
    def process_question(self, question):
        """处理用户问题"""
        try:
            # 首先尝试RAG搜索
            rag_result = self.rag_tool.search(question)
            
            # 判断RAG结果质量
            if rag_result['success'] and self._is_rag_result_good(rag_result['answer']):
                return {
                    'answer': rag_result['answer'],  # 不再清理响应
                    'strategy': 'intelligent_agent_rag',
                    'sources': self._format_rag_sources(rag_result),
                    'method': 'RAG检索',
                    'raw_response': rag_result['answer']
                }
            
            # RAG结果不好，随机选择策略：45%仅联网搜索，55%两者结合
            print("RAG结果不理想，智能Agent随机选择后续策略...")
            import random
            
            if random.random() < 0.30:
                # 30%概率：仅联网搜索
                print("智能Agent选择：仅联网搜索")
                search_results = self.google_tool.search(question)
                
                if search_results:
                    search_answer = self.search_generator.generate_answer(question, search_results)
                    return {
                        'answer': search_answer,  # 不再清理响应
                        'strategy': 'intelligent_agent_search',
                        'sources': self._format_search_sources(search_results),
                        'method': '网络搜索',
                        'raw_response': search_answer
                    }
                else:
                    # 搜索失败，降级到两者结合
                    print("网络搜索失败，降级到两者结合...")
                    return self._use_combined_strategy(question)
            else:
                # 55%概率：两者结合
                print("智能Agent选择：两者结合")
                return self._use_combined_strategy(question)
            
        except Exception as e:
            return {
                'answer': f"Agent处理失败: {str(e)}",
                'strategy': 'error',
                'sources': [],
                'method': '错误'
            }
    
    def _use_combined_strategy(self, question):
        """两者结合策略"""
        try:
            # 获取RAG结果（即使质量不好也要获取）
            rag_result = self.rag_tool.search(question)
            
            # 获取搜索结果
            search_results = self.google_tool.search(question)
            
            # 使用IntegrationAgent整合信息
            integration_agent = IntegrationAgent(self.llm)
            
            # 准备信息进行整合
            rag_info = None
            if rag_result['success'] and rag_result['answer']:
                rag_info = rag_result
            
            # 整合信息
            if rag_info or search_results:
                integrated_answer = integration_agent.integrate_information(
                    question, rag_info, search_results
                )
                
                # 合并来源
                sources = []
                if rag_info:
                    sources.extend(self._format_rag_sources(rag_info))
                if search_results:
                    sources.extend(self._format_search_sources(search_results))
                
                return {
                    'answer': integrated_answer,  # 不再清理响应
                    'strategy': 'intelligent_agent_combined',
                    'sources': sources,
                    'method': '两者结合',
                    'raw_response': integrated_answer
                }
            
            # 如果都失败了，使用基础LLM回答
            basic_answer = self._basic_llm_answer(question)
            return {
                'answer': basic_answer,  # 不再清理响应
                'strategy': 'intelligent_agent_basic',
                'sources': [],
                'method': '基础回答',
                'raw_response': basic_answer
            }
            
        except Exception as e:
            # 如果组合策略也失败，使用基础LLM回答
            basic_answer = self._basic_llm_answer(question)
            return {
                'answer': basic_answer,  # 不再清理响应
                'strategy': 'intelligent_agent_basic',
                'sources': [],
                'method': '基础回答',
                'raw_response': basic_answer
            }
    
    def _is_rag_result_good(self, answer):
        """判断RAG结果质量"""
        if not answer or len(answer.strip()) < 20:
            return False
        
        # 检查是否包含无关回答的关键词
        bad_indicators = [
            "上下文不足", "无法找到", "没有相关信息", 
            "不在提供的上下文中", "抱歉", "无法回答",
            "无法基于提供的上下文", "没有足够的信息"
        ]
        
        for indicator in bad_indicators:
            if indicator in answer:
                return False
        
        return True
    
    def _clean_response(self, response):
        """不再清理响应，直接返回原始响应"""
        return response if response else ""
    
    def _basic_llm_answer(self, question):
        """基础LLM回答"""
        try:
            prompt = self.system_prompt.format(question=question)
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"基础回答生成失败: {str(e)}"
    
    def _format_rag_sources(self, rag_result):
        """格式化RAG来源"""
        sources = []
        if rag_result.get('source_documents'):
            for doc in rag_result['source_documents'][:3]:
                sources.append({
                    'type': '本地知识库',
                    'content': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
        return sources
    
    def _format_search_sources(self, search_results):
        """格式化搜索来源"""
        sources = []
        for result in search_results[:3]:
            sources.append({
                'type': 'Google搜索',
                'content': f"{result['title']}: {result['snippet'][:100]}..."
            })
        return sources

class DecisionAgent:
    """决策代理 - 智能决定回答策略"""
    def __init__(self, llm):
        self.llm = llm
        self.decision_prompt = """
        你是一个智能决策代理，需要分析用户问题并决定最佳回答策略。

        问题类型分析：
        1. 本地知识问题：关于崩坏星穹铁道的角色、剧情、设定等游戏内容
        2. 实时信息问题：需要最新资讯、版本更新、活动信息等
        3. 综合分析问题：需要结合游戏内容和外部信息的复杂问题

        回答策略：
        - rag_only: 仅使用本地RAG系统（适合游戏设定、角色分析等）
        - google_only: 仅使用Google搜索（适合实时资讯、版本更新等）
        - both: 结合两者（适合需要深度分析的复杂问题）

        用户问题：{question}

        请分析问题类型并选择最佳策略，只返回：rag_only、google_only 或 both
        """
    
    def decide_strategy(self, question):
        """决定回答策略"""
        try:
            prompt = self.decision_prompt.format(question=question)
            response = self.llm.invoke(prompt)
            
            # 提取策略
            strategy = response.strip().lower()
            if strategy in ['rag_only', 'google_only', 'both']:
                return strategy
            
            # 基于关键词的fallback逻辑
            return self._keyword_based_decision(question)
        except Exception as e:
            print(f"决策失败，使用默认策略: {str(e)}")
            return self._keyword_based_decision(question)
    
    def _keyword_based_decision(self, question):
        """基于关键词的决策逻辑"""
        question_lower = question.lower()
        
        # 实时信息关键词
        realtime_keywords = ["最新", "版本", "更新", "活动", "新闻", "今天", "现在", "什么时候"]
        if any(keyword in question_lower for keyword in realtime_keywords):
            return "google_only"
        
        # 游戏内容关键词
        game_keywords = ["角色", "剧情", "设定", "世界观", "技能", "命途", "星神"]
        if any(keyword in question_lower for keyword in game_keywords):
            return "rag_only"
        
        # 默认使用本地RAG
        return "rag_only"

class IntegrationAgent:
    """整合代理 - 整合多源信息"""
    def __init__(self, llm):
        self.llm = llm
        self.search_generator = SearchAnswerGenerator(llm)
        self.integration_prompt = """
        你是信息整合专家，需要将来自不同来源的信息整合成一个完整、准确的答案。

        用户问题：{question}

        本地知识库信息：
        {rag_info}

        网络搜索信息：
        {search_info}

        整合要求：
        1. 综合分析所有信息源
        2. 识别并解决信息冲突
        3. 提供完整、准确的答案
        4. 明确标注信息来源
        5. 如果信息不足或冲突，请说明

        请提供整合后的答案：
        """
    
    def integrate_information(self, question, rag_result=None, search_results=None):
        """整合信息生成最终答案"""
        try:
            # 准备信息
            rag_info = rag_result.get('answer', '无本地信息') if rag_result else '无本地信息'
            
            search_info = "无网络信息"
            if search_results:
                search_texts = []
                for result in search_results:
                    search_texts.append(f"标题：{result['title']}\n内容：{result['snippet']}")
                search_info = "\n\n".join(search_texts)
            
            # 生成整合答案
            prompt = self.integration_prompt.format(
                question=question,
                rag_info=rag_info,
                search_info=search_info
            )
            
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"信息整合失败: {str(e)}"

class AdvancedPromptSystem:
    def __init__(self):
        self.prompt_templates = {
            "default": self._base_template(),
            "character": self._character_template(),
            "storyline": self._storyline_template(),
            "relationship": self._relationship_template(),
            "technical": self._technical_template()
        }
        self.template_keywords = {
            "relationship": ["关系", "互动", "相处", "之间", "与", "和", "什么关系"],
            "character": ["角色", "人物", "性格", "特征", "形象", "是谁", "介绍", "背景"],
            "storyline": ["剧情", "故事", "任务", "情节", "主线", "支线", "发展", "结局", "讲了什么"],
            "technical": ["技术", "项目", "开发", "原理", "机制", "模拟宇宙", "系统"]
        }
        
    def get_template(self, question):
        question_lower = question.lower()
        for key, keywords in self.template_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return self.prompt_templates[key]
        return self.prompt_templates["default"]
    
    def _base_template(self):
        return """
        作为《崩坏：星穹铁道》资深分析师，请基于以下上下文回答问题：
        {context}
        
        问题：{question}
        
        回答要求：
        1. 严格基于提供的上下文
        2. 保持游戏世界观一致性
        3. 使用游戏内术语
        4. 如果上下文不足，明确说明
        """

    def _relationship_template(self):
        return """
        【角色关系分析任务】
        上下文：{context}
        问题：{question}

        分析要求：
        1. 识别关系双方的角色定位
        2. 梳理从初见到当前的关系发展阶段
        3. 引用至少2个关键互动事件
        4. 分析关系对双方角色发展的影响
        5. 评估关系在整体剧情中的重要性

        回答格式：
        - 关系概述：[简要总结]
        - 发展阶段：[按时间线说明]
        - 关键事件：[引用具体互动]
        - 影响分析：[对角色和剧情的影响]
        """

    def _storyline_template(self):
        return """
        作为剧情分析师，请基于以下上下文解析游戏剧情：
        {context}
        
        问题：{question}
        
        回答指南：
        1. 梳理时间线和关键事件
        2. 分析剧情转折点
        3. 解释剧情背后的世界观设定
        4. 预测可能的后续发展
        """

    def _character_template(self):
        return """
        作为角色分析专家，请基于以下上下文分析角色：
        {context}
        
        问题：{question}
        
        回答指南：
        1. 分析角色背景故事
        2. 描述角色性格特征
        3. 说明角色在剧情中的作用
        4. 使用角色经典台词作为佐证
        """

    def _technical_template(self):
        return """
        【技术解析任务】
        上下文：{context}
        问题：{question}

        分析要求：
        1. 明确技术/项目的名称和所属领域
        2. 解释基本工作原理
        3. 说明主要开发者/负责组织
        4. 分析该技术对游戏世界的影响
        5. 对比类似技术（如有）
        6. 如果上下文不足，明确说明

        回答格式：
        - 技术名称：[名称]
        - 原理说明：[分步骤解释]
        - 开发者信息：[相关人物/组织]
        - 影响分析：[对世界观的影响]
        """

class StarRailRAGSystem:
    def __init__(self, config):
        self.config = config
        self.embedding_function = self._init_embedding_function()
        self.llm = Ollama(model=config["generation_model"])
        self.splits = None
        self.prompt_system = AdvancedPromptSystem()
        self.rag_chain = None  # 初始化为None，稍后构建
        
        # 初始化工具
        self.google_tool = GoogleSearchTool()
        self.rag_tool = None  # 稍后初始化
        self.search_generator = SearchAnswerGenerator(self.llm)
        
        # 初始化代理
        self.decision_agent = DecisionAgent(self.llm)
        self.integration_agent = IntegrationAgent(self.llm)
        self.intelligent_agent = None  # 稍后初始化
        
    def _init_embedding_function(self):
        """初始化嵌入函数"""
        class OllamaEmbeddingFunction:
            def __init__(self, model_name, api_url):
                self.model_name = model_name
                self.api_url = api_url
            
            def __call__(self, text):
                payload = {"model": self.model_name, "prompt": text}
                try:
                    response = requests.post(self.api_url, json=payload, timeout=60)
                    response.raise_for_status()
                    return response.json()["embedding"]
                except Exception as e:
                    print(f"嵌入生成失败: {str(e)}")
                    return []
        
        return OllamaEmbeddingFunction(
            model_name=self.config["embedding_model"],
            api_url=self.config["ollama_api_url"]
        )
    
    def load_and_process_documents(self):
        """加载并处理文档"""
        folder_path = self.config["folder_path"]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已创建文件夹: {folder_path}")
            return []
        
        documents = []
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        
        if not txt_files:
            print(f"警告: {folder_path} 中没有找到任何.txt文件")
            return []
        
        for filename in tqdm(txt_files, desc="加载文档"):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"加载文件 {filename} 失败: {str(e)}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        self.splits = splits
        return splits
    
    def create_vectorstore(self, splits):
        """创建向量数据库"""
        texts = [doc.page_content for doc in splits]
        metadatas = [doc.metadata for doc in splits]
        
        embeddings = []
        for text in tqdm(texts, desc="生成嵌入向量"):
            embedding = self.embedding_function(text)
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0]*768)
        
        vectorstore = FAISS.from_embeddings(
            text_embeddings=zip(texts, embeddings),
            embedding=self.embedding_function,
            metadatas=metadatas
        )
        vectorstore.save_local(self.config["vector_db_path"])
        return vectorstore
    
    def load_vectorstore(self):
        """加载已有的向量数据库"""
        return FAISS.load_local(
            self.config["vector_db_path"],
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True
        )
    
    def build_rag_chain(self):
        """构建RAG链"""
        try:
            vectorstore = self.load_vectorstore()
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}，正在重建...")
            if not self.splits:
                self.splits = self.load_and_process_documents()
            if self.splits:
                vectorstore = self.create_vectorstore(self.splits)
            else:
                print("无法创建向量数据库，RAG功能将不可用")
                return None
        
        prompt_template = PromptTemplate(
            template=self.prompt_system.get_template(""),
            input_variables=["context", "question"]
        )
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(k=self.config["retrieve_k"]),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        # 初始化RAG工具和智能Agent
        self.rag_tool = RAGTool(self.rag_chain)
        self.intelligent_agent = IntelligentAgent(self.llm, self.google_tool, self.rag_tool)
        
        return self.rag_chain
    
    def intelligent_answer(self, question, mode=AnswerMode.INTELLIGENT):
        """智能回答 - 集成所有代理功能"""
        try:
            if mode == AnswerMode.INTELLIGENT:
                # 使用智能Agent
                if self.intelligent_agent:
                    return self.intelligent_agent.process_question(question)
                else:
                    # 如果智能Agent未初始化，降级到决策模式
                    strategy = self.decision_agent.decide_strategy(question)
                    mode = AnswerMode(strategy)
            
            # 处理其他模式
            rag_result = None
            search_results = None
            
            # 执行相应策略
            if mode in [AnswerMode.RAG_ONLY, AnswerMode.BOTH]:
                if self.rag_chain:
                    rag_result = self.rag_chain({"query": question})
                else:
                    return {
                        'answer': "RAG系统未初始化，请检查文档是否存在",
                        'strategy': 'error',
                        'sources': []
                    }
            
            if mode in [AnswerMode.GOOGLE_ONLY, AnswerMode.BOTH]:
                search_results = self.google_tool.search(question)
            
            # 生成最终答案
            if mode == AnswerMode.RAG_ONLY:
                return {
                    'answer': rag_result['result'],
                    'strategy': mode.value,
                    'sources': self._format_rag_sources(rag_result)
                }
            elif mode == AnswerMode.GOOGLE_ONLY:
                # 使用LLM生成基于搜索结果的回答，而不是直接列出搜索结果
                if search_results:
                    answer = self.search_generator.generate_answer(question, search_results)
                else:
                    answer = "未找到相关搜索结果"
                
                return {
                    'answer': answer,
                    'strategy': mode.value,
                    'sources': self._format_search_sources(search_results)
                }
            else:  # BOTH
                integrated_answer = self.integration_agent.integrate_information(
                    question, rag_result, search_results
                )
                return {
                    'answer': integrated_answer,
                    'strategy': mode.value,
                    'sources': self._format_combined_sources(rag_result, search_results)
                }
        
        except Exception as e:
            return {
                'answer': f"回答生成失败: {str(e)}",
                'strategy': 'error',
                'sources': []
            }
    
    def _format_search_answer(self, search_results):
        """格式化搜索结果为答案（已弃用）"""
        if not search_results:
            return "未找到相关搜索结果"
        
        answer_parts = ["基于网络搜索结果：\n"]
        for i, result in enumerate(search_results, 1):
            answer_parts.append(f"{i}. {result['title']}")
            answer_parts.append(f"   {result['snippet']}")
            answer_parts.append("")
        
        return "\n".join(answer_parts)
    
    def _format_rag_sources(self, rag_result):
        """格式化RAG来源"""
        sources = []
        if rag_result and 'source_documents' in rag_result:
            for i, doc in enumerate(rag_result['source_documents'], 1):
                sources.append({
                    'type': '本地文档',
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata
                })
        return sources
    
    def _format_search_sources(self, search_results):
        """格式化搜索来源"""
        sources = []
        if search_results:
            for result in search_results:
                sources.append({
                    'type': 'Google搜索',
                    'content': f"{result['title']}: {result['snippet']}",
                    'metadata': {'link': result['link']}
                })
        return sources
    
    def _format_combined_sources(self, rag_result, search_results):
        """格式化组合来源"""
        sources = []
        sources.extend(self._format_rag_sources(rag_result))
        sources.extend(self._format_search_sources(search_results))
        return sources

# 配置参数
script_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "folder_path": os.path.join(script_dir, "star_rail_stories"),
    "embedding_model": "bge-m3:latest",
    "generation_model": "qwen3:8b-q4_K_M",
    "vector_db_path": os.path.join(script_dir, "faiss_index"),
    "ollama_api_url": "http://localhost:11434/api/embeddings",
    "chunk_size": 512,
    "chunk_overlap": 128,
    "retrieve_k": 16
}

# 初始化系统
print("正在初始化星穹铁道智能问答系统...")
start_time = time.time()

rag_system = StarRailRAGSystem(config)

if not os.path.exists(config["vector_db_path"]):
    print("未找到向量数据库，正在创建...")
    splits = rag_system.load_and_process_documents()
    if splits:
        vectorstore = rag_system.create_vectorstore(splits)
        print("向量数据库创建完成")
    else:
        print("警告：没有文档可处理，部分功能可能无法正常运行")
else:
    print("已加载现有向量数据库")

rag_chain = rag_system.build_rag_chain()
print(f"系统初始化完成，耗时: {time.time() - start_time:.2f}秒")

class StarRailGUI:
    def __init__(self, rag_chain, rag_system):
        self.rag_chain = rag_chain
        self.rag_system = rag_system
        
        # 初始化GUI
        self.root = tk.Tk()
        self.root.title("崩坏：星穹铁道智能问答系统 (Agent增强版)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")
        
        # 设置字体
        self.title_font = ("微软雅黑", 16, "bold")
        self.text_font = ("微软雅黑", 12)
        self.small_font = ("微软雅黑", 10)
        
        # 当前模式
        self.current_mode = AnswerMode.INTELLIGENT
        
        # 创建UI元素
        self.create_widgets()
    
    def create_widgets(self):
        """创建界面控件"""
        # 标题
        title_frame = tk.Frame(self.root, bg="#6a0dad")
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(title_frame, text="崩坏：星穹铁道智能问答系统 (Agent增强版)", 
                font=self.title_font, fg="white", bg="#6a0dad").pack(pady=10)
        
        # 模式选择区
        mode_frame = tk.LabelFrame(self.root, text="回答模式", font=self.text_font, 
                                 bg="#f0f0f0", padx=10, pady=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value="intelligent")
        modes = [
            ("智能Agent (推荐)", "intelligent"),
            ("仅本地RAG", "rag_only"),
            ("仅联网搜索", "google_only"),
            ("两者结合", "both")
        ]
        
        for text, value in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=value, font=self.text_font, bg="#f0f0f0",
                          command=self.on_mode_change).pack(side=tk.LEFT, padx=10)
        
        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(status_frame, textvariable=self.status_var, font=self.small_font, 
                bg="#f0f0f0", fg="#333333").pack(side=tk.LEFT)
        
        self.strategy_var = tk.StringVar(value="")
        tk.Label(status_frame, textvariable=self.strategy_var, font=self.small_font,
                bg="#f0f0f0", fg="#666666").pack(side=tk.RIGHT)
        
        # 问题输入区
        question_frame = tk.LabelFrame(self.root, text="提问区", font=self.text_font, 
                                     bg="#f0f0f0", padx=10, pady=10)
        question_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.question_entry = scrolledtext.ScrolledText(question_frame, height=4, 
                                                      font=self.text_font, wrap=tk.WORD)
        self.question_entry.pack(fill=tk.X, expand=True)
        self.question_entry.focus()
        
        # 按钮区
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ask_button = tk.Button(button_frame, text="开始提问", font=self.text_font, 
                             command=self.ask_question, bg="#4CAF50", fg="white")
        ask_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(button_frame, text="清空窗口", font=self.text_font, 
                               command=self.clear_fields, bg="#f44336", fg="white")
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # 回答显示区
        answer_frame = tk.LabelFrame(self.root, text="智能回答", font=self.text_font, 
                                   bg="#f0f0f0", padx=10, pady=10)
        answer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.answer_text = scrolledtext.ScrolledText(answer_frame, height=12, 
                                                   font=self.text_font, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.BOTH, expand=True)
        self.answer_text.config(state=tk.DISABLED)
        
        # 来源显示区
        source_frame = tk.LabelFrame(self.root, text="信息来源", font=self.text_font, 
                                   bg="#f0f0f0", padx=10, pady=10)
        source_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        # 创建Treeview显示来源
        columns = ("type", "content")
        self.source_tree = ttk.Treeview(source_frame, columns=columns, show="headings", height=6)
        self.source_tree.heading("type", text="来源类型")
        self.source_tree.heading("content", text="内容摘要")
        self.source_tree.column("type", width=120, stretch=tk.NO)
        self.source_tree.column("content", stretch=tk.YES)
        
        scrollbar_source = ttk.Scrollbar(source_frame, orient="vertical", command=self.source_tree.yview)
        self.source_tree.configure(yscrollcommand=scrollbar_source.set)
        
        self.source_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_source.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 底部信息
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(bottom_frame, text="智能Agent系统 | RAG优先 | 概率策略 (45%搜索/55%组合)", 
                font=self.small_font, bg="#f0f0f0", fg="#666666").pack(side=tk.LEFT)
    
    def on_mode_change(self):
        """模式改变处理"""
        mode_map = {
            "intelligent": AnswerMode.INTELLIGENT,
            "rag_only": AnswerMode.RAG_ONLY,
            "google_only": AnswerMode.GOOGLE_ONLY,
            "both": AnswerMode.BOTH
        }
        self.current_mode = mode_map[self.mode_var.get()]
        
        mode_names = {
            AnswerMode.INTELLIGENT: "智能Agent模式",
            AnswerMode.RAG_ONLY: "本地RAG模式",
            AnswerMode.GOOGLE_ONLY: "联网搜索模式",
            AnswerMode.BOTH: "混合模式"
        }
        self.status_var.set(f"已切换到{mode_names[self.current_mode]}")
    
    def ask_question(self):
        """提问处理"""
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题")
            return
        
        # 清空之前的回答和来源
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, "智能Agent分析中，请稍候...")
        self.answer_text.config(state=tk.DISABLED)
        
        self.source_tree.delete(*self.source_tree.get_children())
        
        # 在新线程中处理问题
        self.status_var.set("Agent处理中...")
        threading.Thread(target=self.process_question, args=(question,), daemon=True).start()
    
    def process_question(self, question):
        """处理问题并更新UI"""
        try:
            start_time = time.time()
            
            # 使用智能回答系统
            result = self.rag_system.intelligent_answer(question, self.current_mode)
            
            elapsed_time = time.time() - start_time
            
            # 更新策略显示
            strategy_names = {
                'rag_only': '本地RAG',
                'google_only': '联网搜索',
                'both': '混合模式',
                'intelligent_agent': '智能Agent',
                'intelligent_agent_rag': '智能Agent(RAG)',
                'intelligent_agent_search': '智能Agent(搜索)',
                'intelligent_agent_combined': '智能Agent(两者结合)',
                'intelligent_agent_basic': '智能Agent(基础)',
                'error': '错误'
            }
            strategy_display = strategy_names.get(result['strategy'], result['strategy'])
            
            # 构建策略显示信息
            strategy_info = f"使用策略: {strategy_display}"
            if 'method' in result:
                strategy_info += f" | {result['method']}"
            
            self.strategy_var.set(strategy_info)
            
            # 更新回答框
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete("1.0", tk.END)
            
            # 显示回答模式信息
            mode_info = f"【当前模式】: {self.current_mode.value}\n"
            mode_info += f"【AI选择策略】: {strategy_display}\n"
            
            # 只在智能Agent模式下显示处理方法
            if self.current_mode == AnswerMode.INTELLIGENT and 'method' in result:
                mode_info += f"【处理方法】: {result['method']}\n"
            
            mode_info += "\n"
            self.answer_text.insert(tk.END, mode_info)
            
            # 显示答案
            self.answer_text.insert(tk.END, f"【智能回答】:\n{result['answer']}")
            self.answer_text.config(state=tk.DISABLED)
            
            # 更新来源显示
            for source in result['sources']:
                self.source_tree.insert("", "end", 
                                      values=(source['type'], source['content']))
            
            self.status_var.set(f"回答完成 | 耗时: {elapsed_time:.2f}秒")
            
        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, f"处理问题时出错: {str(e)}")
            self.answer_text.config(state=tk.DISABLED)
    
    def clear_fields(self):
        """清空输入和回答"""
        self.question_entry.delete("1.0", tk.END)
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.config(state=tk.DISABLED)
        self.source_tree.delete(*self.source_tree.get_children())
        self.status_var.set("准备就绪")
        self.strategy_var.set("")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

# 创建并运行GUI
gui = StarRailGUI(rag_chain, rag_system)
gui.run()