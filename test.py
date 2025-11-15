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
        # 确保文件夹存在
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
        
        # 文本分块
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
        
        # 生成嵌入向量
        embeddings = []
        for text in tqdm(texts, desc="生成嵌入向量"):
            embedding = self.embedding_function(text)
            if embedding:  # 确保嵌入生成成功
                embeddings.append(embedding)
            else:
                # 处理失败情况，可以添加默认向量或跳过
                embeddings.append([0]*768)  # 假设维度为768
        
        # 创建并保存向量数据库
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
        """构建RAG链（使用高级Prompt系统）"""
        try:
            vectorstore = self.load_vectorstore()
        except Exception as e:
            print(f"加载向量数据库失败: {str(e)}，正在重建...")
            if not self.splits:
                self.splits = self.load_and_process_documents()
            vectorstore = self.create_vectorstore(self.splits)
        
        # 使用固定PromptTemplate解决版本兼容问题
        prompt_template = PromptTemplate(
            template=self.prompt_system.get_template(""),
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(k=self.config["retrieve_k"]),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

# 配置参数 - 使用脚本所在目录的相对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

config = {
    "folder_path": os.path.join(script_dir, "star_rail_stories"),
    "embedding_model": "bge-m3:latest",
    "generation_model": "qwen3:8b-q4_K_M",
    "vector_db_path": os.path.join(script_dir, "faiss_index"),
    "ollama_api_url": "http://localhost:11434/api/embeddings",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieve_k": 20  # 增加到20个文档检索
}

# 初始化系统
print("正在初始化星穹铁道知识问答系统...")
start_time = time.time()

rag_system = StarRailRAGSystem(config)

# 加载或创建向量数据库
if not os.path.exists(config["vector_db_path"]):
    print("未找到向量数据库，正在创建...")
    splits = rag_system.load_and_process_documents()
    if splits:
        vectorstore = rag_system.create_vectorstore(splits)
        print("向量数据库创建完成")
    else:
        print("警告：没有文档可处理，系统可能无法正常运行")
else:
    print("已加载现有向量数据库")

# 构建RAG链
rag_chain = rag_system.build_rag_chain()
print(f"系统初始化完成，耗时: {time.time() - start_time:.2f}秒")

class StarRailGUI:
    def __init__(self, rag_chain, rag_system):
        self.rag_chain = rag_chain
        self.rag_system = rag_system
        
        # 初始化GUI
        self.root = tk.Tk()
        self.root.title("崩坏：星穹铁道知识问答系统")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # 设置字体
        self.title_font = ("微软雅黑", 16, "bold")
        self.text_font = ("微软雅黑", 12)
        self.small_font = ("微软雅黑", 10)
        
        # 创建UI元素
        self.create_widgets()
    
    def create_widgets(self):
        """创建界面控件"""
        # 标题
        title_frame = tk.Frame(self.root, bg="#6a0dad")
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(title_frame, text="崩坏：星穹铁道知识问答系统", font=self.title_font, 
                fg="white", bg="#6a0dad").pack(pady=10)
        
        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(status_frame, textvariable=self.status_var, font=self.small_font, 
                bg="#f0f0f0", fg="#333333").pack(side=tk.LEFT)
        
        # 问题输入区
        question_frame = tk.LabelFrame(self.root, text="提问区", font=self.text_font, 
                                     bg="#f0f0f0", padx=10, pady=10)
        question_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.question_entry = scrolledtext.ScrolledText(question_frame, height=5, 
                                                      font=self.text_font, wrap=tk.WORD)
        self.question_entry.pack(fill=tk.X, expand=True)
        self.question_entry.focus()
        
        # 按钮区
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ask_button = tk.Button(button_frame, text="提问", font=self.text_font, 
                             command=self.ask_question, bg="#4CAF50", fg="white")
        ask_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(button_frame, text="清空", font=self.text_font, 
                               command=self.clear_fields, bg="#f44336", fg="white")
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # 回答显示区
        answer_frame = tk.LabelFrame(self.root, text="回答", font=self.text_font, 
                                   bg="#f0f0f0", padx=10, pady=10)
        answer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.answer_text = scrolledtext.ScrolledText(answer_frame, height=15, 
                                                   font=self.text_font, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.BOTH, expand=True)
        self.answer_text.config(state=tk.DISABLED)
        
        # 来源文档区
        source_frame = tk.LabelFrame(self.root, text="来源文档", font=self.text_font, 
                                   bg="#f0f0f0", padx=10, pady=10)
        source_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        self.source_tree = ttk.Treeview(source_frame, columns=("content",), show="headings")
        self.source_tree.heading("#0", text="序号")
        self.source_tree.heading("content", text="内容摘要")
        self.source_tree.column("#0", width=50, stretch=tk.NO)
        self.source_tree.column("content", stretch=tk.YES)
        
        scrollbar = ttk.Scrollbar(source_frame, orient="vertical", command=self.source_tree.yview)
        self.source_tree.configure(yscrollcommand=scrollbar.set)
        
        self.source_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 底部信息
        bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(bottom_frame, text="基于检索增强生成(RAG)技术 | 高级Prompt系统 | 语义检索", 
                font=self.small_font, bg="#f0f0f0", fg="#666666").pack(side=tk.LEFT)
    
    def ask_question(self):
        """提问处理"""
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题")
            return
        
        # 清空之前的回答和来源
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert(tk.END, "思考中，请稍候...")
        self.answer_text.config(state=tk.DISABLED)
        
        self.source_tree.delete(*self.source_tree.get_children())
        
        # 在新线程中处理问题
        self.status_var.set("思考中...")
        threading.Thread(target=self.process_question, args=(question,), daemon=True).start()
    
    def process_question(self, question):
        """处理问题并更新UI"""
        try:
            start_time = time.time()
            response = self.rag_chain({"query": question})
            
            # 确定使用的模板类型
            template_type = "默认模板"
            for key, keywords in self.rag_system.prompt_system.template_keywords.items():
                if any(keyword in question.lower() for keyword in keywords):
                    template_type = {
                        "relationship": "角色关系模板",
                        "character": "角色分析模板",
                        "storyline": "剧情分析模板",
                        "technical": "技术分析模板"
                    }[key]
                    break
            
            elapsed_time = time.time() - start_time
            
            # 更新回答框
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete("1.0", tk.END)
            self.answer_text.insert(tk.END, f"【使用的模板】: {template_type}\n\n")
            self.answer_text.insert(tk.END, f"【回答】: {response['result']}")
            self.answer_text.config(state=tk.DISABLED)
            
            # 更新来源文档
            if "source_documents" in response and response["source_documents"]:
                for i, doc in enumerate(response["source_documents"], 1):
                    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    self.source_tree.insert("", "end", text=str(i), values=(content,))
            
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
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

# 创建并运行GUI
gui = StarRailGUI(rag_chain, rag_system)
gui.run()