import json
import os
import re
from openai import OpenAI
import requests
import torch
import time

from pypinyin import lazy_pinyin, Style
import wikipedia
from zhipuai import ZhipuAI

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, Tool
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool
from langchain.llms import OpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict

from graphrag_connector import run_graphrag

import chromadb
from chromadb.config import Settings

start_time = time.time()  # 记录开始时间

ZHIPUAI_API_KEY = "1288e6b562440ad7a5c9bc511746dcf0.8fXJ2UA3IBm9TgXO" 
BOCHA_API_KEY = "sk-a9b7646515304531a3ab077c1fc7ed51"

client = ZhipuAI(
    api_key=ZHIPUAI_API_KEY
)

def get_embeddings(texts, model="embedding-3"):
    """
    调用 ZhipuAI 嵌入模型生成文本嵌入向量。
    :param texts: 待生成嵌入的文本列表
    :param model: 嵌入模型名称，默认为 'embedding-3'
    :return: 嵌入向量列表
    """
    try:
        # 假设 client.embeddings.create 是正确的函数调用方式
        response = client.embeddings.create(model=model, input=texts)
        
        # 检查 response 是否有 data 属性
        if hasattr(response, 'data'):
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            return embeddings
        else:
            print(f"Unexpected response structure: {response}")
            return None
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt

prompt_template = """
你是一名优秀的中文智能助教。以下是用户的问题和相关的已知信息。请根据规则作答：

### 给定科目
本次问答仅限以下科目范围内的问题：
{__SUBJECT__}

### 已知信息
以下是与用户问题相关的已知信息，由检索系统提供：
{__INFO__}

### 用户问题
用户问：{__QUERY__}

### 回答规则
1. **范围确认**：
   - 确保回答内容严格限定在指定的科目范围内。
   - 如果问题超出范围，请直接回复：“该问题超出了{__SUBJECT__}范围，我无法回答。”

2. **基于已知信息**：
   - 优先使用已知信息回答问题，确保准确性。
   - 如需补充内容，务必保持客观真实，且严格限定在指定科目范围内。

3. **回答要求**：
   - 语言风格亲切，逻辑清晰，表述详尽。
   - 回答应包括以下结构：
     - **直接回答**：明确回答用户问题。
     - **相关来源**：消息的出处最好备注

4. **无法回答的处理**：
   - 如果无法基于已知信息回答，请回复：“与教材无关，我无法回答您的问题。”

请根据以上规则用中文回答问题。
"""

prompt_question = """
你是一名中文智能出题助手，以下是用户的出题要求和相关的已知信息，请根据规则作答：

### 已知信息
以下是与出题任务相关的已知信息，由检索系统提供：
{__INFO__}

### 给定科目
本次出题仅限以下科目范围内：
{__SUBJECT__}

### 用户要求
1. **题目类型**：{__TYPE__}
2. **题目数量**：{__NUM__}
3. **用户描述**：{__QUERY__}

### 出题规则
1. **类型匹配**：
   - 严格按照用户指定的题目类型（如选择题、判断题或问答题）出题，确保每道题符合格式规范。
   - 示例格式：
     - **选择题**：题干 + 选项 + 答案
     - **判断题**：题干 + 答案（对/错）
     - **问答题**：题干 + 答案（完整详细）
   
2. **数量匹配**：
   - 确保生成的题目数量与用户要求一致。

3. **与已知信息相关**：
   - 确保题目基于已知信息设计，避免胡编乱造。
   - 如需补充内容，须严格限定在指定科目范围内。

4. **答案完整**：
   - 每道题必须附带准确答案。

5. **无法出题时的处理**：
   - 如果无法根据已知信息出题，请回复：“与教材无关，我无法完成您的任务。”

###题目示例，与结果无关：
-选择题
 1. 中国《保险法》于1995年6月30日通过，首次修订是在哪一年？
   A. 2000年
   B. 2005年
   C. 2008年
   D. 2014年
   **答案**：C
- 判断题：2014年8月31日和2015年4月24日，第十二届全国人民代表大会常务委员会两次修正《保险法》。答案：对
- 问答题
根据提供的资料，我国《保险法》自1995年实施以来，哪一次修订对被保险人利益保护和法律责任进行了明确规定？
**答案：** 第一次修订是在2009年2月28日，十一届全国人大常委会第七次会议修订《保险法》。

请根据以上规则生成题目，并用中文返回结果。
"""

def extract_field(json_data, field):
    """
    从 JSON 数据中提取指定字段的值。

    参数:
    json_data (str or dict): JSON 字符串或字典。
    field (str): 需要提取的字段名。

    返回:
    list: 包含指定字段值的列表。
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    def extract(obj, field, results):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == field:
                    results.append(value)
                elif isinstance(value, (dict, list)):
                    extract(value, field, results)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, field, results)

    results = []
    extract(json_data, field, results)
    return results

def check_and_trim_messages(messages, max_tokens=8192):
    """
    检查消息的总token数量，并根据需要裁剪消息内容。
    :param messages: 当前对话历史消息列表
    :param max_tokens: 模型的最大token限制
    :return: 裁剪后的消息列表
    """
    total_tokens = sum(len(msg['content'].split()) for msg in messages)  # 简单估算每条消息的token数量
    print(f"总tokens数量: {total_tokens}")

    # 如果总token数超过最大限制，则裁剪
    if total_tokens > max_tokens:
        # 计算需要裁剪掉的token数
        excess_tokens = total_tokens - max_tokens
        print(f"超出的tokens数量: {excess_tokens}")
        
        # 从消息列表的最前面开始裁剪，直到满足最大限制
        trimmed_messages = []
        for msg in reversed(messages):
            msg_tokens = len(msg['content'].split())
            excess_tokens -= msg_tokens
            trimmed_messages.append(msg)
            if excess_tokens <= 0:
                break
        trimmed_messages.reverse()  # 保持消息顺序
        
        print(f"裁剪后的消息数量: {len(trimmed_messages)}")
        return trimmed_messages

    # 如果没有超出限制，直接返回原消息
    return messages

class MyVectorDBConnector:
    def __init__(self, default_collection_name, embedding_fn, db_path="/home/student/zlh/GuardBot-main/chroma_data"):
        try:
            # 转换集合名称为合法名称
            valid_collection_name = self.chinese_to_valid_name(default_collection_name)
            # 初始化客户端
            if not os.path.exists(db_path):
                os.makedirs(db_path)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.embedding_fn = embedding_fn

            if self.collection_exists(valid_collection_name):
                print(f"Collection '{valid_collection_name}' already exists.")
            else:
                print(f"Collection '{valid_collection_name}' does not exist, creating it.")

            # 默认集合
            self.default_collection = self.chroma_client.get_or_create_collection(name=valid_collection_name)
            self.current_collection = self.default_collection
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            self.current_collection = None

    def collection_exists(self, name):
        collections = self.chroma_client.list_collections()
        return any(col.name == name for col in collections)

    def chinese_to_valid_name(self,chinese_name):
        """将中文转为合法的集合名称"""
        # 中文转拼音
        pinyin_name = ''.join(lazy_pinyin(chinese_name, style=Style.NORMAL))
        # 替换非法字符为下划线
        valid_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", pinyin_name)
        # 确保长度符合要求（3-63个字符）
        valid_name = valid_name[:63].strip("_")
        # 如果长度不足，使用默认名称
        if len(valid_name) < 3:
            valid_name = "default_collection"
        return valid_name

    def switch_collection(self, collection_name):
        '''切换到指定 collection'''
        try:
            self.current_collection = self.chroma_client.get_or_create_collection(name=collection_name)
            print(f"Switched to collection: {collection_name}")
        except Exception as e:
            print(f"Error switching to collection {collection_name}: {e}")

    def add_documents(self, documents):
        '''向当前 collection 添加文档与向量'''
        try:
            embeddings = self.embedding_fn([doc["summary"] for doc in documents])  # 基于 summary 生成向量
            self.current_collection.add(
                embeddings=embeddings,  # 每个文档的向量
                documents=[doc["summary"] for doc in documents],  # 存储的文档是 summary
                ids=[doc["id"] for doc in documents],  # 每个文档的 id
                metadatas=[{"extracted_info": doc["extracted_info"]} for doc in documents]  # 元数据存储 extracted_info
            )
            print("Documents added successfully.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def search(self, query, top_n=5):
        """查询向量数据库并返回完整信息"""
        try:            
            results = self.current_collection.query(
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            print('results======================>:', results)
            detailed_results = [
                {
                    "extracted_info": metadata
                }
                for doc_list, metadata_list in zip(results["ids"], results["metadatas"])
                for doc, metadata in zip(doc_list, metadata_list)
            ]
            return detailed_results
        except Exception as e:
            print(f"Error during search: {e}")
            return None

class GenerateFinalResponse(BaseTool):
    name: str = "generate_final_response"
    description: str = (
        "你最多思考三轮"
        "当你想出来的回答已经达到回答要求时，请立刻马上把你的想法返回，不需要一直思考，"
        "综合分析多领域信息并生成全面、优先级明确的最终答案。"
        "该工具在问题复杂、涉及多轮推导或需要多信息源整合时尤为实用。"
        "通过调用子工具并进行多次迭代分析，确保答案准确性与逻辑清晰。"
        "适合深度解析、跨领域整合以及需要高质量输出的问题。"

    ) 
    subject : str
    max_iterations: int = 1  # 限制最多思考1轮

    def _run(self, query: str) -> str:
        """生成最终答案，考虑优先级和逻辑，最多思考2轮。"""
        try:
            vector_search_tool = VectorSearchTool(subject=self.subject)
            search_web_tool = SearchWeb()
            calculate_tool = Calculate()
            
            responses = []
            tools = [vector_search_tool, search_web_tool, calculate_tool]
            
            # 遍历工具生成回答，但限制最大思考轮数
            for i, tool in enumerate(tools):
                response = tool._run(query)
                if response:
                    responses.append(response)

            if not responses:
                return "没有找到相关信息，请稍后再试。"

            # 组合这些响应为最终答案
            final_answer = " ".join(responses)
            
            return final_answer
        
        except Exception as e:
            return f"处理时发生错误: {str(e)}"
class GenerateQuestionsTool(BaseTool):
    name: str = "generate_questions"
    description: str = (
        "你最多思考三轮"
        "生成与指定主题和领域相关的高质量问题，适用于学习、测试或研究场景。"
        "工具通过结合向量数据库和用户输入内容，生成用户指定的类型的问题（q_type）。"
        "支持根据用户需求（num）生成指定数量的问题，确保问题的相关性与针对性。"
        "适合教育培训、知识验证和考试设计等场景。"
        "生成与主题相关的选择题，并返回完整的题目和答案。"
        "输出格式应包含题目正文、选项以及正确答案的标注，以便用于学习或测试场景。"
        "当你想出来的题目已经达到要求的题目数以及题目要求是，请立刻马上把你的想法返回，不需要一直思考，"
    )  
    subject: str  # 专业领域
    num: int
    q_type: str
    max_iterations: int = 1  # 限制最多思考1轮

    def _run(self, query: str) -> str:
        """生成指定数量和类型的问题，最多思考2轮，并返回格式化的题目、选项和答案。"""
        try:
            # 初始化工具
            vector_search_tool = VectorSearchTool(subject=self.subject)
            calculate_tool = Calculate()

            responses = []
            tools = [vector_search_tool, calculate_tool]
            
            # 遍历工具生成回答，但限制最大思考轮数
            for i, tool in enumerate(tools):
                response = tool._run(query)
                if response:
                    responses.append(response)

            if not responses:
                return "没有找到相关信息，请稍后再试。"

            return response

        except Exception as e:
            return f"处理时发生错误: {str(e)}"

class ThinkTool(BaseTool):
    name: str = "think_tool"
    description: str = (
        "优化生成内容的逻辑结构并提升答案质量的工具。"
        "适用于需要对内容进行深入分析、逻辑审查或结构调整的场景。"
        "工具通过系统化的思维优化，确保答案条理清晰、论证严谨、表达精准。"
        "非常适合复杂问题的答复优化或多步骤推导的改进。"
    )  
    def _run(self, query: str) -> str:
        """对查询进行思考，总结，产生高质量的答案。"""
        # 伪代码示范思考过程
        if query:
            # 根据查询内容思考可能的答案
            return "生成的答案已经过深思熟虑，具有逻辑和条理。"
        else:
            return "没有足够的信息进行思考。"

class GreetingResponseTool(BaseTool):
    name: str = "greeting_response"
    description: str = (
        "处理用户问候并生成热情、得体回应的工具。"
        "支持多语言识别，根据用户的打招呼方式提供定制化回应。"
        "该工具在初次接触或需要引导用户进入对话时表现尤为优秀。"
        "适合提升用户体验并建立友好互动的场景。"
    )  
    def _run(self, query: str) -> str:
        """根据用户的问候语返回合适的回答。"""
        greetings = ["你好", "您好", "早上好", "晚上好", "嗨"]
        if any(greeting in query for greeting in greetings):
            return "你好！我是智能助教，很高兴为您服务！有什么问题可以帮您解答吗？"
        return "您好！如果您有任何问题，我很乐意为您提供帮助。"
class SearchWeb(BaseTool):
    name: str = "search_web"
    description: str = (
        "通过 SerpAPI 实时获取互联网内容以回答用户问题。"
        "工具支持关键词搜索，返回包含标题、链接和摘要的结构化结果。"
        "适合处理需要最新动态、验证信息或超出本地知识范围的问题。"
        "特别适用于查找补充数据、探索新领域或快速响应动态需求的场景。"
    ) 
    def _run(self, query: str) -> str:
        """Run web search using SerpAPI."""
        url = 'https://api.bochaai.com/v1/web-search'
        headers = {
            'Authorization': f'Bearer {BOCHA_API_KEY}',  # 请替换为你的API密钥
            'Content-Type': 'application/json'
        }
        data = {
            "query": query,
            "freshness": "oneYear", # 搜索的时间范围，例如 "oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"
            "summary": True, # 是否返回长文本摘要总结
            "count": 3
        }

        response = requests.post(url, headers=headers, json=data)
        res = []
        if response.status_code == 200:
            json_data = response.json()
            if "data" in json_data and "webPages" in json_data["data"] and "value" in json_data["data"]["webPages"]:
                value_list = json_data["data"]["webPages"]["value"]
                for item in value_list:
                    name = item.get("name", "无标题")
                    url = item.get("url", "无链接")
                    snippet = item.get("snippet", "无摘要")
                    res.append(f"标题: {name}\n链接: {url}\n摘要: {snippet}\n")
        res = '\n'.join(res)
        if response.status_code == 200:
            # 返回给大模型的格式化的搜索结果文本
            # 可以自己对博查的搜索结果进行自定义处理
            return str(res)
        else:
            raise Exception(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
class Calculate(BaseTool):
    name: str = "calculate"
    description: str = (
        "执行各种数学运算并返回精确结果。"
        "支持基础运算（加减乘除）、高级运算（指数、对数、三角函数）以及统计分析。"
        "适合需要快速解决数学计算问题的场景，如工程计算、学术分析或数据建模。"
    )  
    def _run(self, expression: str) -> str:
        """Run calculation."""
        try:
            result = str(eval(expression))
            return result
        except Exception as e:
            return f"Error in calculation: {str(e)}"
class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "利用向量数据库进行高效语义检索，快速定位相关信息。"
        "通过自然语言查询，返回与输入内容语义匹配度高的结果。"
        "适合需要精确信息检索、深度分析和语义匹配的场景。"
        "广泛应用于学术研究、商业分析和教育培训等领域。"
    )  
    subject: str  # 将 subject 定义为类字段，使用时初始化

    def _run(self, query: str) -> str:
        """运行向量搜索。

        Args:
            query: 用户输入的查询语句。

        Returns:
            str: 格式化的搜索结果。
        """
        try:
            # 数据库连接
            vector_db = MyVectorDBConnector(self.subject, get_embeddings)
            results = vector_db.search(query, 3)
            if not results:
                return "未找到任何相关内容，请尝试调整查询语句。"
            
            # 格式化搜索结果
            formatted_results = []
            for idx, result in enumerate(results):
                formatted_results.append(f"结果 {idx + 1}: {result}")
            return "\n".join(formatted_results)
        except ConnectionError:
            return "数据库连接失败，请稍后重试。"
        except Exception as e:
            return f"向量搜索时发生未知错误: {str(e)}"

class GetCompletion:
    def __init__(self, base_url, api_key, model_name, subject):
        self.llm = ChatOpenAI(
            temperature = 1.0,
            model_name=model_name,
            openai_api_key=api_key,
            base_url=base_url,
            request_timeout=30
        )
        self.subject = subject

    def _initialize_toolsQA(self, subject):
        # 初始化问题解答相关工具
        return [
            Tool(
                name="GreetingResponseTool",
                func=GreetingResponseTool().run,
                description="回应用户的问候"
            ),
            Tool(
                name="GenerateFinalResponse",
                func=GenerateFinalResponse(subject=subject).run,
                description="生成最终的答案"
            ),
            Tool(
                name="ThinkTool",
                func=ThinkTool(subject=subject).run,
                description="执行思考"
            ),
            Tool(
                name="VectorSearchTool",
                func=VectorSearchTool(subject=subject).run,
                description="进行向量搜索"
            ),
            Tool(
                name="SearchWeb",
                func=SearchWeb().run,
                description="搜索网络信息"
            ),
            Tool(
                name="Calculate",
                func=Calculate().run,
                description="进行计算"
            ),
        ]

    def _initialize_toolsQQ(self, subject, q_type, num):
        # 初始化问题生成相关工具
        return [
            Tool(
                name="GreetingResponseTool",
                func=GreetingResponseTool().run,
                description="回应用户的问候"
            ),
            Tool(
                name="GenerateQuestionsTool",
                func=GenerateQuestionsTool(subject=subject, q_type=q_type, num=num).run,
                description="生成问题"
            ),
            Tool(
                name="ThinkTool",
                func=ThinkTool(subject=subject).run,
                description="执行思考"
            ),
            Tool(
                name="VectorSearchTool",
                func=VectorSearchTool(subject=subject).run,
                description="进行向量搜索"
            ),
            Tool(
                name="SearchWeb",
                func=SearchWeb().run,
                description="搜索网络信息"
            ),
            Tool(
                name="Calculate",
                func=Calculate().run,
                description="进行计算"
            ),
        ]
    
    def generate_response(self, prompt, messages): 
        tools = self._initialize_toolsQA(self.subject)
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate", 
            max_iterations = 6,
            agent_kwargs={ 
                "prompt_template": """
                请按照要求生成：
                "你最多思考三轮"
                - 当然，如果是打招呼的话，要热情地回应
                - 一旦获取到与问题相关的有效结果，请立即停止进一步思考并输出结果。
                - 仅选择与问题高度相关的搜索结果，如果搜索结果与问题关系不大，请忽略。
                - 确保最终答案尽可能详细，将所有能够想到的相关信息和推论展示出来。
                - 除了最终的回答，不要出现类似Thought: I now know the final answer。
                - 对于任何你认为相关的细节都要加以描述，不要简单的总结，深入挖掘每个环节的具体内容。
                - 如果你发现有多种解决方法或多个相关信息点，请把它们一一列举出来，并明确各自的优势。
                """
            }
        )

        # 裁剪消息
        messages = check_and_trim_messages(messages)

        # 构建 prompt 并进行处理
        formatted_prompt = f"""
            注意：生成的问题应该与提供的内容密切相关，涵盖各个知识点。
            请确保生成的问题能够考察学生对该内容的理解和应用能力。

        问题: {prompt}
        """

        base_messages = []
        for msg in messages:
            if msg["role"] == "user":
                base_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                base_messages.append(AIMessage(content=msg["content"]))

        base_messages.append(HumanMessage(content=formatted_prompt))

        try:
            response = agent.run(base_messages, timeout=10)  # 设置超时限制
            pattern = r"Final Answer: (.*)"

            # 使用正则表达式提取内容
            match = re.search(pattern, response)
            if match:
                response = match.group(1).strip()
            else:
                print("No match found")
            return response
        except Exception as e:
            print("Error in Agent run:", e)
            return f"执行出错: {str(e)}"
    def generate_questions(self, query, q_type="单选", num=3):
        tools = self._initialize_toolsQQ(self.subject,q_type,num)
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            early_stopping_method="generate", 
            max_iterations = 3,
            agent_kwargs={ 
                "prompt_template": """
                请按照要求生成：                
                "你最多思考三轮"
                - 当然，如果是打招呼的话，要热情地回应
                - 当你想出来的题目已经达到要求的题目数以及题目要求是，请立刻马上把你的想法返回，不需要一直思考，
                - 仅选择与问题高度相关的搜索结果，如果搜索结果与问题关系不大，请忽略。
                - 确保最终答案尽可能详细，将所有能够想到的相关信息和推论展示出来。
                - 除了最终的回答，不要出现类似Thought: I now know the final answer。
                - 对于任何你认为相关的细节都要加以描述，不要简单的总结，深入挖掘每个环节的具体内容。
                - 如果你发现有多种解决方法或多个相关信息点，请把它们一一列举出来，并明确各自的优势。
                """
            }
        )
        """生成相关问题。

        Args:
            query (str): 用户查询。
            q_type (str): 问题类型（默认单选）。
            num (int): 要生成的问题数量（默认 5 个）。

        Returns:
            str: 生成的问题列表。
        """
        response = agent.run(query)
        pattern = r"Final Answer: (.*)"

        # 使用正则表达式提取内容
        match = re.search(pattern, response)
        if match:
            response = match.group(1).strip()
        else:
            print("No match found")
        return response

class Chat_Bot:
    def __init__(self, subject,n_results=10):
        self.subject = subject
        self.llm_api = GetCompletion(
                            base_url="http://localhost:8731/v1",
                            api_key="1145141919810",
                            model_name="/home/student/zlh/model/Qwen/Qwen2___5-14B-Instruct",
                            subject = self.subject
                        )
        self.n_results = n_results
        self.messages = []  # 用于存储对话历史 [{"role": "user", "content": ...}, ...]

    def chat(self, user_query):
    # 添加用户消息到对话历史
        self.messages.append({"role": "user", "content": user_query})
        # === 1. 检索 ===
        self.vector_db = MyVectorDBConnector(self.subject, get_embeddings)
        search_results = self.vector_db.search(user_query, self.n_results)
        if not search_results:
            raise ValueError("未检索到相关结果，请尝试更换查询关键词。")

        # 提取检索信息
        search_results_formatted = "\n\n".join(
            [entry['extracted_info']['extracted_info'] for entry in search_results]
        )

        # === 2. 构建 Prompt ===
        prompt = build_prompt(
            prompt_template,
            info=search_results_formatted,
            query=user_query,
            subject=self.subject
        )
        print("生成的 Prompt ===================>")
        print(prompt)

        # === 3. 调用 LLM ===
        response = self.llm_api.generate_response(prompt, self.messages)
        if not response:
            raise ValueError("LLM 返回空响应，请检查 Prompt 或模型设置。")
        print("一般响应 ===================>")
        print(response)

        # === 5. 更新对话历史 ===
        self.messages.append({"role": "assistant", "content": response})
        return response

        # except Exception as e:
        #     error_message = f"错误发生：{str(e)}"
        #     print(error_message)
        #     self.messages.append({"role": "assistant", "content": error_message})
        #     return error_message

    def question(self, user_query, type, num):
        self.vector_db = MyVectorDBConnector(self.subject, get_embeddings)
        search_results = self.vector_db.search(user_query, self.n_results)
        search_results = "\n\n".join([entry['extracted_info']['extracted_info'] for entry in search_results])
        print(search_results)
        prompt = build_prompt(
            prompt_question,
            info=search_results,
            query=user_query,
            type=type,
            num=num,
            subject = self.subject
        )
        print("prompt===================>")
        print(prompt)

        response = self.llm_api.generate_questions(prompt, type, num)
        
        return response

    # def graphrag_query(self, query):
    #     """
    #     调用 GraphRAG 查询功能。
        
    #     Args:
    #         query (str): 用户的查询问题。
        
    #     Returns:
    #         str: 查询结果，或错误信息。
    #     """
    #     result = run_graphrag(query)
    #     if result['stderr']:
    #         return f"GraphRAG 错误: {result['stderr']}"
    #     return result['stdout']

    # def chat_GraphRAG(self, user_query, subject):
    #     self.messages.append({"role": "user", "content": user_query})
    #     try:
    #         # 使用 GraphRAG 获取补充信息
    #         graph_info = self.graphrag_query(user_query)
    #         print("GraphRAG 输出 ===================>")
    #         print(graph_info)
            
    #         # 构建 Prompt
    #         prompt = build_prompt(
    #             prompt_template,
    #             query=user_query,
    #             graph_info=graph_info,
    #             subject=subject
    #         )
            
    #         # 调用 LLM 生成响应
    #         response = self.llm_api(prompt, self.messages)
    #         if not response:
    #             raise ValueError("LLM 返回空响应，请检查 Prompt 或模型设置。")
            
    #         # 更新对话历史
    #         self.messages.append({"role": "assistant", "content": response})
    #         return graph_info
        
    #     except Exception as e:
    #         error_message = f"错误发生：{str(e)}"
    #         self.messages.append({"role": "assistant", "content": error_message})
    #         return error_message


end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time
print(f"执行时间为: {elapsed_time:.4f} 秒")

if __name__ == "__main__":
        # 初始化向量数据库管理器
    
    db_path = "/home/student/zlh/GuardBot-main/chroma_data" 
    # 初始化聊天机器人
    chat_bot = Chat_Bot(subject='保险法')
    user_query = "保险法的发展"

    # # # 与聊天机器人交互
    # response = chat_bot.chat(user_query)
    # print("Bot response:", response)
    
    # 出题示例
    type = "选择题"
    num = 5
    question_response = chat_bot.question(user_query, type, num)
    print("Question response:=========>", question_response)
    
    # # 代码助手示例
    # role = "Python 编程助手"
    # code_query = "如何用 Python 读取 CSV 文件？"
    # code_response = chat_bot.code(role, code_query)
    # print("Code response:", code_response)