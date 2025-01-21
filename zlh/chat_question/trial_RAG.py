from transformers import pipeline
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os
import re
from pypinyin import lazy_pinyin, Style
import chromadb
from chromadb.config import Settings

# 初始化大模型（用于信息提取与摘要生成）
extractor = pipeline("question-answering", model="Qwen-7B")  # 替换为实际的大模型 API 或本地加载
summarizer = pipeline("summarization", model="Qwen-7B")  # 替换为摘要模型

embedding_model = SentenceTransformer("/path/to/embedding/model")  # 替换为具体路径

# 向量生成函数
def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_tensor=False).tolist()

class DocumentProcessor:
    """文档处理类：提取核心信息与摘要"""
    def __init__(self, extractor, summarizer):
        self.extractor = extractor
        self.summarizer = summarizer

    def extract_core_info(self, text):
        """利用大模型提取文档中的核心信息"""
        try:
            result = self.extractor(question="提取文档的核心信息", context=text)
            return result["answer"]
        except Exception as e:
            print(f"Error in extraction: {e}")
            return ""

    def generate_summary(self, core_info):
        """根据提取的核心信息生成摘要"""
        try:
            summary = self.summarizer(core_info, max_length=150, min_length=30, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error in summarization: {e}")
            return ""

class VectorDBManager:
    """向量数据库管理类"""
    def __init__(self, embedding_func, db_path):
        self.embedding_func = embedding_func
        self.db_path = db_path
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        self.client = PersistentClient(path=db_path)
        self.current_collection = None

    def create_collection(self, name):
        """创建或切换到集合"""
        valid_name = self.chinese_to_valid_name(name)
        self.current_collection = self.client.get_or_create_collection(name=valid_name)

    def add_documents(self, summaries, core_infos):
        """添加文档到数据库"""
        embeddings = self.embedding_func(summaries)
        ids = [f"id_{i}" for i in range(len(summaries))]
        try:
            self.current_collection.add(
                embeddings=embeddings,
                documents=summaries,
                metadatas=[{"core_info": info} for info in core_infos],
                ids=ids
            )
            print("Documents added successfully.")
        except Exception as e:
            print(f"Error adding documents: {e}")

    def search(self, query, top_n=5):
        """在数据库中检索"""
        try:
            query_embedding = self.embedding_func([query])
            results = self.current_collection.query(
                query_embeddings=query_embedding,
                n_results=top_n
            )
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return None

    @staticmethod
    def chinese_to_valid_name(name):
        """转换中文名为合法集合名"""
        pinyin_name = ''.join(lazy_pinyin(name, style=Style.NORMAL))
        valid_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", pinyin_name)[:63]
        return valid_name or "default_collection"

# 文件加载器示例
class PDFFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_text(self):
        from PyPDF2 import PdfReader
        reader = PdfReader(self.file_path)
        return " ".join(page.extract_text() for page in reader.pages)

# 示例：处理文档并存储
folder_path = "/home/student/zlh/GuardBot-main/doc_db/保险法"
db_path = "/home/student/zlh/GuardBot-main/chroma_data"
db_manager = VectorDBManager(get_embeddings, db_path)
processor = DocumentProcessor(extractor, summarizer)

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    if file_name.endswith(".pdf"):
        loader = PDFFileLoader(file_path)
        raw_text = loader.load_text()

        # 1. 提取核心信息
        core_info = processor.extract_core_info(raw_text)

        # 2. 生成摘要
        summary = processor.generate_summary(core_info)

        # 3. 存入向量数据库
        db_manager.create_collection("document_summaries")
        db_manager.add_documents([summary], [core_info])

# 检索示例
query = "大数据技术的发展历程"
results = db_manager.search(query, top_n=3)
print("Search Results:", results)
