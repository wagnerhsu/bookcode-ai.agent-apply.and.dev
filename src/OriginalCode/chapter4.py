--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化 FAISS 向量数据库
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.load_local("path/to/index", embedding_model)
# 插入新数据到向量数据库
def add_data_to_index(data, metadata):
    vector_store.add_texts([data], metadatas=[metadata])
# 查询与输入文本最相似的5条记录
def search_similar(query):
    results = vector_store.similarity_search(query, k=5)
    for result in results:
        print(f"内容: {result['text']}, 元数据: {result['metadata']}")
# 示例：添加数据并查询
add_data_to_index("客户订单号为12345", {"category": "订单"})
search_similar("查询订单状态")


from collections import defaultdict
# 倒排索引结构
inverted_index = defaultdict(list)
# 构建倒排索引
def build_inverted_index(documents):
    for doc_id, content in enumerate(documents):
        words = content.split()
        for word in words:
            inverted_index[word].append(doc_id)
# 查询倒排索引
def search_inverted_index(query):
    return inverted_index.get(query, [])
# 示例：构建和查询倒排索引
documents = ["订单已发货", "订单已取消", "订单处理中"]
build_inverted_index(documents)
print(search_inverted_index("订单"))  # 输出: [0, 1, 2]


from langchain.memory import ConversationBufferMemory
# 创建内存缓存，用于存储查询结果
memory = ConversationBufferMemory(memory_key="query_cache")
# 查询时使用缓存
def query_with_cache(query):
    if memory.contains(query):
        print("从缓存中获取结果...")
        return memory.get(query)
    result = search_similar(query)
    memory.set(query, result)
    return result
# 示例：查询并缓存结果
query_with_cache("查询订单状态")


--------------------------------------------------------------------------------------------------------------


from collections import defaultdict
# 倒排索引结构的初始化
inverted_index = defaultdict(list)
# 构建倒排索引
def build_inverted_index(documents):
    for doc_id, content in enumerate(documents):
        words = content.split()  # 将文档内容分词
        for word in words:
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)  # 记录词条所在文档的ID
# 查询倒排索引
def search_inverted_index(query):
    return inverted_index.get(query, [])
# 示例：构建和查询倒排索引
documents = [
    "订单已发货",
    "订单已取消",
    "订单正在处理中",
    "发货状态查询"
]
build_inverted_index(documents)
# 查询“订单”相关的所有文档
result = search_inverted_index("订单")
print(f"包含'订单'的文档编号: {result}")


import concurrent.futures
# 并行构建倒排索引的函数
def build_inverted_index_parallel(documents):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(build_inverted_index, [documents])

# 示例：使用多线程构建倒排索引
build_inverted_index_parallel(documents)
# 查询倒排索引
print(search_inverted_index("发货"))


# 布尔查询优化：多关键词查询示例
def boolean_search(queries):
    result_sets = [set(search_inverted_index(query)) for query in queries]
    return list(set.intersection(*result_sets))  # 求交集，返回所有匹配的文档
# 示例：查询包含“订单”和“发货”的文档
result = boolean_search(["订单", "发货"])
print(f"包含'订单'和'发货'的文档编号: {result}")


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化嵌入模型，用于将文本转换为向量
embedding_model = OpenAIEmbeddings()
# 创建或加载本地的 FAISS 向量数据库
vector_store = FAISS.load_local("your_FAISS", embedding_model)
# 添加文本数据到向量数据库
def add_text_to_vector_store(texts, metadata_list):
    vector_store.add_texts(texts, metadatas=metadata_list)
# 查询与输入内容语义最接近的文本
def search_similar_text(query, top_k=5):
    results = vector_store.similarity_search(query, k=top_k)
    return results
# 示例：添加数据并进行查询
texts = ["客户的订单已经发货", "查询订单的物流信息", "订单已取消", "支付已完成"]
metadata = [{"doc_id": 1}, {"doc_id": 2}, {"doc_id": 3}, {"doc_id": 4}]
add_text_to_vector_store(texts, metadata)
results = search_similar_text("查看我的物流信息")
# 输出查询结果
for result in results:
    print(f"内容: {result['text']}, 元数据: {result['metadata']}")


# 删除向量数据库中的指定文本
def delete_text_from_vector_store(metadata_key, metadata_value):
    vector_store.delete(metadata={metadata_key: metadata_value})
# 示例：删除订单已取消的记录
delete_text_from_vector_store("doc_id", 3)
# 再次查询以验证删除是否成功
results = search_similar_text("订单取消")
print(f"查询结果: {results}")


--------------------------------------------------------------------------------------------------------------


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
# 初始化 LLM 模型
llm = OpenAI(model_name="text-davinci-003")
# 解析订单文本的提示模板
template = """
从以下客户订单描述中提取订单号、产品名称和数量：
描述：{order_text}
"""
prompt = PromptTemplate(input_variables=["order_text"], template=template)
# 示例：解析订单文本
order_text = "客户订购了2台iPhone 14，订单号为ABC123。"
response = llm(prompt.format(order_text=order_text))
print(f"解析结果: {response}")


--------------------------------------------------------------------------------------------------------------


import pandas as pd
# 示例数据：包含多种格式的订单数据
data = [
    {"order_id": "ABC123", "product": "iPhone14", "quantity": "2"},
    {"order_id": "def456", "product": "Samsung Galaxy", "quantity": " 3 "},
    {"order_id": "ABC123", "product": "iPhone14", "quantity": "2"}  # 重复数据
]
# 转换为 DataFrame
df = pd.DataFrame(data)
# 数据清洗：去除重复订单，修正数量格式
df = df.drop_duplicates()
df["quantity"] = df["quantity"].str.strip().astype(int)
# 打印清洗后的数据
print(df)


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化嵌入模型和向量数据库
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.load_local("path/to/index", embedding_model)
# 添加数据到知识库
def add_to_knowledge_base(texts, metadata_list):
    vector_store.add_texts(texts, metadatas=metadata_list)
# 示例：添加客户数据
texts = ["客户的订单号是ABC123", "客户的订单已发货", "客户取消了订单DEF456"]
metadata = [{"doc_id": 1}, {"doc_id": 2}, {"doc_id": 3}]
add_to_knowledge_base(texts, metadata)
# 查询知识库
def query_knowledge_base(query, top_k=2):
    results = vector_store.similarity_search(query, k=top_k)
    return results
# 示例：查询发货状态
results = query_knowledge_base("查询订单ABC123的状态")
for result in results:
    print(f"内容: {result['text']}, 元数据: {result['metadata']}")


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化嵌入模型和向量数据库
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.load_local("path/to/index", embedding_model)
# 实时查询函数
def real_time_query(query_text, top_k=5):
    results = vector_store.similarity_search(query_text, k=top_k)
    return results
# 示例：实时查询
query_result = real_time_query("查询订单状态")
for result in query_result:
    print(f"内容: {result['text']}, 元数据: {result['metadata']}")


--------------------------------------------------------------------------------------------------------------


from langchain.memory import ConversationBufferMemory
# 初始化缓存模块
memory = ConversationBufferMemory(memory_key="query_cache")
# 查询并使用缓存
def cached_query(query):
    if memory.contains(query):
        print("从缓存中获取结果...")
        return memory.get(query)
    result = real_time_query(query)
    memory.set(query, result)
    return result
# 示例：缓存查询
cached_result = cached_query("查询订单状态")
print(f"查询结果: {cached_result}")


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from PIL import Image
import numpy as np
# 初始化向量数据库
vector_store = FAISS.load_local("path/to/index", OpenAIEmbeddings())
# 添加图像向量到数据库
def add_image_to_index(image_path, metadata):
    image = Image.open(image_path)
    image_vector = np.array(image).flatten()  # 简单向量化示例
    vector_store.add_vectors([image_vector], metadatas=[metadata])
# 示例：查询与图像相关的文本
add_image_to_index("sample_image.jpg", {"doc_id": 4, "type": "image"})
results = vector_store.similarity_search("图片描述", k=2)
print(f"查询结果: {results}")


--------------------------------------------------------------------------------------------------------------


import requests
# 从API获取数据并加入知识库
def fetch_and_index_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        text = data.get("description", "")
        vector_store.add_texts([text], [{"api_source": api_url}])
# 示例：从API获取订单状态数据
fetch_and_index_data("https://api.example.com/order_status")


--------------------------------------------------------------------------------------------------------------


import time
import random
import asyncio
import threading
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Any
class CacheManager:
    """缓存管理器，用于查询结果缓存和命中率监控。"""
    def __init__(self):
        self.cache = defaultdict(dict)
    def set(self, key: str, value: Any):
        """将查询结果存入缓存。"""
        self.cache[key] = value
        print(f"Cached: {key}")
    def get(self, key: str):
        """从缓存中获取查询结果。"""
        return self.cache.get(key, None)
    def contains(self, key: str) -> bool:
        """检查缓存中是否存在结果。"""
        return key in self.cache
cache_manager = CacheManager()
def cache_result(func):
    """装饰器：缓存查询结果以减少数据库访问。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = args[0]  # 查询的第一个参数是关键字
        if cache_manager.contains(query):
            print("从缓存中获取结果...")
            return cache_manager.get(query)
        result = func(*args, **kwargs)
        cache_manager.set(query, result)
        return result
    return wrapper
class InvertedIndex:
    """倒排索引类，支持关键词搜索。"""
    def __init__(self):
        self.index = defaultdict(list)
    def build_index(self, documents: List[str]):
        """构建倒排索引。"""
        for doc_id, content in enumerate(documents):
            for word in content.split():
                self.index[word].append(doc_id)
    def search(self, query: str) -> List[int]:
        """在倒排索引中查询关键词。"""
        return self.index.get(query, [])
class VectorIndex:
    """向量索引类，支持语义检索。"""
    def __init__(self):
        self.vectors = defaultdict(dict)
    def add_vector(self, text: str, vector: List[float], metadata: Dict):
        """添加向量数据到索引中。"""
        self.vectors[text] = {"vector": vector, "metadata": metadata}
    def search_vector(self, query_vector: List[float], top_k=2) -> List[Dict]:
        """模拟基于向量的语义检索。"""
        return [{"text": "订单已发货", "metadata": {"status": "shipped"}}]
class QueryHandler:
    """查询处理器，整合倒排索引与向量索引。"""
    def __init__(self):
        self.inverted_index = InvertedIndex()
        self.vector_index = VectorIndex()
    @cache_result
    def search(self, query: str) -> List[Any]:
        """基于倒排索引和向量索引进行搜索。"""
        keyword_results = self.inverted_index.search(query)
        vector_results = self.vector_index.search_vector([random.random() for _ in range(10)])
        return {"keyword_results": keyword_results, "vector_results": vector_results}
def async_update_index(index: InvertedIndex, new_documents: List[str]):
    """使用线程异步索引更新。"""
    def update():
        print("更新索引中...")
        index.build_index(new_documents)
    thread = threading.Thread(target=update)
    thread.start()
    thread.join()
class SecurityManager:
    """安全管理类，处理访问控制。"""
    def __init__(self):
        self.authorized_users = {"admin": "1234"}
    def authenticate(self, user: str, password: str) -> bool:
        """验证用户身份。"""
        return self.authorized_users.get(user) == password
class KnowledgeBase:
    """主知识库类，管理索引、查询和缓存。"""
    def __init__(self):
        self.query_handler = QueryHandler()
        self.security_manager = SecurityManager()
    def query(self, user: str, password: str, query: str):
        """验证用户并执行查询。"""
        if not self.security_manager.authenticate(user, password):
            raise PermissionError("Unauthorized Access")
        return self.query_handler.search(query)
@wraps
def log_execution(func):
    """装饰器：记录函数执行时间。"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds.")
        return result
    return wrapper
@log_execution
def main():
    """主程序入口，模拟系统操作。"""
    kb = KnowledgeBase()
    # 更新索引
    documents = ["订单已发货", "订单处理中", "客户取消了订单"]
    async_update_index(kb.query_handler.inverted_index, documents)
    # 查询系统
    try:
        result = kb.query("admin", "1234", "订单")
        print(f"查询结果: {result}")
    except PermissionError as e:
        print(e)
if __name__ == "__main__":
    main()

