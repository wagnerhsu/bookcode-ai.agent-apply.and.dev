--------------------------------------------------------------------------------------------------------------


from transformers import GPT2Tokenizer, GPT2LMHeadModel
def load_model_and_tokenizer():
    """加载GPT模型和Tokenizer"""
    model_name = "gpt2"  # 或者使用 gpt-neo 等模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
print("模型和分词器加载成功！")
from transformers import GPT2Tokenizer, GPT2LMHeadModel
def load_model_and_tokenizer():
    """加载本地GPT模型和Tokenizer"""
    model_name = "./models/gpt2/"  # 本地路径
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer
model, tokenizer = load_model_and_tokenizer()
print("模型和分词器加载成功！")
from datasets import load_dataset
def prepare_dataset(file_path, tokenizer, block_size=128):
    """加载并预处理数据"""
    dataset = load_dataset('text', data_files={'train': file_path})
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=block_size, padding="max_length")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset['train']
# 加载和预处理数据
file_path = "./financial_data.txt"
dataset = prepare_dataset(file_path, tokenizer)
print("数据集加载并预处理成功！")
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
def get_training_arguments(output_dir="./results"):
    """配置训练参数"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # 根据显存大小调整
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',  # 日志文件路径
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        report_to="none"  # 关闭wandb或其他报告
    )
# 创建数据集的Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
training_args = get_training_arguments()
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)
# 开始训练
trainer.train()
print("模型训练完成！")
# 保存微调后的模型和分词器
model.save_pretrained("./finetuned_gpt2")
tokenizer.save_pretrained("./finetuned_gpt2")
print("微调后的模型已保存！")
def generate_text(prompt, model, tokenizer, max_length=50):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)
# 测试生成
prompt = "债券投资的特点是"
generated_text = generate_text(prompt, model, tokenizer)
print("生成的文本：", generated_text)

--------------------------------------------------------------------------------------------------------------


# 示例：短期记忆的存储
short_term_memory = {"account": "12345678", "last_query": "信用卡账单"}
# 示例：长期记忆的保存和查询
import sqlite3
conn = sqlite3.connect("customer_memory.db")
cursor = conn.cursor()
# 创建表用于保存客户咨询历史
cursor.execute("""
CREATE TABLE IF NOT EXISTS customer_history (
    customer_id TEXT,
    query TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
# 保存用户的历史问题
def save_to_memory(customer_id, query):
    cursor.execute("INSERT INTO customer_history (customer_id, query) VALUES (?, ?)", 
                   (customer_id, query))
    conn.commit()


--------------------------------------------------------------------------------------------------------------


import requests
def get_weather(city):
    """调用天气API获取指定城市的天气信息"""
    api_url = f"http://api.weatherapi.com/v1/current.json?key=your_api_key&q={city}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return f"{city}的温度是 {data['current']['temp_c']}°C"
    else:
        return "无法获取天气信息"
print(get_weather("Beijing"))


--------------------------------------------------------------------------------------------------------------


from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# 使用本地路径加载模型
model = SentenceTransformer('./paraphrase-MiniLM-L6-v2')
knowledge_base = [
    "股票市场是风险投资的主要渠道。",
    "债券投资具有较低风险，适合保守型投资者。",
    "外汇市场波动较大，适合有经验的投资者参与。"
]
knowledge_vectors = model.encode(knowledge_base)
def semantic_search(query):
    query_vector = model.encode([query])
    similarities = cosine_similarity(query_vector, knowledge_vectors)
    closest_idx = np.argmax(similarities)
    return knowledge_base[closest_idx]
print(semantic_search("适合风险偏好低的投资"))


--------------------------------------------------------------------------------------------------------------


import time
import sqlite3
import asyncio
import requests
import numpy as np
from functools import wraps
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Dict, List, Callable, Any
# 全局上下文和内存模拟
short_term_memory = {}
long_term_db = "customer_memory.db"
class MemoryManager:
    """管理上下文和记忆模块"""
    @staticmethod
    def load_long_term_memory():
        conn = sqlite3.connect(long_term_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_history (
                customer_id TEXT, query TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return conn, cursor
    def save_to_memory(self, customer_id: str, query: str):
        conn, cursor = self.load_long_term_memory()
        cursor.execute("INSERT INTO customer_history (customer_id, query) VALUES (?, ?)", (customer_id, query))
        conn.commit()
        conn.close()
    def get_last_query(self, customer_id: str) -> str:
        conn, cursor = self.load_long_term_memory()
        cursor.execute("SELECT query FROM customer_history WHERE customer_id = ? ORDER BY timestamp DESC LIMIT 1", (customer_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "No history found."
memory_manager = MemoryManager()
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
class ModelManager:
    """管理模型加载和生成"""
    def __init__(self):
        self.model, self.tokenizer = self.load_model_and_tokenizer()
    @staticmethod
    def load_model_and_tokenizer():
        print("Loading model and tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
model_manager = ModelManager()
class VectorSearch:
    """语义检索模块"""
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.knowledge_base = [
            "股票市场是风险投资的主要渠道。",
            "债券投资适合保守型投资者。",
            "外汇市场波动较大，适合有经验的投资者参与。"
        ]
        self.knowledge_vectors = self.model.encode(self.knowledge_base)
    @staticmethod
    def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """手动计算余弦相似度"""
        return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    def search(self, query: str) -> str:
        query_vector = self.model.encode([query])[0]
        similarities = [self.cosine_similarity(query_vector, vec) for vec in self.knowledge_vectors]
        closest_idx = np.argmax(similarities)
        return self.knowledge_base[closest_idx]
vector_search = VectorSearch()
class APIManager:
    """API管理器，用于调用外部服务"""
    @staticmethod
    def get_weather(city: str) -> str:
        api_url = f"http://api.weatherapi.com/v1/current.json?key=your_api_key&q={city}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return f"{city}的温度是 {data['current']['temp_c']}°C"
        return "无法获取天气信息"
class FinancialAgent:
    """金融智能体，整合上下文、模型与API"""
    @performance_monitor
    async def handle_query(self, customer_id: str, query: str):
        memory_manager.save_to_memory(customer_id, query)
        if "天气" in query:
            city = query.split(" ")[-1]
            weather_info = APIManager.get_weather(city)
            print(weather_info)
        elif "投资建议" in query:
            advice = vector_search.search(query)
            print(f"智能投资建议: {advice}")
        else:
            response = model_manager.generate_text(query)
            print(f"智能生成: {response}")
async def main():
    agent = FinancialAgent()
    await agent.handle_query("customer_001", "请问北京的天气如何？")
    await agent.handle_query("customer_001", "给我一些投资建议")
if __name__ == "__main__":
    asyncio.run(main())


