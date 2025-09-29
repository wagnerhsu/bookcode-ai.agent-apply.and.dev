--------------------------------------------------------------------------------------------------------------


from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
# 初始化 ChatGPT 模型（使用 gpt-4）
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
# 创建提示模板
template_1 = PromptTemplate(
    input_variables=["question"],
    template="解析问题: {question}"
)
template_2 = PromptTemplate(
    input_variables=["answer"],
    template="基于答案：{answer}，生成后续步骤"
)
# 创建顺序任务链（使用 RunnableSequence）
chain = RunnableSequence(
    first=template_1 | llm,   # 通过管道连接模板和 LLM
    then=template_2 | llm     # 传递数据给下一个模板和 LLM
)
# 执行任务链并获取结果
response = chain.invoke({"question": "如何实现机器学习模型的训练？"})
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
# 初始化 GPT-4 模型
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
# 创建提示模板
template_1 = PromptTemplate(
    input_variables=["query"],
    template="请解析查询：{query}"
)
template_2 = PromptTemplate(
    input_variables=["parsed_result"],
    template="基于解析结果：{parsed_result}，生成响应。"
)
# 将模板与 LLM 结合，创建 LLMChain
chain_1 = LLMChain(llm=llm, prompt=template_1, output_key="parsed_result")
chain_2 = LLMChain(llm=llm, prompt=template_2, output_key="final_response")
# 创建顺序任务链
sequential_chain = SequentialChain(
    chains=[chain_1, chain_2],  # 加入任务链
    input_variables=["query"],  # 初始输入变量
    output_variables=["final_response"],  # 最终输出变量
    verbose=True
)
# 执行任务链并打印结果
response = sequential_chain.run({"query": "如何训练机器学习模型？"})
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# 初始化 GPT-4 模型
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
# 定义贷款任务链
loan_template = PromptTemplate(
    input_variables=["user_input"],
    template="用户询问贷款问题：{user_input}。请详细解释贷款流程。"
)
loan_chain = LLMChain(llm=llm, prompt=loan_template, output_key="loan_response")
# 定义信用卡任务链
credit_card_template = PromptTemplate(
    input_variables=["user_input"],
    template="用户询问信用卡问题：{user_input}。请详细解释信用卡的申请条件。"
)
credit_card_chain = LLMChain(llm=llm, prompt=credit_card_template, output_key="credit_card_response")
# 自定义路由逻辑
def route_task(task_type, user_input):
    """根据任务类型选择相应的任务链并执行。"""
    if task_type == "贷款":
        response = loan_chain.run({"user_input": user_input})
    elif task_type == "信用卡":
        response = credit_card_chain.run({"user_input": user_input})
    else:
        response = "无效的任务类型，请输入'贷款'或'信用卡'。"
    return response
# 测试路由逻辑
task_type = "贷款"
user_input = "如何申请个人贷款？"
response = route_task(task_type, user_input
# 打印响应结果
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import ConversationChain
from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-4")
# 初始化对话链，保持上下文信息
conversation = ConversationChain(
    llm=llm,
    memory=True,  # 开启上下文记忆
    verbose=True
)
# 执行多轮对话
response = conversation.predict(input="查询最近的订单状态")
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain.llms import OpenAI
llm = OpenAI(model_name="gpt-4")
# 调用 LLM 执行推理任务
response = llm("根据最新市场趋势，给出投资建议")
print(response)
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import SequentialChain
# 创建回调处理程序
callback = StdOutCallbackHandler()
# 创建任务链并添加回调
chain = SequentialChain(
    chains=[template_1, template_2],# 输入自定义模板
    input_variables=["question"],
    callbacks=[callback],
    verbose=True
)
# 执行任务链
response = chain({"question": "最新的财务数据是什么？"})



from langchain.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["question"],
    template="请回答以下问题：{question}"
)


from langchain.prompts import PromptTemplate
# 定义客户服务模板
template = PromptTemplate(
    input_variables=["product", "issue"],
    template="请帮忙查询{product}的当前状态，并解决以下问题：{issue}"
)
# 在调用时用具体内容替换占位符
prompt = template.format(product="智能手表", issue="无法开机")
print(prompt)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# 初始化语言模型
llm = OpenAI(model_name="gpt-4")
# 定义任务模板
template_1 = PromptTemplate(
    input_variables=["query"],
    template="请解析以下用户请求：{query}"
)
template_2 = PromptTemplate(
    input_variables=["result"],
    template="根据解析结果：{result}，生成最终响应。"
)
# 创建顺序链
chain = SequentialChain(
    chains=[template_1, template_2],
    input_variables=["query"],
    verbose=True
)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import AsyncSequentialChain
# 定义异步任务链
async_chain = AsyncSequentialChain(
    chains=[template_1, template_2],
    input_variables=["query"],
    verbose=True
)
# 执行异步任务链
await async_chain.run({"query": "查询订单状态"})


--------------------------------------------------------------------------------------------------------------


from langchain.chains import RouterChain
# 根据输入动态选择任务路径
router_chain = RouterChain(
    conditions={"查询订单": order_query_chain, "查询物流": logistics_query_chain}
)
# 运行任务链
router_chain.run({"query": "查询物流信息"})


--------------------------------------------------------------------------------------------------------------


from apscheduler.schedulers.blocking import BlockingScheduler
from langchain.chains import SequentialChain
# 创建顺序任务链
chain = SequentialChain(chains=[...])
# 创建调度器并定义定时任务
scheduler = BlockingScheduler()
scheduler.add_job(lambda: chain.run({"query": "生成日报"}), 'cron', hour=8)
# 启动调度器
scheduler.start()
from langchain.chains import WebhookChain
# 创建 Webhook 任务链
webhook_chain = WebhookChain(webhook_url="https://example.com/webhook")
# 定义任务逻辑
def on_new_order(data):
    chain.run(data)
# 注册事件回调
webhook_chain.on_event(on_new_order)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import ConversationChain
from langchain.llms import OpenAI
# 初始化语言模型
llm = OpenAI(model_name="gpt-4")
# 创建对话任务链
conversation_chain = ConversationChain(llm=llm, verbose=True)
# 用户输入触发任务链
response = conversation_chain.predict(input="查询订单状态")
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import AsyncSequentialChain
# 异步任务链，市场数据与客户分析并行执行
async_chain = AsyncSequentialChain(
    chains=[market_data_chain, client_analysis_chain],
    verbose=True
)
# 执行异步任务链
await async_chain.run(input_data)


--------------------------------------------------------------------------------------------------------------


from langchain.cache import InMemoryCache
# 初始化缓存系统
cache = InMemoryCache()
# 使用缓存查询客户数据
def get_client_preferences(client_id):
    if cache.get(client_id):
        return cache.get(client_id)
    preferences = query_client_preferences(client_id)  # 查询数据库
    cache.set(client_id, preferences)
    return preferences
from langchain.chains import RouterChain
# 根据客户类型动态选择任务路径
router_chain = RouterChain(
    conditions={
        "短期投资": short_term_strategy_chain,
        "长期投资": long_term_strategy_chain,
    }
)
# 执行动态任务链
router_chain.run({"client_type": "短期投资"})


--------------------------------------------------------------------------------------------------------------


from langchain.chains import SequentialChain
def safe_execute_module(module, retries=3):
    for attempt in range(retries):
        try:
            return module.run()
        except Exception as e:
            print(f"错误发生：{e}，重试次数：{attempt + 1}")
    raise Exception("任务执行失败")
# 包装模块执行，确保发生错误时自动重试
safe_execute_module(market_data_chain)


--------------------------------------------------------------------------------------------------------------


from langchain.callbacks import StdOutCallbackHandler
# 初始化回调处理程序
callback = StdOutCallbackHandler()
# 监控任务链执行
chain = SequentialChain(
    chains=[market_data_chain, client_analysis_chain],
    callbacks=[callback],
    verbose=True
)


--------------------------------------------------------------------------------------------------------------


import mysql.connector
# 连接 MySQL 数据库
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="ecommerce"
)
# 查询订单状态
def query_order_status(order_id):
    cursor = db.cursor()
    cursor.execute(f"SELECT status FROM orders WHERE id = {order_id}")
    result = cursor.fetchone()
    return result


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化向量数据库
vector_store = FAISS.load_local("path/to/index", OpenAIEmbeddings())
# 根据查询内容检索相似文档
results = vector_store.similarity_search("查询物流信息", k=5)


--------------------------------------------------------------------------------------------------------------


import requests
# 获取实时股票行情
def get_stock_price(symbol):
    response = requests.get(f"https://api.stock.com/quote?symbol={symbol}")
    return response.json()["price"]


--------------------------------------------------------------------------------------------------------------


query = """
{
  user(id: "123") {
    name
    orderHistory {
      id
      status
    }
  }
}
"""
response = requests.post("https://api.example.com/graphql", json={"query": query})


--------------------------------------------------------------------------------------------------------------


from PyPDF2 import PdfReader
# 读取 PDF 文件并提取文本
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


--------------------------------------------------------------------------------------------------------------


import pandas as pd
# 读取 Excel 文件并解析数据
df = pd.read_excel("report.xlsx")
summary = df.describe()


--------------------------------------------------------------------------------------------------------------


import paho.mqtt.client as mqtt
# 初始化 MQTT 客户端并连接到 IoT 平台
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883)
# 订阅温度传感器数据
def on_message(client, userdata, message):
    temperature = float(message.payload.decode())
    if temperature > 25:
        print("启动空调")
client.on_message = on_message
client.subscribe("home/temperature")
client.loop_start()


--------------------------------------------------------------------------------------------------------------


from langchain.chains import ConversationChain
from langchain.llms import OpenAI
# 初始化对话链
llm = OpenAI(model_name="gpt-4")
conversation = ConversationChain(llm=llm, verbose=True)
# 用户输入与系统响应
response = conversation.predict(input="我订的订单什么时候到？")
print(response)


--------------------------------------------------------------------------------------------------------------


from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# 初始化向量存储
vector_store = FAISS.load_local("path/to/index", OpenAIEmbeddings())
# 存储用户偏好数据
def store_user_preference(user_id, preference):
    vector_store.add_texts([preference], metadatas=[{"user_id": user_id}])
# 检索用户偏好
def retrieve_user_preference(user_id):
    return vector_store.similarity_search(f"用户 {user_id} 的偏好", k=1)


--------------------------------------------------------------------------------------------------------------


from langchain.memory import ConversationBufferMemory
# 初始化上下文管理器，并限制上下文长度
memory = ConversationBufferMemory(memory_key="chat_history", max_length=5)
# 在对话链中使用上下文管理器
conversation = ConversationChain(llm=llm, memory=memory)


--------------------------------------------------------------------------------------------------------------


from langchain.chains import ConversationChain
# 初始化多用户对话链
user_conversations = {}
def get_user_conversation(user_id):
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationChain(llm=llm)
    return user_conversations[user_id]
# 根据用户 ID 获取并执行对话链
response = get_user_conversation("user_123").predict(input="查询订单状态")
print(response)


--------------------------------------------------------------------------------------------------------------


import asyncio
import time
import random
from typing import Dict, List, Any, Callable
from functools import wraps
class FakeLLM:
    """LLM 模型，生成自然语言响应。"""
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
    def __call__(self, prompt: str) -> str:
        """生成的响应内容，来自 GPT 模型。"""
        responses = {
            "Hello, Alice!": "Hello, Alice! Nice to meet you!",
            "Fetching data for Machine Learning.": "Here is the latest data on Machine Learning."
        }
        # 根据 prompt 返回响应
        return responses.get(prompt, "I have no idea what you are asking.")
class LangChainAgent:
    """LangChain驱动的智能体系统，支持多步骤任务链与上下文管理。"""
    def __init__(self, llm: FakeLLM):
        self.context = {}  # 上下文存储，用于任务之间的数据传递
        self.tasks = []  # 任务链容器
        self.llm = llm  # 引入 LLM 实例

    def add_task(self, func: Callable, name: str = None):
        """向任务链中添加任务，并为任务指定名称。"""
        task_name = name if name else func.__class__.__name__
        print(f"Adding task: {task_name}")
        self.tasks.append((func, task_name))

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """依次执行任务链中的所有任务，并返回最终结果。"""
        data = input_data
        for task, name in self.tasks:
            print(f"Executing task: {name}")
            data = await task(data)  # 任务执行
        return data

class TemplateTask:
    """模板化任务，用于解析用户输入并生成响应。"""

    def __init__(self, template: str, llm: FakeLLM):
        self.template = template
        self.llm = llm  # 引入LLM 模型

    async def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用LLM 生成响应并返回结果。"""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        input_value = data.get('name') or data.get('query', 'unknown')
        prompt = self.template.format(name_or_query=input_value)
        response = self.llm(prompt)  # 生成响应
        print(f"Task completed: {response}")
        return {"response": response}
class RouterChain:
    """路由链，根据条件选择任务路径。"""
    def __init__(self, routes: Dict[str, Callable]):
        self.routes = routes
    async def execute(self, condition: str, input_data: Dict[str, Any]):
        """根据条件选择并执行路径中的任务链。"""
        if condition in self.routes:
            print(f"Routing to {condition}...")
            await self.routes[condition](input_data)
        else:
            print(f"No route found for condition: {condition}")

class PerformanceMonitor:
    """性能监控器，记录并报告系统的运行情况。"""
    def __init__(self):
        self.execution_times = []
    def log_time(self, func: Callable, task_name: str = None):
        """装饰器：记录任务的执行时间，并为任务指定名称。"""
        task_name = task_name if task_name else func.__class__.__name__
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            self.execution_times.append(elapsed)
            print(f"Task {task_name} executed in {elapsed:.2f}s")
            return result
        return wrapper
    def report(self):
        """生成性能报告。"""
        total = sum(self.execution_times)
        print(f"Total execution time: {total:.2f}s")
        print(f"Average execution time: {total / len(self.execution_times):.2f}s")
# 初始化 LLM 模型
llm = FakeLLM(temperature=0.5)
# 初始化智能体与性能监控器
monitor = PerformanceMonitor()
agent = LangChainAgent(llm)
# 定义任务模板并添加到任务链
greet_task = TemplateTask(template="Hello, {name_or_query}!", llm=llm)
query_task = TemplateTask(template="Fetching data for {name_or_query}.", llm=llm)
agent.add_task(monitor.log_time(greet_task, "Greet Task"))
agent.add_task(monitor.log_time(query_task, "Query Task"))
# 创建路由链并定义路径
router = RouterChain(routes={
    "greet": lambda data: agent.run(data),
    "query": lambda data: agent.run(data)
})
@monitor.log_time
async def main():
    """主程序入口，执行任务链并生成性能报告。"""
    print("Starting LangChain agent...")
    await router.execute("greet", {"name": "Alice"})
    await router.execute("query", {"query": "Machine Learning"})
    monitor.report()
# 执行主程序
if __name__ == "__main__":
    asyncio.run(main())

