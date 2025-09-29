--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging

# 配置日志记录
logging.basicConfig(
    filename='customer_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 设置OpenAI API密钥（需设置为环境变量）
openai.api_key = os.getenv("OPENAI_API_KEY")

class CustomerServiceBot:
    """智能客服系统，集成订单查询、退换货、FAQ及商品推荐功能"""

    def __init__(self):
        self.context = []  # 存储上下文内容
        self.cache = {}  # 缓存，减少API调用

    def add_to_context(self, text: str):
        """将用户输入或生成的文本添加到上下文"""
        self.context.append(text)
        if len(self.context) > 10:  # 限制上下文长度，避免超出API限制
            self.context.pop(0)
        logging.info(f"Added to context: {text}")

    def get_context_prompt(self) -> str:
        """生成用于API调用的上下文提示词"""
        return "\n".join(self.context)

    def generate_response(self, user_input: str, max_tokens=150) -> str:
        """基于用户输入生成响应"""
        self.add_to_context(user_input)
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=0.7
            )
            reply = response.choices[0].text.strip()
            self.add_to_context(reply)
            return reply
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request."

    def handle_order_query(self, order_id: str) -> str:
        """模拟订单查询逻辑"""
        return f"Order {order_id} is currently being processed and will be shipped soon."

    def handle_return_request(self, order_id: str) -> str:
        """模拟退换货逻辑"""
        return f"Your return request for order {order_id} has been submitted. Please follow the instructions sent to your email."

    def answer_faq(self, question: str) -> str:
        """使用OpenAI API回答常见问题"""
        faq_prompt = f"Answer the following question: {question}"
        return self.generate_response(faq_prompt)

    def interact_with_user(self):
        """用户交互界面"""
        print("欢迎使用智能客服系统！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if "订单" in user_input:
                order_id = user_input.split()[-1]
                response = self.handle_order_query(order_id)
            elif "退货" in user_input:
                order_id = user_input.split()[-1]
                response = self.handle_return_request(order_id)
            else:
                response = self.answer_faq(user_input)

            print(f"助手: {response}\n")

# 初始化并运行客服系统
if __name__ == "__main__":
    bot = CustomerServiceBot()
    bot.interact_with_user()


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging
from flask import Flask, request, jsonify

# 设置日志记录
logging.basicConfig(
    filename='multichannel_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化Flask应用
app = Flask(__name__)

class MultiChannelWritingAssistant:
    """多渠道写作智能体"""

    def __init__(self):
        self.context = []  # 上下文存储

    def add_to_context(self, text: str):
        """将用户输入或响应添加到上下文"""
        self.context.append(text)
        if len(self.context) > 10:
            self.context.pop(0)  # 避免上下文过长导致API调用失败
        logging.info(f"Updated context: {text}")

    def get_context_prompt(self) -> str:
        """生成API调用的完整上下文"""
        return "\n".join(self.context)

    def generate_response(self, user_input: str) -> str:
        """生成内容响应"""
        self.add_to_context(user_input)
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.get_context_prompt(),
                max_tokens=100,
                temperature=0.7
            )
            reply = response.choices[0].text.strip()
            self.add_to_context(reply)
            return reply
        except Exception as e:
            logging.error(f"Error: {e}")
            return "Sorry, I couldn't process your request."

# 初始化智能体
assistant = MultiChannelWritingAssistant()

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    """处理来自第三方渠道的Webhook请求"""
    data = request.json
    user_message = data.get('message', '')
    response = assistant.generate_response(user_message)
    return jsonify({"reply": response})

def terminal_interaction():
    """终端交互模拟"""
    print("智能写作助手 - 输入 'exit' 退出。")
    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("再见！")
            break
        response = assistant.generate_response(user_input)
        print(f"助手: {response}\n")

if __name__ == "__main__":
    # 启动终端交互
    terminal_interaction()

    # 启动Flask服务，监听Webhook
    # app.run(host='0.0.0.0', port=5000)



# 请求（来自第三方平台的POST请求）：
{
  "message": "帮我写一封道歉信"
}
# 响应：
{
  "reply": "尊敬的客户，我们为给您带来的不便深表歉意。我们正在努力改进服务，希望能继续得到您的支持。"
}


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging
from typing import List, Dict

# 设置日志记录
logging.basicConfig(
    filename='conversation_manager.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class IntentRecognitionBot:
    """基于OpenAI API的智能客服系统，支持意图识别和多轮对话管理"""

    def __init__(self):
        self.context = []  # 存储对话上下文
        self.intents = {
            "order_query": ["查询订单", "订单状态", "订单跟踪"],
            "refund_request": ["申请退货", "退货", "退款"],
            "faq": ["配送时间", "支付问题", "折扣", "促销"]
        }

    def add_to_context(self, text: str):
        """将用户输入和系统响应添加到上下文中"""
        self.context.append(text)
        if len(self.context) > 10:
            self.context.pop(0)  # 防止上下文过长
        logging.info(f"Updated context: {text}")

    def get_context_prompt(self) -> str:
        """生成用于API调用的上下文提示"""
        return "\n".join(self.context)

    def identify_intent(self, user_input: str) -> str:
        """识别用户意图"""
        for intent, keywords in self.intents.items():
            if any(keyword in user_input for keyword in keywords):
                return intent
        return "unknown"

    def generate_response(self, prompt: str, max_tokens=100) -> str:
        """基于OpenAI API生成响应"""
        self.add_to_context(prompt)
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=0.7
            )
            reply = response.choices[0].text.strip()
            self.add_to_context(reply)
            return reply
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request."

    def handle_intent(self, user_input: str) -> str:
        """根据识别的意图生成响应"""
        intent = self.identify_intent(user_input)
        if intent == "order_query":
            return self.generate_response(f"查询订单: {user_input}")
        elif intent == "refund_request":
            return self.generate_response(f"处理退货请求: {user_input}")
        elif intent == "faq":
            return self.generate_response(f"回答常见问题: {user_input}")
        else:
            return "I'm sorry, I didn't understand your request."

    def interact_with_user(self):
        """用户交互界面"""
        print("欢迎使用智能客服系统！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            response = self.handle_intent(user_input)
            print(f"助手: {response}\n")

# 初始化并运行客服系统
if __name__ == "__main__":
    bot = IntentRecognitionBot()
    bot.interact_with_user()


--------------------------------------------------------------------------------------------------------------


def test_intent_recognition_bot():
    """自动化测试函数，用于验证意图识别与响应生成模块"""

    # 初始化智能客服机器人
    bot = IntentRecognitionBot()

    # 定义测试案例（输入及预期响应的部分关键词）
    test_cases = [
        {"input": "查询订单 10086", "expected_intent": "order_query"},
        {"input": "我想申请退货", "expected_intent": "refund_request"},
        {"input": "配送时间是多久？", "expected_intent": "faq"},
        {"input": "天气怎么样？", "expected_intent": "unknown"},
    ]

    # 运行测试案例并打印结果
    print("开始测试智能客服系统...")

    for case in test_cases:
        user_input = case["input"]
        expected_intent = case["expected_intent"]

        # 识别意图并生成响应
        identified_intent = bot.identify_intent(user_input)
        response = bot.handle_intent(user_input)

        # 打印测试结果
        print(f"输入: {user_input}")
        print(f"识别的意图: {identified_intent} | 预期意图: {expected_intent}")
        print(f"系统响应: {response}")
        print("测试通过" if identified_intent == expected_intent else "测试失败")
        print("-" * 50)

# 调用测试函数
if __name__ == "__main__":
    test_intent_recognition_bot()


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging
from typing import List

# 设置日志记录
logging.basicConfig(
    filename='multiturn_conversation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class MultiTurnConversationBot:
    """支持多轮交互与上下文保持的智能客服系统"""

    def __init__(self, max_context_length=10):
        self.context: List[str] = []  # 上下文存储
        self.max_context_length = max_context_length  # 上下文最大长度

    def add_to_context(self, text: str):
        """将文本添加到上下文中，并保持长度限制"""
        self.context.append(text)
        if len(self.context) > self.max_context_length:
            self.context.pop(0)  # 超出限制时移除最旧的上下文
        logging.info(f"Updated context: {self.context}")

    def get_context_prompt(self) -> str:
        """生成API调用的上下文提示词"""
        return "\n".join(self.context)

    def generate_response(self, user_input: str, max_tokens=100) -> str:
        """基于上下文生成响应"""
        self.add_to_context(f"用户: {user_input}")
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=0.7
            )
            reply = response.choices[0].text.strip()
            self.add_to_context(f"助手: {reply}")
            return reply
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "对不起，我无法处理您的请求。"

    def interact_with_user(self):
        """用户交互界面，支持多轮对话"""
        print("欢迎使用智能客服系统！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            response = self.generate_response(user_input)
            print(f"助手: {response}\n")

# 初始化并运行客服系统
if __name__ == "__main__":
    bot = MultiTurnConversationBot()
    bot.interact_with_user()


--------------------------------------------------------------------------------------------------------------


import random
import logging
from typing import List, Dict

# 设置日志记录
logging.basicConfig(
    filename='nlp_recommendation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RecommendationSystem:
    """推荐系统，用于根据用户历史和行为进行商品推荐"""

    def __init__(self):
        # 假设的商品数据库
        self.products = {
            "electronics": ["手机", "笔记本电脑", "耳机", "智能手表"],
            "books": ["数据科学导论", "Python编程", "机器学习实战", "深度学习"],
            "clothes": ["T恤", "牛仔裤", "运动鞋", "夹克"]
        }

    def recommend_products(self, category: str) -> List[str]:
        """根据商品类别生成推荐列表"""
        if category not in self.products:
            return ["暂无推荐商品"]
        return random.sample(self.products[category], 2)  # 随机推荐两件商品

class NLPModule:
    """自然语言处理模块，用于解析用户输入和识别意图"""

    def __init__(self):
        self.intents = {
            "recommend_electronics": ["推荐电子产品", "我想买手机", "有哪些笔记本电脑推荐？"],
            "recommend_books": ["推荐书籍", "有哪些Python书？", "我要学习机器学习"],
            "recommend_clothes": ["推荐衣服", "冬天穿什么？", "给我推荐些鞋子"]
        }

    def identify_intent(self, user_input: str) -> str:
        """识别用户的推荐意图"""
        for intent, phrases in self.intents.items():
            if any(phrase in user_input for phrase in phrases):
                return intent
        return "unknown"

class Chatbot:
    """聊天机器人，集成NLP模块和推荐系统"""

    def __init__(self):
        self.nlp_module = NLPModule()
        self.recommendation_system = RecommendationSystem()

    def handle_user_input(self, user_input: str) -> str:
        """处理用户输入并返回推荐结果"""
        intent = self.nlp_module.identify_intent(user_input)
        
        if intent == "recommend_electronics":
            products = self.recommendation_system.recommend_products("electronics")
        elif intent == "recommend_books":
            products = self.recommendation_system.recommend_products("books")
        elif intent == "recommend_clothes":
            products = self.recommendation_system.recommend_products("clothes")
        else:
            return "对不起，我无法理解您的需求。"

        response = f"为您推荐以下商品：{', '.join(products)}"
        logging.info(f"User input: {user_input} | Response: {response}")
        return response

    def interact_with_user(self):
        """与用户进行交互"""
        print("欢迎使用智能客服系统！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            response = self.handle_user_input(user_input)
            print(f"助手: {response}\n")

# 运行聊天机器人
if __name__ == "__main__":
    bot = Chatbot()
    bot.interact_with_user()


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging
import random
from typing import List, Dict

# 配置日志记录
logging.basicConfig(
    filename='integrated_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class RecommendationSystem:
    """推荐系统模块"""
    def __init__(self):
        self.products = {
            "electronics": ["手机", "笔记本电脑", "耳机", "智能手表"],
            "books": ["Python编程", "机器学习导论", "数据科学手册"],
            "clothes": ["T恤", "运动鞋", "夹克"]
        }

    def recommend(self, category: str) -> List[str]:
        """根据类别推荐商品"""
        if category not in self.products:
            return ["暂无推荐商品"]
        return random.sample(self.products[category], 2)

class NLPModule:
    """NLP模块，用于意图识别"""
    def __init__(self):
        self.intents = {
            "order_query": ["查询订单", "订单状态"],
            "refund_request": ["退货", "退款"],
            "recommend_electronics": ["推荐电子产品", "推荐手机"],
            "recommend_books": ["推荐书籍", "推荐Python书"],
            "recommend_clothes": ["推荐衣服", "推荐运动鞋"]
        }

    def identify_intent(self, user_input: str) -> str:
        """识别用户意图"""
        for intent, keywords in self.intents.items():
            if any(keyword in user_input for keyword in keywords):
                return intent
        return "unknown"

class OrderManagement:
    """订单与售后管理模块"""
    def query_order(self, order_id: str) -> str:
        """查询订单状态"""
        return f"订单{order_id}正在处理中，预计将在2天内发货。"

    def process_refund(self, order_id: str) -> str:
        """处理退货申请"""
        return f"退货申请已提交。订单{order_id}将在3天内完成处理。"

class Chatbot:
    """智能体主模块"""
    def __init__(self):
        self.nlp_module = NLPModule()
        self.recommendation_system = RecommendationSystem()
        self.order_management = OrderManagement()
        self.context = []

    def add_to_context(self, text: str):
        """将对话添加到上下文"""
        self.context.append(text)
        if len(self.context) > 10:
            self.context.pop(0)
        logging.info(f"Context updated: {self.context}")

    def get_context_prompt(self) -> str:
        """生成上下文提示"""
        return "\n".join(self.context)

    def generate_response(self, user_input: str) -> str:
        """生成自然语言响应"""
        self.add_to_context(f"用户: {user_input}")
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.get_context_prompt(),
                max_tokens=100,
                temperature=0.7
            )
            reply = response.choices[0].text.strip()
            self.add_to_context(f"助手: {reply}")
            return reply
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "对不起，我无法处理您的请求。"

    def handle_user_input(self, user_input: str) -> str:
        """处理用户输入并调用相应模块"""
        intent = self.nlp_module.identify_intent(user_input)

        if intent == "order_query":
            order_id = user_input.split()[-1]
            return self.order_management.query_order(order_id)

        elif intent == "refund_request":
            order_id = user_input.split()[-1]
            return self.order_management.process_refund(order_id)

        elif intent == "recommend_electronics":
            products = self.recommendation_system.recommend("electronics")
            return f"推荐的电子产品：{', '.join(products)}"

        elif intent == "recommend_books":
            books = self.recommendation_system.recommend("books")
            return f"推荐的书籍：{', '.join(books)}"

        elif intent == "recommend_clothes":
            clothes = self.recommendation_system.recommend("clothes")
            return f"推荐的衣服：{', '.join(clothes)}"

        else:
            return self.generate_response(user_input)

    def interact(self):
        """终端交互"""
        print("欢迎使用集成智能客服系统！输入 'exit' 退出。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            response = self.handle_user_input(user_input)
            print(f"助手: {response}\n")

# 初始化并运行系统
if __name__ == "__main__":
    bot = Chatbot()
    bot.interact()


--------------------------------------------------------------------------------------------------------------


import time
import concurrent.futures
from typing import List

# 初始化智能客服系统
bot = Chatbot()

def run_functional_tests():
    """功能测试：验证各模块是否正常工作"""
    print("开始功能测试...")
    test_cases = [
        {"input": "查询订单 12345", "expected": "订单12345正在处理中"},
        {"input": "我要退货订单 67890", "expected": "退货申请已提交"},
        {"input": "推荐一些电子产品", "expected": "推荐的电子产品"},
        {"input": "推荐一些书籍", "expected": "推荐的书籍"},
        {"input": "推荐衣服", "expected": "推荐的衣服"},
    ]

    for case in test_cases:
        response = bot.handle_user_input(case["input"])
        assert case["expected"] in response, f"测试失败: {case['input']}"
        print(f"测试通过: {case['input']} -> {response}")

def run_load_test(concurrent_users: int = 10):
    """负载测试：模拟多用户同时请求"""
    print("开始负载测试...")

    def simulate_user_request():
        start = time.time()
        response = bot.handle_user_input("查询订单 12345")
        duration = time.time() - start
        print(f"用户请求完成，响应时间：{duration:.2f} 秒")
        return duration

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        results = list(executor.map(lambda _: simulate_user_request(), range(concurrent_users)))

    avg_time = sum(results) / len(results)
    print(f"平均响应时间：{avg_time:.2f} 秒")

def run_error_handling_test():
    """错误处理测试：模拟异常情况"""
    print("开始错误处理测试...")

    try:
        response = bot.handle_user_input("模拟错误")
        assert response != "对不起，我无法处理您的请求。"
        print(f"错误处理成功: {response}")
    except Exception as e:
        print(f"捕获异常：{e}")

def run_response_time_test():
    """响应时间测试：确保响应快速"""
    print("开始响应时间测试...")
    start = time.time()
    response = bot.handle_user_input("推荐一些书籍")
    duration = time.time() - start
    print(f"响应时间：{duration:.2f} 秒 -> {response}")

# 执行测试
if __name__ == "__main__":
    run_functional_tests()
    run_load_test(concurrent_users=5)  # 模拟5个并发用户请求
    run_error_handling_test()
    run_response_time_test()


--------------------------------------------------------------------------------------------------------------


# 基于Python官方镜像
FROM python:3.8-slim
# 设置工作目录
WORKDIR /app
# 复制代码文件到容器中
COPY . /app
# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
# 暴露端口
EXPOSE 5000
# 启动命令
CMD ["python", "main.py"]


>> docker build -t smart-chatbot:latest .
>> docker run -d -p 5000:5000 smart-chatbot:latest
>> sudo apt-get update
>> sudo apt-get install -y docker.io
>> sudo systemctl start docker
>> sudo systemctl enable docker


>> docker tag smart-chatbot:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest
>> docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest
docker pull <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest
docker run -d -p 5000:5000 <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest


# 创建一个.github/workflows/deploy.yml文件：
name: Deploy to AWS
on:
  push:
    branches:
      - main
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Login to AWS ECR
      run: |
        aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com

    - name: Build Docker image
      run: docker build -t smart-chatbot:latest .

    - name: Push Docker image to ECR
      run: |
        docker tag smart-chatbot:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest
        docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest

    - name: Deploy to EC2
      run: |
        ssh -i <your-ec2-key.pem> ubuntu@<ec2-instance-ip> 'docker pull <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest && docker run -d -p 5000:5000 <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/smart-chatbot:latest'


>> docker ps
>> curl http://<ec2-instance-ip>:5000
>> curl http://<ec2-instance-ip>:5000 -d '{"message": "查询订单 12345"}'
>> {
>>   "response": "订单12345正在处理中，预计将在2天内发货。"
>> }
