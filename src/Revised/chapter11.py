--------------------------------------------------------------------------------------------------------------


import openai
import os
from langdetect import detect  # 用于检测输入语言
from typing import Optional

# 确保你已经配置了OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class WritingAssistant:
    """智能写作助手，支持多语言生成与语义校准"""

    def __init__(self, terminology: Optional[dict] = None):
        # 初始化术语表，用于语义校准
        self.terminology = terminology or {
            "artificial intelligence": "AI",
            "machine learning": "ML"
        }

    def detect_language(self, text: str) -> str:
        """检测输入文本的语言"""
        try:
            language = detect(text)
            print(f"Detected language: {language}")
            return language
        except Exception as e:
            print(f"Language detection failed: {e}")
            return "en"  # 默认使用英语

    def generate_text(self, prompt: str, target_language: str = "en") -> str:
        """使用OpenAI API生成文本"""
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                messages=f"Translate the following text to {target_language}:\n{prompt}",
                max_tokens=500,
                temperature=0.7
            )
            generated_text = response.choices[0].text.strip()
            print(f"Generated Text: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Text generation failed: {e}")
            return "Generation error."

    def apply_terminology(self, text: str) -> str:
        """校准生成文本中的术语"""
        for term, abbreviation in self.terminology.items():
            text = text.replace(term, abbreviation)
        print(f"Calibrated Text: {text}")
        return text

    def translate_and_calibrate(self, input_text: str, target_language: str):
        """综合流程：检测语言、生成文本并校准术语"""
        detected_language = self.detect_language(input_text)
        generated_text = self.generate_text(input_text, target_language)
        calibrated_text = self.apply_terminology(generated_text)
        return calibrated_text

# 初始化写作助手
assistant = WritingAssistant()

# 测试示例
input_text = "Artificial intelligence is transforming industries worldwide."
target_language = "zh"  # 目标语言：中文

# 运行完整流程
result = assistant.translate_and_calibrate(input_text, target_language)
print("\nFinal Output:\n", result)


--------------------------------------------------------------------------------------------------------------


# 测试代码：多语言生成与语义校准

test_inputs = [
    ("Artificial intelligence is revolutionizing healthcare.", "zh"),
    ("机器学习正在改变金融行业。", "en"),
    ("L'intelligence artificielle transforme l'éducation.", "en"),
    ("La inteligencia artificial está cambiando el mundo.", "fr")
]

for input_text, target_lang in test_inputs:
    print(f"\nInput: {input_text}")
    print(f"Target Language: {target_lang}")
    result = assistant.translate_and_calibrate(input_text, target_lang)
    print(f"Final Output:\n{result}\n")


--------------------------------------------------------------------------------------------------------------


import openai
import os

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class CustomWritingAssistant:
    """个性化写作助手，支持语气、长度和风格定制"""

    def __init__(self):
        self.default_style = "neutral"  # 默认风格为中性
        self.default_tone = "formal"    # 默认语气为正式
        self.default_length = "medium"  # 默认内容长度为中等

    def generate_text(self, prompt: str, style: str, tone: str, length: str) -> str:
        """根据用户偏好生成个性化文本"""
        # 构建动态提示，根据用户偏好调整内容
        dynamic_prompt = (
            f"Write a {length} {tone} piece in a {style} style about:\n{prompt}"
        )
        print(f"Dynamic Prompt: {dynamic_prompt}")

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=dynamic_prompt,
                max_tokens=500 if length == "detailed" else 100,
                temperature=0.7 if tone == "casual" else 0.3
            )
            generated_text = response.choices[0].text.strip()
            print(f"Generated Text: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Error generating text: {e}")
            return "Error generating text."

# 初始化写作助手
assistant = CustomWritingAssistant()

# 用户输入的参数
prompt = "The impact of artificial intelligence on the job market"
style = input("Choose a style (neutral, academic, creative): ").strip()
tone = input("Choose a tone (formal, casual): ").strip()
length = input("Choose a length (short, medium, detailed): ").strip()

# 根据用户参数生成个性化文本
result = assistant.generate_text(prompt, style, tone, length)
print("\nFinal Output:\n", result)
# 测试不同风格、语气和长度的生成结果
test_cases = [
    ("The future of renewable energy", "academic", "formal", "detailed"),
    ("How to stay motivated", "creative", "casual", "short"),
    ("Benefits of cloud computing", "neutral", "formal", "medium"),
]

for prompt, style, tone, length in test_cases:
    print(f"\nPrompt: {prompt}")
    print(f"Style: {style}, Tone: {tone}, Length: {length}")
    result = assistant.generate_text(prompt, style, tone, length)
    print(f"Final Output:\n{result}\n")


--------------------------------------------------------------------------------------------------------------


import openai
import os
# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
class WritingAssistant:
    """基于OpenAI API的智能写作助手，实现内容生成与续写"""

    def generate_content(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """生成内容，根据提示词生成文本"""
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            print(f"Generated Text: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Error generating content: {e}")
            return "Content generation failed."

    def continue_writing(self, initial_text: str, max_tokens: int = 150, temperature: float = 0.5) -> str:
        """续写内容，根据已有文本扩展生成后续内容"""
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                messages=f"Continue the following text:\n{initial_text}",
                max_tokens=max_tokens,
                temperature=temperature
            )
            continued_text = response.choices[0].text.strip()
            print(f"Continued Text: {continued_text}")
            return continued_text
        except Exception as e:
            print(f"Error continuing text: {e}")
            return "Continuation failed."
# 初始化写作助手
assistant = WritingAssistant()
# 测试：生成新内容
prompt = "The impact of artificial intelligence on the job market"
generated_content = assistant.generate_content(prompt, max_tokens=50, temperature=0.7)
print("\nGenerated Content:\n", generated_content)
# 测试：续写内容
initial_text = "Artificial intelligence is rapidly changing the job market by"
continued_content = assistant.continue_writing(initial_text, max_tokens=100, temperature=0.5)
print("\nContinued Content:\n", continued_content)


# 生成新内容的测试用例
print("\nRunning Content Generation Test:")
prompt_1 = "The future of renewable energy"
result_1 = assistant.generate_content(prompt_1, max_tokens=50, temperature=0.7)
print("\nGenerated Content 1:\n", result_1)

# 测试续写功能
print("\nRunning Continuation Test:")
initial_text_2 = "Machine learning has transformed many industries by"
result_2 = assistant.continue_writing(initial_text_2, max_tokens=80, temperature=0.6)
print("\nContinued Content 2:\n", result_2)


--------------------------------------------------------------------------------------------------------------


import openai
import os

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class ContextualWritingAssistant:
    """基于OpenAI API的写作助手，支持多轮交互与上下文保持"""

    def __init__(self):
        self.context = []  # 用于存储用户输入的上下文

    def add_to_context(self, user_input: str):
        """将用户输入添加到上下文中"""
        self.context.append(user_input)

    def get_context_prompt(self) -> str:
        """构建用于API调用的完整上下文提示词"""
        return "\n".join(self.context)

    def generate_response(self, user_input: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """根据当前上下文生成响应"""
        self.add_to_context(user_input)  # 将用户输入添加到上下文
        prompt = self.get_context_prompt()  # 获取完整的上下文提示词

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            print(f"Generated Response: {generated_text}")

            # 将生成的内容添加到上下文
            self.add_to_context(generated_text)
            return generated_text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Response generation failed."

# 初始化写作助手
assistant = ContextualWritingAssistant()

# 多轮交互示例
print("Welcome to the Writing Assistant. Let's start the conversation!")

while True:
    user_input = input("User: ").strip()  # 获取用户输入
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # 生成响应并打印
    response = assistant.generate_response(user_input, max_tokens=150)
    print(f"Assistant: {response}\n")


# 模拟连续输入的测试用例
test_inputs = [
    "Tell me about the history of artificial intelligence.",
    "What are the key advancements in AI?",
    "How is AI used in healthcare today?",
    "Can you summarize the key challenges faced by AI?"
]
print("\nRunning Multi-turn Interaction Test:")
for user_input in test_inputs:
    print(f"User: {user_input}")
    response = assistant.generate_response(user_input, max_tokens=150)
    print(f"Assistant: {response}\n")



# 中文多轮交互测试用例
test_inputs = [
    "告诉我人工智能的发展历史。",
    "有哪些重要的人工智能技术进展？",
    "人工智能如何应用于医疗领域？",
    "总结一下人工智能面临的主要挑战。"
]

print("\n运行多轮交互测试（中文）:")
for user_input in test_inputs:
    print(f"用户: {user_input}")
    response = assistant.generate_response(user_input, max_tokens=150)
    print(f"助手: {response}\n")


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging

# 配置日志记录
logging.basicConfig(
    filename='writing_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥（需将API密钥设置为环境变量）
openai.api_key = os.getenv("OPENAI_API_KEY")

class WritingAssistant:
    """智能写作助手，集成内容生成、续写、多轮交互和个性化功能"""

    def __init__(self):
        self.context = []  # 存储上下文内容
        self.cache = {}  # 简单缓存机制，用于减少API调用

    def add_to_context(self, text: str):
        """将用户输入或生成的文本添加到上下文中"""
        self.context.append(text)
        logging.info(f"Added to context: {text}")

    def get_context_prompt(self) -> str:
        """生成用于API调用的完整上下文提示词"""
        return "\n".join(self.context)

    def generate_text(self, prompt: str, style: str = "neutral", tone: str = "formal",
                      max_tokens: int = 100, temperature: float = 0.7) -> str:
        """根据用户输入生成个性化文本"""
        # 检查缓存中是否已有该生成结果
        cache_key = (prompt, style, tone)
        if cache_key in self.cache:
            logging.info("Cache hit. Returning cached result.")
            return self.cache[cache_key]

        # 动态构建提示词
        dynamic_messages = f"Write a {tone} text in {style} style about: {prompt}"
        self.add_to_context(dynamic_prompt)

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            self.add_to_context(generated_text)
            self.cache[cache_key] = generated_text  # 缓存结果
            logging.info(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            return f"Error: {e}"

    def continue_text(self, initial_text: str, max_tokens: int = 150,
                      temperature: float = 0.5) -> str:
        """根据已有文本续写内容"""
        self.add_to_context(initial_text)

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=temperature
            )
            continued_text = response.choices[0].text.strip()
            self.add_to_context(continued_text)
            logging.info(f"Continued text: {continued_text}")
            return continued_text
        except Exception as e:
            logging.error(f"Error continuing text: {e}")
            return f"Error: {e}"

    def interact_with_user(self):
        """用户交互界面，支持多轮对话"""
        print("欢迎使用智能写作助手！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
            response = self.generate_text(user_input)
            print(f"助手: {response}\n")

# 初始化写作助手
assistant = WritingAssistant()
assistant.interact_with_user()


--------------------------------------------------------------------------------------------------------------


import openai
import os
from textblob import TextBlob  # 用于拼写检查
from googletrans import Translator  # 用于翻译功能

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class EnhancedWritingAssistant:
    """扩展功能的智能写作助手，支持拼写检查、翻译和摘要生成"""

    def __init__(self):
        self.context = []  # 存储上下文内容
        self.translator = Translator()  # 初始化翻译器

    def add_to_context(self, text: str):
        """将文本添加到上下文中"""
        self.context.append(text)

    def get_context_prompt(self) -> str:
        """生成用于API调用的完整上下文提示"""
        return "\n".join(self.context)

    def generate_text(self, prompt: str, max_tokens=100, temperature=0.7) -> str:
        """根据用户输入生成文本"""
        self.add_to_context(prompt)
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            self.add_to_context(generated_text)
            return generated_text
        except Exception as e:
            return f"Error: {e}"

    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """将文本翻译为目标语言"""
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            return f"Translation Error: {e}"

    def check_spelling(self, text: str) -> str:
        """检查并纠正文本中的拼写错误"""
        corrected_text = TextBlob(text).correct()
        return str(corrected_text)

    def summarize_text(self, text: str) -> str:
        """使用OpenAI API生成文本摘要"""
        summary_messages = f"Summarize the following text:\n{text}"
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=summary_prompt,
                max_tokens=50,
                temperature=0.5
            )
            summary = response.choices[0].text.strip()
            return summary
        except Exception as e:
            return f"Summary Error: {e}"

    def interact_with_user(self):
        """用户交互界面"""
        print("欢迎使用智能写作助手！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            # 生成文本并显示拼写检查、翻译和摘要功能
            generated = self.generate_text(user_input)
            print(f"助手生成的内容: {generated}\n")

            # 拼写检查
            corrected = self.check_spelling(generated)
            print(f"拼写检查后: {corrected}\n")

            # 翻译为中文
            translated = self.translate_text(generated, 'zh-cn')
            print(f"翻译为中文: {translated}\n")

            # 生成摘要
            summary = self.summarize_text(generated)
            print(f"摘要: {summary}\n")

# 初始化写作助手
assistant = EnhancedWritingAssistant()
assistant.interact_with_user()


--------------------------------------------------------------------------------------------------------------


import openai
import os
import logging
from textblob import TextBlob  # 拼写检查
from googletrans import Translator  # 翻译功能

# 配置日志记录
logging.basicConfig(
    filename='writing_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class WritingAssistant:
    """集成功能的智能写作助手"""

    def __init__(self):
        self.context = []  # 存储上下文内容
        self.cache = {}  # 缓存机制，减少API调用
        self.translator = Translator()  # 初始化翻译器

    def add_to_context(self, text: str):
        """将文本添加到上下文中"""
        self.context.append(text)
        if len(self.context) > 10:  # 如果上下文超过10条，则移除最旧的一条
            self.context.pop(0)
        logging.info(f"Added to context: {text}")

    def get_context_prompt(self) -> str:
        """生成API调用的完整上下文提示词"""
        return "\n".join(self.context)

    def generate_text(self, prompt: str, max_tokens=100, temperature=0.7) -> str:
        """生成新内容并添加到上下文中"""
        cache_key = prompt
        if cache_key in self.cache:
            logging.info("Cache hit. Returning cached result.")
            return self.cache[cache_key]

        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                prompt=self.get_context_prompt(),
                max_tokens=max_tokens,
                temperature=temperature
            )
            generated_text = response.choices[0].text.strip()
            self.add_to_context(generated_text)
            self.cache[cache_key] = generated_text  # 缓存结果
            logging.info(f"Generated text: {generated_text}")
            return generated_text
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            return f"Error: {e}"

    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """翻译文本"""
        try:
            translation = self.translator.translate(text, dest=target_language)
            return translation.text
        except Exception as e:
            logging.error(f"Translation Error: {e}")
            return f"Translation Error: {e}"

    def check_spelling(self, text: str) -> str:
        """检查拼写并返回纠正后的文本"""
        corrected_text = TextBlob(text).correct()
        return str(corrected_text)

    def summarize_text(self, text: str) -> str:
        """生成文本摘要"""
        try:
            response = openai.ChatCompletion.create(
                engine="gpt-4-0125-preview",
                messages=f"Summarize the following text:\n{text}",
                max_tokens=50,
                temperature=0.5
            )
            summary = response.choices[0].text.strip()
            return summary
        except Exception as e:
            logging.error(f"Summary Error: {e}")
            return f"Summary Error: {e}"

    def interact_with_user(self):
        """用户交互界面"""
        print("欢迎使用智能写作助手！输入 'exit' 退出程序。")
        while True:
            user_input = input("用户: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            generated = self.generate_text(user_input)
            print(f"助手生成的内容: {generated}\n")

            corrected = self.check_spelling(generated)
            print(f"拼写检查后: {corrected}\n")

            translated = self.translate_text(generated, 'zh-cn')
            print(f"翻译为中文: {translated}\n")

            summary = self.summarize_text(generated)
            print(f"摘要: {summary}\n")

# 初始化并运行助手
if __name__ == "__main__":
    assistant = WritingAssistant()
    assistant.interact_with_user()
