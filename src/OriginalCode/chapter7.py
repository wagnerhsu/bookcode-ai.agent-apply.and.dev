--------------------------------------------------------------------------------------------------------------


import openai
from langdetect import detect
# 设置OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"
# 定义术语词库
TERMINOLOGY = {
    "massive MIMO": "大规模多输入多输出",
    "beamforming": "波束成形",
    "5G network": "5G网络"
}
def translate_text(input_text, target_language="zh"):
    """多语言翻译函数，带术语校准"""
    # 检测输入语言
    source_language = detect(input_text)
    print(f"Detected source language: {source_language}")
    # 构建翻译Prompt
    prompt = f"Translate the following text from {source_language} to {target_language}: \n{input_text}"
    # 调用OpenAI API进行翻译
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=512,
        temperature=0.5
    )   
    # 获取翻译结果
    translated_text = response.choices[0].text.strip()
    print(f"Initial Translation: {translated_text}")
    # 校准术语
    for term, translation in TERMINOLOGY.items():
        translated_text = translated_text.replace(term, translation)
    print(f"Final Translation with Terminology Adjustments: {translated_text}")
    return translated_text
# 测试多语言翻译与术语校准
input_text = "The 5G network uses massive MIMO and beamforming technologies to improve performance."
translated_output = translate_text(input_text, target_language="zh")
print("\nFinal Output:\n", translated_output)


--------------------------------------------------------------------------------------------------------------


# 代码示例：输入预处理与输出优化
import openai
from langdetect import detect
# 设置OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"
def preprocess_input(text):
    """输入预处理：清理和标准化用户输入"""
    cleaned_text = text.strip().replace("\n", " ")
    print(f"Preprocessed Input: {cleaned_text}")
    return cleaned_text
def translate_text(input_text, target_language="zh"):
    """调用大语言模型进行翻译"""
    # 检测输入语言
    source_language = detect(input_text)
    print(f"Detected Source Language: {source_language}")
    # 构建Prompt
    prompt = f"Translate the following text from {source_language} to {target_language}: \n{input_text}"
    # 调用OpenAI模型
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=512,
        temperature=0.5
    )
    # 获取翻译结果
    translated_text = response.choices[0].text.strip()
    print(f"Translated Output: {translated_text}")
    return translated_text
def format_output(text):
    """输出优化：添加格式和标记"""
    formatted_text = f"***Translation Result***\n{text}\n\n---\nThank you for using the translation service!"
    print(f"Formatted Output: {formatted_text}")
    return formatted_text
# 主程序逻辑
def main():
    input_text = """
    The rapid development of 5G networks has revolutionized wireless communication,
    enabling new services such as IoT and smart cities.
    """
    print("Raw Input:", input_text)
    # 输入预处理
    cleaned_text = preprocess_input(input_text)
    # 翻译
    translated_text = translate_text(cleaned_text, target_language="zh")
    # 输出优化
    final_output = format_output(translated_text)
    print("\nFinal Optimized Output:\n", final_output)
# 运行主程序
if __name__ == "__main__":
    main()


--------------------------------------------------------------------------------------------------------------


# 代码示例：多语言模型调用与上下文保持
import openai
# 设置OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"
class TranslationAgent:
    """翻译智能体，支持多轮对话中的上下文保持。"""
    def __init__(self):
        self.conversation_history = []  # 用于存储对话历史
    def add_to_history(self, role, message):
        """将新对话添加到历史记录中。"""
        self.conversation_history.append(f"{role}: {message}")
    def build_prompt(self, user_message):
        """构建包含上下文的Prompt，用于模型调用。"""
        history = "\n".join(self.conversation_history[-5:])  # 只保留最近5条记录
        prompt = f"{history}\nUser: {user_message}\nAI:"
        return prompt
    def translate(self, user_message, target_language="zh"):
        """调用GPT模型进行翻译，并保持上下文。"""
        # 构建Prompt
        prompt = self.build_prompt(user_message)
        # 调用OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=512,
            temperature=0.5
        )
        # 获取翻译结果
        translated_text = response.choices[0].text.strip()
        # 将新对话添加到历史
        self.add_to_history("User", user_message)
        self.add_to_history("AI", translated_text)

        return translated_text
# 初始化智能体
agent = TranslationAgent()
# 模拟多轮对话
print("Starting conversation...")
user_input1 = "What is 5G technology?"
response1 = agent.translate(user_input1, target_language="zh")
print(f"Translation 1: {response1}")

user_input2 = "How does beamforming work in 5G?"
response2 = agent.translate(user_input2, target_language="zh")
print(f"Translation 2: {response2}")

user_input3 = "Explain the advantages of massive MIMO."
response3 = agent.translate(user_input3, target_language="zh")
print(f"Translation 3: {response3}")


--------------------------------------------------------------------------------------------------------------


import openai
import time

# 设置OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"

class RobustTranslationAgent:
    """支持翻译优化和错误处理的智能翻译系统。"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries  # 最大重试次数

    def translate(self, text, target_language="zh"):
        """执行翻译并处理错误，确保系统稳定性。"""
        prompt = f"Translate the following text to {target_language}:\n{text}"
        
        for attempt in range(self.max_retries):
            try:
                # 调用OpenAI API
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.5
                )
                translated_text = response.choices[0].text.strip()
                optimized_text = self.optimize_translation(translated_text)
                return optimized_text
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)  # 等待2秒后重试
        
        # 返回默认提示信息
        return "Translation failed. Please try again later."

    def optimize_translation(self, translated_text):
        """对翻译结果进行优化，如消除多余空格或不自然的句子结构。"""
        optimized_text = translated_text.replace("  ", " ")  # 去除多余空格
        # 其他优化逻辑可在此处扩展
        return optimized_text

# 初始化翻译智能体
agent = RobustTranslationAgent()

# 测试翻译并演示错误处理
input_text = "5G technology is revolutionizing communication networks around the world."
result = agent.translate(input_text, target_language="zh")
print("\nFinal Translation:\n", result)


--------------------------------------------------------------------------------------------------------------


import openai
import time

# 设置OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"

class TranslationAgent:
    """支持Prompt设计、多轮交互和错误处理的智能翻译系统。"""

    def __init__(self, max_retries=3):
        self.conversation_history = []  # 存储对话历史
        self.max_retries = max_retries  # 最大重试次数

    def add_to_history(self, role, message):
        """将对话历史添加到记录中。"""
        self.conversation_history.append(f"{role}: {message}")

    def build_prompt(self, user_input):
        """构建包含历史对话的Prompt。"""
        history = "\n".join(self.conversation_history[-5:])  # 最近5条对话
        prompt = f"{history}\nUser: {user_input}\nAI:"
        return prompt

    def translate(self, user_input, target_language="zh"):
        """执行翻译并支持多轮交互与错误处理。"""
        self.add_to_history("User", user_input)  # 记录用户输入

        prompt = self.build_prompt(user_input)  # 构建Prompt

        for attempt in range(self.max_retries):
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.5
                )
                translated_text = response.choices[0].text.strip()
                self.add_to_history("AI", translated_text)  # 记录模型响应
                return translated_text

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)  # 等待2秒重试

        return "Translation failed. Please try again later."

    def feedback_loop(self, initial_input, target_language="zh"):
        """支持用户多轮反馈与翻译优化的交互循环。"""
        current_input = initial_input

        while True:
            translation = self.translate(current_input, target_language)
            print(f"\nAI Translation:\n{translation}")

            feedback = input("Would you like to modify the translation? (y/n): ").strip().lower()
            if feedback == 'n':
                break

            current_input = input("Please provide your updated input: ").strip()

# 初始化智能体
agent = TranslationAgent()

# 开始多轮交互翻译
print("Welcome to the Translation Assistant.")
initial_text = "What are the key benefits of 5G technology?"
agent.feedback_loop(initial_text, target_language="zh")


--------------------------------------------------------------------------------------------------------------

>> python -m venv myenv 
>> source myenv/bin/activate # Linux/Mac 
>> .\myenv\Scripts\activate # Windows
>> pip install openai langdetect requests
>> export OPENAI_API_KEY="your_openai_api_key"  # Linux/Mac
>> set OPENAI_API_KEY="your_openai_api_key"  # Windows

--------------------------------------------------------------------------------------------------------------


import openai
import os

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai_connection():
    """测试与OpenAI API的连接是否正常。"""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Hello, how are you?",
            max_tokens=50
        )
        print("API Connection Successful!")
        print("Response:\n", response.choices[0].text.strip())
    except Exception as e:
        print("API Connection Failed.")
        print("Error:", str(e))

# 运行测试函数
if __name__ == "__main__":
    test_openai_connection()


--------------------------------------------------------------------------------------------------------------


# 翻译系统主要模块
import openai
import os
import time
from langdetect import detect

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class TranslationSystem:
    """智能翻译系统，支持输入预处理、翻译和术语校准。"""

    def __init__(self):
        self.terminology = {
            "5G network": "5G网络",
            "beamforming": "波束成形",
            "massive MIMO": "大规模多输入多输出"
        }

    def preprocess_input(self, text):
        """预处理输入，去除多余空格和换行符。"""
        cleaned_text = text.strip().replace("\n", " ")
        print(f"Preprocessed Input: {cleaned_text}")
        return cleaned_text

    def translate(self, text, target_language="zh", retries=3):
        """调用OpenAI API进行翻译，并处理错误。"""
        prompt = f"Translate the following text to {target_language}:\n{text}"
        
        for attempt in range(retries):
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.5
                )
                translated_text = response.choices[0].text.strip()
                print(f"Translated Output: {translated_text}")
                return translated_text
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)  # 等待2秒后重试
        
        return "Translation failed. Please try again later."

    def apply_terminology(self, text):
        """对翻译结果进行术语校准。"""
        for term, translation in self.terminology.items():
            text = text.replace(term, translation)
        print(f"Optimized Output: {text}")
        return text

    def format_output(self, text):
        """格式化输出，添加分隔符和结束语。"""
        formatted_text = f"***Translation Result***\n{text}\n\n---\nThank you for using our translation service!"
        print(f"Formatted Output: {formatted_text}")
        return formatted_text

    def translate_and_optimize(self, input_text, target_language="zh"):
        """整合流程：预处理、翻译、术语校准和格式化。"""
        cleaned_input = self.preprocess_input(input_text)
        translated_text = self.translate(cleaned_input, target_language)
        optimized_text = self.apply_terminology(translated_text)
        final_output = self.format_output(optimized_text)
        return final_output

# 初始化系统实例
translator = TranslationSystem()

# 测试：翻译示例文本并输出优化结果
test_input = "5G network and beamforming technology are revolutionizing communication."
result = translator.translate_and_optimize(test_input)
print("\nFinal Output:\n", result)


--------------------------------------------------------------------------------------------------------------


import openai
import os
import time
import speech_recognition as sr  # 用于语音输入
from langdetect import detect

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class TranslationSystem:
    """扩展的智能翻译系统，支持语音输入、缓存和日志记录。"""

    def __init__(self):
        self.terminology = {
            "5G network": "5G网络",
            "beamforming": "波束成形",
            "massive MIMO": "大规模多输入多输出"
        }
        self.translation_cache = {}  # 缓存已翻译的内容

    def preprocess_input(self, text):
        """预处理输入，去除多余空格和换行符。"""
        cleaned_text = text.strip().replace("\n", " ")
        print(f"Preprocessed Input: {cleaned_text}")
        return cleaned_text

    def translate(self, text, target_language="zh", retries=3):
        """调用OpenAI API进行翻译，并处理错误。"""
        if text in self.translation_cache:
            print("Cache Hit! Returning cached translation.")
            return self.translation_cache[text]

        prompt = f"Translate the following text to {target_language}:\n{text}"
        for attempt in range(retries):
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.5
                )
                translated_text = response.choices[0].text.strip()
                self.translation_cache[text] = translated_text  # 缓存翻译结果
                print(f"Translated Output: {translated_text}")
                return translated_text
            
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)

        return "Translation failed. Please try again later."

    def apply_terminology(self, text):
        """对翻译结果进行术语校准。"""
        for term, translation in self.terminology.items():
            text = text.replace(term, translation)
        print(f"Optimized Output: {text}")
        return text

    def format_output(self, text):
        """格式化输出，添加分隔符和结束语。"""
        formatted_text = f"***Translation Result***\n{text}\n\n---\nThank you for using our translation service!"
        print(f"Formatted Output: {formatted_text}")
        return formatted_text

    def log_request(self, input_text, translated_text):
        """记录请求和响应日志。"""
        with open("translation_log.txt", "a") as log_file:
            log_file.write(f"Input: {input_text}\nTranslation: {translated_text}\n---\n")

    def translate_and_optimize(self, input_text, target_language="zh"):
        """整合流程：预处理、翻译、术语校准和格式化。"""
        cleaned_input = self.preprocess_input(input_text)
        translated_text = self.translate(cleaned_input, target_language)
        optimized_text = self.apply_terminology(translated_text)
        final_output = self.format_output(optimized_text)
        self.log_request(input_text, optimized_text)  # 记录日志
        return final_output

    def speech_to_text(self):
        """将语音输入转换为文本。"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say something...")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized Speech: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
# 初始化系统实例
translator = TranslationSystem()
# 测试：语音输入或文本翻译
print("Choose input method:")
print("1. Text Input")
print("2. Speech Input")
choice = input("Enter choice (1 or 2): ").strip()
if choice == "1":
    test_input = "5G network and beamforming technology are revolutionizing communication."
    result = translator.translate_and_optimize(test_input)
    print("\nFinal Output:\n", result)
elif choice == "2":
    speech_text = translator.speech_to_text()
    if speech_text:
        result = translator.translate_and_optimize(speech_text)
        print("\nFinal Output:\n", result)
else:
    print("Invalid choice. Please enter 1 or 2.")


--------------------------------------------------------------------------------------------------------------


import openai
import os
from nltk.corpus import wordnet  # 用于同义词查找
import nltk

# 确保NLTK词库已安装
nltk.download('wordnet')

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class LanguageAssistance:
    """语言辅助模块，支持智能提示、同义词替换和句子优化。"""

    def __init__(self):
        self.history = []  # 存储用户输入的历史

    def suggest_synonyms(self, word):
        """为给定单词提供同义词建议。"""
        synonyms = wordnet.synsets(word)
        synonym_list = set()
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonym_list.add(lemma.name())
        return list(synonym_list)

    def auto_complete(self, text):
        """基于部分输入内容自动补全句子。"""
        prompt = f"Complete the following sentence:\n{text}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        completion = response.choices[0].text.strip()
        print(f"Auto-completion: {completion}")
        return completion

    def optimize_sentence(self, text):
        """通过OpenAI模型优化句子结构。"""
        prompt = f"Optimize the following sentence for clarity and conciseness:\n{text}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.5
        )
        optimized_text = response.choices[0].text.strip()
        print(f"Optimized Sentence: {optimized_text}")
        return optimized_text

    def add_to_history(self, text):
        """将用户输入添加到历史记录中。"""
        self.history.append(text)

    def show_history(self):
        """展示用户输入历史。"""
        print("Input History:")
        for entry in self.history:
            print(f"- {entry}")

# 初始化语言辅助模块
assistant = LanguageAssistance()

# 测试：同义词建议
word = "communication"
synonyms = assistant.suggest_synonyms(word)
print(f"Synonyms for '{word}': {synonyms}")

# 测试：自动补全句子
incomplete_sentence = "The future of technology lies in"
completion = assistant.auto_complete(incomplete_sentence)
print(f"Completed Sentence: {incomplete_sentence} {completion}")

# 测试：句子优化
sentence = "The communication systems of the future are expected to be extremely fast and highly reliable."
optimized_sentence = assistant.optimize_sentence(sentence)
print(f"Optimized Sentence: {optimized_sentence}")

# 展示输入历史
assistant.add_to_history(sentence)
assistant.add_to_history(completion)
assistant.show_history()


--------------------------------------------------------------------------------------------------------------


import openai
import os
from flask import Flask, request, jsonify
from langdetect import detect

# 初始化Flask应用
app = Flask(__name__)

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_text(text, target_language="zh"):
    """调用OpenAI API进行翻译。"""
    prompt = f"Translate the following text to {target_language}:\n{text}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=512,
        temperature=0.5
    )
    translated_text = response.choices[0].text.strip()
    return translated_text

@app.route('/translate', methods=['POST'])
def translate():
    """处理翻译请求的API端点。"""
    data = request.get_json()
    input_text = data.get('text', '')
    target_language = data.get('target_language', 'zh')

    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        translated_text = translate_text(input_text, target_language)
        return jsonify({'translated_text': translated_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
>> python app.py
>> curl -X POST http://localhost:5000/translate \
>> -H "Content-Type: application/json" \
>> -d '{"text": "What is 5G technology?", "target_language": "zh"}'
{
  "translated_text": "什么是5G技术？"
}
{
  "error": "No text provided"
}
import openai
import os
import time
import speech_recognition as sr  # 语音输入库
from flask import Flask, request, jsonify
from langdetect import detect  # 语言检测
import nltk

# 确保NLTK词库安装
nltk.download('wordnet')
from nltk.corpus import wordnet

# 初始化Flask应用
app = Flask(__name__)

# 从环境变量中获取API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

class TranslationSystem:
    """智能翻译系统，支持缓存、术语校准、语音输入和日志记录。"""

    def __init__(self):
        self.terminology = {
            "5G network": "5G网络",
            "beamforming": "波束成形",
            "massive MIMO": "大规模多输入多输出"
        }
        self.translation_cache = {}  # 缓存已翻译的内容

    def preprocess_input(self, text):
        """预处理输入，去除多余空格和换行符。"""
        return text.strip().replace("\n", " ")

    def translate_text(self, text, target_language="zh", retries=3):
        """调用OpenAI API进行翻译，并处理错误。"""
        if text in self.translation_cache:
            print("Cache Hit! Returning cached translation.")
            return self.translation_cache[text]

        prompt = f"Translate the following text to {target_language}:\n{text}"
        for attempt in range(retries):
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.5
                )
                translated_text = response.choices[0].text.strip()
                self.translation_cache[text] = translated_text
                return translated_text
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                time.sleep(2)

        return "Translation failed. Please try again later."

    def apply_terminology(self, text):
        """对翻译结果进行术语校准。"""
        for term, translation in self.terminology.items():
            text = text.replace(term, translation)
        return text

    def log_request(self, input_text, translated_text):
        """记录日志到文件。"""
        with open("translation_log.txt", "a") as log_file:
            log_file.write(f"Input: {input_text}\nTranslation: {translated_text}\n---\n")

    def speech_to_text(self):
        """将语音输入转换为文本。"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say something...")
            audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio."
        except sr.RequestError as e:
            return f"Request failed; {e}"

    def translate_and_optimize(self, input_text, target_language="zh"):
        """整合流程：预处理、翻译、术语校准和日志记录。"""
        cleaned_text = self.preprocess_input(input_text)
        translated_text = self.translate_text(cleaned_text, target_language)
        optimized_text = self.apply_terminology(translated_text)
        self.log_request(input_text, optimized_text)
        return optimized_text

translator = TranslationSystem()  # 初始化系统

@app.route('/translate', methods=['POST'])
def translate():
    """API端点：处理翻译请求。"""
    data = request.get_json()
    input_text = data.get('text', '')
    target_language = data.get('target_language', 'zh')

    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        result = translator.translate_and_optimize(input_text, target_language)
        return jsonify({'translated_text': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Choose input method:")
    print("1. Text Input")
    print("2. Speech Input")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        test_input = "The future of 5G networks is promising."
        result = translator.translate_and_optimize(test_input)
        print("\nFinal Output:\n", result)

    elif choice == "2":
        speech_text = translator.speech_to_text()
        if speech_text:
            result = translator.translate_and_optimize(speech_text)
            print("\nFinal Output:\n", result)

    else:
        print("Invalid choice. Please enter 1 or 2.")

    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000)

