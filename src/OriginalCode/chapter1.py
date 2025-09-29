----------------------------------------------------------------------------------------------------------


from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)
print(completion.choices[0].message)


----------------------------------------------------------------------------------------------------------


import openai
import os
from dotenv import load_dotenv
# 加载 .env 文件中的 API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# 使用 DALL•E API 生成图像
response = openai.Image.create(
    prompt="A cute dog",
    n=2,
    size="1024x1024"}
# 打印生成图像的 URL
for i, image in enumerate(response['data']):
    print(f"Image {i+1}: {image['url']}")


----------------------------------------------------------------------------------------------------------


import time
import openai
openai.api_key = "your_openai_api_key_here"
for attempt in range(5):
    try:
        response = openai.Image.create(
            prompt="A cute dog",
            n=1,
            size="1024x1024"
        )
        print(response['data'][0]['url'])
        break  # 成功时跳出循环
    except openai.error.APIConnectionError as e:
        print(f"连接失败，重试中...（{attempt + 1}/5）")
        time.sleep(2)  # 等待 2 秒后重试
    except Exception as e:
        print(f"发生错误: {e}")
        break
