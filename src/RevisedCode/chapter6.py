--------------------------------------------------------------------------------------------------------------


from langchain_core.tools import StructuredTool
def search_train_ticket(origin, destination, date):
    """查询火车票信息"""
    return [
        {"train": "G1234", "departure": "08:00", "arrival": "12:00", "price": "100.00"},
        {"train": "G5678", "departure": "18:30", "arrival": "22:30", "price": "100.00"},
    ]
# 将该函数封装为LangChain工具
search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,
    name="查询火车票",
    description="根据出发地、目的地和日期查询火车票信息"
)


--------------------------------------------------------------------------------------------------------------


from langchain_core.prompts import PromptTemplate
# 定义Prompt模板，指导智能体思考与行动
template = '''
问题：{input}
思考：{agent_scratchpad}
行动：执行 {tool}，输入：{tool_input}
观察：{observation}
思考：基于观察结果分析下一步
最终答案：{final_answer}
'''
prompt = PromptTemplate.from_template(template)


--------------------------------------------------------------------------------------------------------------


from langchain.memory import ConversationTokenBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
	import random
# 定义记忆模块
class ConversationTokenBufferMemory:
    """记忆模块，用于存储对话记录。"""
    def __init__(self, max_token_limit=1000):
        self.max_token_limit = max_token_limit
        self.memory = []
    def save_context(self, input_data, output_data):
        """保存对话上下文。"""
        self.memory.append((input_data, output_data))
        if len(self.memory) > self.max_token_limit:
            self.memory = self.memory[-self.max_token_limit:]
    def load_memory(self):
        """加载当前记忆。"""
        return self.memory
#  LLM 模型
class FakeChatModel:
    """大语言模型，返回响应。"""
    def __init__(self, model="gpt-4-turbo", temperature=0):
        self.model = model
        self.temperature = temperature
    def generate(self, prompt):
        """返回详细的火车票查询结果。"""
        responses = [
            """
            列车车次：G1234
            出发时间：2024-06-01 08:00
            到达时间：2024-06-01 12:00
            票价：¥100.00
            座位类型：商务座
            运营情况：正常运营
            """,
            """
            列车车次：G5678
            出发时间：2024-06-01 18:30
            到达时间：2024-06-01 22:30
            票价：¥95.00
            座位类型：一等座
            运营情况：预计准点发车
            """,
            """
            列车车次：G9012
            出发时间：2024-06-01 19:00
            到达时间：2024-06-01 23:00
            票价：¥85.00
            座位类型：二等座
            运营情况：运营良好，无故障报告
            """
        ]
        return random.choice(responses)
# 定义输出解析器
class Action(BaseModel):
    """结构化输出解析类。"""
    name: str
    args: dict
class PydanticOutputParser:
    """输出解析器，将响应转为结构化数据。"""
    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object
    def parse(self, response):
        """解析响应，转换为结构化对象。"""
        return self.pydantic_object(name="查询火车票", args={"result": response.strip()})
# 定义代理类
class MyAgent:
    """智能代理类，组合 LLM、记忆和工具。"""
    def __init__(self, llm, memory, tools, prompt_template):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.prompt_template = prompt_template
    def run(self, task):
        """执行任务的主流程。"""
        # 保存任务描述到记忆
        self.memory.save_context({"input": task}, {"output": "任务已收到"})
        # 生成响应
        response = self.llm.generate(self.prompt_template.format(task=task))
        # 解析响应
        parser = PydanticOutputParser(Action)
        result = parser.parse(response)
        # 保存响应到记忆
        self.memory.save_context({"input": task}, {"output": result.args["result"]})
        return result
# 定义任务模板
template = "查询任务：{task}。请为用户生成详细的火车票查询结果。"
# 初始化组件
memory = ConversationTokenBufferMemory(max_token_limit=1000)
llm = FakeChatModel(model="gpt-4-turbo", temperature=0)
# 示例工具
def search_train_ticket_tool(task):
    return {"result": "模拟的火车票查询结果"}
# 初始化代理实例
agent = MyAgent(llm=llm, memory=memory, tools=[search_train_ticket_tool], prompt_template=template)
# 执行任务
task = "查询2024年6月1日从北京到上海的火车票"
result = agent.run(task)
# 打印结果
print(f"任务执行结果：\n{result.args['result']}")


--------------------------------------------------------------------------------------------------------------


from langchain_core.prompts import PromptTemplate
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)


import json
import sys
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID
from langchain.memory import ConversationTokenBufferMemory
from langchain.tools.render import render_text_description
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError


def search_train_ticket(
        origin: str,
        destination: str,
        date: str,
        departure_time_start: str,
        departure_time_end: str
) -> List[dict[str, str]]:
    """按指定条件查询火车票"""

    # 模拟火车票数据
    return [
        {
            "train_number": "G1234",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 8:00",
            "arrival_time": "2024-06-01 12:00",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G5678",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 18:30",
            "arrival_time": "2024-06-01 22:30",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G9012",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 19:00",
            "arrival_time": "2024-06-01 23:00",
            "price": "100.00",
            "seat_type": "商务座",
        }
    ]
def purchase_train_ticket(train_number: str) -> dict:
    """购买火车票"""
    return {
        "result": "success",
        "message": "购买成功",
        "data": {
            "train_number": "G1234",
            "seat_type": "商务座",
            "seat_number": "7-17A"
        }
    }
search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,
    name="查询火车票",
    description="查询指定日期可用的火车票。",
)
purchase_train_ticket_tool = StructuredTool.from_function(
    func=purchase_train_ticket,
    name="购买火车票",
    description="购买火车票。会返回购买结果(result), 和座位号(seat_number)",
)
finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)


--------------------------------------------------------------------------------------------------------------


prompt_text = """
你是强大的AI火车票助手，可以使用工具与指令查询并购买火车票

你的任务是:
{task_description}

你可以使用以下工具或指令，它们又称为动作或actions:
{tools}

当前的任务执行记录:
{memory}

按照以下格式输出：

任务：你收到的需要执行的任务
思考: 观察你的任务和执行记录，并思考你下一步应该采取的行动
然后，根据以下格式说明，输出你选择执行的动作/工具:
{format_instructions}
"""


final_prompt = """
你的任务是:
{task_description}
以下是你的思考过程和使用工具与外部资源交互的结果。
{memory}
你已经完成任务。
现在请根据上述结果简要总结出你的最终答案。
直接给出答案。不用再解释或分析你的思考过程。
"""


import sys
from typing import Optional, Dict, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult
class Action(BaseModel):
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")
class MyPrintHandler(BaseCallbackHandler):
    """自定义LLM CallbackHandler，用于打印大模型返回的思考过程"""
    def __init__(self):
        BaseCallbackHandler.__init__(self)
    def on_llm_new_token(
            self,
            token: str,
            *,
          chunk:Optional[Union[GenerationChunk, ChatGenerationChunk]]=None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        end = ""
        content = token + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return token
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        end = ""
        content = "\n" + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return response


from typing import Optional, Tuple, Dict, Any
from pydantic import ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI
from langchain.tools.render import render_text_description
from uuid import UUID
import json
import sys
class MyAgent:
    def __init__(
            self,
            llm: BaseChatModel = ChatOpenAI(
                model="gpt-4-turbo",  # 使用GPT-4-turbo提升推理能力
                temperature=0,
                model_kwargs={"seed": 42},
            ),
            tools=None,
            prompt: str = "",
            final_prompt: str = "",
            max_thought_steps: Optional[int] = 10,
    ):
        if tools is None:
            tools = []
        self.llm = llm
        self.tools = tools
        self.final_prompt = PromptTemplate.from_template(final_prompt)
        self.max_thought_steps = max_thought_steps
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.prompt = self.__init_prompt(prompt)
        self.llm_chain = self.prompt | self.llm | StrOutputParser()
        self.verbose_printer = MyPrintHandler()
def __init_prompt(self, prompt):
    return PromptTemplate.from_template(prompt).partial(
        tools=render_text_description(self.tools),
        format_instructions=self.__chinese_friendly(
            self.output_parser.get_format_instructions(),
        )
    )
def run(self, task_description):
    thought_step_count = 0
    agent_memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=4000)
    agent_memory.save_context({"input": "\ninit"}, {"output": "\n开始"})
    while thought_step_count < self.max_thought_steps:
        print(f">>>>Round: {thought_step_count}<<<<")
        action, response = self.__step(task_description, agent_memory)
        if action.name == "FINISH":
            break
        observation = self.__exec_action(action)
        print(f"----\nObservation:\n{observation}")
        self.__update_memory(agent_memory, response, observation)
        thought_step_count += 1
    if thought_step_count >= self.max_thought_steps:
        reply = "任务未完成！"
    else:
        final_chain = self.final_prompt | self.llm | StrOutputParser()
        reply = final_chain.invoke({"task_description": task_description, "memory": agent_memory})
    return reply
def __step(self, task_description, memory) -> Tuple[Action, str]:
    response = ""
    for s in self.llm_chain.stream({"task_description": task_description, "memory": memory}, config={"callbacks": [self.verbose_printer]}):
        response += s
    action = self.output_parser.parse(response)
    return action, response
def __exec_action(self, action: Action) -> str:
    observation = "未找到工具"
    for tool in self.tools:
        if tool.name == action.name:
            try:
                observation = tool.run(action.args)
            except ValidationError as e:
                observation = f"Validation Error in args: {str(e)}, args: {action.args}"
            except Exception as e:
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
    return observation
@staticmethod
def __update_memory(agent_memory, response, observation):
    agent_memory.save_context(
        {"input": response},
        {"output": "\n返回结果:\n" + str(observation)}
    )
@staticmethod
def __chinese_friendly(string) -> str:
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)
if __name__ == "__main__":
    my_agent = MyAgent(
        tools=tools,
        prompt=prompt_text,
        final_prompt=final_prompt,
    )
    task = "帮我买2024年10月30日早上去上海的火车票"
    reply = my_agent.run(task)
    print(reply)


--------------------------------------------------------------------------------------------------------------


from langchain_core.prompts import PromptTemplate
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)
import json
import sys
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID
from langchain.memory import ConversationTokenBufferMemory
from langchain.tools.render import render_text_description
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List
from langchain_core.tools import StructuredTool
def search_train_ticket(
        origin: str,
        destination: str,
        date: str,
        departure_time_start: str,
        departure_time_end: str
) -> List[dict[str, str]]:
    """按指定条件查询火车票"""
    # mock train list
    return [
        {
            "train_number": "G1234",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 8:00",
            "arrival_time": "2024-06-01 12:00",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G5678",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 18:30",
            "arrival_time": "2024-06-01 22:30",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G9012",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 19:00",
            "arrival_time": "2024-06-01 23:00",
            "price": "100.00",
            "seat_type": "商务座",
        }
    ]
def purchase_train_ticket(
        train_number: str,
) -> dict:
    """购买火车票"""
    return {
        "result": "success",
        "message": "购买成功",
        "data": {
            "train_number": "G1234",
            "seat_type": "商务座",
            "seat_number": "7-17A"
        }
    }
search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,
    name="查询火车票",
    description="查询指定日期可用的火车票。",
)
purchase_train_ticket_tool = StructuredTool.from_function(
    func=purchase_train_ticket,
    name="购买火车票",
    description="购买火车票。会返回购买结果(result), 和座位号(seat_number)",
)
finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)
tools = [search_train_ticket_tool, purchase_train_ticket_tool, finish_placeholder]
prompt_text = """
你是强大的AI火车票助手，可以使用工具与指令查询并购买火车票
你的任务是:
{task_description}
你可以使用以下工具或指令，它们又称为动作或actions:
{tools}
当前的任务执行记录:
{memory}
按照以下格式输出：
任务：你收到的需要执行的任务
思考: 观察你的任务和执行记录，并思考你下一步应该采取的行动
然后，根据以下格式说明，输出你选择执行的动作/工具:
{format_instructions}
"""
final_prompt = """
你的任务是:
{task_description}
以下是你的思考过程和使用工具与外部资源交互的结果。
{memory}
你已经完成任务。
现在请根据上述结果简要总结出你的最终答案。
直接给出答案。不用再解释或分析你的思考过程。
"""
class Action(BaseModel):
    """结构化定义工具的属性"""
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")
class MyPrintHandler(BaseCallbackHandler):
    """自定义LLM CallbackHandler，用于打印大模型返回的思考过程"""
    def __init__(self):
        BaseCallbackHandler.__init__(self)
    def on_llm_new_token(
            self,
            token: str,
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        end = ""
        content = token + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return token
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        end = ""
        content = "\n" + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return response
class MyAgent:
    def __init__(
            self,
            llm: BaseChatModel = ChatOpenAI(
                model="gpt-4-turbo",  # agent用GPT4效果好一些，推理能力较强
                temperature=0,
                model_kwargs={
                    "seed": 42
                },
            ),
            tools=None,
            prompt: str = "",
            final_prompt: str = "",
            max_thought_steps: Optional[int] = 10,
    ):
        if tools is None:
            tools = []
        self.llm = llm
        self.tools = tools
        self.final_prompt = PromptTemplate.from_template(final_prompt)
        self.max_thought_steps = max_thought_steps  # 最多思考步数，避免死循环
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.prompt = self.__init_prompt(prompt)
        self.llm_chain = self.prompt | self.llm | StrOutputParser()  # 主流程的LCEL
        self.verbose_printer = MyPrintHandler()
        def __init_prompt(self, prompt):
            return PromptTemplate.from_template(prompt).partial(
                tools=render_text_description(self.tools),
                format_instructions=self.__chinese_friendly(
                    self.output_parser.get_format_instructions(),
                )
            )
    def run(self, task_description):
        """Agent主流程"""
        # 思考步数
        thought_step_count = 0
        # 初始化记忆
        agent_memory = ConversationTokenBufferMemory(
            llm=self.llm,
            max_token_limit=4000,
        )
        agent_memory.save_context(
            {"input": "\ninit"},
            {"output": "\n开始"}
        )
        # 开始逐步思考
        while thought_step_count < self.max_thought_steps:
            print(f">>>>Round: {thought_step_count}<<<<")
            action, response = self.__step(
                task_description=task_description,
                memory=agent_memory
            )
            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                break
            # 执行动作
            observation = self.__exec_action(action)
            print(f"----\nObservation:\n{observation}")
            # 更新记忆
            self.__update_memory(agent_memory, response, observation)
            thought_step_count += 1
        if thought_step_count >= self.max_thought_steps:
            # 如果思考步数达到上限，返回错误信息
            reply = "任务未完成！"
        else:
            # 否则，执行最后一步
            final_chain = self.final_prompt | self.llm | StrOutputParser()
            reply = final_chain.invoke({
                "task_description": task_description,
                "memory": agent_memory
            })
            return reply

    def __step(self, task_description, memory) -> Tuple[Action, str]:
        """执行一步思考"""
        response = ""
        for s in self.llm_chain.stream({
            "task_description": task_description,
            "memory": memory
        }, config={
            "callbacks": [
                self.verbose_printer
            ]
        }):
            response += s
        action = self.output_parser.parse(response)
        return action, response
    def __exec_action(self, action: Action) -> str:
        observation = "没有找到工具"
        for tool in self.tools:
            if tool.name == action.name:
                try:
                    # 执行工具
                    observation = tool.run(action.args)
                except ValidationError as e:
                    # 工具的入参异常
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    # 工具执行异常
                    observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
        return observation
    @staticmethod
    def __update_memory(agent_memory, response, observation):
        agent_memory.save_context(
            {"input": response},
            {"output": "\n返回结果:\n" + str(observation)}
        )
    @staticmethod
    def __chinese_friendly(string) -> str:
        lines = string.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('{') and line.endswith('}'):
                try:
                    lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
                except:
                    pass
        return '\n'.join(lines)
if __name__ == "__main__":
    my_agent = MyAgent(
        tools=tools,
        prompt=prompt_text,
        final_prompt=final_prompt,
    )
    task = "帮我买24年10月30日早上去上海的火车票"
    reply = my_agent.run(task)
    print(reply)
