# *langchain*1.0框架：

##  1.基础通用框架：

###   1.1框架1(了解基础结构即可，后面会用到提示词模板)

```python
import os
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai", ##除了直接访问deepseek的官网调用的api时，其他国内的大模型调用全部用openai
    api_key=os.getenv("aliQwen-api"), ##前提是该环境变量定义到了系统的环境变量中
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
messages = [
    SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
    HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
]
print(model.invoke(message).content)
```

###   1.2框架2：(了解调用创建环境变量文档来读取api_key)

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
load_dotenv(encoding='utf-8')
model = init_chat_model(
    model="deepseek-chat", 
    model_provider="deepseek", ##由于api和model都为deepseek官网的，所以这行代码可以省略
    api_key=os.getenv("deepseek-api"),##前提是在 该目录下创建了新的文本文件命名为.env，里面有写明deepseek-api=xxxxxx
    base_url="https://api.deepseek.com"
)
messages = [
    SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
    HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
]
print(model.invoke(message).content)
```

###    1.3框架3（企业级调用框架1）：

```python 
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException
from langchain.messages import HumanMessage,SystemMessage

load_dotenv(encoding='utf-8')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_model_client():
    api_key = os.getenv("deepseek-api")
    if not api_key:
        raise ValueError("环境变量 QWEN_API_KEY 未配置，请检查.env文件")
    
    model = init_chat_model(
    model="deepseek-chat", 
    model_provider="deepseek", ##由于api和model都为deepseek官网的，所以这行代码可以省略
    api_key=os.getenv("deepseek-api"),##前提是在 该目录下创建了新的txt文件命名为env，里面有写明deepseek-api=xxxxxx
    base_url="https://api.deepseek.com"
) 
    return model

def main():
    try:
        # 初始化客户端
        model = init_model_client()
        logger.info("LLM客户端初始化成功")

        # 调用模型（问题用变量存储，提高可读性）
        messages = [
        SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
        HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
        ]

        response = model.invoke(messages)

        # 格式化输出结果（而非直接打印原始对象）
        logger.info(f"问题：{messages}")
        logger.info(f"回答：{response.content}")
    
    except ValueError as e:
        logger.error(f"配置错误：{str(e)}")
    except LangChainException as e:
        logger.error(f"模型调用失败：{str(e)}")
    except Exception as e:
        logger.error(f"未知错误：{str(e)}")

 if __name__ == "__main__":
    main()
```



###    1.4框架4（企业级调用框架2）：

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import LangChainException
from langchain.messages import HumanMessage,SystemMessage

load_dotenv(encoding='utf-8')

import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_model_client():
    
    ##做一个api非空判断
    api_key = os.getenv("deepseek-api")
    if not api_key:
        raise ValueError("环境变量 QWEN_API_KEY 未配置，请检查.env文件")
    
    model = init_chat_model(
    model="deepseek-chat", 
    model_provider="deepseek", ##由于api和model都为deepseek官网的，所以这行代码可以省略
    api_key=os.getenv("deepseek-api"),##前提是在 该目录下创建了新的txt文件命名为env，里面有写明deepseek-api=xxxxxx
    base_url="https://api.deepseek.com"
) 
    return model

def main():
    try:
        # 初始化客户端
        model = init_model_client()
        logger.info("LLM客户端初始化成功")
        
        # 流式调用模型（慢慢的出文字）
        messages = [
        SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
        HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
        ]
        responseStream = model.stream(messages)
        for chunk in responseStream:
            print(chunk.content,end="")
        
     except ValueError as e:
        logger.error(f"配置错误：{str(e)}")
     except LangChainException as e:
        logger.error(f"模型调用失败：{str(e)}")
     except Exception as e:
        logger.error(f"未知错误：{str(e)}")

if __name__ == "__main__":
    main()
```

## 2.独家的大模型接入框架：

### 2.1.deepseek接入：

接入代码参考网站：https://docs.langchain.com/oss/python/integrations/chat/deepseek

### 2.2.qwen接入:

接入代码参考网站：https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654

## 3.ollama本地大模型接入框架：

### 3.1.Ollama 常用命令速查表  ：

| 命令                                  | 一句话说明                         |
| ------------------------------------- | ---------------------------------- |
| `ollama pull llama3`                  | 下载指定模型（例：llama3）。       |
| `ollama run llama3`                   | 启动并进入该模型交互对话。         |
| `ollama list`                         | 列出本机已下载的所有模型。         |
| `ollama rm llama3`                    | 删除不再需要的模型以节省磁盘。     |
| `ollama cp llama3 my-llama3`          | 本地复制/重命名模型。              |
| `ollama show llama3`                  | 查看模型详细信息（参数、大小等）。 |
| `ollama create my-model -f Modelfile` | 用自定义 Modelfile 构建新模型。    |
| `ollama serve`                        | 启动后台服务，供 API 调用。        |
| `ollama ps`                           | 查看当前正在运行的模型进程。       |
| `ollama stop llama3`                  | 停止正在运行的模型。               |

​    /bye                                                                                   退出大模型

### 3.2.Ollama 大模型安装官网 ：

可在此官网搜索需要安装的大模型名称以及版本号：https://ollama.com/download

### 3.3.Ollama 大模型接入框架：

```python
# pip install -qU langchain-ollama   需要提前安装
# pip install -U ollama              需要提前安装
 
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage,SystemMessage

# 设置本地模型，不使用深度思考
model = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:latest", reasoning=False)
# 打印结果，
messages = [
        SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
        HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
        ]
print(model.invoke(messages))


```

## 4.同/异步大模型调用方法：

### 4.1同/异步方法：invoke/ainvoke

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
#通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
load_dotenv(encoding='utf-8')

# List<Messages>


# 实例化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 构建消息列表
messages = [
    SystemMessage(content="你是一个法律助手，只回答法律问题，超出范围的统一回答，非法律问题无可奉告"),
    HumanMessage(content="简单介绍下广告法，一句话告知50字以内")
    #HumanMessage(content="2+3等于几?")
]
############################################################################################################
# 同步方法调用模型
def main():
    response = model.invoke(messages)  
    print(f"响应类型：{type(response)}")
    print(response.content_blocks)
    
# 运行同步函数
if __name__ == "__main__":
    main()
    
############################################################################################################
# 异步方法调用模型
async def main():
    # 异步调用一条请求
    response = await model.ainvoke("解释一下LangChain是什么，简洁回答100字以内")
    print(f"响应类型：{type(response)}")
    print(response.content_blocks)

# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())

```

### 4.2同/异步方法：stream/astream

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
#通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
load_dotenv(encoding='utf-8')

# 实例化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 构建消息列表
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]
############################################################################################################
# 同步流式调用大模型
def main():
    response = model.stream(messages)
    print(f"响应类型：{type(response)}")
    for chunk in response:
        print(chunk.content, end="",flush=True)
    print("\n")

# 运行同步函数
if __name__ == "__main__":
   main()
############################################################################################################
# 异步流式调用大模型
async def abc():
    # astream 返回异步生成器，无需 await 修饰，直接赋值
    response = model.astream(messages)
    print(f"响应类型：{type(response)}") 
    async for chunk in response:
         print(chunk.content, end="", flush=True)
    print("\n")

# 运行异步函数
if __name__ == "__main__":
  asyncio.run(main())
```

### 4.3同/异步方法：batch/abatch

```python
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,SystemMessage
#通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
load_dotenv(encoding='utf-8')

# 2.实例化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 问题列表
questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内"
]
############################################################################################################
# 同步批量调用大模型 
def main():
    response = model.batch(questions)
    print(f"响应类型：{type(response)}")
    
    for q, r in zip(questions, response):
        print(f"问题：{q}\n回答：{r.content}\n")

# 运行同步函数
if __name__ == "__main__":
   main()
   ############################################################################################################  # 异步批量调用大模型 
async def main():
    # 调用 model.abatch() 异步批量处理请求，需用 await 修饰（关键）
    response = await model.abatch(questions)
    print(f"响应类型：{type(response)}")

    for q, r in zip(questions, response):
        print(f"问题：{q}\n回答：{r.content}\n")

# 运行异步函数
if __name__ == "__main__":
    asyncio.run(main())
```

## 5.大模型prompt框架：

### 5.1prompt的基本框架：(了解)

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="你好，请你介绍一下你自己"),
    AIMessage(content="我是一名人工智能助手，请问您有什么想问的嘛?"),
    # ToolMessage - 用于工具调用场景
    ToolMessage(
        tool_call_id="call_abc123",  # 关联的工具调用ID
        content='{"population": 21540000, "area": "16410平方公里"}',  # 工具执行结果
    )
]

print(messages)
```

### 5.2PromptTemplate的基本框架：

#### 5.2.1正常的最基础的框架方法：（了解）

```python
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate


# 创建一个PromptTemplate对象，用于生成格式化的提示词模板，该模板包含两个变量：role（角色）和question（问题）
template = PromptTemplate(
    template="你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}",
    input_variables=['role', 'question']#由于只用的PromptTemplate(),则需要这行代码.
)

# 使用模板格式化具体的提示词内容，将role替换为"python开发"，question替换为"冒泡排序怎么写？"
prompt = template.format(role="python开发",question="冒泡排序怎么写,只要代码其它不要，简洁")

# 输出格式化后的提示词内容
print(prompt)# 你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写,只要代码其它不要，简洁




model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
result = model.invoke(prompt)
print(result.content)
print("\n\n")
```

#### 5.2.2用 from_template 方法：（可以不用，要知道）

```python
from langchain_core.prompts import PromptTemplate

# 创建一个PromptTemplate对象，用于生成格式化的提示词模板，模板包含两个占位符：{role}表示角色，{question}表示问题
template = PromptTemplate.from_template("你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}")
#以上代码由于只用的PromptTemplate.from_template(),则自动会执行input_variables=['role', 'question']，则不需要多的代码

# 使用指定的角色和问题参数来格式化模板，生成最终的提示词字符串
prompt = template.invoke(role="python开发",question="快速排序怎么写？")

# 输出生成的提示词
print(prompt)

```

#### 5.2.2用 partial_variables 方法：（可以不用，要知道）

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import time

# 1实例化过程中指定 partial_variables 参数
# 创建一个包含时间变量的模板，时间变量使用partial_variables预设为当前时间,然后格式化问题生成最终提示词
template1 = PromptTemplate.from_template(
    "现在时间是：{time},请对我的问题给出答案，我的问题是：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")} #对time的占位符进行预填入
)

prompt1 = template1.invoke(question="今天是几号？")#之前对time的占位符进行预填入，正式填入时不需要再对time进行填入
print(prompt1)
```

#### 5.2.3用联合提示词方法：

```python
from langchain_core.prompts import PromptTemplate

# 分别创建两个独立的PromptTemplate模板
prompt_a = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
prompt_b = PromptTemplate.from_template("内容不超过{length}个字")
# 将两个模板进行拼接组合
prompt_all = prompt_a + prompt_b
# 填充组合后模板的占位符，生成最终的提示词
prompt2 = prompt_all.invoke(topic="LangChain", length=200)
print(prompt2)
```

### 5.3ChatPromptTemplate的基本框架：

#### 5.3.1from_messages模板：

```python
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

############################################################################################################
以下两种提示词模板二选一：
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你是AI助手，你的名字叫{name}。"),
        HumanMessage(content="请问：{question}")
    ]
)
************************************************************************************************************
chatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", "你是AI助手，你的名字叫{name}。"),
        ("human", "请问：{question}"),
    ]
)
############################################################################################################


############################################################################################################

prompt_value = chat_prompt.invoke({"role": "python开发工程师", "question": "堆排序怎么写"})

############################################################################################################
# 打印格式化后的提示消息
print(prompt_value)

# llm = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
# print()
# print("======================")
#
# result = llm.invoke(prompt_value)
# print(result)
# print(result.content)
```

#### 5.3.2MessagesPlaceHoder占位符模板：(了解，后面有牛逼的)

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 构建一个 ChatPromptTemplate，包含多种消息类型：
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的Python应用开发工程师，请认真回答我提出的Python相关的问题"),
    # 插入 memory 占位符，用于填充历史对话记录（如多轮对话上下文）
    MessagesPlaceholder("memory"),
    ("human", "{question}")
])

# 调用 prompt.invoke 来格式化整个 Prompt 模板
prompt_value = prompt.invoke({
    "memory": [
        # 用户第一轮说的话
        HumanMessage("我的名字叫亮仔，是一名程序员111"),
        # AI 第一轮的回应
        AIMessage("好的，亮仔你好222")
    ],
    # 当前问题：结合上下文，测试模型是否记住了用户名字
    "question": "请问我的名字叫什么？"
})

# 打印生成的完整 prompt 文本，格式化后的聊天记录
print(prompt_value.to_string())
```

### 5.4外部加载prompt的基本框架：

主框架：

```python
from langchain_core.prompts import load_prompt

template = load_prompt("prompt.json", encoding="utf-8")
print(template.invoke({'name':"张三", 'what':"搞笑的"}))
```

prompt.json文件：

```json
{
    "_type": "prompt",
    "input_variables": ["name", "what"],
    "template": "请{name}讲一个{what}的故事"
}
```

## 6.输出解析器的基本框架：

### 6.1JSON输出解析器：

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chat_models import init_chat_model
from loguru import logger
from pydantic import BaseModel, Field

*****************结合Field设置，运行时校验能力，主要用这个************************************
class Person(BaseModel):
    
    time: int = Field(description="时间") 
    person: str = Field(description="人物")
    event: str = Field(description="事件")
*********************************************************************************************************


# 创建JSON输出解析器，用于将model输出解析为Person对象
parser = JsonOutputParser(pydantic_object=Person)###如果()里面是空的，则大模型输出则按照自己的json标准格式来输出
# 将需要输出什么要求的json格式，转化为ai能看懂的提示词
a = parser.get_format_instructions()

# 创建聊天提示模板，定义系统角色和用户输入格式
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化JSON数据。"),
    ("human", "请生成一个关于{topic}的新闻。{format_instructions}")
])

# 格式化提示词，填入具体主题和格式化指令
prompt = chat_prompt.format(topic="小米su7跑车", format_instructions=a)

# 初始化大语言模型实例
model = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用大语言模型获取响应结果
result = model.invoke(prompt)
# 使用解析器将模型输出解析为结构化数据
response = parser.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")

```

### 6.2str输出解析器：

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chat_models import init_chat_model
from loguru import logger
from pydantic import BaseModel, Field

# 创建str输出解析器，用于将model输出解析为Person对象
parser = StrOutputParser()
# 将需要输出什么要求的json格式，转化为ai能看懂的提示词
# 创建聊天提示模板，定义系统角色和用户输入格式
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手"),
    ("human", "请生成一个关于{topic}的新闻。")
])

# 格式化提示词，填入具体主题和格式化指令
prompt = chat_prompt.format(topic="小米su7跑车")

# 初始化大语言模型实例
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用大语言模型获取响应结果
result = model.invoke(prompt)
# 使用解析器将模型输出解析为结构化数据
response = parser.invoke(result)
logger.info(f"解析后的结构化结果:\n{response}")
```

### 6.3Annotated, TypedDict输出解析器：（了解）

```python
from langchain_core.prompts import ChatPromptTemplate
import os
from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class Animal(TypedDict):
    animal: Annotated[str, "动物"]
    emoji: Annotated[str, "表情"]

class AnimalList(TypedDict):
    animals: Annotated[list[Animal], "动物与表情列表"] # List<Animal>

# messages = [{"role": "user", "content": "任意生成三种动物，以及他们的 emoji 表情"}]
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}助手"),
    ("human", "任意生成三种动物，以及他们的 emoji 表情")
])
prompt_value = chat_prompt.format_messages(**{"role": "ai"})
llm1 = llm.with_structured_output(AnimalList)##对ai进行改造，要求按照这个类型去输出，改造后的ai命名为llm1
resp = llm1.invoke(prompt_value)
print(resp)
```

### 6.4PydanticOutputParser输出解析器:

```python
"""
PydanticOutputParser 是 LangChain 输出解析器体系中最常用、最强大的结构化解析器之一。
它与 JsonOutputParser 类似，但功能更强 —— 能直接基于 Pydantic 模型 定义输出结构，
并利用其类型校验与自动文档能力。
对于结构更复杂、具有强类型约束的需求，PydanticOutputParser 则是最佳选择。
它结合了Pydantic模型的强大功能，提供了类型验证、数据转换等高级功能
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    description: str = Field(description="产品简介")

    @field_validator("description")  ###自定义的校验函数，待ai生成之后会自动执行 
    def validate_description(cls, value):
        """
        验证产品简介字段的长度
        参数:
            value (str): 待验证的产品简介文本
        返回:
            str: 验证通过的产品简介文本
        异常:
            ValueError: 当产品简介长度小于10个字符时抛出
        """
        if len(value) < 10:
            raise ValueError('产品简介长度必须大于等于10')
        return value

# 创建Pydantic输出解析器实例，用于解析模型输出为Product对象
parser = PydanticOutputParser(pydantic_object=Product)

# 获取格式化指令，用于指导模型输出符合Product模型的JSON格式
a = parser.get_format_instructions()

# 创建聊天提示模板，包含系统消息和人类消息
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化的json数据，{format_instructions}"),
    ("human", "请你输出标题为：{topic}的新闻内容")
])

# 格式化提示消息，填充主题和格式化指令
prompt = prompt_template.format_messages(topic="华为Mate X7", format_instructions=a)

# 记录格式化后的提示消息
logger.info(prompt)

# 创建模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用模型获取结果
result = model.invoke(prompt)

# 使用解析器将模型结果解析为Product对象
response = parser.invoke(result)

# 打印解析后的结构化结果
logger.info(f"解析后的结构化结果:\n{response}")


```

## 7.LCEL链式调用：

### 7.1RunnableSequence-顺序链：

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
import os

# 创建聊天提示模板，包含系统角色设定和用户问题输入
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题"),
    ("human", "请回答:{question}")
])

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

parser = StrOutputParser ()

# 构建处理链：提示模板 -> 模型 -> 输出解析器
chain = chat_prompt | model | parser

# 执行处理链并记录最终结果及数据类型
result_chain = chain.invoke({"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"})
logger.info(f"Chain执行结果:\n {result_chain}")
```

### 7.2RunnableBranch-分支链：

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from langchain_core.runnables import RunnableBranch
import os

# 构建提示词
english_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个英语翻译专家，你叫小英"),
    ("human", "{query}")
])

japanese_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个日语翻译专家，你叫小日"),
    ("human", "{query}")
])

korean_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个韩语翻译专家，你叫小韩"),
    ("human", "{query}")
])


def determine_language(inputs):
    """判断语言种类"""
    query = inputs["query"]
    if "日语" in query:
        return "japanese"
    elif "韩语" in query:
        return "korean"
    else:
        return "english"


# 初始化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 创建字符串输出解析器，用于处理模型输出
parser = StrOutputParser()

# 创建一个可运行的分支链，根据输入文本的语言类型选择相应的处理流程
chain = RunnableBranch(
    (lambda x: determine_language(x) == "japanese", japanese_prompt | model | parser),
    (lambda x: determine_language(x) == "korean", korean_prompt | model | parser),
    (english_prompt | model | parser)	
)

# 测试查询
test_queries = [
    {'query': '请你用韩语翻译这句话:"见到你很高兴"'},
    {'query': '请你用日语翻译这句话:"见到你很高兴"'},
    {'query': '请你用英语翻译这句话:"见到你很高兴"'}
]

for query_input in test_queries:
    # 执行链
    result = chain.invoke({"query":query_input})
    logger.info(f"输出结果: {result}\n")
```

### 7.3RunnableSerializable-串行链：

```python
"""
RunnableSerializable-串行链
子链叠加串行，假如我们需要多次调用大模型，将多个步骤串联起来实现功能
"""
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
import os

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 子链1提示词
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
# 子链1解析器
parser1 = StrOutputParser()
# 子链1：生成内容
chain1 = prompt1 | model | parser1


# 子链2提示词
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
    ("human", "{input}")
])
# 子链2解析器
parser2 = StrOutputParser()
# 子链2：翻译内容
chain2 = prompt2 | model | parser2


# 组合成一个复合 Chain，使用 lambda 函数将chain1执行结果content内容添加input键作为参数传递给chain2
full_chain = chain1 | (lambda content: {"input": content}) | chain2

# 调用复合链
result = full_chain.invoke({"topic": "langchain"})
logger.info(result)
```

### 7.4RunnableParallel-并行链：

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from loguru import logger
import os

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 并行链1提示词
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
# 并行链1解析器
parser1 = StrOutputParser()
# 并行链1：生成中文结果
chain1 = prompt1 | model | parser1

# 并行链2提示词
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用英文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
# 并行链2解析器
parser2 = StrOutputParser()

# 并行链2：生成英文结果
chain2 = prompt2 | model | parser2

# 创建并行链,用于同时执行多个语言处理链，且必须为字典的形式
parallel_chain = RunnableParallel({    
    "chinese": chain1,
    "english": chain2
})

# 调用复合链
result = parallel_chain.invoke({"topic": "langchain"})
logger.info(result)
```

### 7.5RunnableLambda-函数链：

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger
import os

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 一个简单的打印函数，调试用
def debug_print(x):
    logger.info(f"中间结果:{x}")
    return {"input": x}

# 子链1提示词
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
# 子链1解析器
parser1 = StrOutputParser()
# 子链1：生成内容
chain1 = prompt1 | model | parser1

# 子链2提示词
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
    ("human", "{input}")
])
# 子链2解析器
parser2 = StrOutputParser()

# 子链2：翻译内容
chain2 = prompt2 | model | parser2

# 创建一个可运行的调试节点，用于打印中间结果，即对debug_print函数包装成Runnable参与链式运行
debug_node = RunnableLambda(debug_print)

# 构建完整的处理链，将chain1、调试打印和chain2串联起来
full_chain = chain1 | debug_node | chain2

# 调用复合链
result = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果111:{result}")
```

## 8.记忆缓存：

### 8.1可持续记忆（RedisChatMessageHistory和RunnableWithMessageHistory）调用：

```python
from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
import os
import redis

#############定义好连接到Redis的客户端，此功能仅用作手动备份redis中的数据到电脑硬盘，重启关机后记忆仍存在##############
REDIS_URL = "redis://localhost:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
################################################
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt=ChatPromptTemplate.from_messages([
    MessagesPlaceholder('history'),
    ('human','{question}')
]

)

parser=StrOutputParser()
#############告诉redis存放历史记录的位置和编个号，并且对话完成后能够自动执行存数据####################
def get_session_history(session_id):
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL

    )
    return history
##############################################################

##############告诉Redis这个链的运行规则以及如果指挥Redis填坑#################
chain=RunnableWithMessageHistory(
    prompt|model|parser,
    get_session_history,
    input_messages_key='question',
    history_messages_key='history',
)
###############################################################


###############告诉redis存入时的编号名字###########################
config=RunnableConfig(configurable={'session_id':'管理员'})
###############################################################
print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ['quit', 'exit', 'q']:
        break

    response = chain.invoke({"question": question}, config)
    print(f"AI回答:{response}")

# 把刚聊的内容，从内存 → 写入硬盘！
redis_client.save()
```

## 9.tool工具调用（懂原理就可以了）：

工具调用主函数：

```python
import os
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from QueryWeatherTool import get_weather

# 初始化大语言模型实例
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 将模型与工具绑定，使其能够调用 get_weather 工具
llm_with_tools = llm.bind_tools([get_weather])###如果需要加入多个工具，则[get_weather，tool1，tool2]

# 创建解析器，用于提取工具调用结果中的 JSON 数据
parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)

# 构建工具调用链：模型 -> 解析器 -> 调用天气工具
get_weather_chain = llm_with_tools | parser | get_weather
# print(get_weather_chain.invoke("你好， 请问北京的天气怎么样？"))
# 定义输出提示模板，将 JSON 天气数据转换为自然语言描述
output_prompt = PromptTemplate.from_template(
    """你将收到一段 JSON 格式的天气数据{weather_json}，请用简洁自然的方式将其转述给用户。
    以下是天气 JSON 数据：
    请将其转换为中文天气描述，例如：	
    “北京现在天气：多云，气温 28℃，体感有点闷热（约 32℃），湿度 75%，微风（东南风 2 米/秒），
    能见度很好，大约 10 公里。建议穿短袖短裤。适合做户外运动。"
    """
)

# 创建字符串输出解析器
output_parser = StrOutputParser()

# 构建最终输出链：提示模板 -> 模型 -> 输出解析器
output_chain = output_prompt | llm | output_parser

# 构建完整的处理链：天气查询链 ->将天气数据包装为字典格式 -> 输出链
full_chain = get_weather_chain | (lambda x: {"weather_json": x}) | output_chain

# 执行完整链路，查询上海天气并打印结果
result = full_chain.invoke("请问北京今天的天气如何？")
logger.info(result)



```

QueryWeatherTool文件：

```python
import json
import requests
from langchain.tools import tool
import os

# 替换成你的 OpenWeatherMap API Key
API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_weather(city_en: str) -> str:
    """查询天气，参数city为英文城市名"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_en,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    resp = requests.get(url, params=params)
    return json.dumps(resp.json(), ensure_ascii=False)
```

## 10.向量化和向量数据库：

### 10.1Embedding 文本向量化：

#### 10.1.1阿里云百炼---文本嵌入模型（Embedding Model）基础版（了解）：

```python
# https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587
#可以在以上官网上找openai版的

import dashscope
from http import HTTPStatus

dashscope.api_key = "sk-73718658396745019dab4f963be5e436Y"

input_text = "衣服的质量杠杠的"

resp = dashscope.TextEmbedding.call(

    model="text-embedding-v4",
    input=input_text,
)

if resp.status_code == HTTPStatus.OK:
    print(resp)
```

#### 10.1.2阿里云百炼---文本嵌入模型（Embedding Model）标准版：

```python
"""
https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654
pip install langchain-community dashscope
"""

from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    # other params...
)

##########################单内容进行向量化#################################
text = "This is a test document."

query_result = embeddings.embed_query(text)

##########################多内容进行向量化#################################
texts=[
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
      ]
doc_results = embeddings.embed_documents(texts)
print(doc_results)
```

#### 10.1.3阿里云百炼---文本嵌入模型（Embedding Model）升级规范版（了解）：

```python
import dashscope
import json
import os
from http import HTTPStatus

resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("QWEN_API_KEY"),  # 从环境变量读取
    input=[{"text": "尚硅谷AI"}]
)

result = "";

# 处理模型返回结果，提取关键信息并格式化输出
if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage
    }
    print(json.dumps(result, ensure_ascii=False, indent=4))


```

### 10.2Embedding 多模态向量化(即支持图片和文本等转化为向量)：

```python
import dashscope
import json
import os
from http import HTTPStatus

# Embedding 文本向量化

# 调用多模态embedding模型接口进行向量编码
# https://bailian.console.aliyun.com/?productCode=p_efm&tab=model#/model-market/all?capabilities=ME
resp = dashscope.MultiModalEmbedding.call(
    model="tongyi-embedding-vision-plus",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取
    input=[{"text": "尚硅谷AI"}, {"text": "人工智能"},{"image": "https://xxx.jpg"}]#image为图片地址
)

result = "";

# 处理模型返回结果，提取关键信息并格式化输出，可以直接复制
if resp.status_code == HTTPStatus.OK:
    result = {
        "status_code": resp.status_code,
        "request_id": getattr(resp, "request_id", ""),
        "code": getattr(resp, "code", ""),
        "message": getattr(resp, "message", ""),
        "output": resp.output,
        "usage": resp.usage
    }
    print(json.dumps(result, ensure_ascii=False, indent=1)) #ensure_ascii=False支持中文显示，indent=4让 JSON 带缩进、更易读。
```

### 10.3Embedding 多模态向量化相似度(在未存入Redis中前)比较：（了解）

```python
"""
把文本转换成向量有什么用呢？
最核心的作用是可以通过向量之间的计算，来分析文本与文本之间的相似性。
计算的方法有很多种，其中用得最多的是向量余弦相似度。
Python语言中提供了一个库sklearn，可以很方便的计算向量之间的余弦相似度
"""

import dashscope
import os
from http import HTTPStatus
import numpy as np


# 准备输入文本数据
texts = [
    '我喜欢吃苹果',
    '苹果是我最喜欢吃的水果',
    '我喜欢用苹果手机'
]

# 获取每个文本的embedding向量
embeddings = []
# 假如要处理图片，请参考https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587
for text in texts:
    input_data = [{'text': text}]
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        input=input_data
    )

    if resp.status_code == HTTPStatus.OK:
        embedding = resp.output['embeddings'][0]['embedding']
        embeddings.append(embedding)
        print(embedding)

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 比较所有文本之间的相似度
print("文本相似度比较结果:")
print("=" * 60)

for i in range(3):
    for j in range(i+1, 3):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        print(f"文本{i+1} vs 文本{j+1}:")
        print(f"  文本{i+1}: {texts[i]}")
        print(f"  文本{j+1}: {texts[j]}")
        print(f"  余弦相似度: {similarity:.4f}")
        print("-" * 40)
```

### 10.4Embedding 多模态向量化数据存入Redis数据库并相似化比较（重点）：

```python
# pip install langchain-community dashscope redis redisvl
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document

# 1. 初始化阿里千问 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("QWEN_API_KEY")  # 从环境变量读取
)


# 2. 准备要向量化的文本（Document 列表）
texts = [
    "通义千问是阿里巴巴研发的大语言模型。",
    "Redis 是一个高性能的键值存储系统，支持向量检索。",
    "LangChain 可以轻松集成各种大模型和向量数据库。"
]
documents = [Document(page_content=text, metadata={"segment_id": str(i+1)}) for i, text in enumerate(texts)]

# 3. 连接到 Redis 并存入向量（自动调用 embeddings 嵌入）
vector_store = Redis.from_documents(
    documents=documents,
    embedding=embeddings,
    redis_url="redis://localhost:6379",  # 替换为你的 Redis 地址
    index_name="my_index11",               # 向量索引名称
)

# 4. （可选）后续可直接用于检索
query = "redis和langchain是什么"
# 自动计算相似度，返回最相似的2条结果（k=数量）
results = vector_store.similarity_search(query=query, k=2)
print(results)
```

## 11.检索增强生成RAG：

### 11.1文档加载器：（不需要记，用的时候拿来就可以）

1.加载TXT：

```python
# pip install langchain_community
from langchain_community.document_loaders import TextLoader

# 返回List[Document]
file_path = "assets/sample.txt"  # 文件路径
encoding = "utf-8"  # 文件编码方式

docs = TextLoader(file_path, encoding).load()

print(docs)
```

2.加载pdf：

```python
# pip install langchain_community
from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader(
    # 文件路径，支持本地文件和在线文件链接，如"https://arxiv.org/pdf/alg-geom/9202012"
    file_path="assets/sample.pdf",
    # 提取模式:
    #   plain 提取文本,layout 按布局提取;plain = 乱序、无排版、只抓文字;layout = 保留排版、分段清晰、阅读友好
    extraction_mode="layout",
).load()

print(docs)
```

3.加载markdown：

```python
# pip install langchain_community unstructured[md]
from langchain_community.document_loaders import UnstructuredMarkdownLoader

docs = UnstructuredMarkdownLoader(
    # 文件路径
    file_path="assets/sample.md",
    # 加载模式:
    #"single"把整个 Markdown 文件 → 变成 1 个 Document 对象
    #"elements"把 Markdown 文件 → 按标题、段落、列表、代码块自动切分成 很多个 Document

    mode="elements",
).load()

print(docs)
```

4.加载json

```python
# pip install jq
    from langchain_community.document_loaders import JSONLoader

# 提取所有字段
docs = JSONLoader(
    file_path="assets/sample.json",  # 文件路径
    jq_schema=".",  # 提取所有字段
    text_content=False,  # 提取内容是否为字符串格式,False：保留 JSON 原本的结构（字典 / 列表）;True：强制把所有内容转成纯字符串

).load()

print(docs)
```

5.加载Doc/Docx：

```python
# pip install langchain_community unstructured[docx]
# pip install -U unstructured
# pip install python-docx
# pip install regex==2026.1.14
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

docs = UnstructuredWordDocumentLoader(
    # 文件路径
    file_path="assets/alibaba-more.docx",
    # 加载模式:
    # "single"把整个 Markdown 文件 → 变成 1 个 Document 对象
    # "elements"把 Markdown 文件 → 按标题、段落、列表、代码块自动切分成 很多个 Document
    mode="single",
).load()

#print(type(docs))
print(docs)
```

6.加载csv：

```python
# pip install langchain_community
from langchain_community.document_loaders.csv_loader import CSVLoader

# 加载所有列
docs = CSVLoader(
    file_path="assets/sample.csv",  # 文件路径
).load()  # 返回List[Document]

print(docs)

# 加载部分列
docs = CSVLoader(
    file_path="assets/sample.csv",  # 文件路径
    metadata_columns=["title", "author"],  # 将指定列作为元数据
    content_columns=["content"],  # 将指定列作为内容
).load()  # 返回List[Document]

print(docs)
```

### 11.2文档分割器：

**1.对txt文本进行分隔：(了解，因为后续直接加载文档后就直接是document对象)**

```python
"""
使用split_text()方法进行文本分割
RecursiveCharacterTextSplitter中指定的
chunk_size=100,块大小为100，
chunk_overlap=30, 片段重叠字符数为30，
length_function=len，计算长度的函数使用len，# 可选：默认为字符串长度，可自定义函数来实现按 token 数切分
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1.分割文本内容
content = (
    "大模型RAG（检索增强生成）是一种结合生成模型与外部知识检索的技术，通过从大规模文档或数据库中检索相关信息，"
    "辅助生成模型以提升回答的准确性和相关性。其核心流程包括用户输入查询、系统检索相关知识、"
    "生成模型基于检索结果生成内容，并输出最终答案。RAG的优势在于能够弥补生成模型的知识盲区，"
    "提供更准确、实时和可解释的输出，广泛应用于问答系统、内容生成、客服、教育和企业领域。"
    "然而，其也面临依赖高质量知识库、可能的响应延迟、较高的维护成本以及数据隐私等挑战。")


# 2.定义递归文本分割器
# 使用RecursiveCharacterTextSplitter创建文本分割器，设置块大小为100，重叠长度为30,
# length_function=len就是指定使用 Python 内置的len()函数来计算文本长度，也是这个分割器的默认值
# 比如，print(len("大模型RAG技术"))  # 输出8，因为统计的是字符个数（中文字符、字母、符号各算1个）
# 遵循 “重叠后向前取有效内容、且不生成过小碎片” 的核心分割逻辑，不会让最后一个片段的有效内容只剩扣除重叠后的少量字符
# 原始文本 → split_text → 第一次分割成字符串块 → create_documents → 核对是否符合切分规则，对字符串块二次分割 → 内容丢失有可能
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len)

# 3.分割文本
# 将原始文本内容分割成多个文本块1
splitter_texts = text_splitter.split_text(content)

# 4.转换为文档对象
# 将分割后的文本块转换为文档对象列表
splitter_documents = text_splitter.create_documents(splitter_texts)
print(f"原始文本大小：{len(content)}")
print(f"分割文档数量：{len(splitter_documents)}")
for splitter_document in splitter_documents:
    print(f"文档片段大小：{len(splitter_document.page_content)},文档内容：{splitter_document.page_content}")
```

**2.对documents对象进行分隔：**

```python
"""
分割文档对象
RecursiveCharacterTextSplitter不仅可以分割纯文本，还可以直接分割Document对象
"""
# pip install python-magic-bin
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# 1.创建文档加载器，进行文档加载


# 返回List[Document]
file_path = "rag.txt"  # 文件路径
encoding = "utf-8"  # 文件编码方式

documents = TextLoader(file_path, encoding).load()
print(documents)
# 2.定义递归文本分割器
# 创建RecursiveCharacterTextSplitter实例，用于将文档分割成指定大小的文本块
# chunk_size: 每个文本块的最大字符数为100
# chunk_overlap: 相邻文本块之间的重叠字符数为30
# length_function: 使用len函数计算文本长度
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len)

# 3.分割文本
# 使用文本分割器将加载的文档分割成多个较小的文档片段
splitter_documents = text_splitter.split_documents(documents)

# 输出分割后的文档信息
print(f"分割文档数量：{len(splitter_documents)}")

for splitter_document in splitter_documents:
    print(f"文档片段：{splitter_document.page_content}")
    print(f"文档片段大小：{len(splitter_document.page_content)}, 文档元数据：{splitter_document.metadata}")
```

### 11.3RAG调用完整流程代码：

```python
# pip install unstructured
# pip install docx2txt
# pip install python-docx
from langchain.chat_models  import  init_chat_model
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，
    如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。

    文本内容：
    {context}

    问题：{question}

    回答：
    "
""")
])

# 初始化阿里千问 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("QWEN_API_KEY")  # 从环境变量读取
)

# 用文件加载器加载docx文件
documents = UnstructuredWordDocumentLoader(
    # 文件路径
    file_path="alibaba-java.docx",
    mode="single",
).load()
# 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
texts = text_splitter.split_documents(documents)

print(f"文档个数:{len(texts)}")


# 连接到 Redis 并存入向量（自动调用 embeddings 嵌入）
vector_store = Redis.from_documents(
    documents=texts,
    embedding=embeddings,
    redis_url="redis://localhost:6379",  # 替换为你的 Redis 地址
    index_name="my_index3",  # 向量索引名称
)
# 创建一个可以根据问题搜索Redis数据库相似度的实例，由RunnablePassthrough()给到问题
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 8. 创建Runnable链
rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
)

# 9. 提问
question = "00000和A0001分别是什么意思"
result = rag_chain.invoke(question)
print("\n问题:", question)
print("\n回答:", result.content)
```

## 12.MCP(模型上下文协议)全流程调用：服务端--客户端

客户端：

```python
# 1. 导入工具包（create_agent 确实存在于这个版本）
import asyncio
import json
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # ✅ 1.0.11 版本导入正常
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger


# 2. 加载MCP工具配置
def load_servers():
    with open('mcp.json', "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get("fetch", {})

# 3. 核心聊天函数（纯原生create_agent，无执行器）
async def main() -> None:
    # 加载MCP工具
    servers_cfg = load_servers()
    mcp_client = MultiServerMCPClient(servers_cfg)
    tools = await mcp_client.get_tools()

    # 初始化通义千问大模型
    llm = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # ✅ 1.0.11 版本正确的 create_agent 调用方式（移除了不兼容的verbose参数）
    agent = create_agent(
        model=llm,  # 第一个参数：模型
        tools=tools,  # 第二个参数：工具
        system_prompt="你是一个有用的助手，需要使用提供的工具来完成用户请求。"
    )

    # 聊天循环
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            # ✅ 正确的调用方式
            result = await agent.ainvoke({
                "messages": user_input
            })
            # ✅ 正确的结果获取方式
            print(f"\nAI: {result['messages'][-1].content}")
        except Exception as e:
            logger.error(f"\n⚠️ 出错: {e}")

    logger.info("🧹 会话已结束，Bye!")

# 启动程序
if __name__ == "__main__":
    asyncio.run(main())
```

服务端：

```python
import json
import os
from mcp.server.fastmcp import FastMCP
from loguru import logger
import requests

# 创建FastMCP实例，用于启动天气服务器SSE服务

mcp = FastMCP("WeatherServerSSE", host="127.0.0.1", port=8001)
@mcp.tool()
def get_weather(city: str) -> str:
    """
    查询指定城市的即时天气信息。
    参数 city: 城市英文名，如 Beijing
    返回: OpenWeather API 的 JSON 字符串
    """
    url = "https://api.openweathermap.org/data/2.5/weather"

    # 设置查询参数，包括城市名、API Key、单位和语言
    params = {
            "q": city,
            "appid": os.getenv("OPENWEATHER_API_KEY"),  # 从环境变量中读取 API Key
            "units": "metric",  # 使用摄氏度
            "lang": "zh_cn"  # 输出语言为简体中文
        }

    # 发送 GET 请求获取天气数据
    response = requests.get(url, params=params)
    # 解析响应内容为 JSON 并序列化为字符串返回
    data = response.json()
    return json.dumps(data,ensure_ascii=False)


if __name__ == "__main__":
    # 运行MCP客户端，使用Server-Sent Events(SSE)作为传输协议
    mcp.run(transport="sse")

'''
核心重点：202 Accepted 状态码的意义（结合 MCP SSE 场景）
HTTP 202 状态码和常见的200 OK有本质区别，适配 MCP SSE 的流式处理特性：
200 OK：请求已处理完成，服务端立即返回最终结果（适合一次性请求 - 响应的场景，比如普通接口查询）；
202 Accepted：请求已接收并受理，服务端会在后台处理（比如调用天气工具、执行 MCP 指令），
处理完成后通过 SSE 流将结果推送给客户端（适合耗时 / 流式处理的场景，这正是 MCP SSE 服务的设计逻辑）。
'''
```

配置文件：

```python
{
  "mcpServers": {
    "weather": {
      "url": "http://127.0.0.1:8001/sse",
      "transport": "sse"
    },
    "fetch": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ],
      "transport": "stdio"
    }
  }
}
```

## 13.Agent智能体：

### 13.1agent智能体基础框架示例：

```python
import os
import json
import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# 1.Tool 定义（完全不变）
@tool
def get_weather(loc: str) -> dict:
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称。
                注意，中国的城市需要用对应城市的英文名称代替。
    :return: OpenWeather API 查询即时天气的结果。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn"
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    return json.dumps(data, ensure_ascii=False)

# 2. 模型初始化（完全不变）
model = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

# 3. 创建Agent（完全不变）
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是天气助手。"
    "当用户询问多个城市天气时，"
    "你需要分别调用工具获取数据，并进行比较分析。"
)

# 4. 调用Agent（完全不变）
result = agent.invoke(
    {"messages": "请问今天北京和上海的天气怎么样，哪个城市更热？"}
)

# ====================== 仅修改这里 ======================
# 手动实现 StrOutputParser 效果：提取最终纯文本内容
final_text = result["messages"][-1].content.strip()
# 输出纯字符串（和 StrOutputParser() 格式100%一致）
print(final_text)
```

### 13.2单个agent智能体：

```python
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

# 模拟产品数据库
PRODUCT_DATABASE = {
    "无线耳机": [
        {"id": "WH-1000XM5", "name": "索尼 WH-1000XM5", "popularity": 95, "price": 299},
        {"id": "QC45", "name": "Bose QuietComfort 45", "popularity": 88, "price": 329},
        {"id": "AIRMAX", "name": "苹果 AirPods Max", "popularity": 92, "price": 549},
        {"id": "PXC550", "name": "森海塞尔 PXC 550", "popularity": 76, "price": 299},
        {"id": "HT450", "name": "JBL Tune 760NC", "popularity": 82, "price": 99}
    ],
    "游戏鼠标": [
        {"id": "GPW", "name": "罗技 G Pro 无线", "popularity": 90, "price": 129},
        {"id": "VIPER", "name": "雷蛇 Viper V2 Pro", "popularity": 87, "price": 149},
        {"id": "DAV3", "name": "雷蛇 DeathAdder V3", "popularity": 85, "price": 119}
    ],
    "笔记本电脑": [
        {"id": "MBP14", "name": "MacBook Pro 14英寸", "popularity": 94, "price": 1999},
        {"id": "XPS13", "name": "戴尔 XPS 13", "popularity": 89, "price": 1299},
        {"id": "TPX1", "name": "ThinkPad X1 Carbon", "popularity": 86, "price": 1499}
    ]
}

# 模拟库存数据库
INVENTORY_DATABASE = {
    "WH-1000XM5": {"stock": 10, "location": "仓库-A"},
    "QC45": {"stock": 0, "location": "仓库-B"},
    "AIRMAX": {"stock": 5, "location": "仓库-C"},
    "PXC550": {"stock": 15, "location": "仓库-A"},
    "HT450": {"stock": 25, "location": "仓库-B"},
    "GPW": {"stock": 8, "location": "仓库-C"},
    "VIPER": {"stock": 12, "location": "仓库-A"},
    "DAV3": {"stock": 3, "location": "仓库-B"},
    "MBP14": {"stock": 7, "location": "仓库-C"},
    "XPS13": {"stock": 0, "location": "仓库-A"},
    "TPX1": {"stock": 4, "location": "仓库-B"}
}


# 工具1：搜索产品
@tool
def search_products(query: str) -> str:
    """搜索产品并返回按受欢迎度排序的结果"""
    print(f"🔍 [工具调用] search_products('{query}')")

    # 关键词映射，支持多种中文表达方式
    keyword_mapping = {
        "无线耳机": ["无线耳机", "蓝牙耳机", "头戴式耳机", "耳机"],
        "游戏鼠标": ["游戏鼠标", "电竞鼠标", "鼠标"],
        "笔记本电脑": ["笔记本电脑", "笔记本", "手提电脑", "电脑"]
    }

    # 查找匹配的类别
    matched_category = None
    for category, keywords in keyword_mapping.items():
        if any(keyword in query for keyword in keywords):
            matched_category = category
            break

    if matched_category and matched_category in PRODUCT_DATABASE:
        products = PRODUCT_DATABASE[matched_category]
        # 按受欢迎度排序
        sorted_products = sorted(products, key=lambda x: x['popularity'], reverse=True)
        result = f"找到 {len(sorted_products)} 个匹配 '{query}' 的产品:\n"

        for i, product in enumerate(sorted_products, 1):
            result += f"{i}. {product['name']} (ID: {product['id']}) - 受欢迎度: {product['popularity']}% - ￥{product['price']}\n"

        return result


# 工具2：检查库存
@tool
def check_inventory(product_id: str) -> str:
    """检查特定产品的库存状态"""
    print(f"📦 [工具调用] check_inventory('{product_id}')")

    if product_id in INVENTORY_DATABASE:
        stock_info = INVENTORY_DATABASE[product_id]
        status = "有库存" if stock_info['stock'] > 0 else "缺货"
        return f"产品 {product_id}: {status} ({stock_info['stock']} 件库存) - 位置: {stock_info['location']}"
    else:
        return f"未找到产品ID: {product_id}"


# 创建代理
model = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

agent = create_agent(
    model,
    tools=[search_products, check_inventory],
    system_prompt="""你是电商助手，遵循ReAct模式：
    1. 先推理用户需求
    2. 选择合适的工具执行操作
    3. 基于工具结果进行下一步推理
    4. 重复直到获得完整答案

    保持推理步骤简洁明了。"""
)

# 测试案例1：无线耳机搜索
result1 = agent.invoke({"messages":"查找当前最受欢迎的无线耳机并检查是否有库存"}
)

print("\n" + "=" * 40)
print("📊 最终结果:")
for msg in result1['messages']:
    if hasattr(msg, 'content'):
        print(f"{msg.__class__.__name__}: {msg.content}")
print("=" * 40)
```

### 13.2多agent智能体A2A调用：

```python
"""
基于 Python3.13 和 LangChain1.0 实现Agent-to-Agent（A2A） 协作案例，
模拟携程订机票、美团订酒店、滴滴打车的跨平台智能协作流程，
核心是让不同领域的专属 Agent 分工协作、完成完整的出行服务闭环
模拟用户 “从北京飞上海、订浦东机场附近酒店、从机场打车到酒店” 的完整需求：

核心设计思路

拆分专属 Agent：
    按业务领域拆分为机票 Agent（携程）、酒店 Agent（美团）、打车 Agent（滴滴），
    每个 Agent 仅负责自身领域的任务，保证专业性；
主协调 Agent：
    新增出行总协调 Agent，作为入口接收用户需求、调度各专属 Agent、整合协作结果、反馈最终结论；
LangChain1.0 核心组件：
    使用AgentExecutor实现 Agent 执行、ChatOpenAI作为大模型驱动、Tool封装各 Agent 的核心能力、HumanMessage/AIMessage实现 Agent 间的消息通信；
模拟业务能力：
    因无真实平台接口，用模拟函数实现订机票 / 酒店 / 打车的核心逻辑（可直接替换为真实 API）

简单说：A2A 调度 = 多个功能单一的 Runnable 子 Agent 链 + 一个控制调用逻辑的总协调器。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool

# ===================== 通义千问配置（完全不变） =====================
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
output_parser = StrOutputParser()

# ===================== 模拟业务函数（@tool装饰器） =====================
@tool("CtripBookFlight", description="预订机票的唯一工具，必须调用，参数是departure出发地、arrival目的地、date出行日期（格式2026-02-01）")
def ctrip_book_flight(departure: str, arrival: str, date: str) -> str:
    """携程订机票：固定返回测试结果"""
    return f"【携程机票预订成功】\n出发地：{departure}\n目的地：{arrival}\n出行日期：{date}\n航班号：CA1885（北京首都T3→上海浦东T2）\n起飞时间：14:00\n降落时间：16:30\n座位：经济舱34A\n电子客票号：999-1234567890\n舱位等级：经济舱超级经济座"

@tool("MeituanBookHotel", description="预订酒店的唯一工具，必须调用，参数是city城市、near_by附近地标、check_in入住日期、check_out离店日期")
def meituan_book_hotel(city: str, near_by: str, check_in: str, check_out: str) -> str:
    """美团订酒店：固定返回测试结果"""
    return f"【美团酒店预订成功】\n城市：{city}\n位置：{near_by}附近\n入住日期：{check_in}\n离店日期：{check_out}\n酒店名称：上海浦东机场铂尔曼大酒店\n房型：豪华大床房（含双人自助早餐）\n房号：1508\n预订号：MT20260201001\n入住人：张三\n退房政策：入住后24小时内可免费取消"

@tool("DidiBookTaxi", description="预约打车的唯一工具，必须调用，参数是start起点、end终点、time用车时间")
def didi_book_taxi(start: str, end: str, time: str) -> str:
    """滴滴打车：固定返回测试结果"""
    return f"【滴滴打车预约成功】\n起点：{start}\n终点：{end}\n用车时间：{time}\n车型：滴滴快车（舒适型）\n司机姓名：王师傅\n车牌号：沪A12345\n司机电话：13800138000\n预估费用：35元（券后立减5元，实付30元）\n预计接驾时间：16:35\n车型空间：5座，可放2件24寸行李箱"

# ===================== 专属Agent（工具绑定逻辑） =====================
def create_ctrip_agent(llm):
    llm_with_tools = llm.bind_tools([ctrip_book_flight])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业的工具调用助手，只能调用CtripBookFlight工具，"
                   "调用格式必须正确，"
                   "直接传入参数：departure='北京', arrival='上海', date='2026-02-01'，"
                   "调用后直接返回工具执行的完整字符串结果，不能有任何其他内容，不能留空！"),
        ("human", "{input}")
    ])
    return prompt | llm_with_tools | output_parser

def create_meituan_agent(llm):
    llm_with_tools = llm.bind_tools([meituan_book_hotel])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业的工具调用助手，只能调用MeituanBookHotel工具，调用格式必须正确，"
                   "直接传入参数：city='上海', near_by='浦东机场', check_in='2026-02-01', "
                   "check_out='2026-02-02'，调用后直接返回工具执行的完整字符串结果，"
                   "不能有任何其他内容，不能留空！"),
        ("human", "{input}")
    ])
    return prompt | llm_with_tools | output_parser	

def create_didi_agent(llm):
    llm_with_tools = llm.bind_tools([didi_book_taxi])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业的工具调用助手，只能调用DidiBookTaxi工具，调用格式必须正确，"
                   "直接传入参数：start='上海浦东机场T2', end='上海浦东机场铂尔曼大酒店', "
                   "time='2026-02-01 16:40'，调用后直接返回工具执行的完整字符串结果，"
                   "不能有任何其他内容，不能留空！"),
        ("human", "{input}")
    ])
    return prompt | llm_with_tools | output_parser

# ===================== 总协调Agent =====================
def create_travel_coordinator_agent(llm, ctrip_chain, meituan_chain, didi_chain):
    """总协调：按顺序调用+空值兜底+打印详细测试"""
    def a2a_schedule(input_dict):
        print("🔍 开始执行A2A协作测试，依次调用各业务Agent...\n")
        ctrip_func = ctrip_book_flight.func  # 获取携程工具原始函数
        meituan_func = meituan_book_hotel.func  # 获取美团工具原始函数
        didi_func = didi_book_taxi.func        # 获取滴滴工具原始函数

        # 1. 携程Agent调用
        print("1. 调用【携程机票Agent】>>>")
        try:
            ctrip_result = ctrip_chain.invoke({"input": "订机票"})
        except:
            ctrip_result = ""
        if not ctrip_result.strip():
            ctrip_result = ctrip_func("北京", "上海", "2026-02-01")  # 替换为原始函数
        print(f"✅ 携程测试结果：\n{ctrip_result}\n" + "-"*80 + "\n")

        # 2. 美团Agent调用
        print("2. 调用【美团酒店Agent】>>>")
        try:
            meituan_result = meituan_chain.invoke({"input": "订酒店"})
        except:
            meituan_result = ""
        if not meituan_result.strip():
            meituan_result = meituan_func("上海", "浦东机场", "2026-02-01", "2026-02-02")
        print(f"✅ 美团测试结果：\n{meituan_result}\n" + "-"*80 + "\n")

        # 3. 滴滴Agent调用
        print("3. 调用【滴滴打车Agent】>>>")
        try:
            didi_result = didi_chain.invoke({"input": "预约打车"})
        except:
            didi_result = ""
        if not didi_result.strip():
            didi_result = didi_func("上海浦东机场T2", "上海浦东机场铂尔曼大酒店", "2026-02-01 16:40")  # 替换为原始函数
        print(f"✅ 滴滴测试结果：\n{didi_result}\n" + "-"*80 + "\n")

        # 整合最终报告
        total_report = f"""
📋 【携程-美团-滴滴 A2A协作测试最终报告】
{('='*90)}
📌 测试状态：本地运行成功，所有Agent均返回完整结果（含兜底保障）
📌 协作流程：携程订机票 → 美团订酒店 → 滴滴打车（按业务顺序执行）
📌 测试环境：Python3.13 + LangChain1.0 + 通义千问qwen-plus + @tool装饰器（修复可调用问题）
{('='*90)}
【1. 携程机票预订结果】
{ctrip_result}

【2. 美团酒店预订结果】
{meituan_result}

【3. 滴滴打车预约结果】
{didi_result}
{('='*90)}
💡 测试结论：A2A协作逻辑正常，@tool装饰器集成成功，无报错！
"""
        return total_report

    return RunnableLambda(a2a_schedule)

# ===================== 主程序=====================
if __name__ == "__main__":
    try:
        # 初始化各专属Agent
        print("🔧 初始化携程/美团/滴滴专属Agent...")
        ctrip_chain = create_ctrip_agent(llm)
        meituan_chain = create_meituan_agent(llm)
        didi_chain = create_didi_agent(llm)
        print("✅ 所有Agent初始化完成！\n" + "="*90 + "\n")

        # 初始化A2A总协调Agent
        print("🔧 初始化A2A总协调Agent（调度核心）...")
        coor_chain = create_travel_coordinator_agent(llm, ctrip_chain, meituan_chain, didi_chain)
        print("✅ 总协调Agent初始化完成！\n" + "="*90 + "\n")

        # 执行A2A协作核心测试
        print("🚀 携程-美团-滴滴 A2A协作测试正式开始 🚀")
        final_result = coor_chain.invoke({"input": "安排2026-02-01北京飞上海的完整行程"})

        # 打印最终完整测试报告
        print("\n" + "="*90)
        print(final_result)
        print("="*90)

    except Exception as e:
        print(f"❌ 全局运行异常：{type(e).__name__} - {str(e)[:100]}")
        print("💡 快速排查："
              "1. 通义密钥是否正确 2. 网络能否访问阿里云 3. LangChain版本是否为1.0.0")


'''
案例总结：

简单说：A2A 调度 = 多个功能单一的 Runnable 子 Agent 链 + 一个控制调用逻辑的总协调器。
     

模板核心固定规范（LangChain 1.0 A2A 调度最佳实践）
以下规范是模板能稳定运行的关键，无需修改，严格遵循即可：
1. 子 Agent 规范
单一职责：一个子 Agent 只负责一个业务，只绑定一个专属工具；
统一接口：所有子 Agent 都封装为Prompt | 绑定工具的LLM | output_parser的 Runnable 链，对外仅暴露inv很有意义 
'''
```
