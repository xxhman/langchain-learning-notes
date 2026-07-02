# LangGraph框架：

首要安装：pip install -U langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple

## 1.langgraph可视化代码：

```python
#1. 打印图的ascii可视化结构,其中app为代码最终的输出结果
print(app.get_graph().print_ascii())

#2. 打印图的Mermaid代码可视化结构并通过https://www.processon.com/mermaid编辑器查看
print(app.get_graph().draw_mermaid())

```

## 2.langgraph基础框架流程：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import uuid

# =====================
# 1. 定义【状态模板】，把后面要用的内容定义进去
# =====================
class StateSchema(TypedDict):
    username: str    # 原名name，更直观
    text_content: str# 原名greeting，无歧义


# =====================
# 2. 定义两个节点函数
# =====================
def build_base_text(current_state: StateSchema) -> dict:
    user_name = current_state["username"]
    return {"text_content": f"你好,{user_name}"}


def append_emoji(current_state: StateSchema) -> dict:
    content = current_state["text_content"]
    return {"text_content": content + "  。。。😄"}


# =====================
# 3. 搭建工作流图
# =====================
# 绑定我们的状态规则模板
workflow = StateGraph(StateSchema)

# 对节点函数进行节点绑定
workflow.add_node("生成基础文案", build_base_text)
workflow.add_node("添加表情符号", append_emoji)

# 对绑定后的节点连线：规定执行顺序
workflow.add_edge(START, "生成基础文案")
workflow.add_edge("生成基础文案", "添加表情符号")
workflow.add_edge("添加表情符号", END)

# =====================
# 4. 编译、运行
# =====================
run_app = workflow.compile()

# 传入执行的内容参数
final_result = run_app.invoke({"username": "z3"})

# 打印结果
print(final_result)
print(final_result["text_content"])

# 打印图的ascii可视化结构
print(run_app.get_graph().print_ascii())
# 打印图的Mermaid代码可视化结构并通过https://www.processon.com/mermaid编辑器查看
print(run_app.get_graph().draw_mermaid())


```

## 3.langgraph加入LLM基础框架流程：

```python
"""
LangGraph的灵魂：State(状态) + Nodes(节点) + Edges(边) + Graph(图)
"""

import uuid
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


# ========== 1. 定义状态（State） ==========
# 存储对话消息
class AtguiguState(TypedDict):
    # messages 是一个消息列表，Annotated + add_messages 表示支持自动追加消息
    messages: list

# ========== 2. 定义大模型 ==========
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ========== 3. 定义节点函数 ==========
# 节点：调用大模型，并把回复加入到 state["messages"] 里
def model_node(state: AtguiguState):
    reply = llm.invoke(state["messages"])   # 输入历史消息，调用模型
    return {"messages": [reply]}            # 返回新消息，自动加到 state

# ========== 4. 构建图结构 ==========
graph = StateGraph(AtguiguState)            # 初始化图，指定 State 类型

graph.add_node("model", model_node)         # 添加一个节点，名字叫 "model"

graph.add_edge(START, "model")      # 从 START 到 "model"
graph.add_edge("model", END)        # 从 "model" 到 END

# ========== 5. 编译==========
app = graph.compile()

# ========== 6. 运行 ==========
#result = app.invoke({"messages": [HumanMessage(content="请用一句话解释什么是 LangGraph。")]})
result = app.invoke({"messages": "请用一句话解释什么是 LangGraph。"})

# 打印模型的最后一条回复

print("模型回答：", result["messages"][-1].content)


# =========================
#1. 打印图的ascii可视化结构
print(app.get_graph().print_ascii())
print("="*50)

#2. 打印图的Mermaid代码可视化结构并通过https://www.processon.com/mermaid编辑器查看
print(app.get_graph().draw_mermaid())
print("="*50)
```

## 4.langgraph中state(状态)的基础框架流程：

### 4.1 schema(模式)基础框架流程（了解）：

```python
"""
LangGraph 图输入输出模式和私有状态传递演示

该演示展示了：
1. 如何定义图的输入和输出模式
"""

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# 定义输入状态模式
class InputState(TypedDict):
    question: str

# 定义输出状态模式
class OutputState(TypedDict):
    answer: str

# 定义整体状态模式，结合输入和输出
class OverallState(InputState, OutputState):
    pass


# 定义处理节点
def answer_node(state: InputState):
    """
    处理输入并生成答案的节点
    Args:
        state: 输入状态
    Returns:
        dict: 包含答案的字典
    """
    print(f"执行 answer_node 节点:")
    print(f"  输入: {state}")

    # 示例答案
    answer = "再见" if "bye" in state["question"].lower() else "你好"
    result = {"answer": answer, "question": state["question"]}

    print(f"  输出: {result}")
    return result


def demo_input_output_schema():
    """演示输入输出模式"""
    print("=== 演示输入输出模式 ===")

    # 使用指定的输入和输出模式构建图
    builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    builder.add_node("answer_node", answer_node)  # 添加答案节点

    builder.add_edge(START, "answer_node")  # 定义起始边
    builder.add_edge("answer_node", END)  # 定义结束边
    graph = builder.compile()  # 编译图

    # 使用输入调用图并打印结果
    result = graph.invoke({"question": "你好"})
    print(f"图调用结果: {result}")
    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
    print()


def main():
    """主函数"""
    print("=== LangGraph 图输入输出模式===\n")

    # 演示输入输出模式
    demo_input_output_schema()

    print("=== 演示完成 ===")


if __name__ == "__main__":
    main()
```



### 4.2 Reducers(规约函数)基础框架流程）：

#### 4.2.1 default：未指定Reducer时使用覆盖更新：

```python
'''
如果未明确指定reducer函数，则默认对该键的更新是覆盖行为。
LangGraph Reducer函数演示 - 默认Reducer（覆盖更新）

直接覆盖：
如果没有为状态字段指定 Reducer，默认会覆盖更新。
也就是说，后执行的节点返回的值会直接覆盖先执行节点的值，
即下一个节点的State数据是上一个节点的返回。
'''

from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 默认Reducer（覆盖更新）
# 未指定合并策略，默认覆盖，上一个节点的返回是下一个节点的值
class DefaultReducerState(TypedDict):
    foo: int
    bar: List[str]

def node_default_1(state: DefaultReducerState) -> dict:
    print(state["foo"])
    print(state["bar"])
    return {"foo": 22}

def node_default_2(state: DefaultReducerState) -> dict:
    print()
    print(state["foo"])
    print(state["bar"])
    return {"bar": ["bye1","bye2","bye3"]}


def main():
    print("1. 默认Reducer（覆盖更新）演示:\n")
    builder = StateGraph(DefaultReducerState)

    builder.add_node("node1", node_default_1)
    builder.add_node("node2", node_default_2)

    builder.add_edge(START, "node1")
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", END)

    graph = builder.compile()

    result = graph.invoke(input={"foo": 1, "bar": ["hi"]})
    #print(f"初始状态: {{'foo': 1, 'bar': ['hi']}}")
    print(f"执行结果: {result}\n")


if __name__ == "__main__":
    main()
```

#### 4.2.2 add_messages：用于消息列表追加：

```python
"""
LangGraph Reducer函数演示 - add_messages Reducer（消息列表专用）
"""

from typing import Annotated, List

from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 2. add_messages Reducer（消息列表专用）
class AddMessagesState(TypedDict):
    """
    引入的 Annotated 类型，它允许给类型添加额外的元数据。
    messages: Annotated[List, add_messages]
    表示:
    - messages 我的状态里只有一个字段叫 messages，类型是是 List列表类型,
    - add_messages  这里的 add_messages 是一个函数，用于修改 messages 列表
                    每当节点返回对 messages 的“局部更新”时，
                    请用 add_messages 规约器把它合并到旧列表上（追加，而不是覆盖）
    总结：
    节点永远只 return 增量字典，不用手动把旧列表读出来再拼接。
    add_messages 在后台帮你完成“追加”动作；如果换成默认 reducer，旧消息会被整份替换掉
    """
    messages: Annotated[List, add_messages]

def chat_node_1(state: AddMessagesState) -> dict:
    return {"messages": [("assistant", "Hello from node 1")]}

def chat_node_2(state: AddMessagesState) -> dict:
    return {"messages": [("assistant", "Hello from node 2")]}

def run_demo():
    print("2. add_messages Reducer（消息列表专用）演示:")
    builder = StateGraph(AddMessagesState)
    builder.add_node("chat1", chat_node_1)
    builder.add_node("chat2", chat_node_2)

    builder.add_edge(START, "chat1")
    builder.add_edge(START, "chat2")  # 并行执行
    builder.add_edge("chat1", END)
    builder.add_edge("chat2", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [("user", "Hi there!")]})
    print(f"初始状态: {{'messages': [('user', 'Hi there!')]}}")
    print(f"执行结果: {result}\n")

    print("*" * 60)

    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())

if __name__ == "__main__":
    run_demo()
```

#### 4.2.3 add_messages：用于消息列表追加：

1.列表追加：

```python
"""
LangGraph Reducer函数演示 - operator.add Reducer（列表追加）
"""

import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 3. operator.add Reducer（列表追加）
class ListAddState(TypedDict):
    #data: Annotated[List[int], None]  #默认覆盖
    data: Annotated[List[int], operator.add] # （列表追加）


def producer_1(state: ListAddState) -> dict:
    return {"data": [1, 2]}


def producer_2(state: ListAddState) -> dict:
    return {"data": [3, 4]}


def run_demo():
    builder = StateGraph(ListAddState)
    # 注册节点
    builder.add_node("producer1", producer_1)
    builder.add_node("producer2", producer_2)
    # 顺序执行边
    builder.add_edge(START, "producer1")
    builder.add_edge("producer1", "producer2")
    builder.add_edge("producer2", END)

    graph = builder.compile()
    result = graph.invoke({"data": [0]})
    print(f"初始状态: {{'data': [0]}}")
    print(f"执行结果: {result}\n")


if __name__ == "__main__":
    run_demo()
```

2.字符串连接：

```python
"""
LangGraph Reducer函数演示 - 字符串连接Reducer
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 6. 字符串连接Reducer
class StringConcatState(TypedDict):
    text: Annotated[str, operator.add]


def add_text_1(state: StringConcatState) -> dict:
    return {"text": "Hello "}


def add_text_2(state: StringConcatState) -> dict:
    return {"text": "World!"}


def run_demo():
    print("3.2 字符串连接Reducer演示:")

    builder = StateGraph(StringConcatState)

    builder.add_node("add_text_1", add_text_1)
    builder.add_node("add_text_2", add_text_2)

    builder.add_edge(START, "add_text_1")
    builder.add_edge(START, "add_text_2")  # 并行执行
    builder.add_edge("add_text_1", END)
    builder.add_edge("add_text_2", END)

    graph = builder.compile()

    result = graph.invoke({"text": "Say: "})
    print(f"初始状态: {{'text': 'Say: '}}")
    print(f"执行结果: {result}\n")


if __name__ == "__main__":
    run_demo()
```

3.数值累加：

```python
"""
LangGraph Reducer函数演示 - 数值累加Reducer
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 7. 数值累加Reducer
class NumberAddState(TypedDict):
    count: Annotated[int, operator.add]


def increment_1(state: NumberAddState) -> dict:
    return {"count": 5}


def increment_2(state: NumberAddState) -> dict:
    return {"count": 3}


def run_demo():
    print("3.3 数值累加Reducer演示:")
    builder = StateGraph(NumberAddState)
    builder.add_node("increment_1", increment_1)
    builder.add_node("increment_2", increment_2)
    # 顺序执行边
    builder.add_edge(START, "increment_1")
    builder.add_edge("increment_1", "increment_2")
    builder.add_edge("increment_2", END)

    graph = builder.compile()

    result = graph.invoke({"count": 10})
    print(f"初始状态: {{'count': 10}}")
    print(f"执行结果: {result}\n")


if __name__ == "__main__":
    run_demo()
```

#### 4.2.4 自定义Reducer：支持用户自定义合并逻辑：（原本的数值相乘有bug，需要自己去重新定义函数）

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


def MyOperatorMul(current: float, update: float) -> float:
    """自定义乘法reducer，处理初始值为1.0"""
    # 如果是第一次调用，current会是默认值0.0
    if current == 0.0:
        # 对于乘法，恒等元应该是1.0或者 return 1.0 * update
        print(f"current:{current}")
        print(f"update:{update}")
        return 1.0 * update
    return current * update

class MultiplyState(TypedDict):
    factor: Annotated[float, MyOperatorMul]

def multiplier(state: MultiplyState) -> dict:
    return {"factor": 2.0}

def run_demo():
    print("使用自定义reducer解决乘法问题:")
    builder = StateGraph(MultiplyState)
    builder.add_node("multiplier", multiplier)
    builder.add_edge(START, "multiplier")
    builder.add_edge("multiplier", END)
    graph = builder.compile()

    result = graph.invoke({"factor": 5.0})
    print(f"初始状态: {{'factor': 5.0}}")
    print(f"执行结果: {result}")  # 应该是 {'factor': 10.0}
    print(f"解释: 5.0 * 2.0 = 10.0\n")

if __name__ == "__main__":
    run_demo()
```

## 5. Graph API之Node(节点)：

### 5.1节点缓存：

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

# 定义状态类，也就是你的业务实体entity
class State(TypedDict):
    x: int
    result: int

# 创建图
builder = StateGraph(State)

# 定义节点：模拟耗时计算（sleep3秒）
def expensive_node(state: State) -> dict[str, int]:
    time.sleep(3)
    return {"result": state["x"] * 2}

# 添加节点，cache_policy代表缓存节点策略添加，ttl=8代表缓存保存8s
builder.add_node("expensive_node",expensive_node,cache_policy=CachePolicy(ttl=8))

# 设置入口和出口
builder.set_entry_point("expensive_node") #等价于stateGraph.add_edge(START, "expensive_node")
builder.set_finish_point("expensive_node") #等价于stateGraph.add_edge("expensive_node", END)

# 编译图，指定内存缓存，提供缓存的存储容器可以修改成其他的，具体如何修改用ai
app = builder.compile(cache=InMemoryCache())

# 第一次执行：耗时3秒（无缓存）
print("第一次执行（无缓存，耗时3秒）：")
print(app.invoke({"x": 5}))
# 第二次执行：瞬间返回（利用缓存，8秒内有效）
print("第二次运行利用缓存并快速返回：")
print(app.invoke({"x": 5}))
```

### 5.2 为节点添加重试策略：

```python
from functools import partial
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from requests import RequestException, Timeout


# 定义状态
class GraphState(TypedDict):
    process_data: dict # 默认更新策略


# 定义一个节点，入参为state
def input_node(state: GraphState) -> GraphState:
    print(f'input_node收到的初始值:{state}')
    return {"process_data": {"input": "input_value"}}

# 定义带参数的node节点
def process_node(state: dict, param1: int, param2: str) -> dict:
    print(state, param1, param2)
    return {"process_data": {"process": "process_value"}}


# 重试策略,add_node方法时可选
retry_policy = RetryPolicy(
    max_attempts=3,                       # 最大重试次数
    initial_interval=1,                   # 初始间隔
    jitter=True,                          # 抖动（添加随机性避免重试风暴）
    backoff_factor=2,                     # 退避乘数（每次重试间隔时间的增长倍数）
    retry_on=[RequestException, Timeout]  # 只重试这些异常
)


stateGraph = StateGraph(GraphState)
# 添加inpu节点
stateGraph.add_node("input", input_node)
# 给process_node节点绑定参数
process_with_params = partial(process_node, param1=100, param2="test")
# 添加带参数的node节点，并且为节点添加重试策略
stateGraph.add_node("process", process_with_params,retry_policy=retry_policy)

# 定义节点之间的执行顺序 edges
# 设置节点间的依赖关系，形成执行流程图
stateGraph.add_edge(START, "input")
stateGraph.add_edge("input", "process")
stateGraph.add_edge("process", END)

# 编译图构建器生成计算图
graph = stateGraph.compile()

# 打印图的可视化结构
print(graph.get_graph().print_ascii())

# 调用graph对象的invoke方法，传入初始状态，执行图计算流程
result= graph.invoke({"process_data": 5})
print(f"最后的结果是:{result}")
```

### 5.3 重试机制分类模板：

```python
"""
LangGraph 节点重试策略演示

默认重试策略：max_attempts=5，对Exception重试、对ValueError/TypeError等不重试，异常过滤列表完全相同；
自定义重试策略：max_attempts=5 + custom_retry_on，仅对包含{模拟API调用失败}的异常重试；throw new RuntimeExp("模拟API调用失败")
不可重试测试：ValueError直接抛错，无重试，max_attempts=3
"""
from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# 定义状态类型
class AtguiguState(TypedDict):
    result: str

# 全局计数器：记录API尝试次数
attempt_counter = 0


# 工具函数
def build_retry_graph(node_name: str, node_func, retry_policy: RetryPolicy):
    builder = StateGraph(AtguiguState)
    #为节点添加重试策略，需要在add_node中设置retry_policy参数。
    # retry_policy参数接受一个RetryPolicy命名元组对象。
    # 默认情况下，retry_on参数使用default_retry_on函数，该函数会在遇到任何异常时重试
    builder.add_node(node_name, node_func, retry_policy=retry_policy)
    builder.add_edge(START, node_name)
    builder.add_edge(node_name, END)
    return builder.compile()


# 模拟不稳定的API调用，使用全局变量跟踪尝试次数
def unstable_api_call(state: AtguiguState) -> Dict[str, Any]:
    """模拟不稳定API：前2次失败，第3次成功（全局计数器记录尝试次数）"""
    global attempt_counter
    attempt_counter += 1
    # 纯文本打印尝试次数
    print(f"尝试调用API，这是第 {attempt_counter} 次尝试")

    # 模拟失败/成功逻辑：前2次抛异常，第3次返回结果
    if attempt_counter < 3:
        raise Exception(f"模拟API调用失败abcd (尝试 {attempt_counter})")
    return {"result": f"API调用成功，经过 {attempt_counter} 次尝试"}

# 自定义重试条件判断函数
def custom_retry_on(exception: Exception) -> bool:
    """自定义重试规则：只对包含「模拟API调用失败」的异常重试"""
    err_msg = str(exception)
    if "模拟API调用失败" in err_msg:
        print(f"捕获到可重试异常: {err_msg}")
        return True
    print(f"捕获到不可重试异常: {err_msg}")
    return False

# 测试方法1：默认重试策略
def test_default_retry():
    global attempt_counter
    print("测试默认重试策略:")
    attempt_counter = 0  # 重置计数器
    default_graph = build_retry_graph(
        node_name="unstable_api",
        node_func=unstable_api_call,
        retry_policy=RetryPolicy(max_attempts=5)  # 最多5次尝试，足够重试成功
    )
    try:
        result = default_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 测试方法2：自定义重试策略（输出完全匹配要求）
def test_custom_retry():
    global attempt_counter
    print("测试自定义重试策略:")
    attempt_counter = 0  # 重置计数器
    custom_graph = build_retry_graph(
        node_name="custom_retry_api",
        node_func=unstable_api_call,
        retry_policy=RetryPolicy(max_attempts=5, retry_on=custom_retry_on) #retry_on代表引人了自定义的重试策略
    )
    try:
        result = custom_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 主演示函数
def run_demo():
    print("=== LangGraph 节点重试策略完整演示===")
    print("-" * 80 + "\n")
    test_default_retry() #默认重试策略
    test_custom_retry() #自定义重试策略


# 程序入口
if __name__ == "__main__":
    run_demo()
```

## 6.Graph API之Edge(边)：

### 6.1 条件边：

```python
from typing import Optional, TypedDict  # 替换：导入TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger

# ====================== 【核心修改1】状态改为 TypedDict ======================
class MyState(TypedDict):
    """
    定义TypedDict状态（字典格式，无实例化）
    """
    x: int
    result: str

# ====================== 【核心修改2】所有函数：.属性 → ["键"] ======================
def check_x(state: MyState) -> MyState:
    logger.info(f"[check_x] Received state: {state}")
    return state  # 字典直接返回，无需实例化

def is_even(state: MyState) -> bool:
    # 原 state.x → 改为 state["x"]
    return state["x"] % 2 == 0

def handle_even(state: MyState) -> MyState:
    logger.info("[handle_even] x 是偶数")
    # 原 return MyState(...) → 直接返回字典！
    return {"x": state["x"], "result": "even"}

def handle_odd(state: MyState) -> MyState:
    logger.info("[handle_odd] x 是奇数")
    # 直接返回字典
    return {"x": state["x"], "result": "odd"}

# ====================== 图构建逻辑 完全不变 ======================
builder = StateGraph(MyState)
builder.add_node("check_x", check_x)
builder.add_node("handle_even", handle_even)
builder.add_node("handle_odd", handle_odd)

builder.add_conditional_edges("check_x", is_even, {
    True: "handle_even",
    False: "handle_odd"
})

builder.add_edge(START, "check_x")
builder.add_edge("handle_even", END)
builder.add_edge("handle_odd", END)

graph = builder.compile()

# 打印流程图
print(graph.get_graph().print_ascii())

# ====================== 【核心修改3】invoke传入字典，而非实例 ======================
# 测试偶数4
logger.info("输入 x=4（偶数）")
graph.invoke({"x": 4})  # 直接传字典！

# 测试奇数3
logger.info("输入 x=3（奇数）")
graph.invoke({"x": 3})
```

### 6.2 条件入口点：

```python
'''
LangGraph中条件入口点的典型应用场景
完整展示了条件入口点的核心概念：根据输入内容动态决定从START节点去往哪个处理节点。
'''
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义简单的状态
class SimpleState(TypedDict):
    user_input: str
    response: str
    node_visited: str


# 2. 路由函数 - 决定从START去哪
def route_input(state: SimpleState) -> str:
    """根据用户输入决定去哪个节点"""
    text = state["user_input"].lower()

    if "hello" in text or "hi" in text:
        return "greeting"  # 返回路由键
    elif "bye" in text or "exit" in text:
        return "farewell"  # 返回路由键
    else:
        return "question"  # 返回路由键


# 3. 各个处理节点
def handle_greeting(state: SimpleState) -> SimpleState:
    """处理问候"""
    state["response"] = "你好！很高兴见到你！"
    state["node_visited"] = "greeting_node"
    return state


def handle_farewell(state: SimpleState) -> SimpleState:
    """处理告别"""
    state["response"] = "再见！祝你有个美好的一天！"
    state["node_visited"] = "farewell_node"
    return state


def handle_question(state: SimpleState) -> SimpleState:
    """处理问题"""
    state["response"] = "我听到了你的问题，需要更多帮助吗？"
    state["node_visited"] = "question_node"
    return state


# 4. 创建图
def create_simple_graph():
    """创建一个简单的图"""
    stateGraph = StateGraph(SimpleState)

    # 添加节点
    stateGraph.add_node("greeting_node", handle_greeting)
    stateGraph.add_node("farewell_node", handle_farewell)
    stateGraph.add_node("question_node", handle_question)

    stateGraph.add_conditional_edges(
        START,  # 起点
        route_input,  # 路由函数
        # 路由映射（可选）：路由函数的返回值 -> 节点名
        {
            "greeting": "greeting_node",  # route_input返回"greeting"时，去greeting_node
            "farewell": "farewell_node",  # route_input返回"farewell"时，去farewell_node
            "question": "question_node"  # route_input返回"question"时，去question_node
        }
    )
	
    # 所有节点都到END
    stateGraph.add_edge("greeting_node", END)
    stateGraph.add_edge("farewell_node", END)
    stateGraph.add_edge("question_node", END)

    return stateGraph.compile()


# 5. 使用示例
def run_example():
    # 创建图
    graph = create_simple_graph()
    # 测试不同的输入
    test_inputs = [
        "Hello everyone!",
        "Goodbye now",
        "What time is it?"
    ]

    for user_input in test_inputs:
        print(f"\n输入: {user_input}")
        print("-" * 30)

        # 创建初始状态
        initial_state = SimpleState(
            user_input=user_input,
            response="",
            node_visited=""
        )

        # 执行图
        result = graph.invoke(initial_state)

        print(f"路由决策: {route_input(initial_state)}")
        print(f"访问的节点: {result['node_visited']}")
        print(f"响应: {result['response']}")

    print()
    # 打印图的ascii可视化结构
    print(graph.get_graph().print_ascii())
    print("=================================")
    print()
    # 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
    print(graph.get_graph().draw_mermaid())


# 运行示例
if __name__ == "__main__":
    print("简单条件入口点示例")
    print("=" * 40)
    run_example()


```

## 7. Graph API之Send/Command/Runtime context：

### 7.1 send：

```python
"""
SendDemo.py
LangGraph Map-Reduce 模式演示
通过使用 Send 对象，LangGraph 提供了一种优雅的方式来实现这种动态图结构，
使得我们可以根据运行时状态来决定执行路径。

解释：
（1）首先执行 generate_subjects主题列表节点，生成主题列表：['猫', '狗', '程序员']
（2）然后通过条件边函数 map_subjects_to_jokes 为每个主题创建一个 Send 对象
（3）make_joke 节点被并行执行3次，每次处理一个主题
（4）最终将所有生成的笑话合并到一个列表中
这种模式非常适合处理动态数量的任务
"""
import operator
from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 定义状态
class AtguiguState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]  # 使用列表合并的方式


# 第一个节点：生成需要处理的主题列表
def generate_subjects(state: AtguiguState) -> dict:
    subjects = ["猫", "狗", "程序员"]
    return {"subjects": subjects}


# Map节点：为每个主题生成笑话
def make_joke(state: AtguiguState) -> dict:
    subject = state.get("subject", "未知")
    # 根据主题生成相应笑话
    jokes_map = {
        "猫": "为什么猫不喜欢在线购物？因为它们更喜欢实体店！",
        "狗": "为什么狗不喜欢计算机？因为它们害怕被鼠标咬！",
        "程序员": "为什么程序员喜欢洗衣服？因为他们在寻找bugs！",
        "未知": "这是一个关于未知主题的神秘笑话。"
    }

    joke = jokes_map.get(subject, f"这是一个关于{subject}的即兴笑话。")
    return {"jokes": [joke]}


# 条件边函数：根据主题列表生成Send对象列表
def map_subjects_to_jokes(state: AtguiguState) -> List[Send]:
    """将主题列表映射到joke生成任务"""
    subjects = state["subjects"]
    # 为每个主题创建一个Send对象，指向make_joke节点
    # 每个Send对象包含节点名称和传递给该节点的状态
    send_list = [Send("make_joke", {"subject": subject}) for subject in subjects]
    return send_list


def main():
    """演示Map-Reduce模式"""

    # 创建图
    builder = StateGraph(AtguiguState)

    # 添加节点
    builder.add_node("generate_subjects", generate_subjects)
    builder.add_node("make_joke", make_joke)

    # 添加条件边，使用Send对象实现map-reduce
    builder.add_conditional_edges(
        "generate_subjects",  # 源节点
        map_subjects_to_jokes  # 路由函数，返回Send对象列表
    )

    # 添加边
    builder.add_edge(START, "generate_subjects")
    builder.add_edge("make_joke", END)

    # 编译图
    graph = builder.compile()
    print(graph.get_graph().print_ascii())

    # 执行图
    initial_state = {"subjects": [], "jokes": []}
    result = graph.invoke(initial_state)
    print(f"\n最终结果: {result}")


if __name__ == "__main__":
    main()
```

### 7.2 command：

```python
"""
CommandDemo.py | LangGraph 1.0.6 正式版
Command 基础演示：状态更新+流程控制+动态路由
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# 全局常量：统一递归限制，便于维护
RECURSION_LIMIT = 50

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # 自动合并消息
    current_agent: str
    task_completed: bool

# 决策代理（核心路由节点）
def decision_agent(state: AgentState) -> Command[AgentState]:
    """根据消息内容路由代理，任务完成则直接终止"""
    print("执行节点: decision_agent")
    # 优先终止流程（核心防循环逻辑）
    if state["task_completed"]:
        print("✅ 检测到任务已完成，直接终止流程")
        return Command(
            update={"messages": [("system", "所有任务处理完成，流程正常结束")]},
            goto=END
        )
    # 提取消息文本（兼容空消息）
    last_message = state["messages"][-1] if state["messages"] else ("", "")
    last_msg_content = last_message[1]
    print(f"最新消息文本: {last_msg_content}")

    # 动态路由
    if "数学" in last_msg_content:
        print("✅ 检测到数学任务，路由到数学代理")
        return Command(
            update={"messages": [("system", "路由到数学代理")], "current_agent": "math_agent"},
            goto="math_agent"
        )
    elif "翻译" in last_msg_content:
        print("✅ 检测到翻译任务，路由到翻译代理")
        return Command(
            update={"messages": [("system", "路由到翻译代理")], "current_agent": "translation_agent"},
            goto="translation_agent"
        )
    else:
        print("❌ 未识别任务类型，标记任务完成并终止")
        return Command(
            update={"messages": [("system", "任务完成")], "task_completed": True},
            goto=END
        )


# 数学代理（业务节点）
def math_agent(state: AgentState) -> Command[AgentState]:
    """处理数学计算任务，完成后返回决策代理"""
    print("执行节点: math_agent")
    result = "2 + 2 = 4"
    print(f"计算结果: {result}")
    return Command(
        update={
            "messages": [("assistant", f"数学计算结果: {result}")],
            "current_agent": "decision_agent",
            "task_completed": True
        },
        goto="decision_agent"
    )


# 翻译代理（业务节点）
def translation_agent(state: AgentState) -> Command[AgentState]:
    """处理中英翻译任务，完成后返回决策代理"""
    print("执行节点: translation_agent")
    translation = "Hello -> 你好"
    print(f"翻译结果: {translation}")
    return Command(
        update={
            "messages": [("assistant", f"翻译结果: {translation}")],
            "current_agent": "decision_agent",
            "task_completed": True
        },
        goto="decision_agent"
    )


def main():
    """演示Command基础用法：状态更新+动态路由+流程终止"""
    print("=== Command 基础演示（LangGraph 1.0.6）===\n")

    # 1. 构建状态图
    builder = StateGraph(AgentState)
    builder.add_node("decision_agent", decision_agent)
    builder.add_node("math_agent", math_agent)
    builder.add_node("translation_agent", translation_agent)

    # 2. 定义边（完整节点关系）
    builder.add_edge(START, "decision_agent")
    builder.add_edge("math_agent", "decision_agent")
    builder.add_edge("translation_agent", "decision_agent")
    builder.add_edge("decision_agent", END)

    # 3. 编译图
    graph = builder.compile()

    # 测试1：数学任务
    print("【测试1: 数学任务】")
    initial_state = {"messages": [("user", "我需要计算数学题")], "current_agent": "user", "task_completed": False}
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print("最终状态(简化):", {k: v for k, v in result.items() if k != "messages"})  # 简化输出
    print("\n" + "-" * 50 + "\n")
    print(result)
    # 测试2：翻译任务
    print("【测试2: 翻译任务】")
    initial_state = {"messages": [("user", "我需要翻译文本")], "current_agent": "user", "task_completed": False}
    print("初始状态:", initial_state)
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print("最终状态(简化):", {k: v for k, v in result.items() if k != "messages"})
    print("\n" + "-" * 50 + "\n")

    # 测试3：未识别任务
    print("【测试3: 未识别任务类型】")
    initial_state = {"messages": [("user", "你好")], "current_agent": "user", "task_completed": False}
    print("初始状态:", initial_state)
    result = graph.invoke(initial_state, recursion_limit=RECURSION_LIMIT)
    print("最终状态(简化):", {k: v for k, v in result.items() if k != "messages"})

    # 新增：可视化图结构（教学演示必备）
    print("\n=== 图结构可视化 ===")
    print(graph.get_graph().draw_mermaid())
    print(graph.get_graph().print_ascii())

if __name__ == "__main__":
    main()
```

### 7.3 Runtime context运行时上下文：

```python
"""
RuntimeContextDemo.py

LangGraph Context Schema 演示

演示如何在 LangGraph 1.0 中使用 context_schema 向节点传递不属于图表状态的信息。
这在传递模型名称、数据库连接等依赖项时非常有用。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage
from dataclasses import dataclass


# 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    response: str


# 定义上下文结构
@dataclass
class ContextSchema:
    model_name: str
    db_connection: str
    api_key: str


# 节点函数：处理用户消息
def process_message(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """处理用户消息的节点，使用context中的信息"""
    print("执行节点: process_message")

    # 获取最新的用户消息
    last_message = state["messages"][-1].content if state["messages"] else ""
    print(f"用户消息: {last_message}")
    print("=========以下是从RuntimeContext中获得信息=========")
    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    db_connection = runtime.context.db_connection
    api_key = runtime.context.api_key

    print(f"使用的模型: {model_name}")
    print(f"数据库连接: {db_connection}")
    print(f"API密钥前缀: {api_key[:5]}***")  # 只显示前5位，隐藏其余部分

    # 模拟使用这些信息处理请求
    response = f"使用 {model_name} 处理了您的请求，已连接到 {db_connection}"

    return {
        "messages": [AIMessage(content=response)],
        "response": response
    }


# 节点函数：生成最终响应
def generate_response(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    """生成最终响应的节点"""
    print("执行节点: generate_response")

    # 使用runtime.context中的信息
    model_name = runtime.context.model_name
    print(f"使用模型 {model_name} 生成最终响应")

    # 获取之前的结果
    previous_response = state["response"]

    # 生成更详细的响应
    final_response = f"{previous_response}\n\n这是使用 {model_name} 生成的完整响应。"

    return {
        "messages": [AIMessage(content=final_response)],
        "response": final_response
    }


def main():
    """演示 context_schema 的使用"""
    print("=== Context Schema 演示 ===\n")

    # 定义上下文
    context = ContextSchema(
        model_name="gpt-4-turbo",
        db_connection="postgresql://user:pass@localhost:5432/orders_db",
        api_key="sk-abcdefghijklmnopqrstuvwxyz123456"
    )

    # 创建图，指定state_schema和context_schema
    builder = StateGraph(AgentState, context_schema=ContextSchema)

    # 添加节点
    builder.add_node("process_message", process_message)
    builder.add_node("generate_response", generate_response)

    # 添加边
    builder.add_edge(START, "process_message")
    builder.add_edge("process_message", "generate_response")
    builder.add_edge("generate_response", END)

    # 编译图
    graph = builder.compile()

    # 定义初始状态
    initial_state = {
        "messages": [HumanMessage(content="请帮我查询最新的订单信息")],
        "response": ""
    }

    print("初始状态:", initial_state)
    print()
    print("上下文信息:\n", {
        "model_name": context.model_name,
        "db_connection": context.db_connection,
        "api_key": f"{context.api_key[:5]}***"
    })
    print("\n" + "-" * 50 + "\n")

    # 执行图，通过context参数传递上下文
    result = graph.invoke(initial_state, context=context)

    print("\n" + "=" * 50)
    print("最终状态:", result)
    print("\n最终响应:")
    print(result["response"])


if __name__ == "__main__":
    main()
```



## 8.高级特性之流式处理(Streaming)：

作用：
①**实时看流程状态**：比如知道 AI 现在在处理哪个步骤、当前的结果是什么（比如 “正在细化主题”“已生成笑话初稿”）；

②**实时看子流程结果**：如果你的 AI 流程里嵌套了小流程（子图），也能同步看到子流程的进度；

③**实时看 LLM 输出的每一个字**：比如 AI 写笑话时，每个词、每句话实时蹦出来，不是最后一次性显示；

④**自定义实时消息**：比如让 AI 干活时，实时发 “进度 30%”“正在调用工具查数据” 这种自定义提示；

⑤**调试用**：能看到流程里的详细细节，方便找问题。

可选模式：

①**values**：每步结束后，输出完整的当前状态（比如 “主题：冰淇淋和猫；笑话：xxx”）；
②**updates**：每步结束后，只输出变化的部分（比如只显示 “主题新增了‘和猫’”）；
③**messages**：专门实时输出 LLM 的每一个字 / 词，还带相关信息（比如是哪个步骤调用的 LLM）；
④**custom**：只输出你自定义的消息（比如进度提示）；
⑤**debug**：输出所有细节，方便调试。

### 8.1流图状态(Stream graph state)：利用updates和values模式

```python
"""
StreamGraphState.py

流图状态
使用流模式，并在图执行时流式传输其状态。updatesvalues
updates在图的每一步后，将更新流向状态。
values在图的每一步后，流出状态的---->全部值。
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AtguiguState(TypedDict):
  topic: str
  joke: str


def refine_topic(state: AtguiguState):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: AtguiguState):
    return {"joke": f"This is a joke about {state['topic']}"}

def main():
    graph = (
      StateGraph(AtguiguState)
      .add_node(refine_topic)
      .add_node(generate_joke)

      .add_edge(START, "refine_topic")
      .add_edge("refine_topic", "generate_joke")
      .add_edge("generate_joke", END)

      .compile()
    )

    # updates在图的每一步后，将更新流向状态，代码框架固定，模式可以改变
    for chunk in graph.stream({"topic": "ice cream"},stream_mode="updates"):
        print(chunk)

    print()

    # values在图的每一步后，流出状态的全部值。
    for chunk in graph.stream({"topic": "ice cream"},stream_mode="values"):
        print(chunk)

if __name__ == "__main__":
    main()
```

### 8.2LLM令牌(LLM tokens）：逐Token流式输出，message模式

```python
'''
StreamLLMTokens.py

使用messages流模式，从图中的任何部分（包括节点、工具、子图或任务）逐token流式传输大型语言模型（LLM）的输出。
messages模式的流式输出是一个元组(message_chunk, metadata)，其中：
   message_chunk：来自大语言模型（LLM）的令牌或消息片段。
   metadata：一个包含图节点和大语言模型调用详情的字典元数据。

'''

from typing import TypedDict
from langgraph.graph import StateGraph,START
from langchain.chat_models import init_chat_model
import os




class State(TypedDict):
    query:str
    answer:str

def node(state:State):
    print("开始调用node节点")

    model = init_chat_model(model="qwen-plus",
                            model_provider="openai",
                            api_key=os.getenv("QWEN_API_KEY"),
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    llm_result = model.invoke( [("user",state["query"])] )

    return {"answer":llm_result}

def main():
    graph = (
        StateGraph(state_schema=State)
        .add_node(node)
        .add_edge(START,"node")
        .compile()
    )

    inputs = {"query":"帮我生成一个200字的小学生作文，主题为我的一天"}

    # stream_mode="messages"从任何调用了大语言模型的图节点流式传输二元组（大语言模型token，元数据）。
    '''messages模式的流式输出是一个元组(message_chunk, metadata)，其中：
        message_chunk：来自大语言模型（LLM）的令牌或消息片段。
        metadata：一个包含图节点和大语言模型调用详情的字典元数据。'''
    for chunk,meta_data in graph.stream(inputs,stream_mode="messages"):
        print(chunk.content,end="")
        

if __name__ == '__main__':
    main()
```

### 8.3 流式传输自定义数据：custom模式

```python
'''
StreamCustomDataSimple.py

要从LangGraph节点或工具内部发送自定义用户定义数据，请遵循以下步骤：
   使用get_stream_writer访问流写入器并发送自定义数据。
   调用.stream()或.astream()时，设置stream_mode="custom"以在流中获取自定义数据。
你可以组合多种模式（例如["updates", "custom"]），但至少有一种模式必须是"custom"。

LangGraph 自定义数据流式传输演示
展示如何从节点内部发送自定义用户定义数据
'''

from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START,END

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"custom_key": "欢迎来到尚硅谷线上Agent班级学习，O(∩_∩)O"})
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .add_edge("node",END)
    .compile()
)

# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream({"query": "example"}, stream_mode=["custom"]):
    print(chunk)
#
# for chunk in graph.stream({"query": "example"}, stream_mode=["updates", "custom"]):
#     print(chunk)
#
# for chunk in graph.stream({"query": "example"}, stream_mode=["values", "custom"]):
#     print(chunk)
```

## 9.高级特性之状态持久化(Persistence）：

### 9.1 内存检查点：放入自带的内存中，关机会丢失

```python
"""
MemoryPersistence.py
langgraph-checkpoint：检查点保存器（BaseCheckpointSaver）
的基础接口以及序列化/反序列化接口（SerializerProtocol）。
包含用于实验的内存中检查点实现（InMemorySaver）。
LangGraph 已内置 langgraph-checkpoint。


LangGraph 1.0 持久化存储演示 - 内存存储 (In-Memory)

特点：
- 数据暂存于内存，程序关闭后丢失
- 无需额外配置
- 适用于本地测试和临时验证工作流逻辑
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator


# 定义状态
class PersistenceDemoState(TypedDict):
    # operator.add：将元素追加到现有元素中，支持列表、字符串、数值类型的追加
    messages: Annotated[list, operator.add]
    step_count: Annotated[int, operator.add]


# 节点函数
def step_one(state: PersistenceDemoState) -> dict:
    print("执行步骤 1")
    return {
        "messages": ["执行了步骤 1"],
        "step_count": 1
    }


def step_two(state: PersistenceDemoState) -> dict:
    print("执行步骤 2")
    return {
        "messages": ["执行了步骤 2"],
        "step_count": 1
    }


def step_three(state: PersistenceDemoState) -> dict:
    print("执行步骤 3")
    return {
        "messages": ["执行了步骤 3"],
        "step_count": 1
    }


# 构建图
def create_graph():
    builder = StateGraph(PersistenceDemoState)

    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_node("step_three", step_three)

    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", "step_three")
    builder.add_edge("step_three", END)

    return builder


def main():
    print("=== LangGraph 1.0 内存持久化存储演示 ===\n")

    # 编译图并使用内存存储
    graph = create_graph()
    app = graph.compile(checkpointer=InMemorySaver())

    # 配置线程ID用于存储状态
    config = {"configurable": {"thread_id": "user_13811112222"}}

    print("1. 首次执行工作流:")
    result1 = app.invoke({
        "messages": ["开始执行"],
        "step_count": 0
    }, config)

    print(f"执行结果result: {result1}\n")

    print("2. 检查存储的状态:")
    saved_state = app.get_state(config)
    print(f"保存的状态: {saved_state.values}")
    print(f"下一个节点: {saved_state.next}\n")

    # 获取指定线程的完整执行历史（正序：从最早到最晚,第一步在栈底）
    history = app.get_state_history(config)
    for checkpoint in history:
        print("=" * 50)
        print(f"当前状态: {checkpoint.values}")

    print("=" * 80)
    print("3. 恢复执行工作流:")
    # 由于工作流已经完成，这里会直接返回最终结果
    result2 = app.invoke(None, config)
    print(f"恢复执行结果: {result2}\n")


if __name__ == "__main__":
    main()
```

### 9.2内存检查点：放入自带的内存中，关机会丢失（加入大模型LLM）

```python
import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ==========定义大模型 ==========
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    temperature=0.0,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义短期记忆使用内存（生产可以换 RedisSaver/PostgresSaver）
checkpointer = InMemorySaver()
agent = create_agent(model=llm,checkpointer=checkpointer)
# 多轮对话配置，同一 thread_id 即同一会话
config = {"configurable": {"thread_id": "user-001"}}

msg1 = agent.invoke({"messages": [("user", "你好，我叫张三，喜欢足球，60字内简洁回复")]}, config)
msg1["messages"][-1].pretty_print()

# 6. 第二轮（继续同一 thread）
msg2 = agent.invoke({"messages": [("user", "我叫什么？我喜欢做什么？")]}, config)
msg2["messages"][-1].pretty_print()
```

### 9.3 数据库检查点：放入创建好的数据库中，持久保存

```python
'''
SqlitePersistence.py
在底层，检查点功能由符合BaseCheckpointSaver接口的检查点对象提供支持。
LangGraph提供了多种检查点实现，所有这些实现都通过独立的、可安装的库来完成，数据库类型的有：
   langgraph-checkpoint-sqlite：使用SQLite数据库（SqliteSaver / AsyncSqliteSaver）存储检查点。
非常适合实验和本地工作流程。需要单独安装。
   langgraph-checkpoint-postgres：使用Postgres数据库（PostgresSaver / AsyncPostgresSaver）
存储检查点，用于LangSmith。非常适合在生产环境中使用。需要单独安装。
......

本次案例，安装sqlite所需依赖
pip install langgraph-checkpoint-sqlite

'''

import sqlite3
import operator
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph,START,END



class MyState(TypedDict):
    messages:Annotated[list,operator.add]

def node_1(state:MyState):

    return {"messages":["abc","def"]}

def main():
    # 数据存储到D:\\44目录下面，需要目录存在
    conn = sqlite3.connect(database="D:\\44\\sqlite_data.db",check_same_thread=False)
    sqliteDB = SqliteSaver(conn=conn)

    builder = StateGraph(MyState)
    builder.add_node("node_1",node_1)

    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)

    graph = builder.compile(checkpointer=sqliteDB)
    # 同一个用户id下，每次执行都会插入一次新数据，上课时记得修改用户编号或者直接删除D:\\44\\sqlite_data.db
    config = {"configurable": {"thread_id": "user-001"}}

    initial_state = graph.get_state(config)
    print(f"Initial state: {initial_state}")

    # 执行图
    result = graph.invoke({"messages":[]}, config)
    print(f"Result: {result}")

    print()
    print("====================查看执行后的状态====================")
    # 查看执行后的状态
    final_state = graph.get_state(config)
    print()
    print(f"Final state: {final_state}")

    conn.close()

if __name__ == '__main__':
    main()
```

## 10.高级特性之时间回溯(Time-Travel):即修改执行节点的其中一步的内容，并且顺着修改的这步继续往下执行



```python
"""
要在LangGraph中使用时间旅行：
（1）使用invoke或stream方法，以初始输入来运行图表。
（2）识别现有线程中的检查点：使用get_state_history方法检索特定thread_id的执行历史，
    并找到所需的checkpoint_id。然后，你可以找到截至该中断记录的最新检查点。
（3）更新图状态（可选）：使用update_state方法在检查点修改图的状态，并从替代状态恢复执行。
（4）从检查点恢复执行：使用invoke或stream方法，输入为None，配置中包含适当的thread_id和检查点ID

LangGraph 时间旅行演示

该演示展示了更复杂的时间旅行功能，包括：
1. 运行图并生成多个状态
2. 查看历史状态
3. 从不同历史点恢复执行
4. 比较不同执行路径的结果
"""

import uuid
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver


class StoryState(TypedDict):
    """故事状态定义"""
    character: NotRequired[str]  # character（角色/人物）
    setting: NotRequired[str]    # setting（场景/背景）
    plot: NotRequired[str]       # plot（情节/剧情）
    ending: NotRequired[str]     # ending（结局/结尾）


def create_character(state: StoryState):
    """
    创建故事角色
    Args:
        state: 当前状态
    Returns:
        dict: 更新后的状态
    """
    print("执行节点: create_character")

    # 模拟LLM调用
    mock_character = "一只会说话的猫"
    print(f"创建的角色: {mock_character}")
    return {"character": mock_character}


def set_setting(state: StoryState):
    """
    设置故事背景
    Args:
        state: 当前状态
    Returns:
        dict: 更新后的状态
    """
    print("执行节点: set_setting")

    # 模拟LLM调用
    mock_setting = "在一个神秘的图书馆里"
    print(f"设置的背景: {mock_setting}")
    return {"setting": mock_setting}


def develop_plot(state: StoryState):
    """
    发展故事情节
    Args:
        state: 当前状态
    Returns:
        dict: 更新后的状态
    """
    print("执行节点: develop_plot")

    # 模拟LLM调用
    character = state.get("character", "未知角色")
    setting = state.get("setting", "未知背景")
    mock_plot = f"{character}在{setting}发现了一本会发光的书"
    print(f"发展的剧情: {mock_plot}")
    return {"plot": mock_plot}


def write_ending(state: StoryState):
    """
    编写故事结局
    Args:
        state: 当前状态
    Returns:
        dict: 更新后的状态
    """
    print("执行节点: write_ending")

    # 模拟LLM调用
    plot = state.get("plot", "未知剧情")
    mock_ending = f"当{plot}时，整个图书馆都被魔法光芒照亮了"
    print(f"编写的结局: {mock_ending}")
    return {"ending": mock_ending}


def main():
    """主函数 - 演示高级时间旅行功能"""
    print("=== LangGraph 高级时间旅行演示 ===\n")

    # 构建工作流
    workflow = StateGraph(StoryState)

    # 添加节点
    workflow.add_node("create_character", create_character)
    workflow.add_node("set_setting", set_setting)
    workflow.add_node("develop_plot", develop_plot)
    workflow.add_node("write_ending", write_ending)

    # 添加边来连接节点
    workflow.add_edge(START, "create_character")
    workflow.add_edge("create_character", "set_setting")
    workflow.add_edge("set_setting", "develop_plot")
    workflow.add_edge("develop_plot", "write_ending")
    workflow.add_edge("write_ending", END)

    # 编译
    graph = workflow.compile(checkpointer=InMemorySaver())

    # 1. 运行图表生成第一个故事
    print("1. 生成第一个故事...")
    config1 = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    story1 = graph.invoke({}, config1)
    print(story1)
    # 2. 查看历史状态
    states1 = list(graph.get_state_history(config1))

    print("历史状态:")
    for i, state in enumerate(states1):
        print(f"  {i}. 下一步节点: {state.next}")
        if state.values:
            print(f"     状态值: {state.values}")
        print()

    # 3. 从中间状态恢复执行，创建第二个故事
    character_state = states1[2]  # 索引2对应create_character执行后的状态
    print(f"选中的状态: {character_state.next}")
    print(f"选中的状态值: {character_state.values}")
    # 更新状态，改变角色
    new_config = graph.update_state(
        character_state.config,
        values={"character": "一只会飞的龙"}
    )

    # 4. 从新检查点恢复执行

    story2 = graph.invoke(None, new_config)
    print(story2)

if __name__ == "__main__":
    main()
```

## 11.高级特性之子图：主图里面嵌入子图

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# ====================== 1. 定义不同结构的父子图状态 ======================
# 父图状态：仅包含用户查询和最终答案（与子图状态完全不同）
class ParentState(TypedDict):
    user_query: str  # 父图独有：用户输入的查询
    final_answer: str | None  # 父图独有：子图处理后的最终结果

# 子图状态：专注于分析逻辑（与父图状态无重叠字段）
class SubgraphState(TypedDict):
    analysis_input: str  # 子图独有：分析输入
    analysis_result: str  # 子图独有：分析结果
    intermediate_steps: list  # 子图独有：中间步骤（私有数据）


# ====================== 2. 定义子图核心逻辑 ======================
def subgraph_analysis_node(state: SubgraphState) -> SubgraphState:
    """子图核心节点：处理分析逻辑，生成结果"""
    # 模拟子图的分析过程
    query = state["analysis_input"]
    state["intermediate_steps"] = [f"解析查询：{query}", "执行分析逻辑", "生成结果"]
    state["analysis_result"] = f"针对「{query}」的分析结果：这是子图处理后的内容"
    return state


def build_subgraph() -> StateGraph:
    """构建并编译子图"""
    sub_builder = StateGraph(SubgraphState)
    sub_builder.add_node("subgraph_analysis_node", subgraph_analysis_node)
    sub_builder.add_edge(START, "subgraph_analysis_node")
    sub_builder.add_edge("subgraph_analysis_node", END)
    return sub_builder.compile()


# 提前编译子图（供父图代理节点调用）
compiled_subgraph = build_subgraph()


# ============ 3. 定义父图代理节点（核心：状态转换+调用子图）从节点调用图=======
def call_subgraph_proxy(state: ParentState) -> ParentState:

    # 步骤1：父图状态 → 子图输入（状态转换）,提取父图的user_query，转换为子图需要的analysis_input
    subgraph_input = {
        "analysis_input": state["user_query"],

    }
    # 步骤2：手动调用编译后的子图，手动调用子图（而非直接将子图作为父图节点）
    subgraph_response = compiled_subgraph.invoke(subgraph_input)

    # 步骤3：提取子图的analysis_result，赋值给父图的final_answer
    return {
        "user_query": state["user_query"],  # 保留父图原有字段
        "final_answer": subgraph_response["analysis_result"]
    }


def build_parent_graph() -> StateGraph:
    """构建并编译父图（添加代理节点，而非直接添加子图）"""
    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("call_subgraph_proxy", call_subgraph_proxy)
    parent_builder.add_edge(START, "call_subgraph_proxy")
    parent_builder.add_edge("call_subgraph_proxy", END)
    return parent_builder.compile()


# ====================== 4. 主方法 ======================
def main():
    """主函数：执行父图，验证跨图状态转换逻辑"""
    # 1. 构建父图
    parent_graph = build_parent_graph()

    # 2. 定义父图初始状态（仅包含user_query，符合父图状态结构）
    initial_state = {
        "user_query": "请分析Python中StateGraph的使用场景",
        "final_answer": None
    }

    # 3. 执行父图，实际而言父图调用了call_subgraph_proxy
    final_state = parent_graph.invoke(initial_state)

    # 4. 输出结果
    print("\n父图最终状态：", final_state)
    print("\n子图处理后的最终答案：", final_state["final_answer"])


if __name__ == "__main__":
    main()
```

## 12.A2A：AI架构的协议革命

### 12.1 supervisor（主管）模式：

```python
import os
import re
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

# 初始化：pip install langgraph-supervisor

# 1. 初始化大语言模型
def init_llm_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1,
        max_tokens=1024
    )


# 2. Tools（必须有 docstring）
def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班工具。根据出发机场和到达机场预订一张机票，并返回预订结果。"""
    return f"✅ 成功预订了从 {from_airport} 到 {to_airport} 的航班"


def book_hotel(hotel_name: str) -> str:
    """预订酒店工具。根据酒店名称完成酒店预订，并返回预订结果。"""
    return f"✅ 成功预订了 {hotel_name} 的住宿"


# 3. 子 Agent
flight_assistant = create_agent(
    model=init_llm_model(),
    tools=[book_flight],
    name="flight_assistant"
)

hotel_assistant = create_agent(
    model=init_llm_model(),
    tools=[book_hotel],
    name="hotel_assistant"
)

# 4. 创建 Supervisor，协调者主管
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=init_llm_model(),
    prompt=(
        "你是旅行预订系统的调度主管，负责协调航班预订和酒店预订。\n\n"
        "当用户提出航班和酒店预订请求时，你的工作流程是：\n"
        "1. 首先调用flight_assistant来预订航班\n"
        "2. 然后调用hotel_assistant来预订酒店\n"
        "3. 收到两个助手的结果后，汇总并向用户报告\n"
        "4. 完成后结束对话\n\n"
        "重要规则：\n"
        "- 每个助手只能调用一次\n"
        "- 不要重复任何内容\n"
        "- 不要输出任何英文\n"
        "- 所有通信都使用中文\n"
    )
).compile()


# 5. 消息过滤器，就是一个工具类，处理大模型返回的重复废话，直接用可以不看
def filter_messages(chunk: dict) -> str:
    """提取并过滤消息，只返回中文内容，去除重复和英文"""
    output = ""

    if isinstance(chunk, dict):
        for role, payload in chunk.items():
            if isinstance(payload, dict) and "messages" in payload:
                for msg in payload["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content.strip()

                        # 过滤英文系统消息
                        if (content and
                                not content.startswith("Successfully") and
                                not content.startswith("Transferring") and
                                "Successfully transferred" not in content and
                                "transferred back to" not in content and
                                not content.startswith("帮我预订从")):

                            # 只保留中文内容
                            chinese_content = re.sub(r'[^\u4e00-\u9fff，。！？：；""、\s\d✅]', '', content)
                            if chinese_content and len(chinese_content.strip()) > 5:
                                output += f"{role}: {chinese_content.strip()}\n"

    return output


# 6. 主程序
def main():

    # 1. 询问出发机场
    from_airport = input("1. 您的出发机场是哪里？: ").strip()
    while not from_airport:
        print("请输入有效的出发机场名称")
        from_airport = input("1. 您的出发机场是哪里？: ").strip()

    # 2. 询问到达机场
    to_airport = input("\n2. 您的到达机场是哪里？: ").strip()
    while not to_airport:
        print("请输入有效的到达机场名称")
        to_airport = input("2. 您的到达机场是哪里？: ").strip()

    # 3. 询问酒店名称
    hotel_name = input("\n3. 您要预订的酒店名称是什么？: ").strip()
    while not hotel_name:
        print("请输入有效的酒店名称")
        hotel_name = input("3. 您要预订的酒店名称是什么？: ").strip()

    # 构造更明确的用户请求
    user_request = (
        f"请帮我预订以下旅行安排：\n"
        f"1. 航班：从 {from_airport} 飞往 {to_airport}\n"
        f"2. 酒店：{hotel_name}\n"
        f"请完成这两个预订。"
    )

    print("\n" + "=" * 60)
    print("正在处理您的预订请求...")
    print("=" * 60)
    print()

    # 准备输入数据
    # 创建一个字典，包含一个messages键
    # messages是一个列表，包含一个消息字典
    #每个消息字典包含role（角色）和content（内容）字段
    input_data = {  "messages": [  {"role": "user","content": user_request}  ]  }

    # 使用流式处理

        # 创建一个空集合，用于记录已经打印过的消息内容，避免重复显示
    seen_contents = set()

    for chunk in supervisor.stream(input_data):
            # 调用filter_messages函数处理当前chunk，提取并过滤其中的消息
            filtered_output = filter_messages(chunk)
            # 如果filtered_output不为空（即有过滤后的消息内容）


            if filtered_output:
                # 将过滤后的输出按行分割成列表 strip() 去除首尾空白字符，split('\n') 按换行符分割
                lines = filtered_output.strip().split('\n')
                # 遍历每一行
                for line in lines:
                    # 检查该行是否非空且不在已见过内容的集合中
                    if line and line not in seen_contents:
                        print(line)
                        # 将该行内容添加到已见过集合中，确保不会重复打印
                        seen_contents.add(line)




# 7. 运行主程序
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except Exception as e:
        print(f"\n系统出现错误: {e}")
```

### 12.2 handoffs（交接）模式：

```python
import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command, Send


# ===============================
# 1. 初始化大语言模型
# ===============================
def init_llm_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.1,
        max_tokens=1024,
    )


model = init_llm_model()


# ===============================
# 2. 通用 Handoff 工具工厂
# ===============================
def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"移交给 {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str,"描述下一个 Agent 应该做什么，包括所有必要信息"],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {
            "role": "user",
            "content": task_description,
        }
        agent_input = {
            **state,
            "messages": [task_description_message],
        }

        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# ===============================
# 3. 业务工具（必须有 docstring）
# ===============================

def book_flight(from_airport: str, to_airport: str) -> str:
    """预订航班，根据出发地和目的地完成机票预订"""
    print(f"✅ 成功预订了从 {from_airport} 到 {to_airport} 的航班")
    return f"成功预订了从 {from_airport} 到 {to_airport} 的航班。"



def book_hotel(hotel_name: str) -> str:
    """预订酒店，根据酒店名称完成预订"""
    print(f"✅ 成功预订了 {hotel_name} 的住宿")
    return f"成功预订了 {hotel_name} 的住宿。"


# ===============================
# 4. Handoff 工具
# ===============================
transfer_to_flight_assistant = create_task_description_handoff_tool(
    agent_name="flight_assistant",
    description="将任务移交给航班预订助手",
)

transfer_to_hotel_assistant = create_task_description_handoff_tool(
    agent_name="hotel_assistant",
    description="将任务移交给酒店预订助手",
)


# ===============================
# 5. 定义 Agent（create_agent 新接口）
# create_agent 不再显式接收 prompt，而是：
# 通过 tool schema + tool 名称 + tool docstring
# 通过 graph 上下文（handoff 描述）
# 通过 MessagesState 历史消息
# ===============================
flight_assistant = create_agent(
    model=model,
    tools=[book_flight, transfer_to_hotel_assistant], # 包含移交工具
    name="flight_assistant",
)

hotel_assistant = create_agent(
    model=model,
    tools=[book_hotel, transfer_to_flight_assistant], # 包含移交工具
    name="hotel_assistant",
)


# ===============================
# 6. 构建多 Agent Graph
# ===============================
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .add_edge("flight_assistant",END)
    .compile()
)


# ===============================
# 7. 运行
# ===============================
if __name__ == "__main__":
    result = multi_agent_graph.invoke(
        {
            "messages": "帮我预订从北京到上海的航班，并预订如家酒店"

        }
    )

    for msg in result["messages"]:
        if msg.type in ("human", "ai"):
            print(msg.content)
```

