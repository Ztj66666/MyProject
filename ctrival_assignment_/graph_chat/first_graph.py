import uuid

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from graph_chat.assistant import create_assistant_node, safe_tools, sensitive_tools, sensitive_tool_names

from graph_chat.draw_png import draw_graph
from graph_chat.state import State
from tools.flights_tools import fetch_user_flight_information
from tools.init_db import update_dates
from tools.tools_handler import create_tool_node_with_fallback, _print_event

#定义了一个流程图的构建对象
builder = StateGraph(State)

def get_user_info(state:State):
    '''
    获取用户航班信息并更新字典
    :param state: 当前状态的字典
    :return: 包含用户信息的新状态字典
    '''
    return {"user_info":fetch_user_flight_information.invoke({})}

builder.add_node('fetch_user_info',get_user_info)
builder.add_edge(START,'fetch_user_info')
builder.add_node('assistant',create_assistant_node())
builder.add_node('safe_tools',create_tool_node_with_fallback(safe_tools))
builder.add_node('sensitive_tools',create_tool_node_with_fallback(sensitive_tools))
builder.add_edge('fetch_user_info',"assistant")
def route_conditional_tools(state:State)->str:
    '''
    根据当前状态决定下一个要执行的节点
    :param state: 当前的状态
    :return: str下一个要执行节点的名字
    '''
    next_node = tools_condition(state)
    if next_node == END:
        return END
    ai_message = state['messages'][-1]
    tool_call = ai_message.tool_calls[0]
    if tool_call['name'] in sensitive_tool_names:
        return "sensitive_tools"
    else:
        return 'safe_tools'
builder.add_conditional_edges(
    "assistant",
    route_conditional_tools,
    ['safe_tools','sensitive_tools',END]
)
builder.add_edge('safe_tools','assistant')
builder.add_edge('sensitive_tools','assistant')
memory = MemorySaver()
graph = builder.compile(checkpointer=memory,interrupt_before=['sensitive_tools'],)

draw_graph(graph,'graph3.png')

session_id = str(uuid.uuid4())

update_dates()
config = {
    "configurable" : {
        "passenger_id" : "3443587242",
        "thread_id": session_id
    }
}

_printed = set()
while True:
    question = input("用户:")
    if question.lower() in ['q','exit','quit']:
        print("对话结束")
        break
    else:
        events = graph.stream({"messages":('user',question)},config,stream_mode='values')
        for event in events:
            _print_event(event,_printed)
        current_state = graph.get_state(config)
        # ... 第一次 stream 后的代码
        if current_state.next:
            # ⚠️ 修正 A: 必须使用 input() 获取用户输入
            user_response = input("您是否批准上面操作，输入y继续否则说明您的更改：\n")

            if user_response.strip().lower() == "y":
                # 继续执行
                events = graph.stream(None, config, stream_mode='values')
                for event in events:
                    _print_event(event, _printed)
            else:
                # 拒绝工具调用，并提供拒绝原因作为 ToolMessage 的 content
                # LangGraph 期望 ToolMessage 的 content 是工具的返回结果或拒绝信息。
                # 假设当前状态的下一个是 "tools"，并且是由于 'assistant' 节点返回了 tool_calls 导致的。

                # 提取第一个工具调用 ID
                tool_call_id = current_state.next[0][1][0].tool_calls[0]['id']

                # 准备 ToolMessage
                tool_message_input = {
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"Tool调用被拒绝。原因是'{user_response}'"
                        )
                    ]
                }

                new_events = graph.stream(tool_message_input, config, stream_mode='values')
                # ⚠️ 修正 B: 迭代新的 stream 结果
                for event in new_events:
                    _print_event(event, _printed)