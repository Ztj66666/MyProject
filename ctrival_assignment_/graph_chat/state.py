from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """
    对话状态栈的更新函数（Reducer）。
    用于管理用户当前处于哪一个子任务流程中。

    :param left: 当前的状态列表（现有状态栈）
    :param right: 动作指令。
                  - None: 保持现状
                  - 'pop': 任务结束，返回上一层
                  - 字符串: 进入新的子任务流程
    :return: 更新后的状态列表
    """
    if right is None:
        return left
    if right == 'pop':
        # 弹出栈顶元素，回到上一个对话上下文
        return left[:-1]
    # 将新状态压入栈顶，例如从 'assistant' 进入 'book_hotel'
    return left + [right]


class State(TypedDict):
    """
    LangGraph 图的状态定义。
    定义了在节点之间传递的所有数据结构。
    """

    # 聊天消息列表。Annotated 与 add_messages 结合，
    # 确保新消息是追加(append)到列表中，而不是替换整个列表。
    messages: Annotated[list[AnyMessage], add_messages]

    # 存储基础用户信息（通常是字符串形式，如从数据库检索到的 JSON）
    user_info: str

    # 对话堆栈，记录用户当前在哪个功能模块中。
    # Annotated 绑定了上面的 update_dialog_stack 函数，
    # 使得每次对 dialog_state 赋值时，都会调用该函数进行逻辑处理。
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",  # 默认主助手
                "update_flight",  # 修改航班流程
                "book_car_rental",  # 租车流程
                "book_hotel",  # 订酒店流程
                "book_excursion"  # 订旅行项目流程
            ]
        ],
        update_dialog_stack,  # 指定更新逻辑
    ]