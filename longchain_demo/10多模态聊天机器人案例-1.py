from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.stores import InMemoryStore
from langchain_core.messages import HumanMessage

from langchain_community.chat_message_histories import SQLChatMessageHistory
from longchain_demo.my_llm import llm

prompt = ChatPromptTemplate.from_messages(
    [('system','你是一个乐于助人的ai机器人助手，尽力所能回答所有的问题，提供的聊天历史包含与你对话用户的相关信息'),
    MessagesPlaceholder(variable_name='chat_history',optional=True),
    ('human','{input}')]
)

chain = prompt | llm


def get_session_history(session_id:str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string= 'sqlite:///chat.db',
    )

chain_with_message_history =RunnableWithMessageHistory(chain,
                                                       get_session_history,
                                                       input_messages_key='input',
                                                       history_messages_key='chat_history',
                                       )

result1 = chain_with_message_history.invoke({'input':'你好我是变形金刚'},config={"configurable":{"session_id":"user123"}})
print(result1)
result2 = chain_with_message_history.invoke({'input':'我是什么'},config={"configurable":{"session_id":"user123"}})
print(result2)
