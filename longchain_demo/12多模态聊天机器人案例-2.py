import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_core.messages import HumanMessage
import gradio as gr
from langchain_community.chat_message_histories import SQLChatMessageHistory
from openai import OpenAI

from env_utils import OPENAI_AUDIO_KEY, GROQ_GUDIO_KEY
from longchain_demo.my_llm import llm

prompt = ChatPromptTemplate.from_messages(
    [('system',"{system_message}"),
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
def summarized_messages(current_input):
    session_id = current_input['config']['configurable']['session_id']
    if not session_id:
        raise Exception('session_id is required')
    chat_history = get_session_history(session_id)
    stored_messages = chat_history.messages
    if len(stored_messages) <=2:
        return {
            "original_message": stored_messages,
            "summary_message": None,
        }
    last_two_messages = stored_messages[-2:]
    message_to_summarize = stored_messages[:-2]
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","请将以下历史对话压缩为一条保留关键信息的摘要信息"),
            ("placeholder","{chat_history}"),
            ("human","请生成包含上述核心对话内容的摘要，保留重要事实和决策"),

        ]
    )
    summarization_chain = summarization_prompt | llm
    summary_message = summarization_chain.invoke({'chat_history':message_to_summarize})
    return {
        "original_message" :last_two_messages,
        "summary_message":summary_message,
    }


final_chain = RunnablePassthrough.assign(message_summarized = summarized_messages) | RunnablePassthrough.assign(
    input = lambda x:x['input'],
    chat_history = lambda x:x['message_summarized']['original_message'],
    system_message =lambda x:f"你是一个乐于助人的助手，尽可能回答你能回答的问题。:摘要:{x['message_summarized']['summary_message'].content}" if x['message_summarized'].get('summary_message') else "无摘要"

) | chain_with_message_history

client = OpenAI(api_key=OPENAI_AUDIO_KEY)
GROQ_API_KEY = GROQ_GUDIO_KEY

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1" # 必须有 https://
)

def read_audio(audio_message):
    if audio_message:
        try:
            print("正在通过 Groq 转录...")
            with open(audio_message, "rb") as audio_file:
                # Groq 支持 whisper-large-v3，比 whisper-1 更强
                resp = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    temperature=0
                )
            return resp.text
        except Exception as e:
            print(f"报错了: {e}")
            return f"Error: {e}"
    return ''
def add_message(chat_history,user_message):
    if user_message:
        chat_history.append({'role':'user','content':user_message})
    return chat_history,''

def execute_chain(chat_history):
    input = chat_history[-1]
    result = final_chain.invoke({'input': input['content'], "config": {"configurable": {"session_id": "user123"}}},
                                config={"configurable": {"session_id": "user123"}})
    chat_history.append({"role":"assistant","content":result.content})
    return chat_history

with gr.Blocks(title="多模态机器人")as block:
    gr.Markdown("# 🤖 多模态聊天机器人")
    # 移除 type='messages'
    chatbot = gr.Chatbot(height=500, label='聊天机器人')
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder='请输入文本...',label="文字输入",max_lines=5)
            submit_butten = gr.Button('发送',variant="primary")
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=['microphone'],label='语音输入',type='filepath',format='wav')
    chat_msg = user_input.submit(add_message,[chatbot,user_input],[chatbot,user_input])
    chat_msg.then(execute_chain,chatbot,chatbot)

    audio_input.change(read_audio,[audio_input],[user_input])
    submit_butten.click(add_message,[chatbot,user_input],[chatbot,user_input]).then(execute_chain,chatbot,chatbot)
if __name__ == '__main__':
    block.launch()