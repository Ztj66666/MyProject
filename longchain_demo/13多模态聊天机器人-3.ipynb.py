import base64
import io
import uuid
import gradio as gr
from PIL import Image
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from longchain_demo.my_llm import llm

prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"你是一个多模态ai助手，可以处理文本，音频和图像输入"),
        MessagesPlaceholder(variable_name="message")
    ]
)
chain = prompt | llm

def get_session_history(session_id:str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string= 'sqlite:///muti_chat.db',
    )

chain_with_message_history =RunnableWithMessageHistory(chain,
                                                       get_session_history,
                                                       input_messages_key='input',
                                                       history_messages_key='chat_history',
                                       )

chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
)

config = {"configurable":{"session_id":str(uuid.uuid4())}}

def transcribe_audio(audio_path):
    """使用Base64处理语音转为"""

    # 目前多模态大模型： 支持两个传参方式，1、base64（字符串）（本地）。2、网络访问的url地址（外网的服务器上） http://sxxxx.com/11.mp3
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_message = {  # 把音频文件，封装成一条消息
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_data}",
                "duration": 30  # 单位：秒（帮助模型优化处理）
            }
        }

        return audio_message
    except Exception as e:
        print(e)
        return {}


def transcribe_image(image_path):
    """
    将任意格式的图片转换为base64编码的data URL
    :param image_path: 图片路径
    :return: 包含base64编码的字典
    """
    with Image.open(image_path) as img:
        # 获取原始图片格式（如JPEG/PNG）
        img_format = img.format if img.format else 'JPEG'

        buffered = io.BytesIO()
        # 保留原始格式（避免JPEG强制转换导致透明通道丢失）
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format.lower()};base64,{image_data}",
                "detail": 'low'
            }
        }

def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置,并返回后面的所有user消息"""
    if not history:
        return None
    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后查找第一个user
        return history[last_assistant_idx + 1:]

def add_message(history,messages):
    for m in messages['files']:
        print(m)
        history.append({"role":"user","content":{'path':m}})
    if messages['text'] is not None:
        print(messages['text'])
        history.append({"role":"user","content":messages['text']})
    return history,''

def submit_messages(history):
    print(history[0]['content'])
    user_messages = get_last_user_after_assistant(history)
    print(user_messages)
    content=[]
    if user_messages:
        for m in user_messages:
            print(m['content'][0]['type'])
            if (m['content'][0]['type']=='text'):
                print("字符串")
                content.append({'type':"text",'text':m['content']})
            elif (m['content'][0]['type']=='file'):
                file_path = m['content'][0]
                if file_path.endswith(".wav"):
                    print("音频")
                    file_message = transcribe_audio(file_path)
                elif file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
                    file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                print(m)
                pass
    input_message = HumanMessage(content=content)
    resp = chain_history.invoke({'message':input_message},config=config)
    history.append({"role":"assistant","content":resp.content})

with gr.Blocks(title="多模态机器人")as block:
    gr.Markdown("# 🤖 多模态聊天机器人")
    # 移除 type='messages'
    chatbot = gr.Chatbot(height=500, label='聊天机器人')
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_types=['image','.wav','.mp4'],
        file_count="multiple",
        placeholder="请输入信息或者上传文件",
        show_label=False,
        sources=["microphone","upload"]
    )
    chat_input.submit(
        add_message,
        [chatbot,chat_input],
        [chatbot,chat_input]
    ).then(
        submit_messages,
        [chatbot],
        [chatbot]
    ).then(
        lambda : gr.MultimodalTextbox(interactive=True,),
        None,
        [chat_input]
    )
if __name__ == '__main__':
    block.launch()
