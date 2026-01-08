import streamlit as st
import time

from src.graph import app  # 确保 app.py 在 src 目录

st.set_page_config(page_title="Axcelerate-Ops: AI Consultant", layout="wide", page_icon="🤖")

# 自定义 CSS 提升界面专业感
st.markdown("""
    <style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stStatusWidget { border-left: 5px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Axcelerate-Ops: 数字化劳动力咨询助手")
st.caption("基于 AWS Well-Architected 框架与多智能体协同（LangGraph + FAISS）")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏：Agent 实时监控器
with st.sidebar:
    st.header("🧠 Agent 思考流")
    thought_log = st.container()
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入处理
if prompt := st.chat_input("请询问关于 AWS 架构或 GenAI 安全的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # ✅ 初始化关键变量，确保所有路径下都有定义
        final_ans = ""
        last_node_retry_flag = False
        last_node_retry_count = 0

        # 初始状态设置
        max_limit = 1  # 根据你的需要设置上限
        inputs = {
            "question": prompt,
            "context": [],
            "steps": [],
            "retry_count": 0,
            "max_retries": max_limit,
            "needs_retry": False
        }

        # 运行图流
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    # 1. 实时更新思考流侧边栏
                    if "steps" in value and value["steps"]:
                        thought_log.info(f"📍 **{key.upper()}**: {value['steps'][-1]}")

                    # 2. 捕获重试状态，用于后续判断是否彻底失败
                    if "needs_retry" in value:
                        last_node_retry_flag = value["needs_retry"]
                    if "retry_count" in value:
                        last_node_retry_count = value["retry_count"]

                    # 3. 捕获生成的答案
                    if "answer" in value and value["answer"]:
                        final_ans = value["answer"]
                        response_placeholder.markdown(final_ans)

            # ✅ 循环结束后的“重写限制”逻辑判定
            # 情况 A：如果最终没有产生 answer，且最后一步标记仍需重写
            if not final_ans or (last_node_retry_flag and last_node_retry_count >= max_limit):
                final_ans = f"""抱歉，我尝试了 {max_limit + 1} 次检索优化，但目前的文档库中似乎没有与您提问直接相关的内容。
                \n\n**建议操作：**
                \n1. 检查提问是否包含错别字。
                \n2. 尝试使用更专业的 AWS 术语（如：'IAM 策略'、'Bedrock 成本'）。
                \n3. **请在下方重新输入您的问题。**"""
                response_placeholder.warning(final_ans)

        except Exception as e:
            final_ans = f"❌ **系统执行异常**: {str(e)}"
            response_placeholder.error(final_ans)

        # 将最终结果存入历史
        st.session_state.messages.append({"role": "assistant", "content": final_ans})


# def generate_graph_image(output_path="graph_structure.png"):
#     try:
#         # 获取图的结构
#         graph_data = app.get_graph().draw_mermaid_png()
#
#         # 将二进制流写入文件
#         with open(output_path, "wb") as f:
#             f.write(graph_data)
#         print(f"✅ 架构图已生成至: {output_path}")
#     except Exception as e:
#         print(f"❌ 生成图片失败（可能缺少 pygraphviz）: {e}")
#         print("💡 建议使用方法二生成 Mermaid 代码。")
#
#
# if __name__ == "__main__":
#     generate_graph_image()