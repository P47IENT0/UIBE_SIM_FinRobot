import streamlit as st
from localApp import Chat_Bot
import base64
# Streamed response emulator
def response_generator(user_query, chat_bot):
    response = chat_bot.chat(user_query)
    # 首先检查是否包含final answer
    final_answer_start = response.find("Final Answer: ")
    if final_answer_start != -1:
        full_response = response[final_answer_start + len("Final Answer: "):].strip()
    else:
        full_response = response

    # 然后检查full_response是否以错误信息开头
    if full_response.startswith("执行出错:could not parse Llm output;"):
        # 如果以错误信息开头，去掉错误信息
        full_response = full_response[len("执行出错:could not parse Llm output;"):].strip()
        
    if full_response.startswith("执行出错: Could not parse LLM output: `"):
        # 如果以错误信息开头，去掉错误信息
        full_response = full_response[len("执行出错: Could not parse LLM output: `"):].strip()
    
    yield full_response
def sidebar_bg(side_bg):
    side_bg_ext = 'jpg'  # 根据实际图片格式，这里假设为jpg，可以改为png等
    with open(side_bg, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid='stSidebarContent'] {{
            background: url(data:image/{side_bg_ext};base64,{encoded_string});
            background-repeat: no-repeat;
            background-size: cover;
        }}
        
        </style>
        """,
        unsafe_allow_html=True,
    )

def app():
    # 设置页面标题
    st.markdown("""
    <style>
        .title {{
            color: #ff6b6b; 
            font-size: 50px;
            text-align: center;
            margin-top: 20px;
        }}
        .stButton>button:hover {{
            background-color: #8e44ad; 
            border-color: #6a0dad;
            transform: scale(1.05);
        }}
      
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="title">🎄GuardBot—用代码编织保险知识</h1>', unsafe_allow_html=True)
    
    # 调用sidebar_bg函数设置侧边栏背景
    sidebar_bg('/home/student/zlh/GuardBot-main/class_assistant/pic/R-C (1).jpg')  

    # 科目选择框
    subjects = ["保险法", "保险会计学", "风险管理与理财规划", "风险评估和风险融资", 
                "理财规划与保险", "利息理论", "企业年金与补充养老保险", "社会保险", "生存模型"]
    if "chat_bot" not in st.session_state:
        selected_subject = st.selectbox("🎁 请选择科目", subjects)
        if st.button("确认选择"):
            st.session_state.chat_bot = Chat_Bot(subject=selected_subject)
            st.session_state.messages = []  # 初始化消息记录
            st.success(f"已选择科目：{selected_subject}，GuardBot 初始化完成！")
    else:
        st.sidebar.markdown(f"当前科目：**{st.session_state.chat_bot.subject}**")
    
    # 只有在 Chat_Bot 初始化完成后才能聊天
    if "chat_bot" in st.session_state:
        # 显示聊天记录
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 接受用户输入
        if user_query := st.chat_input("💬 请输入..."):
            # 显示用户消息
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # 显示 Chat_Bot 响应
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                for partial_response in response_generator(user_query, st.session_state.chat_bot):
                    response_placeholder.markdown(partial_response)  # 直接显示final answer或错误信息后的内容
                    
                st.session_state.messages.append({"role": "assistant", "content": partial_response})
    else:
        st.info("请选择一个科目唤醒GuardBot后开始聊天。")
    
    # 侧边栏内容
    with st.sidebar:
        
        if st.button('🎁 Christmas Gift'):
            st.balloons()
        gif_path = '/home/student/zlh/GuardBot-main/class_assistant/pic/d3hfZm10PWdpZg==.gif'
        st.image(gif_path, width=100)
 
if __name__ == "__main__":
    app()