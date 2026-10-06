"""
智能客服（问答）的 Streamlit 前端。

提供聊天界面：用户输入问题，页面调用 RagService 的链进行流式回答，
并把多轮对话保存在 session_state 中用于界面回显。

启动方式（在 app 目录下）：
    streamlit run app_qa.py
"""

import time

import config_data as config
from rag import RagService

import streamlit as sl

# 页面标题与分隔线
sl.title("智能客服")
sl.divider()

# 底部聊天输入框，回车后返回用户输入（无输入时为 None）
prompt = sl.chat_input()

# 初始化聊天记录：首条为欢迎语，供页面渲染
if "message" not in sl.session_state:
    sl.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

# 缓存 RagService 实例。Streamlit 会重跑脚本，
# 缓存可避免每次交互都重新构建整条 RAG 链。
if "rag" not in sl.session_state:
    sl.session_state["rag"] = RagService()

# 回显历史消息（按角色渲染为用户/助手气泡）
for message in sl.session_state["message"]:
    sl.chat_message(message["role"]).write(message["content"])

if prompt:
    # 先把用户提问显示出来并记入会话状态
    sl.chat_message("user").write(prompt)
    sl.session_state["message"].append({"role": "user", "content": prompt})

    # 调用 RAG 链生成回答，stream 表示流式返回，逐字显示
    with sl.spinner("AI思考中......"):
        res_stream = sl.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
        # write_stream 会边接收边渲染，并返回拼接后的完整回答
        full_response = sl.chat_message("assistant").write_stream(res_stream)
        sl.session_state["message"].append({"role": "assistant", "content": full_response})
