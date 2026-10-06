"""
知识库更新服务的 Streamlit 前端。

提供一个网页界面：用户上传 .txt 文件，页面读取内容后调用
KnowledgeBaseService 写入向量库，并展示入库结果。

启动方式（在 app 目录下）：
    streamlit run app_file_uploader.py
"""

import time

import streamlit
from knowledge_base import KnowledgeBaseService

# 页面标题
streamlit.title("知识库更新服务")

# 文件上传控件：仅接受单个 .txt 文件，返回上传对象或 None
uplaod_file = streamlit.file_uploader(
    label="请上传你的文本文件",
    type=['txt'],
    accept_multiple_files=False,
)

# Streamlit 每次交互都会重跑整个脚本，用 session_state 缓存服务实例，
# 避免每次都重建 Chroma 连接。
if "service" not in streamlit.session_state:
    streamlit.session_state["service"] = KnowledgeBaseService()

if uplaod_file is not None:
    # 读取上传文件的基本信息用于展示
    file_name = uplaod_file.name
    file_size = uplaod_file.size / 1024          # 字节转 KB
    file_type = uplaod_file.type

    streamlit.subheader(f"文件名：{file_name}")
    streamlit.write(f"文件格式：{file_type}|文件大小：{file_size:.2f} KB")

    # 读取文件内容并按 UTF-8 解码为字符串
    text = uplaod_file.getvalue().decode("utf-8")

    # 显示加载动画，并在其中执行入库（可能较慢）
    with streamlit.spinner("载入知识库中......"):
        time.sleep(1)  # 稍作停顿，让加载动画可见
        result = streamlit.session_state["service"].upload_by_str(text, file_name)
        streamlit.write(result)
