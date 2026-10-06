"""
FastAPI 应用入口。

目前只是一个最小示例，提供根路径和健康检查两个接口。
RAG 相关功能实际由 app/ 下的 Streamlit 页面承载。
"""

from fastapi import FastAPI

# 创建 FastAPI 应用实例，title/version 会展示在自动生成的 /docs 文档中
app = FastAPI(title="FastAPI Demo", version="0.1.0")


@app.get("/")
async def root():
    """根路径：返回一条简单的问候信息，用于快速确认服务已启动。"""
    return {"msg": "hello"}


@app.get("/health")
async def health():
    """健康检查接口：供探针/负载均衡判断服务是否存活。"""
    return {"status": "ok"}
