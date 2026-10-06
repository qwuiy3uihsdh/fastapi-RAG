"""
RAG（检索增强生成）核心链路。

把"向量检索"与"大模型对话"串成一条 LangChain 链：
  用户提问 -> 检索相关知识库片段 -> 拼装提示词 -> 大模型生成 -> 纯文本输出

同时通过 RunnableWithMessageHistory 接入本地对话历史，
实现带上下文的多轮问答。
"""

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_dashscope import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from file_history_store import get_history
from vector_service import VectorStoreService

import config_data as config


class RagService(object):
    """
    问答服务：封装提示词模板、大模型与整条 RAG 链。

    使用时实例化一次，反复调用 self.chain.stream/invoke 即可。
    """

    def __init__(self):
        # 向量检索服务（负责把问题变成相关文档片段）
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        # 提示词模板：由若干条消息构成
        #   system 段 1：角色约束 + 注入检索到的参考资料 {context}
        #   system 段 2：说明接下来会给出对话历史
        #   MessagesPlaceholder：占位，运行时填入历史消息
        #   user 段：用户本次提问 {input}
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )

        # 对话大模型。走 OpenAI 兼容接口，因此用 ChatOpenAI 客户端，
        # 但 base_url 指向阿里云百炼的兼容端点，密钥从环境变量读取。
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://ws-slyo0j61dbkx5gs6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        )

        # 构建完整链
        self.chain = self.__get_chain()

    def __get_chain(self):
        """
        组装并返回完整的 RAG 对话链。

        数据流（每一步的输出即下一步的输入）：

          {"input": 问题, "history": [...]}          <- 由 RunnableWithMessageHistory 注入
            │
            ├─ 分支 A "input":  RunnablePassthrough()  原样保留整个入参 dict
            └─ 分支 B "context": 取问题 -> 检索 -> 格式化文档
            ▼
          {"input": 入参dict, "context": "参考资料字符串"}
            ▼ format_for_prompt_template  拆包成模板需要的三个字段
          {"input": 问题字符串, "context": 参考资料, "history": 历史}
            ▼ prompt_template  套用提示词模板
            ▼ print_prompt     打印调试
            ▼ chat_model       调用大模型生成
            ▼ StrOutputParser  取出纯文本
        """
        retriever = self.vector_service.get_retriever()

        chain = (
            {
                # 并行执行两个分支，组成一个 dict 再往下传
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document,
            }
            | RunnableLambda(format_for_prompt_template)  # 整理成模板字段
            | self.prompt_template                        # 生成提示词
            | RunnableLambda(print_prompt)                # 调试打印
            | self.chat_model                             # 大模型生成
            | StrOutputParser()                           # 抽取出纯文本
        )

        def format_document(docs: list[Document]):
            """
            把检索到的文档列表拼成一段可读文本，供提示词中的 {context} 使用。

            :param docs: retriever 返回的 Document 列表
            :return: 拼接后的参考资料字符串；无结果时返回固定提示语
            """
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for doc in docs:
                # 同时保留正文与元数据，便于模型引用来源
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"

            return formatted_str

        def format_for_retriever(value: dict) -> str:
            """
            检索分支的入参适配：从整条链的输入 dict 中取出用户的原始问题。

            :param value: {"input": 问题, "history": [...]}
            :return: 用于检索的查询字符串
            """
            return value["input"]

        def format_for_prompt_template(value: dict) -> dict:
            """
            把并行分支的产物整理成提示词模板所需的字段。

            :param value: {"input": 原始入参dict, "context": 参考资料}
            :return: {"input": 问题, "context": 参考资料, "history": 历史消息}
            """
            new_value = {}
            # "input" 分支保留的是整个入参 dict，这里再取一层拿到问题字符串
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        def print_prompt(prompt):
            """
            调试用的中间节点：在链中打印最终发给大模型的提示词。

            接收提示词对象并原样返回，因此插入链中不会影响数据流。

            :param prompt: ChatPromptValue，可调用 to_string() 查看完整文本
            :return: 原样返回的 prompt
            """
            print("=" * 20)
            print(prompt.to_string())
            print("=" * 20)

            return prompt

        # 用管道符 | 依次串联各节点

        # 包一层历史记录能力：
        #   get_history 按 session_id 找到对应历史（见 file_history_store.py），
        #   input_messages_key 指定入参中哪个字段是用户消息，
        #   history_messages_key 指定占位符名，要与此前模板里的 "history" 一致。
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == "__main__":
    # 手动调试：就某个问题发起一次多轮问答（会读取/写入 user_001 的历史文件）。
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "再跟我确认一遍，我的身高是多少"}, session_config)
    print(res)
