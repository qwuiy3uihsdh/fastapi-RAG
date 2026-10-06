"""
基于本地文件的对话历史存储。

LangChain 需要一份"按 session_id 存取聊天记录"的实现，才能让
RunnableWithMessageHistory 记住多轮对话。这里用最简单的 JSON 文件
落盘方案：每个会话对应 chat_history 目录下的一个文件。
"""

import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages import message_to_dict
from langchain_core.messages import messages_from_dict


def get_history(session_id):
    """
    RunnableWithMessageHistory 的工厂函数。

    链在每次调用时会拿着 config 中的 session_id 调用它，
    从而拿到"该会话专属"的历史记录对象。

    :param session_id: 会话标识，来自 config["configurable"]["session_id"]
    :return: FileChatMessageHistory 实例
    """
    return FileChatMessageHistory(session_id=session_id, storage_path="./chat_history")


class FileChatMessageHistory(BaseChatMessageHistory):
    """
    把聊天历史以 JSON 形式存到本地文件的实现。

    继承 BaseChatMessageHistory 后，需要实现三个成员：
      - messages 属性：读取全部历史消息
      - add_messages：追加新消息
      - clear：清空历史
    文件内容为消息字典（message_to_dict 的产物）组成的 JSON 数组。
    """

    def __init__(self, session_id, storage_path):
        """
        :param session_id:   会话标识，用作文件名
        :param storage_path: 历史文件所在目录
        """
        self.session_id = session_id
        self.storage_path = storage_path

        # 该会话对应的历史文件完整路径，例如 ./chat_history/user_001
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保目录存在（首次运行时 chat_history 可能还不存在）
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, message: Sequence[BaseMessage]) -> None:
        """
        向历史中追加一批消息并整体覆写文件。

        注意这里是"读全量 -> 拼接 -> 重写"，而不是追加写文件，
        因为 JSON 数组必须保持格式完整，不能简单地 append。

        :param message: 待追加的消息序列（用户提问、模型回答等）
        """
        all_messages = list(self.messages)      # 先取出已有历史
        all_messages.extend(message)            # 再拼接本次新增的消息

        # 把所有消息对象转换成可 JSON 序列化的字典
        new_messages = [message_to_dict(message) for message in all_messages]

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    def messages(self) -> list[BaseMessage]:
        """
        读取当前会话的全部历史消息。

        以 property 形式提供，调用方直接写 history.messages 即可。
        文件不存在（首次会话）时返回空列表，而不是抛异常。

        :return: BaseMessage 列表，按时间先后顺序排列
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                # 把 JSON 字典还原成 LangChain 的消息对象
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            # 该会话还没有任何历史记录
            return []

    def clear(self) -> None:
        """
        清空当前会话的历史记录（把文件写成空 JSON 数组）。
        """
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
