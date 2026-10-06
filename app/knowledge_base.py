"""
知识库写入服务。

负责把原始文本"加工"成向量库能存储的形态：
  1. 用 MD5 去重，避免同一份内容被重复入库；
  2. 长文本按配置切分成若干 chunk；
  3. 为每个 chunk 附加元数据（来源文件、时间、操作人等）；
  4. 调用 embedding 模型向量化并写入 Chroma。

读取/检索侧的逻辑在 vector_service.py，二者共用同一份 config 配置。
"""

import os
import hashlib
from datetime import datetime

import config_data as config

from langchain_dashscope import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def check_md5(md5_str: str):
    """
    检查某个 MD5 是否已经记录在去重文件中。

    文件不存在时会先创建空文件，再返回 False（视为"尚无记录"）。

    :param md5_str: 待检查的 MD5 十六进制字符串
    :return: True 表示已存在（应跳过入库），False 表示未见过
    """
    # 去重文件还不存在：创建空文件，说明知识库是空的
    if not os.path.exists(config.md5_text):
        open(config.md5_text, "w", encoding="utf-8").close()
        return False
    else:
        # 逐行比对，命中即已入库
        for line in open(config.md5_text, "r", encoding="utf-8").readlines():
            line = line.strip()
            if line == md5_str:
                return True

        return False


def save_md5(md5_str: str):
    """
    把新入库内容的 MD5 追加写入去重文件，供后续 check_md5 比对。

    :param md5_str: 已成功入库内容的 MD5
    """
    with open(config.md5_text, "a", encoding='utf-8') as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    """
    计算字符串的 MD5 十六进制摘要。

    :param input_str: 待计算的内容
    :param encoding:  编码方式，默认 utf-8
    :return: 32 位十六进制 MD5 字符串
    """
    # 先编码成字节串，hashlib 只能处理 bytes
    str_bytes = input_str.encode(encoding=encoding)

    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()

    return md5_hex


class KnowledgeBaseService(object):
    """
    知识库写入（上传）服务。

    构造时建立与 Chroma 的连接并准备切分器，之后可复用它多次上传。
    """

    def __init__(self):
        # 确保持久化目录存在，否则 Chroma 初始化可能失败
        os.makedirs(config.persist_directory, exist_ok=True)

        # 连接持久化向量库（写入端）。注意这里用的 embedding 模型
        # 必须与检索端（vector_service.py）保持一致。
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
            persist_directory=config.persist_directory,
        )

        # 递归字符切分器：按 config 中的分隔符优先级与长度参数切分文本
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,          # 每块目标长度
            chunk_overlap=config.chunk_overlap,    # 相邻块重叠长度
            separators=config.separators,          # 切分优先级
            length_function=len,                   # 以字符数衡量长度
        )

    def upload_by_str(self, data: str, filename):
        """
        把一段文本上传到知识库。

        流程：MD5 去重 -> 按需切分 -> 组织元数据 -> 写入向量库 -> 记录 MD5。

        :param data:     待入库的原始文本内容
        :param filename: 来源文件名，会写入元数据便于溯源
        :return: 面向用户的结果描述文本（跳过 / 成功）
        """
        # 1) 内容去重：同一份文本只入库一次
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在于知识库中"

        # 2) 长文本切分；短文本无需切分，直接整体入库
        if len(data) > config.max_spilt_char_number:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        # 3) 元数据：会随每个 chunk 一起存储，检索时可一并返回
        metadata = {
            "source": filename,                                         # 来源文件
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 入库时间
            "operator": "小草",                                          # 操作人
        }

        # 4) 写入向量库：每个 chunk 复用同一份元数据
        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 5) 记录本次内容的 MD5，供后续去重
        save_md5(md5_hex)

        return "[成功]内容成功上传向量库"


if __name__ == "__main__":
    # 手动调试：直接上传一段测试文本，验证写入链路是否正常。
    service = KnowledgeBaseService()
    result = service.upload_by_str("周杰伦222", "testfile")
    print(result)
