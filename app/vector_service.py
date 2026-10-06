"""
向量检索服务。

本模块负责连接 Chroma 向量库，并把"向量库"包装成 LangChain 的
Retriever（检索器），供 RAG 链在回答前召回相关文档片段。
"""

from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    """
    对 Chroma 向量库的轻量封装。

    只做两件事：
      1. 用指定的 embedding 模型连接（或创建）持久化的向量库；
      2. 暴露一个 retriever，供链式调用按相似度检索。
    """

    def __init__(self, embedding):
        """
        :param embedding: 向量化模型实例（如 DashScopeEmbeddings），
                          必须与写入向量库时使用的模型一致，否则向量空间不匹配。
        """
        self.embedding = embedding

        # 连接持久化向量库。目录/集合不存在时，Chroma 会自动创建。
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """
        返回一个检索器对象。

        as_retriever 会把向量库包装成标准 Retriever 接口，
        调用 retriever.invoke(query) 即可返回与 query 最相似的文档列表。
        search_kwargs 中的 k 由 config.similarity_threshold 控制召回条数。

        :return: LangChain Retriever
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == "__main__":
    # 手动调试：直接用一句话检索，打印召回的文档片段，便于验证知识库效果。
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v3")).get_retriever()

    res = retriever.invoke("我的体重180斤，尺码推荐")
    print(res)
