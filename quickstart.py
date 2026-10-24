import asyncio
from pathlib import Path

from chunk_graph import graph as chunker
from index_graph import graph as indexer
from retrieval_graph import graph as agent
from shared.utils import chat_client_session


async def main():
    
    config = {
        "configurable": {
            "retriever_provider": "chroma",
            "embedding_model": "BAAI/bge-m3",  # 或本地模型目录
            "embedding_device": "cpu",        # 有可用 CUDA 环境时可改为 cuda
            "chroma_path": "./chroma_db",
            "chroma_collection": "langchain_index",
            "search_kwargs": {"k": 8},
            "query_model": "qwen/qwen3.8-max",
            "response_model": "qwen/qwen3.8-max",
            "splitter_configuration": {
                "chunk_size": 512,
                "chunk_overlap": 64,
            },
        }
    }

    chunked = await chunker.ainvoke(
        {"source": str(Path("tests/fixtures/SRNO.pdf").resolve())},config
    )
    if not chunked["chunks"]:
        raise RuntimeError("未提取到有效文本，请检查 PDF 是否含可提取的文本层。")

    await indexer.ainvoke({"docs": chunked["chunks"]}, config)

    result = await agent.ainvoke(
        {"messages": [("user", "SRNO 如何将神经算子用于图像超分辨？")]}, config
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    async def run():
        async with chat_client_session():
            await main()

    asyncio.run(run())
