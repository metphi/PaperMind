import asyncio
from langchain_core.runnables import RunnableConfig





config = RunnableConfig(
    configurable = {
        'retriever_provider': 'chroma',
        'loader_provider': 'pdf',
        'splitter_provider': 'recursive',
        'chroma_collection': 'test_collection',
        'loader_configuration':{},
        'splitter_configuration':{
            'chunk_size': 512,
            'chunk_overlap': 64
        }
    }
)

pdf_path = "tests/fixtures/LIIF.pdf"

async def main():
    from chunk_graph import graph as chunk_graph
    from index_graph import graph as index_graph
    chunker_state = {"source": pdf_path}
    chunker_result = await chunk_graph.ainvoke(chunker_state, config)
    print(f"Chunker result: {len(chunker_result.get('chunks', []))} chunks")
    print(len(chunker_result["chunks"]))
    # indexer_state = {"docs": chunker_result.get("chunks", [])}
    
    
    
if __name__ == "__main__":
    asyncio.run(main())