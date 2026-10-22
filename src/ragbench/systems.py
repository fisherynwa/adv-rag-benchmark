"""The RAG systems compared in the benchmark, built from ready-made LlamaIndex parts."""
from llama_index.core import PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import (HierarchicalNodeParser, SentenceSplitter, SentenceWindowNodeParser,
                                          get_leaf_nodes)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever

TOP_K = 6
QA_PROMPT = PromptTemplate(
    "Context:\n{context_str}\n\nAnswer the question briefly, using only the context above. "
    "If the answer is not in the context, say that the report does not say.\nQuestion: {query_str}\nAnswer: ")


def basic_rag(document, llm, embed, chunk_size=256, chunk_overlap=30, top_k=TOP_K):
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents([document])
    index = VectorStoreIndex(nodes, embed_model=embed)
    return index.as_query_engine(llm=llm, similarity_top_k=top_k, text_qa_template=QA_PROMPT)


def sentence_window(document, llm, embed, window_size=3, top_k=TOP_K):
    nodes = SentenceWindowNodeParser.from_defaults(window_size=window_size).get_nodes_from_documents([document])
    index = VectorStoreIndex(nodes, embed_model=embed)          # each node is one sentence
    return index.as_query_engine(llm=llm, similarity_top_k=top_k, text_qa_template=QA_PROMPT,
                                 node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")])


def auto_merging(document, llm, embed, chunk_sizes=(2048, 512, 128), top_k=TOP_K):
    nodes = HierarchicalNodeParser.from_defaults(chunk_sizes=list(chunk_sizes)).get_nodes_from_documents([document])
    storage = StorageContext.from_defaults()
    storage.docstore.add_documents(nodes)
    index = VectorStoreIndex(get_leaf_nodes(nodes), storage_context=storage, embed_model=embed)
    retriever = AutoMergingRetriever(index.as_retriever(similarity_top_k=top_k), storage)   # merges at 0.5
    return RetrieverQueryEngine.from_args(retriever, llm=llm, text_qa_template=QA_PROMPT)


def build_systems(document, llm, embed):
    """Name -> function that builds the query engine (None = full context: the whole report in the prompt)."""
    return {
        "basic RAG": lambda: basic_rag(document, llm, embed),
        "sentence window 1": lambda: sentence_window(document, llm, embed, 1),
        "sentence window 3": lambda: sentence_window(document, llm, embed, 3),
        "sentence window 5": lambda: sentence_window(document, llm, embed, 5),
        "auto-merging 2048/512/128": lambda: auto_merging(document, llm, embed),
        "full context": None,
    }
