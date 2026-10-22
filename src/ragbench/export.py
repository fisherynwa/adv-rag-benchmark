"""Save the best system (index + settings) and load it again, e.g. for production or the Colab test-retest."""
import json
import os

from llama_index.core import PromptTemplate, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from .systems import QA_PROMPT, TOP_K


def export_sentence_window(document, embed, out_dir, window_size, extra_config=None):
    """Build and save a sentence-window index plus config.json. Returns the number of nodes."""
    nodes = SentenceWindowNodeParser.from_defaults(window_size=window_size).get_nodes_from_documents([document])
    index = VectorStoreIndex(nodes, embed_model=embed)
    os.makedirs(out_dir, exist_ok=True)
    index.storage_context.persist(os.path.join(out_dir, "index"))
    config = {"system": f"sentence window {window_size}", "kind": "sentence_window", "window_size": window_size,
              "top_k": TOP_K, "qa_prompt": QA_PROMPT.template, **(extra_config or {})}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return len(nodes)


def load_engine(out_dir, llm, embed):
    """Rebuild the query engine from a saved sentence-window export."""
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg["kind"] != "sentence_window":
        raise ValueError(f"unsupported kind: {cfg['kind']}")
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=os.path.join(out_dir, "index")),
                                    embed_model=embed)
    return index.as_query_engine(llm=llm, similarity_top_k=cfg["top_k"],
                                 text_qa_template=PromptTemplate(cfg["qa_prompt"]),
                                 node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")])