"""Every system builds and answers, offline, with LlamaIndex's mock models."""
import zlib

import pytest
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM

from ragbench.export import export_sentence_window, load_engine
from ragbench.systems import TOP_K, build_systems

SENTENCES = [f"Sentence number {i} says the value is {i * 10}." for i in range(60)]


class WordHashEmbedding(BaseEmbedding):
    """Deterministic offline embedding: counts of hashed words, so texts sharing words are similar."""
    def _vector(self, text):
        v = [0.0] * 512
        for word in text.lower().split():
            v[zlib.crc32(word.strip(".?,").encode()) % 512] += 1.0
        return v
    def _get_query_embedding(self, query): return self._vector(query)
    def _get_text_embedding(self, text): return self._vector(text)
    async def _aget_query_embedding(self, query): return self._vector(query)


@pytest.fixture(scope="module")
def setup():
    return Document(text=" ".join(SENTENCES), id_="toy"), MockLLM(max_tokens=8), WordHashEmbedding()


def test_all_systems_are_defined(setup):
    systems = build_systems(*setup)
    assert list(systems) == ["basic RAG", "sentence window 1", "sentence window 3", "sentence window 5",
                             "auto-merging 2048/512/128", "full context"]
    assert systems["full context"] is None


@pytest.mark.parametrize("name", ["basic RAG", "sentence window 1", "sentence window 3", "sentence window 5",
                                  "auto-merging 2048/512/128"])
def test_system_answers_with_sources(setup, name):
    response = build_systems(*setup)[name]().query("What does sentence 7 say?")
    assert isinstance(str(response), str)
    assert 0 < len(response.source_nodes) <= TOP_K


@pytest.mark.parametrize("window", [1, 3])
def test_sentence_window_replaces_the_sentence_with_its_window(setup, window):
    response = build_systems(*setup)[f"sentence window {window}"]().query("What does sentence 7 say?")
    for hit in response.source_nodes:
        text = hit.node.get_content()
        assert hit.node.metadata["original_text"].strip() in text
        assert sum(s in text for s in SENTENCES) <= 2 * window + 1


def test_export_and_load_give_the_same_retrieval(setup, tmp_path):
    document, llm, embed = setup
    n_nodes = export_sentence_window(document, embed, tmp_path, window_size=5, extra_config={"llm": "mock"})
    assert n_nodes == len(SENTENCES)
    original = build_systems(document, llm, embed)["sentence window 5"]().query("value is 370")
    reloaded = load_engine(tmp_path, llm, embed).query("value is 370")
    best = lambda response: response.source_nodes[0]
    assert best(original).node.get_content() == best(reloaded).node.get_content()   # the same best match
    assert best(original).score == best(reloaded).score                             # the same embeddings


def test_retrieval_finds_the_matching_sentence(setup):
    response = build_systems(*setup)["sentence window 1"]().query("value is 370")
    assert any("value is 370." in n.node.metadata["original_text"] for n in response.source_nodes)
