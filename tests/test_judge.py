"""The judge wrapper, including a regression test for the bug that left every second judge call unscored."""
import asyncio
import types

import numpy as np
import pytest

from ragbench.judge import make_judge


def test_verdict_becomes_a_float():
    judge = make_judge(llm=object())
    assert judge(lambda: types.SimpleNamespace(passing=True)) == 1.0
    assert judge(lambda: types.SimpleNamespace(score=0.75), "score") == 0.75


def test_unreadable_reply_is_missing_not_wrong():
    judge = make_judge(llm=object())
    assert np.isnan(judge(lambda: types.SimpleNamespace(passing=None)))


def test_judge_error_is_missing_not_a_crash():
    def broken():
        raise RuntimeError("Event loop is closed")
    assert np.isnan(make_judge(llm=object())(broken))


def test_fresh_async_client_before_each_call():
    llm = types.SimpleNamespace(_async_client="old connection")
    make_judge(llm)(lambda: types.SimpleNamespace(passing=True))
    assert llm._async_client is None


def _judge_calls(base_url, fix):
    """Run the benchmark's judge calls, in its order, the way they run in a Jupyter notebook."""
    import nest_asyncio
    from llama_index.core.evaluation import (AnswerRelevancyEvaluator, ContextRelevancyEvaluator,
                                             CorrectnessEvaluator, FaithfulnessEvaluator)
    from llama_index.llms.ollama import Ollama

    llm = Ollama(model="fake", base_url=base_url, request_timeout=30, thinking=False, context_window=24576)
    judge = make_judge(llm) if fix else make_judge(object())     # object() has no client to reset
    correctness, faithfulness = CorrectnessEvaluator(llm=llm), FaithfulnessEvaluator(llm=llm)
    answer_relevancy, context_relevancy = AnswerRelevancyEvaluator(llm=llm), ContextRelevancyEvaluator(llm=llm)
    q, a, ref, ctx = "Design life of the T3?", "15 years.", "15 years.", ["It has a design life of 15 years."]
    calls = [
        (lambda: correctness.evaluate(query=q, response=a, reference=ref), "passing"),
        (lambda: faithfulness.evaluate(response=a, contexts=ctx), "passing"),
        (lambda: answer_relevancy.evaluate(query=q, response=a), "score"),
        (lambda: context_relevancy.evaluate(query=q, contexts=ctx), "score"),
    ] * 2 + [(lambda: correctness.evaluate(query=q, response=a, reference=ref), "passing")] * 4

    async def notebook():                    # a notebook cell runs inside an already running event loop
        return [judge(run, field) for run, field in calls]
    nest_asyncio.apply()
    return asyncio.run(notebook())


def test_every_judge_call_succeeds_with_the_fix(fake_ollama_url):
    verdicts = _judge_calls(fake_ollama_url, fix=True)
    assert not np.isnan(verdicts).any(), verdicts


@pytest.mark.xfail(strict=False, reason="documents the original bug; XPASS means the libraries fixed it upstream")
def test_judge_calls_fail_without_the_fix(fake_ollama_url):
    verdicts = _judge_calls(fake_ollama_url, fix=False)
    assert not np.isnan(verdicts).any()
