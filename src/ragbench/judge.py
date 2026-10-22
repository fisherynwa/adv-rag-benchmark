"""The LLM-as-a-judge wrapper, with the fix for the failing judge calls."""
import numpy as np


def make_judge(llm):
    """Return judge(evaluate, field): the evaluator's verdict as a float, or NaN if the reply can't be read.

    Each LlamaIndex evaluate() call runs in its own short-lived event loop. Ollama's client would reuse a connection
    from the previous (closed) loop, so every second judge call failed. A fresh async client per call fixes that.
    """
    def judge(evaluate, field="passing"):
        if hasattr(llm, "_async_client"):
            llm._async_client = None
        try:
            value = getattr(evaluate(), field)
            return np.nan if value is None else float(value)
        except Exception:
            return np.nan
    return judge
