from __future__ import annotations
import apd_policy_chatbot.llm as llm


def test_llm_answer_question_returns_dict(monkeypatch):
    monkeypatch.setattr(
        llm,
        "answer_question",
        lambda *_a, **_kw: {"answer": "stub answer", "pages": [1]},
    )

    result = llm.answer_question(None, "dummy")

    assert isinstance(result, dict)
    assert result["answer"] == "stub answer"
    assert result["pages"] == [1]
