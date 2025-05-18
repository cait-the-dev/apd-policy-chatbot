from __future__ import annotations

import pytest

import apd_policy_chatbot.chunking as ch

def test_sentence_split_recognises_end_punctuation():
    text = "One. Two!  Three?"
    assert ch.sentence_split(text) == ["One.", "Two!", "Three?"]


def test_sentence_split_ignores_multiple_spaces():
    text = "Hello   world.  Next sentence."
    assert ch.sentence_split(text) == ["Hello   world.", "Next sentence."]

@pytest.mark.parametrize("sent_multiplier", [1, 2, 3])
def test_chunk_page_respects_size(sent_multiplier):
    """Chunk length never exceeds CHUNK_CHAR_SIZE."""
    long_sentence = "x" * (ch.CHUNK_CHAR_SIZE // 2)
    text = (long_sentence + ". ") * sent_multiplier
    chunks = ch.chunk_page(text, page_no=7)
    for c in chunks:
        assert len(c["text"]) <= ch.CHUNK_CHAR_SIZE
        assert c["page"] == 7


def test_chunk_page_handles_oversized_sentence():
    """A single sentence longer than the limit should still appear once."""
    gigantic = "z" * (ch.CHUNK_CHAR_SIZE * 2)
    chunks = ch.chunk_page(gigantic, page_no=99)
    # should not silently drop, should return one chunk (could be > limit)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("z")
    assert chunks[0]["page"] == 99
