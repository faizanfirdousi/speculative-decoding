import sys
import pytest
import torch
from spec_decoding_4060.streamer import LiveTokenStreamer


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            1: "Hello",
            2: " world",
            3: "!",
            4: " Spec",
            5: "ulative",
            6: " decoding",
            7: "🚀",
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.vocab.get(tid, f"<{tid}>") for tid in token_ids)


def test_streamer_autoregressive(capsys):
    tok = MockTokenizer()
    streamer = LiveTokenStreamer(tok, label="Test AR", is_speculative=False, show_stats=True)
    streamer.start()

    streamer.put_token(1)
    streamer.put_token(2)
    streamer.put_token(3)
    stats = streamer.end()

    captured = capsys.readouterr()
    assert "Hello world!" in captured.out
    assert stats["tokens"] == 3
    assert stats["is_speculative"] is False
    assert stats["tps"] > 0
    assert "Throughput Speed" in captured.out


def test_streamer_speculative(capsys):
    tok = MockTokenizer()
    streamer = LiveTokenStreamer(
        tok, label="Test Spec", is_speculative=True, colorize=False, show_stats=True
    )
    streamer.start()

    # Prefill first token
    streamer.put_token(1)

    # Chunk 1: 2 tokens accepted from draft, 1 bonus target token
    accepted = torch.tensor([2, 4])
    authoritative = torch.tensor([5])
    streamer.put_speculative_chunk(accepted, authoritative, drafted_count=3)

    stats = streamer.end()
    captured = capsys.readouterr()

    assert "Hello world Speculative" in captured.out
    assert stats["tokens"] == 4
    assert stats["is_speculative"] is True
    assert stats["total_accepted"] == 2
    assert stats["total_drafted"] == 3
    assert stats["verification_cycles"] == 1
    assert "Draft Acceptance" in captured.out


def test_streamer_colorization(capsys):
    tok = MockTokenizer()
    streamer = LiveTokenStreamer(
        tok, label="Test Colors", is_speculative=True, colorize=True, show_stats=False
    )
    streamer.start()

    accepted = torch.tensor([1])
    authoritative = torch.tensor([2])
    streamer.put_speculative_chunk(accepted, authoritative, drafted_count=2)
    streamer.end()

    captured = capsys.readouterr()
    # Check that cyan (\033[96m) and green (\033[92m) ANSI escapes are present
    assert "\033[96m" in captured.out
    assert "\033[92m" in captured.out
