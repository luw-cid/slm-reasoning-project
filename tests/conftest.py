import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FakeTokenizer:
    """Tokenizer giả: tách theo khớp dài nhất trên một vocab nhỏ cố định (không cần mạng/torch)."""

    VOCAB = [
        "<think>", "</think>", "Wait", " Wait", "Waiting", "\n", "\n\n", ".\n\n", ".", " .", ",", " ,",
        ";", "?", "!", " ", "a", "b", "c", "1", "2", "7", "\\boxed{", "}", "x", "<eos>",
    ]

    def __init__(self):
        self.id_of = {tok: i for i, tok in enumerate(self.VOCAB)}

    def __len__(self):
        return len(self.VOCAB)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            best = None
            for tok, tid in self.id_of.items():
                if text.startswith(tok, i) and (best is None or len(tok) > len(self.VOCAB[best])):
                    best = tid
            if best is None:
                raise ValueError(f"FakeTokenizer: không token hóa được {text[i:]!r}")
            ids.append(best)
            i += len(self.VOCAB[best])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.VOCAB[i] for i in ids)
