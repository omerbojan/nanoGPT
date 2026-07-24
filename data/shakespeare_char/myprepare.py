import torch
from pathlib import Path

#-----------------------------------------------------------------------------------
class Utf8Tokenizer:
    def __init__(self):
        self.dvocub = 256
    def tokenize(self, data):
        """Encode text as a one-dimensional tensor of UTF-8 byte values."""
        return torch.tensor(list(data.encode("utf-8")), dtype=torch.long)
    def detokenize(self, tokens):
        """Decode byte tokens to text, replacing incomplete UTF-8 sequences."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return bytes(tokens).decode("utf-8", errors="replace")


def load_data(input_file_path=None):
    """Load and tokenize Tiny Shakespeare, returning shifted train/eval pairs."""
    if input_file_path is None:
        input_file_path = Path(__file__).parents[1] / "input.txt"
    input_file_path = Path(input_file_path)
    if not input_file_path.exists():
        raise FileNotFoundError(
            f"{input_file_path} does not exist; run data/shakespeare_char/prepare.py first"
        )

    text = input_file_path.read_text(encoding="utf-8")
    split = int(len(text) * 0.9)
    tokenizer = Utf8Tokenizer()
    train = tokenizer.tokenize(text[:split])
    eval_data = tokenizer.tokenize(text[split:])
    return tokenizer, train[:-1], train[1:], eval_data[:-1], eval_data[1:]
