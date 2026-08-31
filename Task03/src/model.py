"""
model.py
--------
Deep-learning model definitions for AI music generation.

Architectures available:
  1. MusicLSTM         — stacked LSTM with embedding + dropout + linear head
  2. MusicLSTMAttention — same but with an additive self-attention layer
                          over the LSTM output sequence
  3. build_model()     — factory function: returns the right model from a
                          config dict so every other script stays generic

All models accept integer token indices as input (not one-hot or floats),
use an Embedding layer to learn dense pitch representations, and output
a logit vector of size `vocab_size` for next-token prediction.

Usage (standalone test):
    python src/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Core LSTM model
# ---------------------------------------------------------------------------

class MusicLSTM(nn.Module):
    """
    Stacked LSTM for next-token music generation.

    Input  : (batch, seq_len)   — integer token indices
    Output : (batch, vocab_size) — unnormalised logits for the next token

    Architecture:
        Embedding → LSTM stack → Dropout → Linear
    """

    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int  = 128,
        hidden_size:   int  = 256,
        num_layers:    int  = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.vocab_size    = vocab_size
        self.embed_dim     = embed_dim
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Learnable token embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,              # index 0 = <PAD>
        )

        # LSTM stack
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection
        lstm_out_size = hidden_size * self.num_directions
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(lstm_out_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier / orthogonal initialisation for stable training."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget-gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zeroed (h0, c0) for a fresh sequence."""
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c0 = h0.clone()
        return h0, c0

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Args:
            x      : (batch, seq_len)  — long tensor of token indices
            hidden : optional (h, c) tuple from previous step

        Returns:
            logits : (batch, vocab_size)
            hidden : updated (h, c) tuple
        """
        batch_size = x.size(0)

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # (batch, seq_len) → (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # lstm_out: (batch, seq_len, hidden * directions)
        lstm_out, hidden = self.lstm(emb, hidden)

        # Use only the last time-step's output
        last_out = lstm_out[:, -1, :]           # (batch, hidden * directions)
        out      = self.dropout(last_out)
        logits   = self.fc(out)                 # (batch, vocab_size)

        return logits, hidden

    def extra_repr(self):
        return (
            f"vocab={self.vocab_size}, embed={self.embed_dim}, "
            f"hidden={self.hidden_size}, layers={self.num_layers}, "
            f"bidir={self.bidirectional}"
        )


# ---------------------------------------------------------------------------
# 2. Additive (Bahdanau-style) self-attention over LSTM outputs
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Learnable context vector attention over a sequence of LSTM states.

    Computes a weighted sum of all time steps, where weights are learned
    via a small 2-layer MLP.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn  = nn.Linear(hidden_size, hidden_size)
        self.v     = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out : (batch, seq_len, hidden_size)
        Returns:
            context  : (batch, hidden_size)  — attention-weighted sum
        """
        # energy: (batch, seq_len, 1)
        energy  = self.v(torch.tanh(self.attn(lstm_out)))
        weights = F.softmax(energy, dim=1)          # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)   # (batch, hidden_size)
        return context


# ---------------------------------------------------------------------------
# 3. LSTM + Attention model
# ---------------------------------------------------------------------------

class MusicLSTMAttention(nn.Module):
    """
    MusicLSTM enhanced with an additive attention mechanism.

    Architecture:
        Embedding → LSTM stack → AdditiveAttention → Dropout → Linear
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 128,
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = h0.clone()
        return h0, c0

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        emb      = self.embedding(x)                    # (B, T, E)
        lstm_out, hidden = self.lstm(emb, hidden)       # (B, T, H)
        context  = self.attention(lstm_out)             # (B, H)
        out      = self.dropout(context)
        logits   = self.fc(out)                         # (B, V)
        return logits, hidden

    def extra_repr(self):
        return (
            f"vocab={self.vocab_size}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, attention=additive"
        )


# ---------------------------------------------------------------------------
# 4. Factory function
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "architecture": "lstm_attention",   # "lstm" | "lstm_attention"
    "embed_dim":    128,
    "hidden_size":  256,
    "num_layers":   2,
    "dropout":      0.3,
    "bidirectional": False,             # only used by "lstm"
}


def build_model(vocab_size: int, config: dict = None) -> nn.Module:
    """
    Instantiate a music generation model from a config dict.

    Args:
        vocab_size : size of the token vocabulary
        config     : dict with keys matching DEFAULT_CONFIG
                     (missing keys fall back to defaults)

    Returns:
        nn.Module (either MusicLSTM or MusicLSTMAttention)
    """
    cfg  = {**DEFAULT_CONFIG, **(config or {})}
    arch = cfg["architecture"].lower()

    if arch == "lstm":
        model = MusicLSTM(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )
    elif arch in ("lstm_attention", "attention"):
        model = MusicLSTMAttention(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Choose 'lstm' or 'lstm_attention'."
        )

    return model


# ---------------------------------------------------------------------------
# 5. Utility — count parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, seq_len: int = 64, batch_size: int = 2):
    """Print a concise model summary with a dummy forward pass."""
    device = next(model.parameters()).device
    x      = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    print(f"\n{'='*55}")
    print(f"  Model   : {model.__class__.__name__}")
    print(f"  Params  : {count_parameters(model):,}")
    print(f"{'='*55}")
    print(model)

    with torch.no_grad():
        logits, _ = model(x)
    print(f"\n  Dummy input  : {tuple(x.shape)}")
    print(f"  Output logits: {tuple(logits.shape)}")
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VOCAB = 200
    SEQ   = 64

    print("--- MusicLSTM ---")
    m1 = build_model(VOCAB, {"architecture": "lstm", "hidden_size": 128})
    model_summary(m1, seq_len=SEQ)

    print("--- MusicLSTMAttention ---")
    m2 = build_model(VOCAB, {"architecture": "lstm_attention", "hidden_size": 256})
    model_summary(m2, seq_len=SEQ)

    # Check output shape
    x = torch.randint(0, VOCAB, (4, SEQ))
    logits, _ = m2(x)
    assert logits.shape == (4, VOCAB), f"Unexpected output shape: {logits.shape}"
    print("Output shape check passed.")
