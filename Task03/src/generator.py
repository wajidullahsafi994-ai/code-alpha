"""
generator.py
------------
Inference module: loads a trained model checkpoint and generates new
music sequences, then writes them as MIDI files.

Sampling strategies:
  - temperature  : scales logits before softmax (lower = more conservative)
  - top-k        : restricts sampling to the k highest-probability tokens
  - nucleus/top-p: restricts sampling to the smallest set of tokens whose
                   cumulative probability exceeds p

Token → MIDI conversion:
  Tokens like "C4_1.0", "F#5_0.5", "C4.E4.G4_1.0", "rest_0.5" are
  decoded back into music21 Note / Chord / Rest objects and assembled
  into a Score, which is then written to a .mid file.

Usage:
    python src/generator.py
    python src/generator.py --length 128 --temperature 0.8 --top-k 10
    python src/generator.py --seed-song bach_invention_1.mid --count 3
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from model import build_model
from preprocessor import load_processed, PROCESSED_DIR, REST_TOKEN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = SRC_DIR.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
OUTPUT_DIR = BASE_DIR / "output" / "midi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Sampling helpers
# ---------------------------------------------------------------------------

def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature (higher T = more random)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-out all logits except the top-k."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold  = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def nucleus_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Zero-out tokens outside the nucleus (top-p) set."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Restore original ordering
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(-1, sorted_idx, sorted_logits)
    return result


def sample_next_token(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 0,
    top_p:       float = 1.0,
) -> int:
    """
    Apply temperature + top-k + nucleus filtering, then multinomial sample.

    Args:
        logits      : (vocab_size,) unnormalised logits
        temperature : sampling temperature
        top_k       : top-k filtering (0 = disabled)
        top_p       : nucleus threshold (1.0 = disabled)

    Returns:
        sampled token index (int)
    """
    logits = temperature_scale(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = nucleus_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ---------------------------------------------------------------------------
# 2.  Model loader
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    vocab_size:      int,
    device:          torch.device,
) -> tuple:
    """
    Load a model from a checkpoint file.

    Returns: (model, model_config)
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    config = ckpt.get("config", {})
    if not config:
        # Fall back to model_config.json in the same directory
        cfg_path = checkpoint_path.parent / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = json.load(f)

    model = build_model(vocab_size, config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, config


# ---------------------------------------------------------------------------
# 3.  Seed sequence builder
# ---------------------------------------------------------------------------

def build_seed_sequence(
    token_to_int: dict,
    int_to_token: dict,
    all_tokens:   list,
    seq_len:      int,
    seed_song:    str  = None,
    midi_dir:     Path = None,
    random_seed:  int  = None,
) -> list:
    """
    Build a starting integer sequence to prime the model.

    Priority:
      1. Parse `seed_song` MIDI file if provided
      2. Random contiguous window from all_tokens
      3. Fallback: seq_len zeros (PAD tokens)
    """
    if random_seed is not None:
        random.seed(random_seed)

    unk_idx = token_to_int.get("<UNK>", 1)

    # Option 1 — seed from a specific MIDI file
    if seed_song and midi_dir:
        from preprocessor import parse_file
        seed_path = midi_dir / seed_song
        if seed_path.exists():
            tokens = parse_file(seed_path)
            if len(tokens) >= seq_len:
                # Use the first seq_len tokens
                window = tokens[:seq_len]
                return [token_to_int.get(t, unk_idx) for t in window]
            else:
                print(f"[generator] Seed file too short ({len(tokens)} tokens); "
                      f"falling back to random window.")

    # Option 2 — random window from training corpus
    if all_tokens and len(all_tokens) >= seq_len:
        start = random.randint(0, len(all_tokens) - seq_len)
        window = all_tokens[start : start + seq_len]
        return [token_to_int.get(t, unk_idx) for t in window]

    # Option 3 — zero padding
    print("[generator] Warning: using zero-padded seed sequence.")
    return [0] * seq_len


# ---------------------------------------------------------------------------
# 4.  Autoregressive generation
# ---------------------------------------------------------------------------

def generate_sequence(
    model:        torch.nn.Module,
    seed_indices: list,
    length:       int,
    token_to_int: dict,
    temperature:  float = 1.0,
    top_k:        int   = 0,
    top_p:        float = 1.0,
    device:       torch.device = torch.device("cpu"),
) -> list:
    """
    Autoregressively generate `length` new tokens given a seed sequence.

    Returns:
        List of token index integers (length = len(seed_indices) + length)
    """
    pad_idx = token_to_int.get("<PAD>", 0)
    model.eval()

    # Prime with the seed
    generated = list(seed_indices)

    with torch.no_grad():
        hidden = None
        seq_len = len(seed_indices)

        for step in range(length):
            # Use the most recent seq_len tokens as context
            context = generated[-seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)

            logits, hidden = model(x, hidden)
            logits_1d = logits[0]  # (vocab_size,)

            # Do not sample PAD or UNK
            logits_1d[pad_idx] = float("-inf")
            unk_idx = token_to_int.get("<UNK>", 1)
            logits_1d[unk_idx] = float("-inf")

            next_idx = sample_next_token(
                logits_1d,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated.append(next_idx)

            # Detach hidden states to avoid growing the graph
            if isinstance(hidden, tuple):
                hidden = (hidden[0].detach(), hidden[1].detach())

    # Return only the newly generated portion
    return generated[seq_len:]


# ---------------------------------------------------------------------------
# 5.  Token sequence → MIDI file
# ---------------------------------------------------------------------------

def tokens_to_midi(
    token_indices: list,
    int_to_token:  dict,
    output_path:   Path,
    instrument:    int   = 0,      # General MIDI program (0 = Acoustic Grand Piano)
    tempo_bpm:     float = 120.0,
) -> Path:
    """
    Convert a list of token indices back to a MIDI file via music21.

    Token format handled:
        "C4_1.0"         → single Note
        "C4.E4.G4_1.0"  → Chord
        "rest_0.5"       → Rest

    Args:
        token_indices : list of int (generated token indices)
        int_to_token  : reverse vocabulary map
        output_path   : destination .mid path
        instrument    : GM program number (0–127)
        tempo_bpm     : playback tempo

    Returns:
        Path to the written MIDI file.
    """
    try:
        from music21 import stream, note, chord, tempo as m21tempo, instrument as m21inst
    except ImportError:
        raise RuntimeError("music21 is required. Run: pip install music21")

    part  = stream.Part()
    instr = m21inst.instrumentFromMidiProgram(instrument)
    part.append(instr)
    part.append(m21tempo.MetronomeMark(number=tempo_bpm))

    skipped = 0
    for idx in token_indices:
        token = int_to_token.get(idx, "")
        if not token or token in ("<PAD>", "<UNK>"):
            skipped += 1
            continue

        # Split into pitch_part and duration_part
        if "_" not in token:
            skipped += 1
            continue

        *pitch_parts, dur_str = token.rsplit("_", 1)
        pitch_part = "_".join(pitch_parts)   # handles edge cases

        try:
            dur = float(dur_str)
        except ValueError:
            skipped += 1
            continue

        if dur <= 0:
            dur = 0.25

        try:
            if pitch_part.lower().startswith("rest"):
                element = note.Rest(quarterLength=dur)

            elif "." in pitch_part:
                # Chord: "C4.E4.G4"
                pitch_names = pitch_part.split(".")
                element = chord.Chord(pitch_names, quarterLength=dur)

            else:
                # Single note: "C#4"
                element = note.Note(pitch_part, quarterLength=dur)

            part.append(element)

        except Exception:
            skipped += 1

    if skipped:
        print(f"[midi] Skipped {skipped} unparseable tokens.")

    score = stream.Score([part])
    score.write("midi", fp=str(output_path))
    print(f"[midi] Saved MIDI → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 6.  High-level generate() function
# ---------------------------------------------------------------------------

def generate(
    length:       int   = 128,
    temperature:  float = 1.0,
    top_k:        int   = 10,
    top_p:        float = 0.9,
    tempo_bpm:    float = 120.0,
    instrument:   int   = 0,
    seed_song:    str   = None,
    count:        int   = 1,
    use_cuda:     bool  = True,
    random_seed:  int   = None,
    output_prefix:str   = "generated",
    checkpoint:   str   = "best_model.pt",
    processed_dir:Path  = PROCESSED_DIR,
    models_dir:   Path  = MODELS_DIR,
    output_dir:   Path  = OUTPUT_DIR,
    midi_dir:     Path  = None,
) -> list:
    """
    End-to-end generation pipeline.

    Returns:
        List of Path objects pointing to the generated MIDI files.
    """
    if midi_dir is None:
        midi_dir = BASE_DIR / "data" / "midi"

    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )
    print(f"\n[generate] Device : {device}")

    # ---- Load processed vocab ------------------------------------------
    print("[generate] Loading vocabulary …")
    data        = load_processed(processed_dir)
    token_to_int = data["token_to_int"]
    int_to_token = data["int_to_token"]
    all_tokens   = data["all_tokens"]
    vocab_size   = data["vocab_size"]
    seq_len      = data["seq_len"]

    # ---- Load model -------------------------------------------------------
    ckpt_path = models_dir / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}.\n"
            "Run  python src/train.py  first."
        )

    print(f"[generate] Loading model from {ckpt_path} …")
    model, config = load_model(ckpt_path, vocab_size, device)
    print(f"[generate] Model architecture : {config.get('architecture', 'unknown')}")
    print(f"[generate] Vocab size         : {vocab_size}")

    # ---- Generate `count` pieces -----------------------------------------
    output_paths = []

    for i in range(count):
        piece_seed = None if random_seed is None else random_seed + i
        print(f"\n[generate] Piece {i+1}/{count}  "
              f"(length={length}, T={temperature}, top_k={top_k}, top_p={top_p})")

        seed_indices = build_seed_sequence(
            token_to_int=token_to_int,
            int_to_token=int_to_token,
            all_tokens=all_tokens,
            seq_len=seq_len,
            seed_song=seed_song,
            midi_dir=midi_dir,
            random_seed=piece_seed,
        )

        generated_indices = generate_sequence(
            model=model,
            seed_indices=seed_indices,
            length=length,
            token_to_int=token_to_int,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )

        out_path = output_dir / f"{output_prefix}_{i+1:03d}.mid"
        tokens_to_midi(
            token_indices=generated_indices,
            int_to_token=int_to_token,
            output_path=out_path,
            instrument=instrument,
            tempo_bpm=tempo_bpm,
        )
        output_paths.append(out_path)

    print(f"\n[generate] Done. {len(output_paths)} MIDI file(s) written to {output_dir}")
    return output_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate new music sequences from a trained model."
    )
    parser.add_argument("--length",      type=int,   default=128,
                        help="Number of tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k",       type=int,   default=10,
                        help="Top-k filtering (0=off, default: 10)")
    parser.add_argument("--top-p",       type=float, default=0.9,
                        help="Nucleus/top-p filtering (1.0=off, default: 0.9)")
    parser.add_argument("--tempo",       type=float, default=120.0,
                        help="Output MIDI tempo in BPM (default: 120)")
    parser.add_argument("--instrument",  type=int,   default=0,
                        help="GM instrument number 0-127 (default: 0 = piano)")
    parser.add_argument("--seed-song",   type=str,   default=None,
                        help="MIDI filename in data/midi/ to use as seed")
    parser.add_argument("--count",       type=int,   default=1,
                        help="Number of pieces to generate (default: 1)")
    parser.add_argument("--checkpoint",  type=str,   default="best_model.pt",
                        help="Checkpoint filename in models/saved/ (default: best_model.pt)")
    parser.add_argument("--prefix",      type=str,   default="generated",
                        help="Output filename prefix (default: generated)")
    parser.add_argument("--no-cuda",     action="store_true")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Random seed (default: None = random)")

    args = parser.parse_args()

    try:
        generate(
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            tempo_bpm=args.tempo,
            instrument=args.instrument,
            seed_song=args.seed_song,
            count=args.count,
            use_cuda=not args.no_cuda,
            random_seed=args.seed,
            output_prefix=args.prefix,
            checkpoint=args.checkpoint,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
