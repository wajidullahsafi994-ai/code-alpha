# AI Music Generation with LSTM

A complete deep-learning pipeline that collects MIDI training data,
preprocesses it with **music21**, trains a stacked **LSTM** (with optional
attention), generates new music sequences, and exports them as playable
MIDI / WAV files.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Stages](#pipeline-stages)
   - [Stage 1 — Data Collection](#stage-1--data-collection)
   - [Stage 2 — Preprocessing](#stage-2--preprocessing)
   - [Stage 3 — Training](#stage-3--training)
   - [Stage 4 — Generation](#stage-4--generation)
   - [Stage 5 — Audio Output](#stage-5--audio-output)
6. [Model Architecture](#model-architecture)
7. [Configuration Reference](#configuration-reference)
8. [Audio Playback Backends](#audio-playback-backends)
9. [Tips & Tuning](#tips--tuning)
10. [File Reference](#file-reference)

---

## Project Structure

```
music_ai/
├── main.py                  # End-to-end pipeline entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_collector.py    # Stage 1 — MIDI data collection
│   ├── preprocessor.py      # Stage 2 — Tokenisation & sequence building
│   ├── model.py             # Model definitions (MusicLSTM, MusicLSTMAttention)
│   ├── train.py             # Stage 3 — Training loop
│   ├── generator.py         # Stage 4 — Autoregressive generation
│   └── audio_player.py      # Stage 5 — MIDI → audio conversion & playback
│
├── data/
│   ├── midi/                # Raw MIDI / MusicXML training files
│   └── processed/           # Tokenised numpy arrays + vocabulary
│
├── models/
│   └── saved/               # Checkpoints, config JSON, loss curve PNG
│
└── output/
    ├── midi/                # Generated MIDI files
    └── audio/               # Exported WAV files
```

---

## How It Works

```
 MIDI files          music21 parser        Vocabulary
 (Bach, Jazz, ──────► Note / Chord / ──────► token ↔ int     ──────►  LSTM
  synthetic …)        Rest tokens           mapping                    model
                      e.g. "C4_1.0"
                           "C4.E4.G4_0.5"
                           "rest_0.25"

 Trained model  ──────► Autoregressive ──────► Token sequence ──────► MIDI file
                         sampling               (temperature,          (music21)
                         (top-k / nucleus)       top-k, top-p)
                                                                 ──────► WAV
                                                                         (FluidSynth /
                                                                          pygame /
                                                                          sine synth)
```

**Token encoding** — each event becomes `"<pitch>_<duration>"`:

| Event | Token |
|---|---|
| Middle C, quarter note | `C4_1.0` |
| C-major chord, half note | `C4.E4.G4_2.0` |
| Quarter rest | `rest_0.25` |

A sliding window of length `seq_len` creates `(X, y)` training pairs where
`y` is the next token — standard language-model style.

---

## Installation

### 1. Python 3.9+

```bash
python --version   # 3.9, 3.10, 3.11, or 3.12
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core packages installed:

| Package | Purpose |
|---|---|
| `torch` | LSTM model, training, inference |
| `music21` | MIDI / MusicXML parsing & MIDI export |
| `numpy` | Sequence arrays |
| `pygame` | MIDI playback |
| `soundfile` + `numpy` | WAV export (sine synth fallback) |
| `mido` | Low-level MIDI parsing in audio backend |
| `matplotlib` | Loss curve & token distribution plots |
| `tqdm` | Progress bars |

### 4. (Optional) FluidSynth for high-quality audio

```bash
# Windows (Chocolatey)
choco install fluidsynth

# macOS (Homebrew)
brew install fluidsynth

# Ubuntu / Debian
sudo apt install fluidsynth
```

You also need a SoundFont file (`.sf2`). A free one:
[GeneralUser GS](https://schristiancollins.com/generaluser.php)

---

## Quick Start

### Option A — Full pipeline (recommended first run)

```bash
cd music_ai
python main.py
```

This runs all five stages with default settings. Expect ~10–30 minutes
of training on CPU (50 epochs, 90 synthetic MIDI files).

### Option B — Fast demo (~2–3 minutes on CPU)

```bash
python main.py --quick
```

Uses reduced settings (10 epochs, smaller model, 30 synthetic files)
to verify the whole pipeline works end-to-end quickly.

### Option C — Run a single stage

```bash
python main.py --stage collect
python main.py --stage preprocess
python main.py --stage train
python main.py --stage generate
python main.py --stage play
```

---

## Pipeline Stages

### Stage 1 — Data Collection

```bash
python src/data_collector.py [options]
```

Three data sources:

| Source | Description | Flag to disable |
|---|---|---|
| **music21 corpus** | Bach, Mozart, Beethoven, Schubert, Handel | `--no-corpus` |
| **URL downloads** | 10 public-domain MIDI files from midiworld.com | `--no-download` |
| **Synthetic** | Algorithmically generated classical / jazz / folk MIDI | always included |

The synthetic generator creates valid MIDI files using scale patterns
(major, minor, pentatonic, blues, dorian …) even with no internet access.

```bash
# Only synthetic data, 50 files per genre
python src/data_collector.py --source synthetic --synthetic-count 50

# Only corpus (requires music21 corpus download on first use)
python src/data_collector.py --source corpus --corpus-limit 100
```

---

### Stage 2 — Preprocessing

```bash
python src/preprocessor.py [--seq-len 64] [--min-freq 2] [--plot]
```

What it does:
1. Parses every `.mid` / `.xml` / `.mxl` file in `data/midi/` with music21
2. Converts notes, chords, and rests into string tokens
3. Builds a vocabulary (token → integer index)
4. Applies a sliding window to produce `(X, y)` pairs
5. Saves to `data/processed/`

Output files:

| File | Contents |
|---|---|
| `X.npy` | Input sequences `(N, seq_len)` int32 |
| `y.npy` | Target next-token `(N,)` int32 |
| `X_norm.npy` | `X` normalised to `[0, 1]` float32 |
| `token_to_int.pkl` | Vocabulary dict |
| `int_to_token.pkl` | Reverse vocabulary dict |
| `all_tokens.pkl` | Flat list of every token across all songs |
| `metadata.txt` | vocab_size, seq_len, num_samples |

```bash
# Longer sequences (more context, more memory)
python src/preprocessor.py --seq-len 128

# Save a token frequency chart
python src/preprocessor.py --plot
```

---

### Stage 3 — Training

```bash
python src/train.py [options]
```

Training details:
- **Loss**: Cross-Entropy (PAD token ignored)
- **Optimiser**: Adam with weight decay 1e-5
- **LR schedule**: Cosine annealing from `lr` down to `lr × 0.01`
- **Gradient clipping**: norm ≤ 5.0 (configurable)
- **Early stopping**: stops if validation loss doesn't improve for `patience` epochs
- **Checkpointing**: saves `best_model.pt` every time validation loss improves

```bash
# Larger model, more epochs
python src/train.py --epochs 150 --hidden 512 --layers 3

# Resume from a previous run
python src/train.py --resume

# CPU only
python src/train.py --no-cuda

# Simple LSTM (no attention)
python src/train.py --arch lstm
```

Outputs saved to `models/saved/`:

| File | Contents |
|---|---|
| `best_model.pt` | Best checkpoint (lowest val loss) |
| `final_model.pt` | State at the last epoch |
| `model_config.json` | Architecture hyperparameters |
| `loss_curve.png` | Training & validation loss plot |

---

### Stage 4 — Generation

```bash
python src/generator.py [options]
```

Sampling strategies:

| Strategy | Flag | Effect |
|---|---|---|
| Temperature | `--temperature 0.8` | < 1.0 = conservative; > 1.0 = creative |
| Top-k | `--top-k 10` | Only sample from the 10 most likely tokens |
| Nucleus (top-p) | `--top-p 0.9` | Only sample from tokens covering 90% probability mass |

```bash
# Conservative, piano, 200 tokens
python src/generator.py --temperature 0.7 --top-k 5 --length 200

# Jazz feel (instrument 25 = Acoustic Guitar)
python src/generator.py --instrument 25 --tempo 140 --temperature 1.2

# Prime from a Bach piece and generate 3 variations
python src/generator.py --seed-song bach_invention_1.mid --count 3

# Use a specific checkpoint
python src/generator.py --checkpoint final_model.pt
```

Generated files are saved to `output/midi/` as `generated_001.mid`, etc.

---

### Stage 5 — Audio Output

```bash
python src/audio_player.py [midi_file] [options]
```

```bash
# Play the first generated MIDI (auto-detected)
python src/audio_player.py

# Play a specific file
python src/audio_player.py output/midi/generated_001.mid

# Export to WAV
python src/audio_player.py output/midi/generated_001.mid --export-wav

# High-quality WAV with FluidSynth + SoundFont
python src/audio_player.py output/midi/generated_001.mid \
    --export-wav --soundfont "C:/soundfonts/GeneralUser.sf2"

# Batch convert all generated MIDI to WAV
python src/audio_player.py --batch
```

---

## Model Architecture

### MusicLSTMAttention (default)

```
Input tokens (batch, seq_len)
       │
       ▼
Embedding(vocab_size → embed_dim)       # learns dense pitch representations
       │
       ▼
LSTM × num_layers                       # captures temporal patterns
  - input:  embed_dim
  - hidden: hidden_size
  - dropout between layers
       │
       ▼
AdditiveAttention                       # weighted sum over all time steps
  - attn:  Linear(H → H)
  - v:     Linear(H → 1)
       │
       ▼
Dropout(p=dropout)
       │
       ▼
Linear(hidden_size → vocab_size)        # logit over next token
```

### MusicLSTM (simpler baseline)

Same as above but uses only the **last time step** output instead of
attention, and supports `--bidirectional`.

### Default hyperparameters

| Parameter | Default | Flag |
|---|---|---|
| Embedding dim | 128 | `--embed-dim` |
| Hidden size | 256 | `--hidden` |
| LSTM layers | 2 | `--layers` |
| Dropout | 0.3 | `--dropout` |
| Architecture | lstm_attention | `--arch` |

---

## Configuration Reference

```bash
python main.py --help
```

Key flags grouped by stage:

```
Pipeline control:
  --stage {collect,preprocess,train,generate,play,all}
  --quick               Fast demo mode (reduced settings)
  --skip-collect        Skip Stage 1
  --skip-preprocess     Skip Stage 2
  --skip-train          Skip Stage 3
  --skip-generate       Skip Stage 4
  --skip-play           Skip Stage 5
  --resume              Resume training from existing checkpoint
  --no-cuda             Force CPU
  --seed INT            Global random seed (default: 42)

Data collection:
  --synthetic-count N   Synthetic files per genre (default: 30)
  --corpus-limit N      Max corpus files (default: 60)

Preprocessing:
  --seq-len N           Sequence window length (default: 64)

Training:
  --epochs N            (default: 50)
  --batch-size N        (default: 64)
  --lr FLOAT            Learning rate (default: 0.001)
  --hidden N            LSTM hidden units (default: 256)
  --patience N          Early stopping patience (default: 10)

Generation:
  --length N            Tokens per piece (default: 128)
  --temperature FLOAT   Sampling temperature (default: 1.0)
  --top-k N             Top-k filter (default: 10)
  --top-p FLOAT         Nucleus filter (default: 0.9)
  --gen-count N         Number of pieces (default: 1)
  --tempo FLOAT         BPM (default: 120)
  --instrument N        GM program 0-127 (default: 0 = piano)

Audio:
  --audio-backend       auto | fluidsynth | pygame | synth
  --export-wav          Export WAV instead of playing
  --soundfont PATH      SoundFont for FluidSynth
```

---

## Audio Playback Backends

The player tries backends in this order:

```
1. FluidSynth  ──  best quality, requires binary + SoundFont
2. pygame      ──  good, uses OS MIDI synthesiser, no extra install
3. Sine synth  ──  always works, pure Python, basic quality
```

Force a specific backend:

```bash
python src/audio_player.py generated_001.mid --backend pygame
python src/audio_player.py generated_001.mid --backend synth --export-wav
```

---

## Tips & Tuning

**Better music quality:**
- Train for more epochs: `--epochs 150`
- Use a larger model: `--hidden 512 --layers 3`
- Add more real MIDI data to `data/midi/` before preprocessing

**More variety in generated output:**
- Raise temperature: `--temperature 1.2`
- Disable top-k: `--top-k 0`

**More coherent output:**
- Lower temperature: `--temperature 0.7`
- Tighten nucleus: `--top-p 0.8`

**Genre-specific results:**
- Prime with a genre file: `--seed-song synthetic_jazz_000_dorian.mid`

**Out of memory during training:**
- Reduce batch size: `--batch-size 32`
- Use a smaller model: `--hidden 128 --layers 1`
- Add `--no-cuda` to use CPU

**Short generated pieces:**
- Increase length: `--length 256`

---

## File Reference

| File | Run with |
|---|---|
| `main.py` | `python main.py` |
| `src/data_collector.py` | `python src/data_collector.py` |
| `src/preprocessor.py` | `python src/preprocessor.py` |
| `src/train.py` | `python src/train.py` |
| `src/generator.py` | `python src/generator.py` |
| `src/audio_player.py` | `python src/audio_player.py` |
| `src/model.py` | `python src/model.py` (self-test) |
