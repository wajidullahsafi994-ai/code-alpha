"""
main.py
-------
End-to-end pipeline for the AI Music Generation project.

Stages:
  1. collect    — download / generate MIDI training data
  2. preprocess — parse MIDI into tokenised sequences
  3. train      — train the LSTM model
  4. generate   — sample new music from the trained model
  5. play       — convert to audio and play / export WAV

Run the full pipeline:
    python main.py

Run individual stages:
    python main.py --stage collect
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage generate
    python main.py --stage play

Quick demo (fast settings, good for testing):
    python main.py --quick

Skip stages already completed:
    python main.py --skip-collect --skip-preprocess

All training hyper-parameters and generation settings can be tuned
via command-line flags — run  `python main.py --help`  for the full list.
"""

import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup so src/ modules are importable from the project root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Stage helpers — thin wrappers that import lazily so missing packages
# only fail at the relevant stage, not on startup.
# ---------------------------------------------------------------------------

def _banner(title: str):
    width = 58
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s)//60}m {int(s)%60}s"


# ---------------------------------------------------------------------------
# Stage 1 — Data collection
# ---------------------------------------------------------------------------

def stage_collect(args) -> int:
    _banner("Stage 1 / 5 — Data Collection")

    from data_collector import (
        collect_from_corpus,
        download_midi_files,
        generate_synthetic_midi,
        print_dataset_summary,
    )

    t0    = time.time()
    total = 0

    if not args.no_corpus:
        total += collect_from_corpus(limit=args.corpus_limit)

    if not args.no_download:
        total += download_midi_files()

    # Always generate synthetic data as a reliable baseline
    for genre in ["classical", "jazz", "folk"]:
        total += generate_synthetic_midi(
            genre=genre,
            count=args.synthetic_count,
            seed=args.seed,
        )

    print_dataset_summary()
    print(f"\n[collect] Done in {_elapsed(t0)}. {total} total files.")
    return total


# ---------------------------------------------------------------------------
# Stage 2 — Preprocessing
# ---------------------------------------------------------------------------

def stage_preprocess(args) -> dict:
    _banner("Stage 2 / 5 — Preprocessing")

    from preprocessor import preprocess, PROCESSED_DIR

    t0 = time.time()
    result = preprocess(
        seq_len=args.seq_len,
        min_freq=args.min_freq,
        min_files=args.min_files,
    )

    if args.plot_tokens:
        from preprocessor import plot_token_distribution
        plot_token_distribution(
            result["token_to_int"],
            result["all_tokens"],
            save_path=PROCESSED_DIR / "token_distribution.png",
        )

    print(f"\n[preprocess] Done in {_elapsed(t0)}.")
    print(f"  Vocab size : {result['vocab_size']}")
    print(f"  Samples    : {len(result['X']):,}")
    return result


# ---------------------------------------------------------------------------
# Stage 3 — Training
# ---------------------------------------------------------------------------

def stage_train(args) -> dict:
    _banner("Stage 3 / 5 — Model Training")

    from train import train

    t0     = time.time()
    result = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        architecture=args.arch,
        clip_grad=args.clip_grad,
        patience=args.patience,
        use_cuda=not args.no_cuda,
        seed=args.seed,
        resume=args.resume,
    )

    print(f"\n[train] Done in {_elapsed(t0)}.")
    print(f"  Best val loss : {result['best_val_loss']:.4f}  "
          f"(epoch {result['best_epoch']})")
    return result


# ---------------------------------------------------------------------------
# Stage 4 — Music generation
# ---------------------------------------------------------------------------

def stage_generate(args) -> list:
    _banner("Stage 4 / 5 — Music Generation")

    from generator import generate

    t0    = time.time()
    paths = generate(
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tempo_bpm=args.tempo,
        instrument=args.instrument,
        seed_song=args.seed_song,
        count=args.gen_count,
        use_cuda=not args.no_cuda,
        random_seed=args.seed,
        output_prefix=args.prefix,
        checkpoint=args.checkpoint,
    )

    print(f"\n[generate] Done in {_elapsed(t0)}.")
    for p in paths:
        print(f"  → {p}")
    return paths


# ---------------------------------------------------------------------------
# Stage 5 — Audio playback / export
# ---------------------------------------------------------------------------

def stage_play(args, midi_paths: list = None):
    _banner("Stage 5 / 5 — Audio Output")

    from audio_player import play_midi, convert_to_wav, batch_convert
    from pathlib import Path

    output_midi_dir = BASE_DIR / "output" / "midi"

    if args.export_wav:
        # Batch convert every generated MIDI to WAV
        wav_paths = batch_convert(
            midi_dir=output_midi_dir,
            soundfont=args.soundfont,
            backend=args.audio_backend,
        )
        if wav_paths:
            print(f"\n[play] {len(wav_paths)} WAV file(s) saved to output/audio/")
        else:
            print("\n[play] WAV export failed — check backend availability.")
        return

    # Determine which files to play
    if midi_paths:
        targets = midi_paths
    else:
        targets = sorted(output_midi_dir.glob("*.mid"))
        if not targets:
            print("[play] No MIDI files found in output/midi/")
            print("       Run  python main.py --stage generate  first.")
            return

    if args.play_all:
        to_play = targets
    else:
        to_play = targets[:1]   # play just the first by default

    for midi_path in to_play:
        print(f"\n[play] {midi_path.name}")
        play_midi(
            midi_path,
            backend=args.audio_backend,
            soundfont=args.soundfont,
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Music Generation — end-to-end pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Pipeline control -------------------------------------------------
    p.add_argument(
        "--stage",
        choices=["collect", "preprocess", "train", "generate", "play", "all"],
        default="all",
        help="Which pipeline stage to run",
    )
    p.add_argument("--quick", action="store_true",
                   help="Use fast settings for a quick demo / smoke-test")
    p.add_argument("--skip-collect",    action="store_true")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train",      action="store_true")
    p.add_argument("--skip-generate",   action="store_true")
    p.add_argument("--skip-play",       action="store_true")
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from an existing checkpoint")
    p.add_argument("--no-cuda",         action="store_true",
                   help="Disable GPU even if available")
    p.add_argument("--seed",            type=int, default=42)

    # ---- Data collection --------------------------------------------------
    g_collect = p.add_argument_group("Data collection")
    g_collect.add_argument("--no-corpus",       action="store_true",
                           help="Skip music21 corpus collection")
    g_collect.add_argument("--no-download",     action="store_true",
                           help="Skip URL-based MIDI downloads")
    g_collect.add_argument("--corpus-limit",    type=int, default=60)
    g_collect.add_argument("--synthetic-count", type=int, default=30,
                           help="Synthetic MIDI files per genre")

    # ---- Preprocessing ----------------------------------------------------
    g_pre = p.add_argument_group("Preprocessing")
    g_pre.add_argument("--seq-len",    type=int,   default=64)
    g_pre.add_argument("--min-freq",   type=int,   default=2)
    g_pre.add_argument("--min-files",  type=int,   default=3)
    g_pre.add_argument("--plot-tokens", action="store_true",
                       help="Save a token-frequency bar chart after preprocessing")

    # ---- Training ---------------------------------------------------------
    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs",     type=int,   default=50)
    g_train.add_argument("--batch-size", type=int,   default=64)
    g_train.add_argument("--lr",         type=float, default=0.001)
    g_train.add_argument("--hidden",     type=int,   default=256)
    g_train.add_argument("--embed-dim",  type=int,   default=128)
    g_train.add_argument("--layers",     type=int,   default=2)
    g_train.add_argument("--dropout",    type=float, default=0.3)
    g_train.add_argument("--arch",       type=str,   default="lstm_attention",
                         choices=["lstm", "lstm_attention"])
    g_train.add_argument("--patience",   type=int,   default=10)
    g_train.add_argument("--clip-grad",  type=float, default=5.0)

    # ---- Generation -------------------------------------------------------
    g_gen = p.add_argument_group("Generation")
    g_gen.add_argument("--length",      type=int,   default=128,
                       help="Number of tokens to generate per piece")
    g_gen.add_argument("--temperature", type=float, default=1.0)
    g_gen.add_argument("--top-k",       type=int,   default=10)
    g_gen.add_argument("--top-p",       type=float, default=0.9)
    g_gen.add_argument("--tempo",       type=float, default=120.0)
    g_gen.add_argument("--instrument",  type=int,   default=0,
                       help="GM instrument number 0-127")
    g_gen.add_argument("--gen-count",   type=int,   default=1,
                       help="Number of pieces to generate")
    g_gen.add_argument("--seed-song",   type=str,   default=None)
    g_gen.add_argument("--checkpoint",  type=str,   default="best_model.pt")
    g_gen.add_argument("--prefix",      type=str,   default="generated")

    # ---- Audio ------------------------------------------------------------
    g_audio = p.add_argument_group("Audio")
    g_audio.add_argument("--audio-backend", type=str, default="auto",
                         choices=["auto", "fluidsynth", "pygame", "synth"])
    g_audio.add_argument("--export-wav",    action="store_true",
                         help="Export generated MIDI as WAV instead of playing")
    g_audio.add_argument("--soundfont",     type=str, default=None,
                         help="Path to a .sf2/.sf3 SoundFont for FluidSynth")
    g_audio.add_argument("--play-all",      action="store_true",
                         help="Play all generated pieces (default: first only)")

    return p


# ---------------------------------------------------------------------------
# Quick-demo preset
# ---------------------------------------------------------------------------

def apply_quick_preset(args):
    """Override settings for a fast smoke-test run (~2-3 min on CPU)."""
    args.synthetic_count = 10
    args.no_corpus       = True
    args.no_download     = True
    args.seq_len         = 32
    args.epochs          = 10
    args.batch_size      = 32
    args.hidden          = 128
    args.embed_dim       = 64
    args.layers          = 1
    args.patience        = 5
    args.length          = 64
    args.gen_count       = 1
    print("[quick] Fast demo mode — reduced settings for a quick test.\n")
    return args


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args):
    stage     = args.stage
    run_all   = (stage == "all")

    midi_paths = []

    # ---- Stage 1: collect ------------------------------------------------
    if run_all and not args.skip_collect or stage == "collect":
        midi_dir = BASE_DIR / "data" / "midi"
        existing = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
        if existing and stage != "collect":
            print(f"[pipeline] Skipping collection — "
                  f"{len(existing)} MIDI files already in data/midi/")
        else:
            stage_collect(args)
        if stage == "collect":
            return

    # ---- Stage 2: preprocess ---------------------------------------------
    if run_all and not args.skip_preprocess or stage == "preprocess":
        processed_dir = BASE_DIR / "data" / "processed"
        if (processed_dir / "X.npy").exists() and stage != "preprocess":
            print("[pipeline] Skipping preprocessing — processed data already exists.")
        else:
            stage_preprocess(args)
        if stage == "preprocess":
            return

    # ---- Stage 3: train --------------------------------------------------
    if run_all and not args.skip_train or stage == "train":
        model_path = BASE_DIR / "models" / "saved" / "best_model.pt"
        if model_path.exists() and not args.resume and stage != "train":
            print("[pipeline] Skipping training — checkpoint already exists. "
                  "Use --resume to continue training.")
        else:
            stage_train(args)
        if stage == "train":
            return

    # ---- Stage 4: generate -----------------------------------------------
    if run_all and not args.skip_generate or stage == "generate":
        midi_paths = stage_generate(args)
        if stage == "generate":
            return

    # ---- Stage 5: play / export ------------------------------------------
    if run_all and not args.skip_play or stage == "play":
        stage_play(args, midi_paths=midi_paths if midi_paths else None)
        if stage == "play":
            return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.quick:
        args = apply_quick_preset(args)

    print("=" * 58)
    print("  AI Music Generation Pipeline")
    print("  Code Alpha Internship — Task 03")
    print("=" * 58)
    print(f"  Stage    : {args.stage}")
    print(f"  Arch     : {args.arch}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Hidden   : {args.hidden}")
    print(f"  Seq len  : {args.seq_len}")
    print(f"  Device   : {'CUDA' if not args.no_cuda else 'CPU (forced)'}")
    print("=" * 58)

    t_global = time.time()

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n[pipeline] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[pipeline] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n[pipeline] All done in {_elapsed(t_global)}.")


if __name__ == "__main__":
    main()
