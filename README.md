# llm-from-scratch

A small, educational codebase and notebook that implements and documents core concepts behind large language models (LLMs) from first principles. This repository collects notes, math derivations, and simple implementations so you can learn the building blocks of modern LLMs and experiment with training & inference.

This repo is intended for learning and research — not a production LLM.

## Quick start (fish shell)

1. Create and activate a virtual environment

   ```fish
   python3 -m venv .venv
   source .venv/bin/activate.fish
   ```
2. Install dependencies

   ```fish
   pip install -r requirements.txt
   ```
3. Try the demo (simple runner)

   ```fish
   python demo/main.py
   ```
4. Open the interactive tutorial

   - The primary notebook is `tutorial.ipynb` — open it with Jupyter or JupyterLab.

     ```fish
     jupyter notebook tutorial.ipynb
     ```

## What you'll find here

- `tutorial.ipynb` — the core learning notebook with notes, math, and runnable code.
- `models/` — simple model code and experimental model wrappers (e.g. `modeling_gemma.py`, `paligemma.py`, `siglip.py`).
- `training/` — training scripts for text and image experiments (`train_text.py`, `train_image.py`).
- `demo/` — small demo runner (`main.py`) to exercise parts of the project.
- `data/` — data utilities (see `data/clean.py`) for cleaning/preparing data used by experiments.
- `checkpoints/` — directory to save and load model checkpoints (gitignored large files).
- `explanability/` — supporting code that documents explainability concepts used in the repo.

Below is a short, practical summary of what the codebase actually implements and what the tutorial covers:

- Code implemented (what's done)

  - Lightweight transformer components: input embeddings, positional encodings, multi-head self-attention, feed-forward blocks, layer-norm + residual wrappers (see `tutorial.ipynb` and `models/`).
  - Training loop examples and data pipeline for small experiments using the TinyStories dataset (see `training/` and `data/`).
  - KV caching and simple quantization helpers for inference efficiency (`tutorial.ipynb` shows `KVCache` / `QuantizedKVCache`).
  - A minimal demo runner in `demo/main.py` to try out model routines locally.
  - Explainability visualizations and diagnostics (`explanability/four_pillars.py`) for data and attention analysis.
- What the tutorial covers (high-level)

  - Concept walkthrough: tokenization, embeddings, positional encodings, attention, feed-forward layers, normalization, and the language modeling head.
  - Code walkthroughs for each transformer piece with small runnable examples (forward pass, loss computation, simple training on TinyStories).
  - Data analysis and explainability: topic modeling, readability metrics, and dataset diagnostics to surface biases or imbalances.
  - Practical tips for inference: KV caching, basic quantization, and lightweight optimizations for small-scale experiments.

This repository is aimed at learning and experimentation. If you'd like the README to be even shorter (single-paragraph summary) or to include a one-command demo example, tell me which and I'll adjust it.

## Usage notes

- Training scripts are lightweight examples — they are designed for learning and small-scale experiments.
- Check `training/train_text.py --help` for available flags and configuration options.
- Checkpoints and logs are stored under `checkpoints/` by default; back them up if you want to keep long runs.

## Development & contributions

- If you'd like to contribute, open an issue describing the change or improvement.
- Keep changes small and focused; add tests or notebook examples when possible.

## Requirements

- A recent Python 3 (3.8+) installation. See `requirements.txt` for Python packages used in the project.
