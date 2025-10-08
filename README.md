# Brije: Cognitive Action Detection System

Real-time detection of cognitive actions in language models using linear probes on Gemma 3 4B.

## Overview

Brije detects 45 cognitive actions (like analyzing, reconsidering, divergent_thinking) during text generation. It uses trained probes on Gemma 3 4B's internal representations to identify active cognitive patterns in real-time.

The system includes:
- Activation capture from Gemma 3 4B
- Binary probe training (one-vs-rest approach)
- Real-time inference during generation
- Multi-agent conversation GUI with live visualization

## Quick Start

### Option A: Google Colab (Recommended)

1. Open [`Brije_Full_Pipeline_Colab.ipynb`](./Brije_Full_Pipeline_Colab.ipynb)
2. Select Runtime → GPU (A100 recommended)
3. Run all cells

Time: ~3-4 hours | Cost: Free with Colab

### Option B: Local Installation

```bash
# Install dependencies
pip install torch transformers nnsight h5py scikit-learn tqdm

# Capture activations (~2-3 hours)
cd src/probes
python capture_activations.py \
    --dataset ../../third_party/datagen/generated_data/stratified_combined_31500.jsonl \
    --output-dir ../../data/activations \
    --model google/gemma-2-3b-it \
    --layers 27

# Train binary probes (~15 hours for all 45)
python train_binary_probes.py \
    --activations ../../data/activations/layer_27_activations.h5 \
    --output-dir ../../data/probes_binary \
    --model-type linear \
    --epochs 20

# Test inference
python multi_probe_inference.py \
    --probes-dir ../../data/probes_binary \
    --model google/gemma-2-3b-it \
    --layer 27 \
    --text "She carefully analyzed the problem."
```

## Usage

### Standalone Inference

```python
from multi_probe_inference import MultiProbeInferenceEngine

engine = MultiProbeInferenceEngine(
    probes_dir="data/probes_binary",
    model_name="google/gemma-2-3b-it",
    layer_idx=27
)

predictions = engine.predict(
    "After reconsidering, she began analyzing the data.",
    top_k=5
)

for pred in predictions:
    print(f"{pred.action_name}: {pred.confidence:.1%}")
```

### Multi-Agent GUI

```bash
cd third_party/liminal_backrooms
python main.py
```

Select "Gemma 3 4B (with Probes)" and watch cognitive actions activate in real-time.

## Project Structure

```
brije/
├── src/probes/                     # Core probe system
│   ├── capture_activations.py     # Extract activations
│   ├── train_binary_probes.py     # Train 45 binary probes
│   ├── multi_probe_inference.py   # Real-time inference
│   ├── probe_models.py            # Probe architectures
│   └── dataset_utils.py           # Data processing
│
├── data/
│   ├── activations/               # HDF5 activation files
│   └── probes_binary/             # Trained probe checkpoints
│
├── third_party/
│   ├── datagen/                   # Cognitive actions dataset
│   │   └── generated_data/        # JSONL training data
│   └── liminal_backrooms/         # Multi-agent GUI
│       ├── main.py                # Conversation orchestrator
│       ├── config.py              # Probe configuration
│       └── gemma_probes.py        # Gemma wrapper
│
└── Brije_Full_Pipeline_Colab.ipynb
```

## Cognitive Actions

The system detects 45 cognitive actions across 6 categories:

- **Metacognitive**: reconsidering, meta_awareness, self_questioning
- **Analytical**: analyzing, comparing, distinguishing, inferring
- **Creative**: divergent_thinking, hypothesis_generation, analogical_thinking
- **Emotional**: emotion_reappraisal, emotion_management
- **Memory**: remembering, recalling, recognizing
- **Evaluative**: evaluating, critiquing, assessing

See `third_party/datagen/variable_pools.py` for the complete taxonomy.

## Configuration

Edit `third_party/liminal_backrooms/config.py`:

```python
ENABLE_PROBES = True
PROBE_MODE = "binary"  # or "multiclass"
PROBES_DIR = "../../data/probes_binary"
PROBE_LAYER = 27
PROBE_TOP_K = 5
PROBE_THRESHOLD = 0.1
```

## Technical Details

### Probe Architecture

Each binary probe is a simple linear classifier:

```
Input (2304) → Dropout (0.1) → Linear (2304→1) → Sigmoid → Confidence
```

### Training Approach

One-vs-rest binary classification:
- 45 separate probes (one per cognitive action)
- Each probe trained to detect "is action X?" vs "not action X"
- Independent confidence scores allow detecting multiple simultaneous actions

### Special Message

All text is augmented with: `"\n\nThe cognitive action being demonstrated here is"`

This creates a consistent extraction point and primes the model to encode cognitive action information.

## Performance

| Metric | Value |
|--------|-------|
| AUC-ROC per probe | 0.85-0.95 |
| Accuracy per probe | 0.75-0.90 |
| Inference speed | ~100-200ms |
| Training time | ~15 hours (all 45 probes) |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU with 16GB+ VRAM (or use Colab)
- ~50GB disk space for activations and probes

## License

MIT License

## Citation

```bibtex
@software{brije_cognitive_probes,
  title={Brije: Cognitive Action Detection System},
  author={Ivan Culo},
  year={2025},
  url={https://github.com/koalacrown/brije}
}
```

## Acknowledgments

- **nnsight** - Model internals access
- **Datagen** - Cognitive actions dataset
- **Liminal Backrooms** - Multi-agent framework
- **Gemma Team (Google)** - Base model