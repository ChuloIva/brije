# Brije: Cognitive Action Detection System

A complete system for detecting and visualizing cognitive actions in real-time using Gemma 3 4B with trained probes, integrated with the liminal_backrooms multi-agent conversation framework.

## 🎯 Overview

This project implements:

1. **Activation Capture**: Extract hidden states from Gemma 3 4B using nnsight
2. **Probe Training**: Train linear/logistic probes on cognitive action dataset
3. **Real-time Inference**: Detect cognitive actions during text generation
4. **Live Visualization**: Display active cognitive patterns in multi-agent conversations

## 📁 Project Structure

```
brije/
├── src/probes/                    # Probe system implementation
│   ├── __init__.py
│   ├── capture_activations.py    # Extract activations from Gemma 3 4B
│   ├── dataset_utils.py           # Load and process cognitive actions data
│   ├── probe_models.py            # Linear and MultiHead probe architectures
│   ├── train_probes.py            # Training pipeline with metrics
│   └── probe_inference.py         # Real-time inference engine
│
├── data/
│   ├── activations/               # Captured activations (HDF5 files)
│   └── probes/                    # Trained probe checkpoints
│
├── third_party/
│   ├── datagen/                   # Cognitive actions dataset generator
│   │   ├── data_generator.py
│   │   ├── variable_pools.py      # 45 cognitive actions taxonomy
│   │   └── generated_data/        # Training data (JSONL files)
│   │
│   ├── nnsight/                   # Model internals access library
│   │   └── src/nnsight/
│   │
│   └── liminal_backrooms/         # Multi-agent conversation GUI
│       ├── main.py                # ✨ Modified: Gemma integration
│       ├── config.py              # ✨ Modified: Probe config + role prompts
│       ├── shared_utils.py        # ✨ Modified: call_gemma_api
│       └── gemma_probes.py        # ✨ New: Gemma wrapper with probes
│
└── notebooks/                     # Analysis notebooks (optional)
```

## 🚀 Quick Start

### Option A: One-Command Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
# Bash script (interactive)
./train_probe_pipeline.sh

# Or Python script
cd src/probes
python run_full_pipeline.py

# With custom options
./train_probe_pipeline.sh --layer 27 --probe-type linear --epochs 20
python run_full_pipeline.py --layer 27 --probe-type multihead --epochs 20
```

This will:
1. ✅ Capture activations from Gemma 3 4B (2-3 hours)
2. ✅ Train the probe (15-30 minutes)
3. ✅ Test inference on example texts
4. ✅ Display performance metrics

**Time:** ~2.5-3.5 hours total on GPU

### Option B: Manual Step-by-Step

### 1. Installation

```bash
# Clone the repository
cd /path/to/brije

# Install dependencies
pip install torch transformers nnsight h5py scikit-learn tqdm

# Ensure you have the datagen dataset
ls third_party/datagen/generated_data/*.jsonl
```

### 2. Capture Activations

Extract activations from Gemma 3 4B on the cognitive actions dataset:

```bash
cd src/probes

python capture_activations.py \
    --dataset ../../third_party/datagen/generated_data/stratified_combined_31500.jsonl \
    --output-dir ../../data/activations \
    --model google/gemma-2-3b-it \
    --layers 7 14 21 27 \
    --format hdf5
```

This will:
- Load 31,500 cognitive action examples (combined stratified dataset)
- Extract activations from layers 7, 14, 21, and 27
- Save train/val/test splits to HDF5 files

**Time:** ~2-3 hours on GPU (larger dataset)

### 3. Train Probes

Train a probe on the captured activations:

```bash
python train_probes.py \
    --activations ../../data/activations/layer_27_activations.h5 \
    --output-dir ../../data/probes \
    --model-type linear \
    --batch-size 32 \
    --epochs 20 \
    --lr 0.001
```

This will:
- Train a linear probe for 20 epochs
- Save best model based on validation accuracy
- Generate test metrics and confusion matrix

**Time:** ~10-20 minutes on GPU

**Expected Accuracy:** 70-85% (45-way classification with 31.5K examples)

### 4. Test Inference

Test the probe on sample text:

```bash
python probe_inference.py \
    --probe ../../data/probes/best_probe.pth \
    --model google/gemma-2-3b-it \
    --layer 27 \
    --text "After receiving feedback, she began reconsidering her initial approach."
```

Output:
```
Detected Cognitive Actions:
  1. reconsidering                   45.3% [metacognitive]
  2. evaluating                       22.1% [evaluative]
  3. analyzing                        12.5% [analytical]
```

### 5. Run Liminal Backrooms with Probes

```bash
cd third_party/liminal_backrooms
python main.py
```

In the GUI:
1. Select **"Gemma 3 4B (with Probes)"** for AI-1 and/or AI-2
2. Choose a prompt style like **"Cognitive Roles - Analyst vs Creative"**
3. Set number of turns (e.g., 5-10)
4. Enter a starting prompt or click **Propagate**
5. Watch cognitive actions activate in real-time!

## 🧠 Cognitive Actions Taxonomy

The system detects 45 cognitive actions across 6 categories:

| Category | Examples |
|----------|----------|
| **Metacognitive** | reconsidering, meta_awareness, self_questioning, suspending_judgment |
| **Analytical** | analyzing, comparing, distinguishing, inferring |
| **Creative** | divergent_thinking, hypothesis_generation, analogical_thinking |
| **Emotional** | emotion_reappraisal, emotion_management, attentional_deployment |
| **Memory** | remembering, recalling, recognizing |
| **Evaluative** | evaluating, critiquing, assessing |

See `third_party/datagen/variable_pools.py` for the complete taxonomy.

## 📊 System Architecture

### Data Flow

```
[Cognitive Actions Dataset]
         ↓
[Gemma 3 4B + nnsight]
         ↓
[Hidden States Extraction] → layers 7, 14, 21, 27
         ↓
[Probe Training] → Linear/MultiHead classifiers
         ↓
[Saved Probe Checkpoint]
         ↓
[Real-time Inference] → During generation
         ↓
[GUI Visualization] → Live bar charts
```

### Probe Architecture

**LinearProbe:**
```
Input (hidden_dim=2304)
  ↓
Dropout (p=0.1)
  ↓
Linear (2304 → 45)
  ↓
Softmax → Probabilities
```

**MultiHeadProbe (more powerful):**
```
Input (2304)
  ↓
Encoder: Linear(2304→512) → ReLU → Dropout → Linear(512→512)
  ↓
Multi-Head Attention (8 heads)
  ↓
Classifier: Linear(512→45)
  ↓
Softmax → Probabilities
```

## 🎨 Role-Based Prompts

Configure different AI personalities to trigger distinct cognitive patterns:

### Analyst vs Creative
- **AI-1 (Analyst)**: Emphasizes analyzing, evaluating, comparing
- **AI-2 (Creative)**: Emphasizes divergent_thinking, imagining, hypothesis_generation

### Skeptic vs Optimist
- **AI-1 (Skeptic)**: Triggers questioning, critiquing, suspending_judgment
- **AI-2 (Optimist)**: Activates reframing, accepting, emotion_reappraisal

### Metacognitive Explorers
- Both AIs focus on meta_awareness, reconsidering, self_questioning

## 📈 Performance Metrics

Typical probe performance on test set:

| Metric | Value (with 31.5K examples) |
|--------|-------|
| Test Accuracy | 70-85% |
| Macro F1 | 0.65-0.80 |
| Top-3 Accuracy | 88-95% |
| Inference Speed | ~50ms per prediction |

## 🔧 Configuration

### Probe Config (`liminal_backrooms/config.py`)

```python
ENABLE_PROBES = True
PROBE_PATH = "../../data/probes/best_probe.pth"
PROBE_LAYER = 27  # Last layer
PROBE_TOP_K = 5   # Show top 5 predictions
PROBE_THRESHOLD = 0.1  # Min confidence
```

### Capture Config

- **Layers**: Evenly spaced (7, 14, 21, 27 for 28-layer model)
- **Pooling**: Mean pooling over sequence length
- **Format**: HDF5 for efficiency

### Training Config

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)

## 🎓 Scientific Foundation

The cognitive actions are based on:

- **Bloom's Taxonomy** - Cognitive process dimensions
- **Guilford's Structure of Intellect** - Divergent/convergent thinking
- **Krathwohl's Affective Domain** - Emotional awareness
- **Metacognitive Frameworks** - Self-awareness and monitoring

See `third_party/datagen/Taxonomies_to_use.md` for details.

## 🔬 Advanced Usage

### Train on Different Layers

```bash
# Train probes for all layers
for layer in 7 14 21 27; do
    python train_probes.py \
        --activations ../../data/activations/layer_${layer}_activations.h5 \
        --output-dir ../../data/probes/layer_${layer} \
        --model-type linear
done
```

### Use MultiHead Probe

```bash
python train_probes.py \
    --activations ../../data/activations/layer_27_activations.h5 \
    --output-dir ../../data/probes \
    --model-type multihead \
    --hidden-dim 512
```

### Analyze Dataset

```bash
python dataset_utils.py \
    ../../third_party/datagen/generated_data/cognitive_actions_7k_final_1759233061.jsonl
```

### Batch Inference

```python
from probe_inference import ProbeInferenceEngine

engine = ProbeInferenceEngine(
    probe_path="data/probes/best_probe.pth",
    layer_idx=27
)

texts = [
    "She was comparing different solutions.",
    "He started generating creative ideas.",
    "They were evaluating the strategy."
]

predictions = engine.predict_batch(texts)

for text, preds in zip(texts, predictions):
    print(f"{text}")
    for pred in preds[:3]:
        print(f"  - {pred.action_name}: {pred.confidence:.1%}")
```

## 🐛 Troubleshooting

### Out of Memory

- Reduce batch size: `--batch-size 16`
- Capture fewer examples: `--max-examples 1000`
- Use smaller layers: `--layers 27` (just the last one)

### Probe Not Loading

- Check path in `config.py` is correct
- Ensure `best_probe.pth` exists in `data/probes/`
- Try absolute path instead of relative

### Low Accuracy

- Try MultiHeadProbe: `--model-type multihead`
- Train longer: `--epochs 50`
- Use later layer: `--layer 27`
- Check class balance in dataset

## 📝 Citation

```bibtex
@software{brije_cognitive_probes,
  title={Brije: Cognitive Action Detection System},
  author={Ivan Culo},
  year={2025},
  url={https://github.com/koalacrown/brije},
  note={Real-time cognitive action detection using probes on Gemma 3 4B}
}
```

## 📄 License

MIT License - Free for research and commercial use.

## 🙏 Acknowledgments

- **nnsight** - For model internals access
- **Datagen** - For cognitive actions dataset
- **Liminal Backrooms** - For multi-agent framework
- **Gemma Team (Google)** - For the base model
