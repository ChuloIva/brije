# Brije: Watching Minds Think (Condensed)

This repository provides a toolkit to detect 45 cognitive actions in real-time as a language model (Gemma 3 4B) generates text. Think of it as an fMRI for an AI's thought process.


## What It Does
- **Detects 45 Cognitive Actions**: Identifies thinking patterns like `analyzing`, `reconsidering`, `divergent_thinking`, and `self_questioning` using trained probes on the model's internal states.
- **Enables Multi-Agent Conversations**: Run simulations (e.g., therapy sessions, debates) between two AI agents and watch their cognitive strategies interact in real-time.
- **Provides an Analysis Suite**: A set of notebooks to analyze cognitive patterns in large datasets, including synchrony, network analysis, and predicting behavioral outcomes.

## Core Features
- **Real-Time Inference**: See cognitive actions light up as text is generated.
- **Multi-Agent GUI**: An interactive interface for running AI-to-AI conversations.
- **Advanced Analytics**: Scripts for deep analysis of cognitive patterns (tested on the AnnoMI therapy dataset).
- **Extensible**: Add new cognitive actions or fine-tune probes on your own data.
- **Hardware Agnostic**: Supports NVIDIA (CUDA), Apple Silicon (MPS), AMD (ROCm), and CPU.

## Getting Started

### The Easy Way: Google Colab
1. Open [`Brije_Full_Pipeline_Colab.ipynb`](./Brije_Full_Pipeline_Colab.ipynb).
2. Set runtime to GPU.
3. Run all cells. This takes ~3-4 hours to train all 45 probes.
4. Download your trained probes from the Colab environment.

### The Local Way (16GB+ VRAM GPU Recommended)
```bash
# 1. Install dependencies
pip install torch transformers nnsight h5py scikit-learn tqdm

# 2. Capture model activations (2-3 hours)
python src/probes/capture_activations.py \
    --model google/gemma-2-3b-it \
    --layer 27 \
    --device auto

# 3. Train all 45 probes (1-2 hours)
python src/probes/train_binary_probes.py \
    --activations data/activations/layer_27_activations.h5 \
    --output-dir data/probes_binary \
    --device auto

# 4. Run inference on a sample text
python src/probes/multi_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "After reconsidering my approach, I began analyzing the problem differently."
```

## Example Outputs

We provide two different views of the cognitive action predictions.

### View 1: Grouped by Action
This view summarizes which actions were detected across all layers for a given text. It's useful for seeing the most prominent cognitive signals.

| Text Snippet | Top Detected Actions (Count of Layers) |
| :--- | :--- |
| `"The quarterly numbers look... interesting..."` | `noticing` (8), `analyzing` (7), `hypothesis_generation` (3) |
| `"What if we completely flipped the script?..."` | `divergent_thinking` (10), `questioning` (3), `convergent_thinking` (3) |
| `"If we launch in Q2 instead of Q1..."` | `counterfactual_reasoning` (10), `evaluating` (6), `reframing` (1) |

**[See full output for 30 examples in `output_example_3.md`](./output_example_3.txt)**

---

### View 2: Layer-by-Layer Breakdown
This view shows the raw confidence scores for each detected action at each specific layer. It's useful for seeing how cognitive processes evolve through the model's depth.

| Text Snippet | Layer-by-Layer Detections (Selected Layers) |
| :--- | :--- |
| `"The quarterly numbers look... interesting..."` | **L22:** `analyzing`(1.0), `noticing`(1.0)<br>**L28:** `evaluating`(1.0), `pattern_recognition`(1.0)<br>**L30:** `questioning`(1.0), `understanding`(1.0) |
| `"What if we completely flipped the script?..."` | **L21:** `divergent_thinking`(1.0), `questioning`(1.0)<br>**L22:** `creating`(0.88)<br>**L28:** `hypothesis_generation`(1.0) |
| `"If we launch in Q2 instead of Q1..."` | **L22:** `counterfactual_reasoning`(1.0)<br>**L26:** `reframing`(1.0)<br>**L30:** `evaluating`(0.93) |

**[See full output for 30 examples in `output_example_4.md`](./output_example_4.txt)

### Network Analysis Visualization
This graph shows the relationships between different cognitive actions, derived from analyzing 133 therapy transcripts. `noticing` is the central hub.

![Cognitive Action Network](output/analysis_AnnoMI/advanced_6_network_analysis.png)

## What's in the Box?
- `src/probes/`: The core scripts for capturing activations, training probes, and running inference.
- `data/`: Where your captured activations and trained probes are stored.
- `third_party/`: Includes the multi-agent GUI (`liminal_backrooms`) and the dataset generator (`datagen`).
- `*.ipynb`: A series of notebooks for both running the pipeline (`Brije_Full_Pipeline_Colab.ipynb`) and for performing advanced analysis (`AnnoMI_*.ipynb`).
- `output/analysis_AnnoMI/`: Contains all generated charts, graphs, and analysis summaries from the notebooks.

## The Big Idea
By making an AI's cognitive processes visible, we can better understand how they "think," analyze their reasoning patterns for safety and alignment, and apply these insights to human domains like therapy, education, and collaboration.


