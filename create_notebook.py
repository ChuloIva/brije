import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Title
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Cognitive Action Probe Testing\n",
        "\n",
        "Test your trained cognitive action probes on text.\n",
        "\n",
        "**Key Feature**: Automatically loads the best-performing layer for each action."
    ]
})

# Setup cell
notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import sys\n",
        "from pathlib import Path\n",
        "import torch\n",
        "\n",
        "# Add src/probes to path\n",
        "sys.path.insert(0, str(Path.cwd() / 'src' / 'probes'))\n",
        "\n",
        "from best_multi_probe_inference import BestMultiProbeInferenceEngine\n",
        "from best_probe_loader import print_performance_summary"
    ]
})

# Performance summary
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. View Probe Performance"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "PROBES_BASE_DIR = Path('data/probes_binary')\n",
        "print_performance_summary(PROBES_BASE_DIR, top_n=10)"
    ]
})

# Load probes
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Load Inference Engine"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "MODEL_NAME = 'google/gemma-2-3b-it'\n",
        "\n",
        "print('Loading probes...')\n",
        "engine = BestMultiProbeInferenceEngine(\n",
        "    probes_base_dir=PROBES_BASE_DIR,\n",
        "    model_name=MODEL_NAME\n",
        ")\n",
        "print('Ready!')"
    ]
})

# Test on text
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Test on Your Text"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "text = '''After receiving feedback, I began reconsidering my approach.\n",
        "I realized I had been making assumptions without fully understanding the constraints.'''\n",
        "\n",
        "predictions = engine.predict(text, top_k=10, threshold=0.1)\n",
        "\n",
        "print('Detected Cognitive Actions:')\n",
        "print('='*70)\n",
        "for i, pred in enumerate(predictions, 1):\n",
        "    marker = '✓' if pred.is_active else '○'\n",
        "    print(f\"{marker} {i:2d}. {pred.action_name:30s} {pred.confidence:6.1%}\")\n",
        "    print(f\"      (Layer {pred.layer}, AUC: {pred.auc:.3f})\")"
    ]
})

# Compare texts
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Compare Two Texts"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "text1 = 'Analyzing the data to identify patterns and trends.'\n",
        "text2 = 'Brainstorming creative solutions to the problem.'\n",
        "\n",
        "comparison = engine.compare_texts(text1, text2, top_k=5)\n",
        "\n",
        "print('TEXT 1:', text1)\n",
        "print('\\nTop actions:')\n",
        "for action, conf in comparison['text1_top_actions'][:5]:\n",
        "    print(f'  - {action:30s} {conf:.1%}')\n",
        "\n",
        "print('\\n' + '='*70)\n",
        "print('TEXT 2:', text2)\n",
        "print('\\nTop actions:')\n",
        "for action, conf in comparison['text2_top_actions'][:5]:\n",
        "    print(f'  - {action:30s} {conf:.1%}')"
    ]
})

# Batch processing
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. Batch Processing"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "texts = [\n",
        "    'Comparing different solutions to find the best approach.',\n",
        "    'Generating creative ideas for the new design.',\n",
        "    'Evaluating the effectiveness of the strategy.'\n",
        "]\n",
        "\n",
        "batch_results = engine.predict_batch(texts, top_k=3, threshold=0.1)\n",
        "\n",
        "for i, (text, preds) in enumerate(zip(texts, batch_results), 1):\n",
        "    print(f'\\n{i}. {text}')\n",
        "    for j, pred in enumerate(preds, 1):\n",
        "        print(f'   {j}. {pred.action_name:30s} {pred.confidence:.1%} (L{pred.layer})')"
    ]
})

# View probe info
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. View Probe Details"]
})

notebook["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "actions = ['analyzing', 'creating', 'metacognitive_regulation']\n",
        "\n",
        "print(f\"{'Action':<35s} {'Layer':>5s} {'AUC':>8s} {'F1':>8s}\")\n",
        "print('-'*60)\n",
        "for action in actions:\n",
        "    info = engine.get_probe_info(action)\n",
        "    print(f\"{info['action']:<35s} {info['layer']:>5d} \"\n",
        "          f\"{info['auc']:>8.4f} {info['f1']:>8.4f}\")"
    ]
})

# Save notebook
with open('/home/koalacrown/Desktop/Code/Projects/brije/Test_Cognitive_Action_Probes.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created successfully!")
