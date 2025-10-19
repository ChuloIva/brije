"""
Analyze sentiment score ranges across different layers to determine normalization strategy
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "probes"))

from streaming_probe_inference import StreamingProbeInferenceEngine

# Initialize engine with sentiment probes
print("Loading model and probes...")
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-3-4b-it',
    sentiment_probes_dir=Path('data/sentiment'),
    include_sentiment=True,
    verbose=False
)

# Test with multiple texts to get better statistics
test_texts = [
    "I was in such a dark place back then, everything felt hopeless and I couldn't see any way forward.",
    "This is the best day of my life! I'm so incredibly happy and excited about everything!",
    "The weather is nice today. I went to the store and bought some groceries.",
    "I'm feeling a bit anxious about the upcoming meeting, but I think it will be okay.",
    "After years of struggle, I finally achieved my dream. The feeling is indescribable!"
]

print(f"\nAnalyzing {len(test_texts)} different texts across {len(engine.sentiment_probes)} sentiment layers...\n")

# Collect statistics per layer
layer_stats = {}
for layer_idx in engine.sentiment_probes.keys():
    layer_stats[layer_idx] = {
        'values': [],
        'min': float('inf'),
        'max': float('-inf'),
        'mean': 0.0,
        'std': 0.0
    }

# Process each text
for text_idx, text in enumerate(test_texts, 1):
    print(f"Processing text {text_idx}/{len(test_texts)}...")

    # Get predictions
    all_predictions = engine.predict_streaming(
        text,
        top_k=len(engine.probes) + len(engine.sentiment_probes),
        threshold=0.0,
        show_realtime=False
    )

    # Extract sentiment predictions
    sentiment_preds = [p for p in all_predictions if p.probe_type == "sentiment"]

    # Collect scores for each layer
    for pred in sentiment_preds:
        layer = pred.layer
        for tok_act in pred.token_activations:
            score = tok_act.confidence
            layer_stats[layer]['values'].append(score)
            layer_stats[layer]['min'] = min(layer_stats[layer]['min'], score)
            layer_stats[layer]['max'] = max(layer_stats[layer]['max'], score)

# Calculate mean and std for each layer
print(f"\n{'='*80}")
print("SENTIMENT SCORE STATISTICS BY LAYER")
print(f"{'='*80}\n")

print(f"{'Layer':<6} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'Range':<10} {'Samples':<8}")
print(f"{'-'*80}")

for layer_idx in sorted(layer_stats.keys()):
    stats = layer_stats[layer_idx]
    if stats['values']:
        values = np.array(stats['values'])
        mean = np.mean(values)
        std = np.std(values)
        score_range = stats['max'] - stats['min']

        stats['mean'] = mean
        stats['std'] = std

        print(f"L{layer_idx:<5} {stats['min']:<10.2f} {stats['max']:<10.2f} {mean:<10.2f} {std:<10.2f} {score_range:<10.2f} {len(values):<8}")

# Analyze overall statistics
all_values = []
for stats in layer_stats.values():
    all_values.extend(stats['values'])

if all_values:
    all_values = np.array(all_values)
    print(f"\n{'-'*80}")
    print(f"OVERALL: {np.min(all_values):<10.2f} {np.max(all_values):<10.2f} {np.mean(all_values):<10.2f} {np.std(all_values):<10.2f} {np.max(all_values) - np.min(all_values):<10.2f} {len(all_values):<8}")

# Check for scale differences between layers
print(f"\n{'='*80}")
print("SCALE ANALYSIS")
print(f"{'='*80}\n")

max_range = max([stats['max'] - stats['min'] for stats in layer_stats.values() if stats['values']])
min_range = min([stats['max'] - stats['min'] for stats in layer_stats.values() if stats['values']])

print(f"Maximum range across layers: {max_range:.2f}")
print(f"Minimum range across layers: {min_range:.2f}")
print(f"Range ratio (max/min): {max_range/min_range if min_range > 0 else float('inf'):.2f}x")

# Suggest normalization approach
print(f"\n{'='*80}")
print("NORMALIZATION RECOMMENDATION")
print(f"{'='*80}\n")

if max_range / min_range > 2.0:
    print("⚠️  Significant scale variation detected between layers!")
    print(f"   Some layers have {max_range/min_range:.1f}x wider range than others.")
    print("\nRecommended approach: Per-layer Z-score normalization")
    print("   normalized_score = (score - layer_mean) / layer_std")
    print("\nThis will:")
    print("   ✓ Make scores comparable across layers")
    print("   ✓ Preserve relative differences within each layer")
    print("   ✓ Center around 0 with std of 1")
else:
    print("✓ Layers have relatively consistent scales")
    print(f"  Range variation is only {max_range/min_range:.1f}x")
    print("\nRecommended approach: Simple min-max normalization")
    print("   normalized_score = (score - min) / (max - min)")

# Show sample normalization for one layer
print(f"\n{'='*80}")
print("SAMPLE NORMALIZATION (Layer 24)")
print(f"{'='*80}\n")

if 24 in layer_stats and layer_stats[24]['values']:
    sample_layer = 24
    sample_values = layer_stats[sample_layer]['values'][:10]  # First 10 values
    layer_mean = layer_stats[sample_layer]['mean']
    layer_std = layer_stats[sample_layer]['std']

    print(f"{'Raw Score':<12} {'Z-score':<12} {'Min-Max (0-1)':<15}")
    print(f"{'-'*40}")

    global_min = min(layer_stats[sample_layer]['values'])
    global_max = max(layer_stats[sample_layer]['values'])

    for val in sample_values:
        z_score = (val - layer_mean) / layer_std if layer_std > 0 else 0
        minmax_norm = (val - global_min) / (global_max - global_min) if global_max > global_min else 0.5
        print(f"{val:<12.2f} {z_score:<12.2f} {minmax_norm:<15.3f}")

print(f"\n{'='*80}\n")
