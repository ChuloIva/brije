"""
Test script to inspect sentiment probe output structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "probes"))

from streaming_probe_inference import StreamingProbeInferenceEngine
import json

# Initialize engine with sentiment probes
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-3-4b-it',
    sentiment_probes_dir=Path('data/sentiment'),
    include_sentiment=True
)

# Short test text
text = "I was in such a dark place back then, everything felt hopeless."
threshold = 0.0005

# Get ALL predictions (cognitive + sentiment)
total_probes = len(engine.probes) + len(engine.sentiment_probes)
print(f"Total probes: {total_probes} (cognitive: {len(engine.probes)}, sentiment: {len(engine.sentiment_probes)})")

all_predictions = engine.predict_streaming(
    text,
    top_k=total_probes,
    threshold=0.0,
    show_realtime=False
)

print(f"\n{'='*80}")
print(f"ALL PREDICTIONS ({len(all_predictions)} total)")
print(f"{'='*80}\n")

# Separate cognitive and sentiment predictions
cognitive_preds = [p for p in all_predictions if p.probe_type == "cognitive"]
sentiment_preds = [p for p in all_predictions if p.probe_type == "sentiment"]

print(f"Cognitive predictions: {len(cognitive_preds)}")
print(f"Sentiment predictions: {len(sentiment_preds)}\n")

# Inspect sentiment predictions structure
print(f"{'='*80}")
print(f"SENTIMENT PREDICTIONS STRUCTURE")
print(f"{'='*80}\n")

for i, sent_pred in enumerate(sentiment_preds[:3], 1):  # Show first 3
    print(f"\n--- Sentiment Prediction {i} ---")
    print(f"  action_name: {sent_pred.action_name}")
    print(f"  action_idx: {sent_pred.action_idx}")
    print(f"  layer: {sent_pred.layer}")
    print(f"  confidence: {sent_pred.confidence}")
    print(f"  probe_type: {sent_pred.probe_type}")
    print(f"  is_active: {sent_pred.is_active}")
    print(f"  peak_confidence: {sent_pred.peak_confidence}")
    print(f"  peak_activation_token: {sent_pred.peak_activation_token}")
    print(f"  Number of token_activations: {len(sent_pred.token_activations)}")

    # Show first few token activations
    print(f"\n  First 3 token activations:")
    for j, tok_act in enumerate(sent_pred.token_activations[:3], 1):
        print(f"    Token {j}: pos={tok_act.token_position}, text='{tok_act.token_text}', "
              f"conf={tok_act.confidence:.3f}, probe_type={tok_act.probe_type}")

# Aggregate predictions
print(f"\n{'='*80}")
print(f"AGGREGATED PREDICTIONS")
print(f"{'='*80}\n")

aggregated = engine.aggregate_predictions(all_predictions, threshold=threshold)
aggregated.sort(key=lambda x: (x.layer_count, abs(x.max_confidence)), reverse=True)

# Find aggregated sentiment predictions
agg_sentiment = [p for p in aggregated if p.action_name == "sentiment"]

print(f"Total aggregated predictions: {len(aggregated)}")
print(f"Aggregated sentiment predictions: {len(agg_sentiment)}\n")

if agg_sentiment:
    sent = agg_sentiment[0]
    print(f"--- Aggregated Sentiment Prediction ---")
    print(f"  action_name: {sent.action_name}")
    print(f"  action_idx: {sent.action_idx}")
    print(f"  layers: {sent.layers}")
    print(f"  layer_count: {sent.layer_count}")
    print(f"  max_confidence: {sent.max_confidence}")
    print(f"  mean_confidence: {sent.mean_confidence}")
    print(f"  best_layer: {sent.best_layer}")
    print(f"  is_active: {sent.is_active}")
    print(f"  peak_confidence: {sent.peak_confidence}")
    print(f"  peak_activation_token: {sent.peak_activation_token}")
    print(f"  Number of layer_predictions: {len(sent.layer_predictions)}")

    # Check structure of layer_predictions
    print(f"\n  Layer predictions structure:")
    for i, layer_pred in enumerate(sent.layer_predictions[:2], 1):
        print(f"    Layer {i}: layer={layer_pred.layer}, action_name={layer_pred.action_name}, "
              f"probe_type={layer_pred.probe_type}, confidence={layer_pred.confidence:.3f}, "
              f"num_token_acts={len(layer_pred.token_activations)}")

# Test what happens when we try to display this in Rich
print(f"\n{'='*80}")
print(f"TESTING RICH DISPLAY")
print(f"{'='*80}\n")

from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

console = Console()

# Try to display aggregated sentiment
if agg_sentiment:
    sent = agg_sentiment[0]
    rprint(f"\n[yellow]Displaying aggregated sentiment object:[/yellow]")
    rprint(sent)  # This might show the raw repr

    # Try to convert to dict to see structure
    print(f"\n[yellow]Converting to dict-like structure:[/yellow]")
    sent_dict = {
        'action_name': sent.action_name,
        'action_idx': sent.action_idx,
        'layer_count': sent.layer_count,
        'max_confidence': sent.max_confidence,
        'probe_type': 'sentiment' if hasattr(sent.layer_predictions[0], 'probe_type') else 'N/A'
    }
    rprint(sent_dict)

print(f"\n{'='*80}")
print(f"TEST COMPLETE")
print(f"{'='*80}\n")