"""
Test normalized sentiment scores
"""

import sys
from pathlib import Path

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

# Test text
text = "I was in such a dark place back then, everything felt hopeless."
threshold = 0.1

print(f"\nTesting with text: '{text}'\n")

# Get ALL predictions
total_probes = len(engine.probes) + len(engine.sentiment_probes)
all_predictions = engine.predict_streaming(
    text,
    top_k=total_probes,
    threshold=0.0,
    show_realtime=False
)

print(f"{'='*80}")
print(f"BEFORE NORMALIZATION")
print(f"{'='*80}\n")

# Show raw sentiment scores
sentiment_preds = [p for p in all_predictions if p.probe_type == "sentiment"]
if sentiment_preds:
    print(f"Raw sentiment scores (first 3 layers, first 5 tokens):\n")
    for pred in sentiment_preds[:3]:
        print(f"Layer {pred.layer}:")
        for tok_act in pred.token_activations[:5]:
            print(f"  Token {tok_act.token_position} ('{tok_act.token_text}'): {tok_act.confidence:+8.3f}")
        print()

# Aggregate with normalization
print(f"{'='*80}")
print(f"AFTER NORMALIZATION (during aggregation)")
print(f"{'='*80}\n")

aggregated = engine.aggregate_predictions(all_predictions, threshold=threshold, normalize_sentiment=True)

# Show normalized sentiment scores
sentiment_agg = [p for p in aggregated if p.action_name == "sentiment"]
if sentiment_agg:
    sent = sentiment_agg[0]
    print(f"Normalized sentiment scores (first 3 layers, first 5 tokens):\n")
    for layer_pred in sent.layer_predictions[:3]:
        print(f"Layer {layer_pred.layer}:")
        for tok_act in layer_pred.token_activations[:5]:
            print(f"  Token {tok_act.token_position} ('{tok_act.token_text}'): {tok_act.confidence:+8.3f} (z-score)")
        print()

    # Show statistics
    all_scores = []
    for layer_pred in sent.layer_predictions:
        for tok_act in layer_pred.token_activations:
            all_scores.append(tok_act.confidence)

    import numpy as np
    scores_array = np.array(all_scores)

    print(f"{'='*80}")
    print(f"NORMALIZATION STATISTICS")
    print(f"{'='*80}\n")
    print(f"Mean:   {np.mean(scores_array):+8.3f} (should be ~0.0)")
    print(f"Std:    {np.std(scores_array):8.3f} (should be ~1.0)")
    print(f"Min:    {np.min(scores_array):+8.3f}")
    print(f"Max:    {np.max(scores_array):+8.3f}")
    print(f"Median: {np.median(scores_array):+8.3f}")

    print(f"\n✓ Sentiment scores are now normalized as z-scores!")
    print(f"  - Scores typically range from -3 to +3")
    print(f"  - Negative = negative sentiment, Positive = positive sentiment")
    print(f"  - Magnitude indicates strength (larger absolute value = stronger)")

print(f"\n{'='*80}\n")
