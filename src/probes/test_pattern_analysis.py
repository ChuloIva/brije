"""
Quick test of pattern analysis on a small sample
"""

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import json
from pathlib import Path
import sys

NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from streaming_probe_inference import StreamingProbeInferenceEngine

# Test configuration
data_path = Path("data/positive_patterns.jsonl")
probes_base_dir = Path("data/probes_binary")
sentiment_probes_dir = Path("data/sentiment")
model_name = "google/gemma-3-4b-it"
threshold = 0.5

print("="*80)
print("TESTING PATTERN ANALYSIS")
print("="*80)

# Load first 3 entries
print("\n1. Loading test data...")
test_entries = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        test_entries.append(json.loads(line.strip()))
print(f"   ✓ Loaded {len(test_entries)} test entries")

# Initialize engine
print("\n2. Initializing inference engine...")
engine = StreamingProbeInferenceEngine(
    probes_base_dir=probes_base_dir,
    model_name=model_name,
    sentiment_probes_dir=sentiment_probes_dir,
    include_sentiment=True,
    layer_range=(15, 30),
    verbose=False  # Reduce noise
)
print(f"   ✓ Engine initialized with {len(engine.probes)} cognitive probes, {len(engine.sentiment_probes)} sentiment probes")

# Test inference on one pattern
print("\n3. Testing inference on first positive pattern...")
entry = test_entries[0]
text = entry['positive_thought_pattern']
print(f"   Text preview: {text[:100]}...")

print("\n   Running streaming inference...")
total_probes = len(engine.probes) + len(engine.sentiment_probes)
all_predictions = engine.predict_streaming(
    text,
    top_k=total_probes,
    threshold=0.0,
    show_realtime=False
)
print(f"   ✓ Got {len(all_predictions)} predictions")

print("\n   Aggregating predictions...")
aggregated = engine.aggregate_predictions(all_predictions, threshold=threshold)
print(f"   ✓ Aggregated to {len(aggregated)} unique actions")

# Split cognitive and sentiment
cognitive_preds = [p for p in aggregated if p.action_name != "sentiment"]
sentiment_preds = [p for p in aggregated if p.action_name == "sentiment"]

print(f"\n   Cognitive predictions: {len(cognitive_preds)}")
print(f"   Sentiment predictions: {len(sentiment_preds)}")

# Show top 5 cognitive
cognitive_preds.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)
print("\n   Top 5 cognitive actions:")
for i, pred in enumerate(cognitive_preds[:5], 1):
    print(f"      {i}. {pred.action_name}: {pred.layer_count} layers, conf={pred.max_confidence:.3f}")

# Show sentiment info
if sentiment_preds:
    avg_sent = sum(p.max_confidence for p in sentiment_preds) / len(sentiment_preds)
    print(f"\n   Sentiment: avg={avg_sent:.3f}, {len(sentiment_preds)} layers active")

print("\n4. Testing all 3 pattern types from first entry...")
pattern_types = ['positive', 'negative', 'transformation']
for pt in pattern_types:
    key_map = {
        'positive': 'positive_thought_pattern',
        'negative': 'reference_negative_example',
        'transformation': 'reference_transformed_example'
    }
    text = entry.get(key_map[pt], '')
    if not text:
        print(f"   ⚠ {pt}: no text")
        continue

    print(f"\n   Testing {pt}...")
    print(f"      Text length: {len(text)} chars")

    all_preds = engine.predict_streaming(text, top_k=total_probes, threshold=0.0, show_realtime=False)
    agg = engine.aggregate_predictions(all_preds, threshold=threshold)
    cog = [p for p in agg if p.action_name != "sentiment"]
    sent = [p for p in agg if p.action_name == "sentiment"]

    cog.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)
    avg_sent = sum(p.max_confidence for p in sent) / len(sent) if sent else 0.0

    print(f"      Cognitive: {len(cog)} actions")
    if cog:
        print(f"         Top: {cog[0].action_name} ({cog[0].layer_count} layers)")
    print(f"      Sentiment: {avg_sent:.3f}")

print("\n" + "="*80)
print("✅ TEST COMPLETE - Analysis pipeline working!")
print("="*80)