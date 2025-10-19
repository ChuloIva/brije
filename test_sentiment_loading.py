"""
Test script to verify sentiment probes are properly loaded and used
"""

import sys
from pathlib import Path

# Add src/probes to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "probes"))

from streaming_probe_inference import StreamingProbeInferenceEngine

print("=" * 80)
print("TEST 1: Engine Initialization")
print("=" * 80)

# Initialize engine with sentiment probes
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-3-4b-it',
    sentiment_probes_dir=Path('data/sentiment'),
    include_sentiment=True,
    verbose=True  # Enable verbose to see loading messages
)

print("\n" + "=" * 80)
print("TEST 2: Verify Sentiment Probes Loaded")
print("=" * 80)

print(f"\nCognitive probes loaded: {len(engine.probes)}")
print(f"Sentiment probes loaded: {len(engine.sentiment_probes)}")
print(f"Include sentiment flag: {engine.include_sentiment}")
print(f"Sentiment probes dir: {engine.sentiment_probes_dir}")

if len(engine.sentiment_probes) == 0:
    print("\n❌ ERROR: No sentiment probes loaded!")
    print("Expected sentiment probes in layers 21-30")

    # Debug: Check what's in the directory
    print("\nDEBUG: Checking sentiment probe directory...")
    for layer_idx in range(21, 31):
        layer_dir = engine.sentiment_probes_dir / f"layer_{layer_idx}"
        probe_path = layer_dir / "sentiment_regression_probe.pth"
        print(f"  Layer {layer_idx}: dir exists={layer_dir.exists()}, probe exists={probe_path.exists()}")
else:
    print(f"\n✓ SUCCESS: {len(engine.sentiment_probes)} sentiment probes loaded")
    print(f"  Layers: {sorted(engine.sentiment_probes.keys())}")

print("\n" + "=" * 80)
print("TEST 3: Run Inference with Sentiment")
print("=" * 80)

# Short test text
test_text = "I am feeling very happy and positive today!"

print(f"\nTest text: '{test_text}'")
print(f"Running inference...")

# Calculate total probes
total_probes = len(engine.probes) + len(engine.sentiment_probes)
print(f"Total probes (cognitive + sentiment): {total_probes}")

# Run prediction
all_predictions = engine.predict_streaming(
    test_text,
    top_k=total_probes,  # Get ALL predictions
    threshold=0.0,
    show_realtime=False
)

print(f"\n✓ Got {len(all_predictions)} predictions")

# Separate cognitive and sentiment predictions
cognitive_preds = [p for p in all_predictions if p.probe_type == "cognitive"]
sentiment_preds = [p for p in all_predictions if p.probe_type == "sentiment"]

print(f"  Cognitive predictions: {len(cognitive_preds)}")
print(f"  Sentiment predictions: {len(sentiment_preds)}")

if len(sentiment_preds) == 0:
    print("\n❌ ERROR: No sentiment predictions generated!")
    print("This means sentiment probes are not being run during inference")
else:
    print(f"\n✓ SUCCESS: {len(sentiment_preds)} sentiment predictions generated")
    print("\nSentiment prediction details:")
    for pred in sentiment_preds[:5]:  # Show first 5
        print(f"  Layer {pred.layer}: confidence={pred.confidence:+.4f}, active={pred.is_active}")

print("\n" + "=" * 80)
print("TEST 4: Aggregate Predictions")
print("=" * 80)

aggregated = engine.aggregate_predictions(all_predictions, threshold=0.0005)

print(f"\n✓ Got {len(aggregated)} aggregated predictions")

# Find sentiment in aggregated
sentiment_agg = [p for p in aggregated if p.action_name == "sentiment"]

if len(sentiment_agg) == 0:
    print("\n❌ ERROR: No sentiment in aggregated predictions!")
else:
    print(f"\n✓ SUCCESS: Sentiment found in aggregated predictions")
    for pred in sentiment_agg:
        print(f"  Action: {pred.action_name}")
        print(f"  Max confidence: {pred.max_confidence:+.4f}")
        print(f"  Mean confidence: {pred.mean_confidence:+.4f}")
        print(f"  Active layers: {pred.layers}")
        print(f"  Layer count: {pred.layer_count}")
        print(f"  Is active: {pred.is_active}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

issues = []

if len(engine.sentiment_probes) == 0:
    issues.append("❌ Sentiment probes not loaded")
else:
    print(f"✓ Sentiment probes loaded: {len(engine.sentiment_probes)}")

if len(sentiment_preds) == 0:
    issues.append("❌ Sentiment predictions not generated")
else:
    print(f"✓ Sentiment predictions generated: {len(sentiment_preds)}")

if len(sentiment_agg) == 0:
    issues.append("❌ Sentiment not in aggregated predictions")
else:
    print(f"✓ Sentiment in aggregated predictions: {len(sentiment_agg)}")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n🎉 ALL TESTS PASSED!")
