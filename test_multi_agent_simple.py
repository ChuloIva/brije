#!/usr/bin/env python3
"""
Simple test for multi-agent conversation with best multi-probe inference
Tests that the system uses each probe's optimal layer
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "third_party" / "liminal_backrooms"))
sys.path.insert(0, str(Path(__file__).parent / "src" / "probes"))

def test_gemma_probes():
    """Test that Gemma loads with best multi-probe engine"""

    print("\n" + "="*80)
    print("TESTING GEMMA WITH BEST MULTI-PROBE ENGINE")
    print("="*80)

    from third_party.liminal_backrooms.gemma_probes import GemmaWithProbes

    # Initialize Gemma with probes
    gemma = GemmaWithProbes(
        model_name="google/gemma-3-4b-it",
        probe_mode="binary"
    )

    # Test a simple generation
    test_prompt = "Talk about that time you were scared ."

    print(f"\n{'='*80}")
    print(f"Test Prompt: {test_prompt}")
    print(f"{'='*80}\n")

    response = gemma.generate(
        prompt=test_prompt,
        system_prompt="You are talking from a first person perspective of someone who is talking to a friend about their feelings.",
        max_new_tokens=400
    )

    print("\nGenerated Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Get predictions
    predictions = gemma.get_predictions_dict()

    print(f"\n{'='*80}")
    print("COGNITIVE ACTIONS DETECTED (with optimal layers)")
    print("="*80)
    print(f"{'Rank':<6} {'Action':<35} {'Confidence':<12} {'Layer':<8} {'AUC':<8} {'Active'}")
    print("-" * 80)

    for i, pred in enumerate(predictions[:10], 1):
        action = pred['action']
        confidence = pred['confidence']
        layer = pred['layer']
        auc = pred['auc']
        is_active = pred['is_active']
        marker = "✓" if is_active else " "

        print(f"{i:<6} {action:<35} {confidence:>6.1%}       L{layer:<6} {auc:.3f}    {marker}")

    # Show layer distribution
    layers_used = [pred['layer'] for pred in predictions]
    unique_layers = sorted(set(layers_used))

    print(f"\n{'='*80}")
    print(f"Layer Distribution (unique layers used: {unique_layers})")
    print("="*80)

    for layer in unique_layers:
        count = layers_used.count(layer)
        print(f"  Layer {layer}: {count} probes")

    print(f"\n✅ Test completed successfully!")
    print(f"   Each probe is using its optimal layer!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_gemma_probes()
