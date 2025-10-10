"""
Multi-probe inference engine for running all 45 binary cognitive action probes
"""

from gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys

# Add nnsight to path
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from nnsight import LanguageModel
from transformers import AutoTokenizer

from probe_models import load_probe
from dataset_utils import get_idx_to_action_mapping


@dataclass
class CognitiveActionPrediction:
    """Single prediction from a probe"""
    action_name: str
    action_idx: int
    confidence: float
    is_active: bool  # Whether confidence >= threshold


class MultiProbeInferenceEngine:
    """
    Inference engine that runs all 45 binary cognitive action probes
    """

    def __init__(
        self,
        probes_dir: Path,
        model_name: str = "google/gemma-2-3b-it",
        layer_idx: int = 27,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize multi-probe inference engine

        Args:
            probes_dir: Directory containing all probe_{action}.pth files
            model_name: Name of the language model
            layer_idx: Layer to extract activations from
            device: Device to run on
        """
        self.probes_dir = Path(probes_dir)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.device = device

        print(f"Initializing MultiProbeInferenceEngine...")
        print(f"  Probes dir: {probes_dir}")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer_idx}")
        print(f"  Device: {device}")

        # Load index to action mapping
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {action: idx for idx, action in self.idx_to_action.items()}
        self.num_actions = len(self.idx_to_action)

        # Load all probes
        self.probes = {}
        self._load_all_probes()

        # Load model and tokenizer
        print(f"Loading language model: {model_name}...")
        self.model = LanguageModel(model_name, device_map=device, dispatch=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"✓ Initialized with {len(self.probes)} probes\n")

    def _load_all_probes(self):
        """Load all 45 binary probes from disk"""
        print(f"Loading {self.num_actions} binary probes...")

        loaded_count = 0
        for action_idx, action_name in self.idx_to_action.items():
            probe_path = self.probes_dir / f"probe_{action_name}.pth"

            if not probe_path.exists():
                print(f"  ⚠ Warning: Probe not found for '{action_name}' at {probe_path}")
                continue

            probe, metadata = load_probe(probe_path, device=self.device)
            probe.eval()  # Set to evaluation mode

            self.probes[action_name] = {
                'probe': probe,
                'metadata': metadata,
                'action_idx': action_idx
            }
            loaded_count += 1

        print(f"✓ Loaded {loaded_count}/{self.num_actions} probes")

        if loaded_count == 0:
            raise RuntimeError(f"No probes found in {self.probes_dir}")

    def extract_activations(self, text: str) -> torch.Tensor:
        """
        Extract activations from the language model for given text

        Args:
            text: Input text

        Returns:
            Activations tensor (1, hidden_dim)
        """
        # Append special message to create consistent extraction point
        # This primes the model to "think about" cognitive actions
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        # Tokenize
        inputs = self.tokenizer(augmented_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        # Extract activations using nnsight
        with self.model.trace(input_ids) as tracer:
            # Get the hidden states from the specified layer
            hidden_states = self.model.model.layers[self.layer_idx].output[0]

            # Use last token representation (similar to paper's approach)
            # This is the representation after the model has "thought about" what cognitive action it is
            activations = hidden_states[:, -1, :].save()

        return activations.value

    def predict(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.1,
        return_all: bool = False
    ) -> List[CognitiveActionPrediction]:
        """
        Run all probes and return predictions

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return
            threshold: Minimum confidence threshold
            return_all: If True, return all predictions (not just top-k)

        Returns:
            List of CognitiveActionPrediction objects, sorted by confidence
        """
        # Extract activations once
        activations = self.extract_activations(text)

        # Run all probes
        predictions = []

        with torch.no_grad():
            for action_name, probe_info in self.probes.items():
                probe = probe_info['probe']
                action_idx = probe_info['action_idx']

                # Get prediction
                logits = probe(activations)
                confidence = torch.sigmoid(logits).item()

                prediction = CognitiveActionPrediction(
                    action_name=action_name,
                    action_idx=action_idx,
                    confidence=confidence,
                    is_active=confidence >= threshold
                )
                predictions.append(prediction)

        # Sort by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)

        # Return top-k or all
        if return_all:
            return predictions
        else:
            # Return top-k that meet threshold
            top_predictions = [p for p in predictions[:top_k] if p.is_active]
            # If none meet threshold, return top-k anyway
            if len(top_predictions) == 0:
                top_predictions = predictions[:top_k]
            return top_predictions

    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.1
    ) -> List[List[CognitiveActionPrediction]]:
        """
        Run predictions on multiple texts

        Args:
            texts: List of input texts
            top_k: Number of top predictions per text
            threshold: Minimum confidence threshold

        Returns:
            List of prediction lists (one per text)
        """
        results = []
        for text in texts:
            preds = self.predict(text, top_k=top_k, threshold=threshold)
            results.append(preds)
        return results

    def get_active_actions(
        self,
        text: str,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Get list of active cognitive actions (above threshold)

        Args:
            text: Input text
            threshold: Confidence threshold

        Returns:
            List of action names that are active
        """
        predictions = self.predict(text, top_k=self.num_actions, threshold=threshold, return_all=True)
        return [p.action_name for p in predictions if p.is_active]

    def compare_texts(
        self,
        text1: str,
        text2: str,
        top_k: int = 10
    ) -> Dict:
        """
        Compare cognitive actions in two texts

        Args:
            text1: First text
            text2: Second text
            top_k: Number of top actions to show

        Returns:
            Dictionary with comparison results
        """
        preds1 = self.predict(text1, top_k=top_k, threshold=0.0, return_all=True)
        preds2 = self.predict(text2, top_k=top_k, threshold=0.0, return_all=True)

        # Create confidence maps
        conf1 = {p.action_name: p.confidence for p in preds1}
        conf2 = {p.action_name: p.confidence for p in preds2}

        # Compute differences
        differences = []
        for action_name in self.idx_to_action.values():
            c1 = conf1.get(action_name, 0.0)
            c2 = conf2.get(action_name, 0.0)
            diff = c2 - c1
            differences.append({
                'action': action_name,
                'text1_confidence': c1,
                'text2_confidence': c2,
                'difference': diff
            })

        # Sort by absolute difference
        differences = sorted(differences, key=lambda x: abs(x['difference']), reverse=True)

        return {
            'text1_top_actions': [(p.action_name, p.confidence) for p in preds1[:top_k]],
            'text2_top_actions': [(p.action_name, p.confidence) for p in preds2[:top_k]],
            'biggest_differences': differences[:top_k]
        }


def main():
    """Example usage of MultiProbeInferenceEngine"""
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-probe inference")
    parser.add_argument(
        "--probes-dir",
        type=str,
        required=True,
        help="Directory containing trained probes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-3b-it",
        help="Language model name"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=27,
        help="Layer to extract activations from"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum confidence threshold"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = MultiProbeInferenceEngine(
        probes_dir=args.probes_dir,
        model_name=args.model,
        layer_idx=args.layer
    )

    # If text provided, analyze it
    if args.text:
        print(f"\nAnalyzing text: \"{args.text}\"\n")
        print("-" * 70)

        predictions = engine.predict(
            args.text,
            top_k=args.top_k,
            threshold=args.threshold
        )

        print(f"\nDetected Cognitive Actions (top {args.top_k}):\n")
        for i, pred in enumerate(predictions, 1):
            marker = "✓" if pred.is_active else " "
            print(f"  {marker} {i}. {pred.action_name:30s} {pred.confidence:.10f}")

        print()

    else:
        # Interactive mode
        print("\nInteractive mode - enter text to analyze (or 'quit' to exit):\n")

        while True:
            text = input("Text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text:
                continue

            predictions = engine.predict(
                text,
                top_k=args.top_k,
                threshold=args.threshold
            )

            print(f"\nTop {args.top_k} predictions:")
            for i, pred in enumerate(predictions, 1):
                marker = "✓" if pred.is_active else " "
                print(f"  {marker} {i}. {pred.action_name:30s} {pred.confidence:.10f}")
            print()


if __name__ == "__main__":
    main()
