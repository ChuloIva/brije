"""
Real-time probe inference engine for live cognitive action detection
"""

from gpu_utils import configure_amd_gpu

# Configure AMD GPU environment if detected (must be before torch import)
configure_amd_gpu()

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add nnsight to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"))
from nnsight import LanguageModel

from probe_models import load_probe, get_probe_predictions
from dataset_utils import get_idx_to_action_mapping


@dataclass
class CognitiveActionPrediction:
    """Single cognitive action prediction with confidence"""
    action_name: str
    action_idx: int
    confidence: float


class ProbeInferenceEngine:
    """
    Real-time inference engine for cognitive action detection

    This class manages:
    - Loading Gemma 3 4B with nnsight
    - Loading trained probes
    - Extracting activations in real-time
    - Running probe inference
    - Returning top-k predictions
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        probe_path: Optional[Path] = None,
        layer_idx: int = 27,  # Last layer by default
        device: str = "auto",
        top_k: int = 5,
        confidence_threshold: float = 0.1
    ):
        """
        Initialize inference engine

        Args:
            model_name: HuggingFace model ID
            probe_path: Path to trained probe model
            layer_idx: Which layer to extract activations from
            device: Device to run on
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence threshold
        """
        print(f"Initializing ProbeInferenceEngine...")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer_idx}")
        print(f"  Device: {device}")

        # Load language model
        # For VLMs like Gemma-3, skip loading vision tower to save memory and enable text-only usage
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only (skipping vision tower)...")
            from transformers import Gemma3ForCausalLM, AutoTokenizer
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=tokenizer)
        else:
            self.model = LanguageModel(model_name, device_map=device)

        self.model_name = model_name
        self.layer_idx = layer_idx

        # Get number of layers
        self.num_layers = len(self.model.model.layers)
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer {layer_idx} out of range (model has {self.num_layers} layers)")

        # Load probe if provided
        self.probe = None
        if probe_path:
            self.load_probe(probe_path)

        # Inference settings
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        # Action mappings
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}

        print(f"✓ Engine initialized with {len(self.idx_to_action)} cognitive actions")

    def load_probe(self, probe_path: Path):
        """Load trained probe model"""
        print(f"Loading probe from {probe_path}...")
        from gpu_utils import get_optimal_device
        device = get_optimal_device()
        self.probe, metadata = load_probe(probe_path, device=device)
        print(f"✓ Probe loaded (metadata: {metadata})")

    def extract_activations(self, text: str) -> torch.Tensor:
        """
        Extract activations from specified layer for input text

        Args:
            text: Input text

        Returns:
            Activations tensor (hidden_size,)
        """
        # Append special message to create consistent extraction point
        # This primes the model to "think about" cognitive actions
        augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

        with self.model.trace(augmented_text) as tracer:
            # Extract activations from the specified layer
            hidden_states = self.model.model.layers[self.layer_idx].output[0].save()

        # Use last token representation (similar to paper's approach)
        # This is the representation after the model has "thought about" what cognitive action it is
        activations = hidden_states[:, -1, :].squeeze(0)

        return activations

    def predict(
        self,
        text: str,
        return_activations: bool = False
    ) -> List[CognitiveActionPrediction]:
        """
        Predict cognitive actions for input text

        Args:
            text: Input text to analyze
            return_activations: Whether to also return raw activations

        Returns:
            List of CognitiveActionPrediction objects, sorted by confidence
        """
        if self.probe is None:
            raise RuntimeError("No probe loaded. Call load_probe() first.")

        # Extract activations
        activations = self.extract_activations(text)

        # Run probe
        activations_batch = activations.unsqueeze(0)  # Add batch dimension
        predictions = get_probe_predictions(
            self.probe,
            activations_batch,
            top_k=self.top_k,
            threshold=self.confidence_threshold
        )[0]  # Get first (and only) batch item

        # Convert to CognitiveActionPrediction objects
        results = [
            CognitiveActionPrediction(
                action_name=self.idx_to_action[idx],
                action_idx=idx,
                confidence=conf
            )
            for idx, conf in predictions
        ]

        if return_activations:
            return results, activations
        return results

    def predict_batch(
        self,
        texts: List[str]
    ) -> List[List[CognitiveActionPrediction]]:
        """
        Predict cognitive actions for batch of texts

        Args:
            texts: List of input texts

        Returns:
            List of prediction lists, one per input text
        """
        if self.probe is None:
            raise RuntimeError("No probe loaded. Call load_probe() first.")

        # Extract activations for all texts
        all_activations = []
        for text in texts:
            act = self.extract_activations(text)
            all_activations.append(act)

        activations_batch = torch.stack(all_activations)

        # Run probe on batch
        batch_predictions = get_probe_predictions(
            self.probe,
            activations_batch,
            top_k=self.top_k,
            threshold=self.confidence_threshold
        )

        # Convert to CognitiveActionPrediction objects
        results = []
        for predictions in batch_predictions:
            text_results = [
                CognitiveActionPrediction(
                    action_name=self.idx_to_action[idx],
                    action_idx=idx,
                    confidence=conf
                )
                for idx, conf in predictions
            ]
            results.append(text_results)

        return results

    def get_action_category(self, action_name: str) -> str:
        """
        Get the category for a cognitive action

        Returns one of: metacognitive, analytical, creative, emotional, memory, evaluative
        """
        # Simple categorization based on action name
        # Could be enhanced with a proper mapping
        metacognitive_keywords = ['meta', 'monitoring', 'regulation', 'questioning', 'reconsidering', 'suspending']
        analytical_keywords = ['analyzing', 'comparing', 'distinguishing', 'deconstructing', 'inferring']
        creative_keywords = ['creating', 'divergent', 'imagining', 'hypothesizing', 'brainstorming']
        emotional_keywords = ['emotion', 'reappraisal', 'accepting', 'suppressing', 'valuing']
        memory_keywords = ['remembering', 'recalling', 'recognizing', 'memorizing']
        evaluative_keywords = ['evaluating', 'critiquing', 'judging', 'assessing']

        action_lower = action_name.lower()

        for keyword in metacognitive_keywords:
            if keyword in action_lower:
                return "metacognitive"

        for keyword in analytical_keywords:
            if keyword in action_lower:
                return "analytical"

        for keyword in creative_keywords:
            if keyword in action_lower:
                return "creative"

        for keyword in emotional_keywords:
            if keyword in action_lower:
                return "emotional"

        for keyword in memory_keywords:
            if keyword in action_lower:
                return "memory"

        for keyword in evaluative_keywords:
            if keyword in action_lower:
                return "evaluative"

        return "other"

    def format_predictions_for_display(
        self,
        predictions: List[CognitiveActionPrediction],
        show_categories: bool = True
    ) -> str:
        """
        Format predictions as a readable string

        Args:
            predictions: List of predictions
            show_categories: Whether to show action categories

        Returns:
            Formatted string
        """
        if not predictions:
            return "No cognitive actions detected above threshold"

        lines = ["Detected Cognitive Actions:"]
        for i, pred in enumerate(predictions, 1):
            category = self.get_action_category(pred.action_name) if show_categories else ""
            category_str = f" [{category}]" if category else ""
            lines.append(
                f"  {i}. {pred.action_name:30s} {pred.confidence:5.1%}{category_str}"
            )

        return "\n".join(lines)


def main():
    """Test the inference engine"""
    import argparse

    parser = argparse.ArgumentParser(description="Test probe inference")
    parser.add_argument(
        "--probe",
        type=str,
        required=True,
        help="Path to trained probe model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID"
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
        default=None,
        help="Text to analyze (if not provided, uses example)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = ProbeInferenceEngine(
        model_name=args.model,
        probe_path=Path(args.probe),
        layer_idx=args.layer,
        top_k=args.top_k
    )

    # Test text
    if args.text:
        test_text = args.text
    else:
        test_text = """After receiving feedback from her colleague, Sarah began reconsidering
        her initial approach to the project. She realized that she had been making assumptions
        without fully understanding the constraints."""

    print("\n" + "="*60)
    print("INPUT TEXT")
    print("="*60)
    print(test_text)

    # Run inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)

    predictions = engine.predict(test_text)

    # Display results
    print("\n" + engine.format_predictions_for_display(predictions, show_categories=True))

    # Test batch inference
    print("\n" + "="*60)
    print("TESTING BATCH INFERENCE")
    print("="*60)

    batch_texts = [
        "She was comparing different solutions to find the best approach.",
        "He started generating creative ideas for the new design.",
        "They were evaluating the effectiveness of their strategy."
    ]

    batch_predictions = engine.predict_batch(batch_texts)

    for i, (text, preds) in enumerate(zip(batch_texts, batch_predictions), 1):
        print(f"\n{i}. {text}")
        if preds:
            for pred in preds[:3]:  # Show top 3
                print(f"   - {pred.action_name}: {pred.confidence:.1%}")


if __name__ == "__main__":
    main()
