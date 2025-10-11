"""
Gemma 3 4B integration with cognitive action probes
"""

import sys
from pathlib import Path

# Add probe modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "probes"))

from probe_inference import ProbeInferenceEngine
from best_multi_probe_inference import BestMultiProbeInferenceEngine, CognitiveActionPrediction
from universal_multi_layer_inference import UniversalMultiLayerInferenceEngine, UniversalPrediction
from config import (
    PROBE_MODE, PROBE_PATH, PROBES_DIR, PROBE_LAYER, PROBE_LAYER_RANGE,
    PROBE_TOP_K, PROBE_THRESHOLD
)
from typing import List, Optional, Union
from collections import defaultdict


class GemmaWithProbes:
    """
    Wrapper for Gemma 3 4B with integrated cognitive action probes
    Supports both binary (one-vs-rest) and multiclass probe modes
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        probe_mode: Optional[str] = None,
        probe_path: Optional[str] = None,
        probes_dir: Optional[str] = None,
        layer_idx: Optional[int] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize Gemma with probes

        Args:
            model_name: HuggingFace model ID
            probe_mode: "binary" or "multiclass" (uses config default if None)
            probe_path: Path to trained probe for multiclass mode
            probes_dir: Directory with all probes for binary mode
            layer_idx: Layer to extract from (uses config default if None)
            top_k: Number of predictions (uses config default if None)
            threshold: Confidence threshold (uses config default if None)
        """
        # Use config defaults if not provided
        probe_mode = probe_mode or PROBE_MODE
        layer_idx = layer_idx if layer_idx is not None else PROBE_LAYER
        top_k = top_k if top_k is not None else PROBE_TOP_K
        threshold = threshold if threshold is not None else PROBE_THRESHOLD

        self.probe_mode = probe_mode
        self.top_k = top_k
        self.threshold = threshold

        print(f"Initializing Gemma with Probes...")
        print(f"  Mode: {probe_mode}")

        if probe_mode == "universal":
            # Use UniversalMultiLayerInferenceEngine (all probes across all layers)
            probes_base_dir = probes_dir or PROBES_DIR

            # Resolve probes base dir relative to this file
            if not Path(probes_base_dir).is_absolute():
                probes_base_dir = Path(__file__).parent / probes_base_dir

            print(f"  Probes base dir: {probes_base_dir}")
            print(f"  Layer range: {PROBE_LAYER_RANGE}")

            self.engine = UniversalMultiLayerInferenceEngine(
                probes_base_dir=Path(probes_base_dir),
                model_name=model_name,
                layer_range=PROBE_LAYER_RANGE
            )
            self.model = self.engine.model

        elif probe_mode == "binary":
            # Use BestMultiProbeInferenceEngine (45 binary probes, each from optimal layer)
            probes_base_dir = probes_dir or PROBES_DIR

            # Resolve probes base dir relative to this file
            # This should point to data/probes_binary (parent of layer_XX dirs)
            if not Path(probes_base_dir).is_absolute():
                probes_base_dir = Path(__file__).parent / probes_base_dir

            print(f"  Probes base dir: {probes_base_dir}")

            self.engine = BestMultiProbeInferenceEngine(
                probes_base_dir=Path(probes_base_dir),
                model_name=model_name
            )
            self.model = self.engine.model

        else:
            # Use ProbeInferenceEngine (single multiclass probe)
            probe_path = probe_path or PROBE_PATH

            # Resolve probe path relative to this file
            if not Path(probe_path).is_absolute():
                probe_path = Path(__file__).parent / probe_path

            print(f"  Probe: {probe_path}")

            self.engine = ProbeInferenceEngine(
                model_name=model_name,
                probe_path=Path(probe_path),
                layer_idx=layer_idx,
                top_k=top_k,
                confidence_threshold=threshold
            )
            self.model = self.engine.model

        self.last_predictions: Union[List[CognitiveActionPrediction], List[UniversalPrediction]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate text and analyze cognitive actions

        Args:
            prompt: Input prompt
            system_prompt: System prompt (optional)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Format prompt with system prompt if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt

        # Generate with the model
        with self.model.generate(full_prompt, max_new_tokens=max_new_tokens) as generator:
            output_tokens = self.model.generator.output.save()

        # Decode the output
        generated_text = self.model.tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        # Extract just the assistant's response if system prompt was used
        if system_prompt and "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()

        # Analyze cognitive actions in the generated text
        if self.probe_mode == "universal":
            # Universal mode: run all probes across all layers
            self.last_predictions = self.engine.predict_all(
                generated_text,
                threshold=self.threshold
            )
        elif self.probe_mode == "binary":
            # Binary mode: run all probes
            self.last_predictions = self.engine.predict(
                generated_text,
                top_k=self.top_k,
                threshold=self.threshold
            )
        else:
            # Multiclass mode: single probe
            self.last_predictions = self.engine.predict(generated_text)

        return generated_text

    def get_last_predictions(self) -> List[CognitiveActionPrediction]:
        """Get cognitive action predictions from last generation"""
        return self.last_predictions

    def get_predictions_dict(self) -> List[dict]:
        """Get last predictions as list of dicts for easy serialization"""
        if self.probe_mode == "universal":
            # Universal mode: group predictions by action
            # Collect all predictions for each action across layers
            action_groups = defaultdict(lambda: {'layers': [], 'confidences': [], 'is_active_layers': []})

            for pred in self.last_predictions:
                action_groups[pred.action_name]['layers'].append(pred.layer)
                action_groups[pred.action_name]['confidences'].append(pred.confidence)
                if pred.is_active:
                    action_groups[pred.action_name]['is_active_layers'].append(pred.layer)

            # Sort actions by number of active layers (count), then by max confidence
            sorted_actions = sorted(
                action_groups.items(),
                key=lambda x: (len(x[1]['is_active_layers']), max(x[1]['confidences'])),
                reverse=True
            )

            # Format as list of dicts in the style of output_example_3.md
            result = []
            for action_name, data in sorted_actions:
                result.append({
                    'action': action_name,
                    'layers': data['is_active_layers'],  # Only layers where it's active
                    'count': len(data['is_active_layers']),
                    'max_confidence': max(data['confidences']),
                    'is_active': len(data['is_active_layers']) > 0
                })

            return result

        elif self.probe_mode == "binary":
            # Binary mode: predictions are CognitiveActionPrediction objects with layer info
            return [
                {
                    'action': pred.action_name,
                    'confidence': pred.confidence,
                    'is_active': pred.is_active,
                    'layer': pred.layer,  # Include which layer this probe is from
                    'auc': pred.auc  # Include the AUC performance metric
                }
                for pred in self.last_predictions
            ]
        else:
            # Multiclass mode: has get_action_category method
            return [
                {
                    'action': pred.action_name,
                    'confidence': pred.confidence,
                    'category': self.engine.get_action_category(pred.action_name)
                }
                for pred in self.last_predictions
            ]


# Global instance (will be initialized when needed)
_gemma_instance: Optional[GemmaWithProbes] = None


def get_gemma_instance() -> GemmaWithProbes:
    """Get or create the global Gemma instance"""
    global _gemma_instance
    if _gemma_instance is None:
        _gemma_instance = GemmaWithProbes()
    return _gemma_instance


def call_gemma_with_probes(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512
) -> dict:
    """
    Call Gemma with probes and return response with predictions

    Args:
        prompt: Input prompt
        system_prompt: Optional system prompt
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with 'response' and 'predictions' keys
    """
    gemma = get_gemma_instance()
    response = gemma.generate(prompt, system_prompt, max_new_tokens)

    return {
        'response': response,
        'predictions': gemma.get_predictions_dict()
    }


if __name__ == "__main__":
    # Test the integration
    print("Testing Gemma with Probes...")

    test_prompt = "What are the key factors to consider when making an important decision?"

    result = call_gemma_with_probes(test_prompt)

    print("\n" + "="*60)
    print("RESPONSE")
    print("="*60)
    print(result['response'])

    print("\n" + "="*60)
    print("COGNITIVE ACTIONS DETECTED")
    print("="*60)
    for pred in result['predictions']:
        print(f"  {pred['action']:30s} {pred['confidence']:5.1%}  [{pred['category']}]")
