"""
Gemma 3 4B integration with cognitive action probes
"""

import sys
from pathlib import Path

# Add probe modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "probes"))

from probe_inference import ProbeInferenceEngine, CognitiveActionPrediction
from config import PROBE_PATH, PROBE_LAYER, PROBE_TOP_K, PROBE_THRESHOLD
from typing import List, Optional


class GemmaWithProbes:
    """
    Wrapper for Gemma 3 4B with integrated cognitive action probes
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        probe_path: Optional[str] = None,
        layer_idx: Optional[int] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize Gemma with probes

        Args:
            model_name: HuggingFace model ID
            probe_path: Path to trained probe (uses config default if None)
            layer_idx: Layer to extract from (uses config default if None)
            top_k: Number of predictions (uses config default if None)
            threshold: Confidence threshold (uses config default if None)
        """
        # Use config defaults if not provided
        probe_path = probe_path or PROBE_PATH
        layer_idx = layer_idx if layer_idx is not None else PROBE_LAYER
        top_k = top_k if top_k is not None else PROBE_TOP_K
        threshold = threshold if threshold is not None else PROBE_THRESHOLD

        # Resolve probe path relative to this file
        if not Path(probe_path).is_absolute():
            probe_path = Path(__file__).parent / probe_path

        print(f"Initializing Gemma with Probes...")
        print(f"  Probe: {probe_path}")

        # Initialize inference engine
        self.engine = ProbeInferenceEngine(
            model_name=model_name,
            probe_path=Path(probe_path),
            layer_idx=layer_idx,
            top_k=top_k,
            confidence_threshold=threshold
        )

        self.model = self.engine.model
        self.last_predictions: List[CognitiveActionPrediction] = []

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
        self.last_predictions = self.engine.predict(generated_text)

        return generated_text

    def get_last_predictions(self) -> List[CognitiveActionPrediction]:
        """Get cognitive action predictions from last generation"""
        return self.last_predictions

    def get_predictions_dict(self) -> List[dict]:
        """Get last predictions as list of dicts for easy serialization"""
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
