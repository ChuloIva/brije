"""
Streaming probe inference engine with token-level activation tracking

Features:
- Real-time probe output during text generation
- Token-by-token activation recording
- Timestamps for each activation
- Visualization of activation patterns
"""

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import sys

# Add nnsight to path
NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from nnsight import LanguageModel
from transformers import AutoTokenizer

from probe_models import load_probe
from dataset_utils import get_idx_to_action_mapping


@dataclass
class TokenActivation:
    """Single token's probe activation"""
    token_id: int
    token_text: str
    token_position: int
    action_name: str
    confidence: float
    layer: int
    timestamp: float
    is_active: bool


@dataclass
class StreamingPrediction:
    """Prediction with streaming metadata"""
    action_name: str
    action_idx: int
    confidence: float
    is_active: bool
    layer: int
    auc: float
    token_activations: List[TokenActivation] = field(default_factory=list)
    peak_activation_token: Optional[str] = None
    peak_confidence: float = 0.0


@dataclass
class AggregatedPrediction:
    """Prediction aggregated across all layers"""
    action_name: str
    action_idx: int
    layers: List[int]  # All layers where this action activated
    layer_count: int  # Number of layers
    max_confidence: float  # Maximum confidence across layers
    mean_confidence: float  # Mean confidence across layers
    best_layer: int  # Layer with highest confidence
    is_active: bool  # Whether max_confidence >= threshold
    layer_predictions: List[StreamingPrediction] = field(default_factory=list)  # Individual layer predictions
    peak_activation_token: Optional[str] = None
    peak_confidence: float = 0.0


class StreamingProbeInferenceEngine:
    """
    Probe inference engine with streaming and token-level tracking

    Capabilities:
    1. Token-by-token activation extraction
    2. Real-time probe output as text is processed
    3. Recording activation history per token
    4. Visualization of activation patterns
    """

    def __init__(
        self,
        probes_base_dir: Path,
        model_name: str = "google/gemma-3-4b-it",
        device: str = None,
        verbose: bool = False,
        layer_range: Tuple[int, int] = (21, 30)
    ):
        """
        Initialize streaming inference engine

        Args:
            probes_base_dir: Base directory containing probes_binary/
            model_name: Name of the language model
            device: Device to run on (auto-detects if None)
            verbose: Whether to print detailed loading info
            layer_range: (start, end) layer range to load (inclusive)
        """
        # Auto-detect device if not provided
        if device is None:
            from gpu_utils import get_optimal_device
            device = get_optimal_device()

        self.probes_base_dir = Path(probes_base_dir)
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.layer_start, self.layer_end = layer_range
        self.layers_needed = list(range(self.layer_start, self.layer_end + 1))

        if verbose:
            print(f"Initializing StreamingProbeInferenceEngine...")
            print(f"  Probes base dir: {probes_base_dir}")
            print(f"  Model: {model_name}")
            print(f"  Device: {device}")
            print(f"  Layer range: {self.layer_start}-{self.layer_end} ({len(self.layers_needed)} layers)\n")

        # Load index to action mapping
        self.idx_to_action = get_idx_to_action_mapping()
        self.action_to_idx = {action: idx for idx, action in self.idx_to_action.items()}

        # Load all probes from all layers (not just best)
        self.probes = self._load_all_probes()

        if verbose:
            print(f"\n✓ Loaded {len(self.probes)} total probes across {len(self.layers_needed)} layers")
            print(f"  ({len(self.probes) // len(self.layers_needed)} actions × {len(self.layers_needed)} layers)")

        # Load model and tokenizer
        if verbose:
            print(f"\nLoading language model: {model_name}...")

        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        # Check if this is a VLM (has vision_config)
        if hasattr(config, 'vision_config'):
            if verbose:
                print("Detected vision-language model. Loading text-only...")
            from transformers import Gemma3ForCausalLM
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=self.tokenizer)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=self.tokenizer)

        if verbose:
            print(f"\n✓ Initialized with {len(self.probes)} probes across {len(self.layers_needed)} layers\n")

    def _load_all_probes(self) -> Dict[Tuple[str, int], Dict]:
        """
        Load all probes from all layers

        Returns:
            Dictionary mapping (action_name, layer) -> probe_info
        """
        probes = {}
        total_loaded = 0
        failed = 0

        for layer_idx in self.layers_needed:
            layer_dir = self.probes_base_dir / f"layer_{layer_idx}"

            if not layer_dir.exists():
                if self.verbose:
                    print(f"  Warning: Layer directory not found: {layer_dir}")
                continue

            # Load all probe files in this layer directory
            probe_files = sorted(layer_dir.glob("probe_*.pth"))

            for probe_path in probe_files:
                # Extract action name from filename: probe_action_name.pth
                action_name = probe_path.stem.replace("probe_", "")

                if action_name not in self.action_to_idx:
                    continue

                try:
                    probe, metadata = load_probe(probe_path, device=self.device)

                    probes[(action_name, layer_idx)] = {
                        'probe': probe,
                        'layer': layer_idx,
                        'action': action_name,
                        'metadata': metadata
                    }
                    total_loaded += 1

                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Failed to load {probe_path}: {e}")
                    failed += 1

            if self.verbose:
                print(f"  Layer {layer_idx}: loaded {len([p for p in probes if p[1] == layer_idx])} probes")

        if failed > 0 and self.verbose:
            print(f"\n  Warning: Failed to load {failed} probes")

        return probes

    def extract_token_activations(
        self,
        text: str,
        callback: Optional[Callable] = None
    ) -> Tuple[List[str], Dict[int, Dict[int, torch.Tensor]]]:
        """
        Extract activations for each token in the text using augmented prompts.

        For each token position, creates text up to that token, appends
        "\\n\\nThe cognitive action being demonstrated here is" and extracts
        the last token's activation (similar to universal_multi_layer_inference).

        Args:
            text: Input text
            callback: Optional callback function called for each token
                     callback(token_pos, token_text, layer_activations)

        Returns:
            Tuple of (tokens, layer_activations_per_token)
            - tokens: List of token strings
            - layer_activations_per_token: Dict[token_pos][layer_idx] -> activations
        """
        # First tokenize the original text to get token boundaries
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Dictionary to store activations per token per layer
        token_layer_acts = {}

        # For each token position, reconstruct text and extract with augmented prompt
        for token_pos in range(len(tokens)):
            # Decode text up to and including this token
            text_prefix = self.tokenizer.decode(input_ids[0, :token_pos + 1], skip_special_tokens=False)

            # Append the cognitive action prompt (like universal_multi_layer_inference)
            augmented_text = f"{text_prefix}\n\nThe cognitive action being demonstrated here is"

            # Extract activations from all layers for this augmented text
            all_layer_activations = {}
            with self.model.trace(augmented_text):
                for layer_idx in self.layers_needed:
                    # Get hidden states from this layer - shape: (batch_size, seq_len, hidden_dim)
                    hidden_states = self.model.model.layers[layer_idx].output[0]
                    # Save the LAST token representation
                    all_layer_activations[layer_idx] = hidden_states[:, -1, :].save()

            # Store activations for this token position
            token_layer_acts[token_pos] = {}
            for layer_idx in self.layers_needed:
                # Squeeze batch dimension
                token_layer_acts[token_pos][layer_idx] = all_layer_activations[layer_idx].squeeze(0)

            # Call callback if provided
            if callback:
                callback(token_pos, tokens[token_pos], token_layer_acts[token_pos])

        return tokens, token_layer_acts

    def predict_streaming(
        self,
        text: str,
        top_k: int = 5,
        threshold: float = 0.1,
        show_realtime: bool = True,
        display_mode: str = "bars"
    ) -> List[StreamingPrediction]:
        """
        Run probes with token-by-token tracking using augmented prompts.

        For each token position, appends "\\n\\nThe cognitive action being demonstrated here is"
        and extracts the last token's activation (similar to universal_multi_layer_inference).

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return
            threshold: Minimum confidence threshold for "active"
            show_realtime: Whether to print activations in real-time
            display_mode: Visual style - "bars", "waves", "matrix", "fire", "pulse"

        Returns:
            List of StreamingPrediction objects with token-level data
        """
        start_time = time.time()

        # Tokenize original text to get token boundaries
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        if show_realtime:
            self._print_header(len(tokens), display_mode)

        # Initialize predictions with token-level tracking
        predictions = {}
        for (action_name, layer_idx), probe_info in self.probes.items():
            key = (action_name, layer_idx)
            predictions[key] = StreamingPrediction(
                action_name=action_name,
                action_idx=self.action_to_idx[action_name],
                confidence=0.0,  # Will be updated
                is_active=False,
                layer=layer_idx,
                auc=probe_info['metadata'].get('auc', 0.0) if 'metadata' in probe_info else 0.0,
                token_activations=[]
            )

        # Process each token with augmented text
        with torch.no_grad():
            for token_pos in range(len(tokens)):
                token_text = tokens[token_pos]
                token_time = time.time()

                if show_realtime:
                    self._print_token_header(token_pos, token_text, display_mode)

                # Decode text up to and including this token
                text_prefix = self.tokenizer.decode(input_ids[0, :token_pos + 1], skip_special_tokens=False)

                # Append cognitive action prompt (like universal_multi_layer_inference)
                augmented_text = f"{text_prefix}\n\nThe cognitive action being demonstrated here is"

                # Extract activations from all layers for this augmented text
                token_layer_activations = {}
                with self.model.trace(augmented_text):
                    for layer_idx in self.layers_needed:
                        # Get hidden states from this layer - shape: (batch_size, seq_len, hidden_dim)
                        hidden_states = self.model.model.layers[layer_idx].output[0]
                        # Save the LAST token representation
                        token_layer_activations[layer_idx] = hidden_states[:, -1, :].save()

                # Collect activations for this token
                token_active_probes = []

                # Run each probe on this token's activation
                for (action_name, layer_idx), probe_info in self.probes.items():
                    probe = probe_info['probe']

                    # Get activation for this token at this layer (last token of augmented text)
                    act = token_layer_activations[layer_idx].squeeze(0)

                    # Run probe
                    logits = probe(act)
                    confidence = torch.sigmoid(logits).item()

                    # Record token activation
                    token_activation = TokenActivation(
                        token_id=input_ids[0, token_pos].item(),
                        token_text=token_text,
                        token_position=token_pos,
                        action_name=action_name,
                        confidence=confidence,
                        layer=layer_idx,
                        timestamp=token_time - start_time,
                        is_active=confidence >= threshold
                    )

                    key = (action_name, layer_idx)
                    predictions[key].token_activations.append(token_activation)

                    # Update peak if this is highest confidence
                    if confidence > predictions[key].peak_confidence:
                        predictions[key].peak_confidence = confidence
                        predictions[key].peak_activation_token = token_text

                    # Collect active probes for display
                    if confidence >= threshold:
                        token_active_probes.append((action_name, confidence, layer_idx))

                # Show real-time activations for this token
                if show_realtime and token_active_probes:
                    self._print_activations(token_active_probes, display_mode)

        # Compute final aggregated confidence (using last token)
        for key, pred in predictions.items():
            if pred.token_activations:
                pred.confidence = pred.token_activations[-1].confidence
                pred.is_active = pred.confidence >= threshold

        # Sort by confidence
        sorted_predictions = sorted(predictions.values(), key=lambda x: x.confidence, reverse=True)

        # Return top-k
        top_predictions = [p for p in sorted_predictions[:top_k] if p.is_active]
        if len(top_predictions) == 0:
            top_predictions = sorted_predictions[:top_k]

        if show_realtime:
            print("\n" + "=" * 80)
            print(f"Processing complete ({time.time() - start_time:.2f}s)")

        return top_predictions

    def aggregate_predictions(
        self,
        predictions: List[StreamingPrediction],
        threshold: float = 0.1
    ) -> List[AggregatedPrediction]:
        """
        Aggregate predictions by action across all layers

        Args:
            predictions: List of StreamingPrediction objects (one per action-layer pair)
            threshold: Minimum confidence threshold for "active"

        Returns:
            List of AggregatedPrediction objects, sorted by layer_count then max_confidence
        """
        from collections import defaultdict

        # Group by action name
        action_groups = defaultdict(list)
        for pred in predictions:
            action_groups[pred.action_name].append(pred)

        # Create aggregated predictions
        aggregated = []
        for action_name, preds in action_groups.items():
            # Sort by confidence to find best layer
            preds_sorted = sorted(preds, key=lambda x: x.confidence, reverse=True)
            best_pred = preds_sorted[0]

            # Get all layers and confidences
            all_layers = [p.layer for p in preds]
            all_confidences = [p.confidence for p in preds]
            
            # Get only ACTIVE layers (above threshold)
            active_layers = [p.layer for p in preds if p.confidence >= threshold]
            active_confidences = [p.confidence for p in preds if p.confidence >= threshold]

            # Calculate aggregates
            max_conf = max(all_confidences)
            mean_conf = sum(all_confidences) / len(all_confidences)

            # Find overall peak activation
            peak_conf = 0.0
            peak_token = None
            for pred in preds:
                if pred.peak_confidence > peak_conf:
                    peak_conf = pred.peak_confidence
                    peak_token = pred.peak_activation_token

            agg_pred = AggregatedPrediction(
                action_name=action_name,
                action_idx=best_pred.action_idx,
                layers=sorted(active_layers),  # Only active layers
                layer_count=len(active_layers),  # Count only active layers
                max_confidence=max_conf,
                mean_confidence=mean_conf,
                best_layer=best_pred.layer,
                is_active=max_conf >= threshold,
                layer_predictions=preds,  # Keep all for reference
                peak_activation_token=peak_token,
                peak_confidence=peak_conf
            )
            aggregated.append(agg_pred)

        # Sort by layer count (descending) then max confidence (descending)
        aggregated.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)

        return aggregated

    def _print_header(self, num_tokens: int, mode: str):
        """Print streaming header based on display mode"""
        if mode == "matrix":
            print(f"\n╔{'═' * 78}╗")
            print(f"║{'  COGNITIVE ACTION MATRIX STREAM':^78}║")
            print(f"║{f'  Processing {num_tokens} tokens...':^78}║")
            print(f"╚{'═' * 78}╝\n")
        elif mode == "fire":
            print(f"\n🔥 {'━' * 76} 🔥")
            print(f"   NEURAL ACTIVATION STREAM - {num_tokens} tokens")
            print(f"🔥 {'━' * 76} 🔥\n")
        elif mode == "waves":
            print(f"\n{'~' * 80}")
            print(f"{'STREAMING COGNITIVE WAVES':^80}")
            print(f"{f'{num_tokens} tokens':^80}")
            print(f"{'~' * 80}\n")
        elif mode == "pulse":
            print(f"\n{'◆' * 40}")
            print(f"{'⚡ LIVE PROBE ACTIVATIONS ⚡':^80}")
            print(f"{f'{num_tokens} tokens to process':^80}")
            print(f"{'◆' * 40}\n")
        else:  # bars
            print(f"\n{'=' * 80}")
            print(f"{'STREAMING PROBE INFERENCE':^80}")
            print(f"{f'Processing {num_tokens} tokens...':^80}")
            print(f"{'=' * 80}\n")

    def _print_token_header(self, pos: int, token: str, mode: str):
        """Print token header based on display mode"""
        if mode == "matrix":
            print(f"\n┌─ Token [{pos:3d}] {'─' * 20}")
            print(f"│  📍 '{token}'")
        elif mode == "fire":
            print(f"\n🔸 Token {pos:3d}: '{token}'")
        elif mode == "waves":
            print(f"\n{'~'*3} Token {pos:3d}: '{token}' {'~'*3}")
        elif mode == "pulse":
            print(f"\n◇ [{pos:3d}] '{token}'")
        else:  # bars
            print(f"\nToken {pos:3d}: '{token}'")

    def _print_activations(self, activations: List[Tuple[str, float, int]], mode: str):
        """Print activations based on display mode"""
        # Sort by confidence
        activations = sorted(activations, key=lambda x: x[1], reverse=True)

        if mode == "matrix":
            for action, conf, layer in activations:
                blocks = self._get_matrix_blocks(conf)
                print(f"│  {blocks} {action[:25]:25s} {conf:5.1%} L{layer}")
            print(f"└{'─' * 60}")

        elif mode == "fire":
            for action, conf, layer in activations:
                flame = self._get_flame_intensity(conf)
                print(f"   {flame} {action[:25]:25s} {conf:5.1%} L{layer}")

        elif mode == "waves":
            for action, conf, layer in activations:
                wave = self._get_wave_pattern(conf)
                print(f"    {wave} {action[:25]:25s} {conf:5.1%} L{layer}")

        elif mode == "pulse":
            for action, conf, layer in activations:
                pulse = self._get_pulse_pattern(conf)
                print(f"  {pulse} {action[:25]:25s} {conf:5.1%} L{layer}")

        else:  # bars
            for action, conf, layer in activations:
                bar = self._get_bar(conf)
                print(f"  {bar} {action[:25]:25s} {conf:5.1%} L{layer}")

    def _get_bar(self, confidence: float) -> str:
        """Classic bar visualization"""
        length = int(confidence * 20)
        return f"✓ {'█' * length}{' ' * (20 - length)}"

    def _get_matrix_blocks(self, confidence: float) -> str:
        """Matrix-style block visualization"""
        chars = ['░', '▒', '▓', '█']
        blocks = []
        for i in range(10):
            threshold = i / 10
            if confidence > threshold:
                idx = min(int((confidence - threshold) * 10 * 4), 3)
                blocks.append(chars[idx])
            else:
                blocks.append(' ')
        return ''.join(blocks)

    def _get_flame_intensity(self, confidence: float) -> str:
        """Fire/flame visualization"""
        if confidence < 0.2:
            return "💨"
        elif confidence < 0.4:
            return "🔥"
        elif confidence < 0.6:
            return "🔥🔥"
        elif confidence < 0.8:
            return "🔥🔥🔥"
        else:
            return "🔥🔥🔥🔥"

    def _get_wave_pattern(self, confidence: float) -> str:
        """Wave/ocean visualization"""
        if confidence < 0.2:
            return "·····"
        elif confidence < 0.4:
            return "～～～～～"
        elif confidence < 0.6:
            return "≈≈≈≈≈"
        elif confidence < 0.8:
            return "∿∿∿∿∿"
        else:
            return "〰〰〰〰〰"

    def _get_pulse_pattern(self, confidence: float) -> str:
        """Pulse/heartbeat visualization"""
        length = int(confidence * 10)
        chars = ['◇', '◆']
        pattern = []
        for i in range(10):
            if i < length:
                # Alternate for pulse effect
                pattern.append(chars[i % 2])
            else:
                pattern.append('·')
        return ''.join(pattern)

    def generate_with_cognitive_tracking(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        threshold: float = 0.1,
        show_realtime: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[str, List[StreamingPrediction]]:
        """
        Generate text while tracking cognitive actions in real-time

        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum tokens to generate
            threshold: Confidence threshold for active predictions
            show_realtime: Show real-time cognitive activations during generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Tuple of (generated_text, predictions_list)
        """
        if show_realtime:
            print(f"\n{'=' * 80}")
            print(f"{'GENERATING WITH COGNITIVE TRACKING':^80}")
            print(f"{'=' * 80}\n")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
            print(f"{'=' * 80}\n")

        # Generate text using the underlying model from the nnsight wrapper
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Access the underlying model from the LanguageModel wrapper
        # The nnsight LanguageModel stores the actual model in ._model
        underlying_model = self.model._model

        # Generate with the underlying model
        with torch.no_grad():
            outputs = underlying_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the generated text to get only the new content
        # Use the tokenizer to properly extract just the generated tokens
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_text = self.tokenizer.decode(outputs[0][len(prompt_tokens):], skip_special_tokens=True)

        if show_realtime:
            print(f"\nGenerated Response:")
            print(f"{'-' * 80}")
            print(response_text)
            print(f"{'-' * 80}\n")
            print(f"\nNow analyzing cognitive actions during generation...\n")

        # Now analyze the full generated text for cognitive actions
        full_text = prompt + "\n" + response_text
        predictions = self.predict_streaming(
            full_text,
            top_k=len(self.probes),
            threshold=threshold,
            show_realtime=show_realtime
        )

        return response_text, predictions

    def visualize_token_activations(
        self,
        predictions: List[StreamingPrediction],
        action_name: Optional[str] = None
    ):
        """
        Visualize token-by-token activations for a specific action

        Args:
            predictions: List of predictions from predict_streaming
            action_name: Specific action to visualize (if None, shows top action)
        """
        # Find the prediction to visualize
        if action_name:
            pred = next((p for p in predictions if p.action_name == action_name), None)
            if not pred:
                print(f"Action '{action_name}' not found in predictions")
                return
        else:
            pred = predictions[0]  # Top prediction

        print(f"\n{'=' * 80}")
        print(f"Token-level activations for: {pred.action_name}")
        print(f"Layer: {pred.layer} | Final confidence: {pred.confidence:.2%} | AUC: {pred.auc:.3f}")
        print(f"{'=' * 80}\n")

        # Print header
        print(f"{'Pos':>4} | {'Token':20} | {'Confidence':>10} | {'Bar':30}")
        print("-" * 70)

        # Print each token with bar visualization
        for tok_act in pred.token_activations:
            bar_length = int(tok_act.confidence * 30)
            bar = "█" * bar_length
            marker = "✓" if tok_act.is_active else " "

            print(f"{tok_act.token_position:4d} | {tok_act.token_text:20} | "
                  f"{tok_act.confidence:9.2%} | {bar} {marker}")

        print(f"\nPeak activation: '{pred.peak_activation_token}' ({pred.peak_confidence:.2%})")

    def export_activations_csv(
        self,
        predictions: List[StreamingPrediction],
        output_path: Path
    ):
        """
        Export token-level activations to CSV for analysis

        Args:
            predictions: List of predictions from predict_streaming
            output_path: Path to save CSV file
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'action_name', 'token_position', 'token_text', 'confidence',
                'layer', 'timestamp', 'is_active'
            ])

            for pred in predictions:
                for tok_act in pred.token_activations:
                    writer.writerow([
                        pred.action_name,
                        tok_act.token_position,
                        tok_act.token_text,
                        tok_act.confidence,
                        tok_act.layer,
                        tok_act.timestamp,
                        tok_act.is_active
                    ])

        print(f"✓ Exported activations to {output_path}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Streaming probe inference with token tracking")
    parser.add_argument(
        "--probes-dir",
        type=str,
        required=True,
        help="Base directory containing probes_binary/"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Language model name"
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
    parser.add_argument(
        "--visualize",
        type=str,
        help="Action name to visualize token activations"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Path to export activations CSV"
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable real-time output"
    )
    parser.add_argument(
        "--display-mode",
        type=str,
        default="bars",
        choices=["bars", "matrix", "fire", "waves", "pulse"],
        help="Visual display mode for streaming output"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = StreamingProbeInferenceEngine(
        probes_base_dir=args.probes_dir,
        model_name=args.model,
        verbose=True
    )

    # Test text
    if args.text:
        test_text = args.text
    else:
        test_text = """After receiving feedback, I began reconsidering my approach.
        I realized I had been making assumptions without fully understanding the constraints."""

    print(f"\nAnalyzing text: \"{test_text}\"\n")

    # Run streaming inference (get ALL predictions, not just top-k)
    all_predictions = engine.predict_streaming(
        test_text,
        top_k=len(engine.probes),  # Get all for aggregation
        threshold=0.0,  # Get all, filter during aggregation
        show_realtime=not args.no_realtime,
        display_mode=args.display_mode
    )

    # Aggregate predictions by action across layers
    aggregated_predictions = engine.aggregate_predictions(all_predictions, threshold=args.threshold)

    # Show summary
    print(f"\n{'=' * 80}")
    print(f"AGGREGATED RESULTS (Top {args.top_k})")
    print(f"{'=' * 80}\n")

    for i, pred in enumerate(aggregated_predictions[:args.top_k], 1):
        marker = "✓" if pred.is_active else "○"
        # Format layer list - show first 3 layers
        layer_str = ','.join([f"L{l}" for l in pred.layers[:3]])
        if len(pred.layers) > 3:
            layer_str += f",+{len(pred.layers)-3}"

        print(f"{marker} {i:2d}. {pred.action_name:30s} {pred.max_confidence:6.2%}  "
              f"({pred.layer_count} layers: {layer_str}, Peak: {pred.peak_confidence:.2%} at '{pred.peak_activation_token}')")

    # Visualize if requested
    if args.visualize:
        engine.visualize_token_activations(all_predictions, args.visualize)
    else:
        # Visualize top prediction by default
        if aggregated_predictions:
            # Get the best layer prediction for the top action
            top_action_preds = aggregated_predictions[0].layer_predictions
            if top_action_preds:
                engine.visualize_token_activations(top_action_preds)

    # Export if requested
    if args.export_csv:
        engine.export_activations_csv(all_predictions, Path(args.export_csv))


if __name__ == "__main__":
    main()