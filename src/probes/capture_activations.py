"""
Activation capture script for Gemma 3 4B using nnsight

This script:
1. Loads Gemma 3 4B from HuggingFace
2. Loads cognitive actions dataset from datagen
3. Uses nnsight to extract activations at multiple layers
4. Saves activations to disk for probe training
"""

# FOR AMD GPU
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AMD_SERIALIZE_KERNEL"] = "3"
os.environ["TORCH_USE_HIP_DSA"] = "1"

import sys
from pathlib import Path
import torch
import pickle
import h5py
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse

# Add nnsight to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"))
from nnsight import LanguageModel

# Import our dataset utilities
from dataset_utils import (
    load_cognitive_actions_dataset,
    create_splits,
    get_action_to_idx_mapping,
    print_dataset_statistics
)


class ActivationCapture:
    """Captures activations from Gemma 3 4B at specified layers"""

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",  # Gemma 3 4B instruction-tuned
        device: str = "auto",
        layers_to_capture: Optional[List[int]] = None
    ):
        """
        Initialize activation capture

        Args:
            model_name: HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', or 'auto')
            layers_to_capture: Which transformer layers to capture (default: [6, 12, 18, 24])
        """
        print(f"Loading model: {model_name}")

        # For VLMs like Gemma-3, skip loading vision tower to save memory and enable text-only usage
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_name)

        # Check if this is a VLM (has vision_config)
        if hasattr(config, 'vision_config'):
            print("Detected vision-language model. Loading text-only (skipping vision tower)...")
            # Load with Gemma3ForCausalLM to skip vision tower
            from transformers import Gemma3ForCausalLM, AutoTokenizer
            base_model = Gemma3ForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype="auto"
            )
            # Load tokenizer separately
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = LanguageModel(base_model, tokenizer=tokenizer)
        else:
            self.model = LanguageModel(model_name, device_map=device)

        self.model_name = model_name

        # Get number of layers in the model
        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        print(f"Model has {self.num_layers} transformer layers")

        # Default to capturing from evenly spaced layers
        if layers_to_capture is None:
            # For gemma-2-3b (28 layers), capture at layers [7, 14, 21, 27]
            self.layers_to_capture = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4,
                self.num_layers - 1
            ]
        else:
            self.layers_to_capture = layers_to_capture

        print(f"Will capture activations from layers: {self.layers_to_capture}")

        self.action_to_idx = get_action_to_idx_mapping()

    def capture_single_example(
        self,
        text: str,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Capture activations for a single text at a specific layer

        Args:
            text: Input text
            layer_idx: Which layer to capture from

        Returns:
            Tensor of activations (hidden_size,) - mean pooled over sequence length
        """
        with self.model.trace(text) as tracer:
            # Access the layer's output
            # Use self.layers to handle both text-only and VLM architectures
            hidden_states = self.layers[layer_idx].output[0].save()

        # hidden_states shape: (batch_size, seq_len, hidden_size)
        # Mean pool over sequence length to get (batch_size, hidden_size)
        # Then squeeze batch dimension
        activations = hidden_states.mean(dim=1).squeeze(0)

        return activations

    def capture_dataset(
        self,
        examples: List,
        layer_idx: int,
        max_examples: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Capture activations for entire dataset at a specific layer

        Args:
            examples: List of CognitiveActionExample objects
            layer_idx: Which layer to capture from
            max_examples: Optional limit on number of examples
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (activations, labels)
            - activations: (num_examples, hidden_size)
            - labels: (num_examples,) integer labels
        """
        if max_examples:
            examples = examples[:max_examples]

        activations_list = []
        labels_list = []

        iterator = tqdm(examples, desc=f"Layer {layer_idx}") if show_progress else examples

        for example in iterator:
            try:
                # Capture activations
                act = self.capture_single_example(example.text, layer_idx)
                activations_list.append(act.cpu())

                # Get label
                label = self.action_to_idx[example.primary_action]
                labels_list.append(label)

            except Exception as e:
                print(f"\nError processing example: {e}")
                print(f"Text: {example.text[:100]}...")
                continue

        # Stack into tensors
        activations = torch.stack(activations_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

        return activations, labels

    def save_activations_pickle(
        self,
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        output_path: Path
    ):
        """
        Save activations to pickle file

        Args:
            activations: Dict mapping split names to (activations, labels) tuples
            output_path: Where to save the pickle file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model_name': self.model_name,
            'layers': self.layers_to_capture,
            'action_to_idx': self.action_to_idx,
            'activations': activations
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nSaved activations to {output_path}")

    def save_activations_hdf5(
        self,
        activations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        output_path: Path
    ):
        """
        Save activations to HDF5 file (more efficient for large datasets)

        Args:
            activations: Dict mapping split names to (activations, labels) tuples
            output_path: Where to save the HDF5 file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['model_name'] = self.model_name
            f.attrs['layers'] = self.layers_to_capture

            # Save each split
            for split_name, (acts, labs) in activations.items():
                grp = f.create_group(split_name)
                grp.create_dataset('activations', data=acts.numpy())
                grp.create_dataset('labels', data=labs.numpy())

        print(f"\nSaved activations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture activations from Gemma 3 4B")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset from datagen"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/activations",
        help="Directory to save activations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to capture (default: evenly spaced)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pickle", "hdf5"],
        default="hdf5",
        help="Output format"
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    examples = load_cognitive_actions_dataset(args.dataset, limit=args.max_examples)
    print_dataset_statistics(examples)

    # Create splits
    print("\nCreating train/val/test splits...")
    train_examples, val_examples, test_examples = create_splits(
        examples,
        stratify=True,
        random_seed=42
    )

    # Initialize capture
    capture = ActivationCapture(
        model_name=args.model,
        device=args.device,
        layers_to_capture=args.layers
    )

    # Capture activations for each layer and split
    all_activations = {}

    for layer_idx in capture.layers_to_capture:
        print(f"\n{'='*60}")
        print(f"Capturing activations from layer {layer_idx}")
        print(f"{'='*60}")

        # Capture for each split
        print("\nProcessing train split...")
        train_acts, train_labels = capture.capture_dataset(train_examples, layer_idx)

        print("\nProcessing validation split...")
        val_acts, val_labels = capture.capture_dataset(val_examples, layer_idx)

        print("\nProcessing test split...")
        test_acts, test_labels = capture.capture_dataset(test_examples, layer_idx)

        # Store
        all_activations[f"layer_{layer_idx}"] = {
            'train': (train_acts, train_labels),
            'val': (val_acts, val_labels),
            'test': (test_acts, test_labels)
        }

        print(f"\nLayer {layer_idx} activation shapes:")
        print(f"  Train: {train_acts.shape}")
        print(f"  Val:   {val_acts.shape}")
        print(f"  Test:  {test_acts.shape}")

    # Save activations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, splits in all_activations.items():
        if args.format == "pickle":
            output_path = output_dir / f"{layer_name}_activations.pkl"
            capture.save_activations_pickle({layer_name: splits}, output_path)
        else:  # hdf5
            output_path = output_dir / f"{layer_name}_activations.h5"
            capture.save_activations_hdf5(splits, output_path)

    print("\n" + "="*60)
    print("ACTIVATION CAPTURE COMPLETE")
    print("="*60)
    print(f"\nActivations saved to: {output_dir}")
    print(f"Layers captured: {capture.layers_to_capture}")
    print(f"Number of classes: {len(capture.action_to_idx)}")


if __name__ == "__main__":
    main()
