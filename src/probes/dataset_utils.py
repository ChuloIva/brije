"""
Dataset utilities for loading and processing cognitive actions data
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys

# Add third_party to path to import variable_pools
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "datagen"))
from variable_pools import COGNITIVE_ACTIONS


@dataclass
class CognitiveActionExample:
    """Single cognitive action training example"""
    text: str
    primary_action: str
    domain: str
    trigger: str
    emotional_state: str
    language_style: str
    sentence_starter: str


def load_cognitive_actions_dataset(
    dataset_path: str,
    limit: Optional[int] = None
) -> List[CognitiveActionExample]:
    """
    Load cognitive actions dataset from JSONL file

    Args:
        dataset_path: Path to JSONL file from datagen
        limit: Optional limit on number of examples to load

    Returns:
        List of CognitiveActionExample objects
    """
    examples = []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path}")

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line.strip())
            example = CognitiveActionExample(
                text=data['text'],
                primary_action=data['cognitive_action'],
                domain=data.get('domain', 'unknown'),
                trigger=data.get('trigger', ''),
                emotional_state=data.get('emotional_state', ''),
                language_style=data.get('language_style', ''),
                sentence_starter=data.get('sentence_starter', '')
            )
            examples.append(example)

    print(f"Loaded {len(examples)} examples")
    return examples


def create_splits(
    examples: List[CognitiveActionExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[List[CognitiveActionExample], List[CognitiveActionExample], List[CognitiveActionExample]]:
    """
    Split dataset into train/val/test sets

    Args:
        examples: List of examples to split
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify: Whether to stratify by primary_action
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    random.seed(random_seed)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    if stratify:
        # Group by primary action
        action_groups: Dict[str, List[CognitiveActionExample]] = {}
        for ex in examples:
            if ex.primary_action not in action_groups:
                action_groups[ex.primary_action] = []
            action_groups[ex.primary_action].append(ex)

        train_examples = []
        val_examples = []
        test_examples = []

        # Split each group proportionally
        for action, group in action_groups.items():
            random.shuffle(group)
            n = len(group)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_examples.extend(group[:train_end])
            val_examples.extend(group[train_end:val_end])
            test_examples.extend(group[val_end:])

        # Shuffle the combined sets
        random.shuffle(train_examples)
        random.shuffle(val_examples)
        random.shuffle(test_examples)

    else:
        # Simple random split
        shuffled = examples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_examples = shuffled[:train_end]
        val_examples = shuffled[train_end:val_end]
        test_examples = shuffled[val_end:]

    print(f"Split sizes - Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    return train_examples, val_examples, test_examples


def get_action_to_idx_mapping() -> Dict[str, int]:
    """
    Get mapping from cognitive action names to indices

    Returns:
        Dictionary mapping action names to integer indices
    """
    actions = sorted(list(COGNITIVE_ACTIONS.keys()))
    return {action: idx for idx, action in enumerate(actions)}


def get_idx_to_action_mapping() -> Dict[int, str]:
    """
    Get mapping from indices to cognitive action names

    Returns:
        Dictionary mapping integer indices to action names
    """
    action_to_idx = get_action_to_idx_mapping()
    return {idx: action for action, idx in action_to_idx.items()}


def compute_class_weights(examples: List[CognitiveActionExample]) -> Dict[str, float]:
    """
    Compute class weights for handling imbalanced data

    Args:
        examples: List of training examples

    Returns:
        Dictionary mapping action names to weights
    """
    from collections import Counter

    action_counts = Counter(ex.primary_action for ex in examples)
    total = len(examples)

    # Inverse frequency weighting
    weights = {}
    for action, count in action_counts.items():
        weights[action] = total / (len(action_counts) * count)

    return weights


def create_binary_labels(labels, target_action_idx: int):
    """
    Convert multi-class labels to binary for one-vs-rest training

    Args:
        labels: Original class indices (0-44) - can be numpy array, torch tensor, or list
        target_action_idx: The action we're training probe for

    Returns:
        Binary labels: 1.0 if label == target_action_idx, else 0.0
        Returns same type as input
    """
    import torch
    import numpy as np

    if isinstance(labels, torch.Tensor):
        return (labels == target_action_idx).float()
    elif isinstance(labels, np.ndarray):
        return (labels == target_action_idx).astype(np.float32)
    else:
        # Assume list or iterable
        return [1.0 if label == target_action_idx else 0.0 for label in labels]


def print_dataset_statistics(examples: List[CognitiveActionExample]):
    """Print statistics about the dataset"""
    from collections import Counter

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nTotal examples: {len(examples)}")

    # Action distribution
    action_counts = Counter(ex.primary_action for ex in examples)
    print(f"\nCognitive actions (top 10):")
    for action, count in action_counts.most_common(10):
        print(f"  {action:30s}: {count:4d} ({count/len(examples)*100:.1f}%)")

    # Domain distribution
    domain_counts = Counter(ex.domain for ex in examples)
    print(f"\nDomains (top 10):")
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain:30s}: {count:4d} ({count/len(examples)*100:.1f}%)")

    # Emotional state distribution
    emotional_counts = Counter(ex.emotional_state for ex in examples)
    print(f"\nEmotional states (top 10):")
    for emotional_state, count in emotional_counts.most_common(10):
        print(f"  {emotional_state:30s}: {count:4d} ({count/len(examples)*100:.1f}%)")

    # Language style distribution
    style_counts = Counter(ex.language_style for ex in examples)
    print(f"\nLanguage styles (top 10):")
    for style, count in style_counts.most_common(10):
        print(f"  {style:30s}: {count:4d} ({count/len(examples)*100:.1f}%)")

    # Text length statistics
    text_lengths = [len(ex.text) for ex in examples]
    print(f"\nText length statistics:")
    print(f"  Mean: {sum(text_lengths)/len(text_lengths):.1f} characters")
    print(f"  Min:  {min(text_lengths)} characters")
    print(f"  Max:  {max(text_lengths)} characters")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_utils.py <path_to_jsonl>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    examples = load_cognitive_actions_dataset(dataset_path)
    print_dataset_statistics(examples)

    train, val, test = create_splits(examples)
    print(f"\nTrain set size: {len(train)}")
    print(f"Val set size: {len(val)}")
    print(f"Test set size: {len(test)}")

    # Print action-to-index mapping
    action_to_idx = get_action_to_idx_mapping()
    print(f"\nNumber of unique actions: {len(action_to_idx)}")
    print("First 10 actions:")
    for i, (action, idx) in enumerate(sorted(action_to_idx.items())[:10]):
        print(f"  {idx:2d}: {action}")
