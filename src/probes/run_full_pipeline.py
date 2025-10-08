"""
Complete pipeline script: Capture → Train → Test
Run the entire probe training workflow with a single command
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path


def print_section(title, color="\033[1;33m"):
    """Print a formatted section header"""
    nc = "\033[0m"
    print(f"\n{color}{'='*65}")
    print(f"{title}")
    print(f"{'='*65}{nc}\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n\033[0;31m✗ Failed: {description}\033[0m")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete probe training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (layer 27, linear probe)
  python run_full_pipeline.py

  # Train multihead probe on layer 21
  python run_full_pipeline.py --layer 21 --probe-type multihead

  # Quick test with fewer epochs
  python run_full_pipeline.py --epochs 5 --batch-size 64

  # Skip activation capture if already done
  python run_full_pipeline.py --skip-capture
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="../../third_party/datagen/generated_data/stratified_combined_31500.jsonl",
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=27,
        help="Layer to extract activations from (default: 27)"
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "multihead"],
        default="linear",
        help="Type of probe to train (default: linear)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cuda, or cpu (default: auto)"
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Skip activation capture (use existing activations)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip inference testing"
    )

    args = parser.parse_args()

    # Paths
    activations_dir = Path("../../data/activations")
    probes_dir = Path("../../data/probes")
    activation_file = activations_dir / f"layer_{args.layer}_activations.h5"
    probe_file = probes_dir / "best_probe.pth"

    # Print configuration
    print_section("COGNITIVE ACTION PROBE TRAINING PIPELINE", "\033[1;34m")

    print("Configuration:")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Layer:       {args.layer}")
    print(f"  Probe Type:  {args.probe_type}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  Device:      {args.device}")
    print(f"  Output:")
    print(f"    Activations: {activation_file}")
    print(f"    Probe:       {probe_file}")
    print()

    start_time = time.time()

    # Step 1: Capture Activations
    if not args.skip_capture or not activation_file.exists():
        print_section("Step 1: Capturing Activations")

        if not Path(args.dataset).exists():
            print(f"\033[0;31m✗ Dataset not found: {args.dataset}\033[0m")
            sys.exit(1)

        cmd = [
            "python", "capture_activations.py",
            "--dataset", args.dataset,
            "--output-dir", str(activations_dir),
            "--layers", str(args.layer),
            "--format", "hdf5",
            "--device", args.device
        ]

        if not run_command(cmd, "Activation capture"):
            sys.exit(1)

        print("\n\033[0;32m✓ Activation capture complete!\033[0m")
    else:
        print_section("Step 1: Using Existing Activations")
        print(f"Found: {activation_file}")

    # Step 2: Train Probe
    print_section("Step 2: Training Probe")

    if not activation_file.exists():
        print(f"\033[0;31m✗ Activations not found: {activation_file}\033[0m")
        sys.exit(1)

    cmd = [
        "python", "train_probes.py",
        "--activations", str(activation_file),
        "--output-dir", str(probes_dir),
        "--model-type", args.probe_type,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--device", args.device
    ]

    if args.probe_type == "multihead":
        cmd.extend(["--hidden-dim", "512"])

    if not run_command(cmd, "Probe training"):
        sys.exit(1)

    print("\n\033[0;32m✓ Probe training complete!\033[0m")

    # Step 3: Test Inference
    if not args.skip_test:
        print_section("Step 3: Testing Inference")

        if not probe_file.exists():
            print(f"\033[0;31m✗ Probe not found: {probe_file}\033[0m")
            sys.exit(1)

        test_examples = [
            "After receiving feedback, Sarah began reconsidering her initial approach.",
            "She was comparing different solutions to find the best strategy.",
            "He started generating creative ideas for the new design.",
            "They were evaluating the effectiveness of their strategy.",
            "She noticed her own thinking patterns and questioned her assumptions."
        ]

        for i, text in enumerate(test_examples, 1):
            print(f"\n\033[1;34mTest {i}/{len(test_examples)}:\033[0m")
            print(f'"{text}"\n')

            cmd = [
                "python", "probe_inference.py",
                "--probe", str(probe_file),
                "--layer", str(args.layer),
                "--text", text,
                "--top-k", "5"
            ]

            run_command(cmd, f"Inference test {i}")

        print("\n\033[0;32m✓ Inference testing complete!\033[0m")

    # Summary
    elapsed = time.time() - start_time
    print_section("PIPELINE COMPLETE! 🎉", "\033[0;32m")

    print(f"Total time: {elapsed/60:.1f} minutes\n")

    print("Files created:")
    print(f"  📊 Activations: {activation_file}")
    print(f"  🧠 Best Probe:  {probe_file}")
    print(f"  📈 Metrics:     {probes_dir / 'test_metrics.json'}")
    print(f"  📝 History:     {probes_dir / 'training_history.json'}")
    print()

    # Load and display metrics
    metrics_file = probes_dir / "test_metrics.json"
    if metrics_file.exists():
        import json

        print("\033[1;34mTest Performance:\033[0m")
        with open(metrics_file) as f:
            metrics = json.load(f)

        print(f"  Accuracy:     {metrics['accuracy']:.1%}")
        print(f"  Macro F1:     {metrics['f1_macro']:.3f}")
        print(f"  Micro F1:     {metrics['f1_micro']:.3f}")
        print(f"  Precision:    {metrics['precision_macro']:.3f}")
        print(f"  Recall:       {metrics['recall_macro']:.3f}")
        print()

    print("\033[1;33mNext Steps:\033[0m")
    print("  1. Test with your own text:")
    print(f"     python probe_inference.py --probe {probe_file} --text 'Your text'")
    print("  2. Use in Liminal Backrooms:")
    print("     cd ../../third_party/liminal_backrooms")
    print("     python main.py")
    print("     (Select 'Gemma 3 4B (with Probes)' in the GUI)")
    print()

    print("\033[0;32mAll done! 🚀\033[0m")


if __name__ == "__main__":
    main()