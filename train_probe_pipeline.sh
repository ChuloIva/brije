#!/bin/bash
# Complete Pipeline: Capture Activations → Train Probe → Test Inference
# Single script to train a cognitive action probe from scratch

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration
DATASET_PATH="./third_party/datagen/generated_data/stratified_combined_31500.jsonl"
ACTIVATIONS_DIR="./data/activations"
PROBES_DIR="./data/probes"
MODEL="google/gemma-3-4b-it"
LAYER=27
PROBE_TYPE="linear"
BATCH_SIZE=32
EPOCHS=20
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --layer)
            LAYER="$2"
            shift 2
            ;;
        --probe-type)
            PROBE_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset PATH       Path to dataset JSONL file"
            echo "                       (default: stratified_combined_31500.jsonl)"
            echo "  --layer NUM          Layer to extract activations from (default: 27)"
            echo "  --probe-type TYPE    Probe type: linear or multihead (default: linear)"
            echo "  --epochs NUM         Training epochs (default: 20)"
            echo "  --batch-size NUM     Batch size for training (default: 32)"
            echo "  --device DEVICE      Device: auto, cuda, or cpu (default: auto)"
            echo "  --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --layer 27 --probe-type linear --epochs 20"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "==================================================================="
echo "  COGNITIVE ACTION PROBE TRAINING PIPELINE"
echo "==================================================================="
echo -e "${NC}"

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET_PATH"
echo "  Model: $MODEL"
echo "  Layer: $LAYER"
echo "  Probe Type: $PROBE_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Output:"
echo "    - Activations: $ACTIVATIONS_DIR"
echo "    - Probes: $PROBES_DIR"
echo ""

# Confirm to proceed
read -p "Start training pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Change to probe directory
cd src/probes

# Step 1: Check if activations already exist
ACTIVATION_FILE="../../$ACTIVATIONS_DIR/layer_${LAYER}_activations.h5"

if [ -f "$ACTIVATION_FILE" ]; then
    echo -e "\n${YELLOW}Step 1: Activations${NC}"
    echo -e "${GREEN}✓ Activations already exist at: $ACTIVATION_FILE${NC}"
    echo "  Skipping activation capture..."

    read -p "Re-capture activations? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SKIP_CAPTURE=false
    else
        SKIP_CAPTURE=true
    fi
else
    SKIP_CAPTURE=false
fi

# Step 1: Capture Activations
if [ "$SKIP_CAPTURE" = false ]; then
    echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}Step 1: Capturing Activations from Layer $LAYER${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

    # Check if dataset exists
    if [ ! -f "../../$DATASET_PATH" ]; then
        echo -e "${RED}ERROR: Dataset not found at ../../$DATASET_PATH${NC}"
        exit 1
    fi

    DATASET_SIZE=$(wc -l < "../../$DATASET_PATH")
    echo "  Dataset size: $DATASET_SIZE examples"
    echo "  Estimated time: 2-3 hours on GPU"
    echo ""

    python capture_activations.py \
        --dataset "../../$DATASET_PATH" \
        --output-dir "../../$ACTIVATIONS_DIR" \
        --model "$MODEL" \
        --layers $LAYER \
        --format hdf5 \
        --device "$DEVICE"

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Activation capture complete!${NC}"
    else
        echo -e "\n${RED}✗ Activation capture failed!${NC}"
        exit 1
    fi
fi

# Step 2: Train Probe
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 2: Training $PROBE_TYPE Probe${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

if [ ! -f "$ACTIVATION_FILE" ]; then
    echo -e "${RED}ERROR: Activations not found at $ACTIVATION_FILE${NC}"
    exit 1
fi

echo "  Input: $ACTIVATION_FILE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Estimated time: 15-30 minutes on GPU"
echo ""

# Build training command
TRAIN_CMD="python train_probes.py \
    --activations $ACTIVATION_FILE \
    --output-dir ../../$PROBES_DIR \
    --model-type $PROBE_TYPE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --device $DEVICE"

# Add hidden dim for multihead probe
if [ "$PROBE_TYPE" = "multihead" ]; then
    TRAIN_CMD="$TRAIN_CMD --hidden-dim 512"
fi

$TRAIN_CMD

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Probe training complete!${NC}"
else
    echo -e "\n${RED}✗ Probe training failed!${NC}"
    exit 1
fi

# Step 3: Test Inference
echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Step 3: Testing Probe Inference${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"

PROBE_FILE="../../$PROBES_DIR/best_probe.pth"

if [ ! -f "$PROBE_FILE" ]; then
    echo -e "${RED}ERROR: Trained probe not found at $PROBE_FILE${NC}"
    exit 1
fi

# Test examples covering different cognitive patterns
TEST_EXAMPLES=(
    "After receiving feedback from her colleague, Sarah began reconsidering her initial approach to the project."
    "She was carefully comparing different solutions to find the most effective strategy."
    "He started generating creative ideas for the new design, exploring unexpected possibilities."
    "They were critically evaluating the effectiveness of their current strategy."
    "In that moment, she noticed her own thinking patterns and began questioning her assumptions."
)

echo "Running inference on test examples..."
echo ""

for i in "${!TEST_EXAMPLES[@]}"; do
    echo -e "${BLUE}Test $((i+1))/${#TEST_EXAMPLES[@]}:${NC}"
    echo "  \"${TEST_EXAMPLES[$i]}\""
    echo ""

    python probe_inference.py \
        --probe "$PROBE_FILE" \
        --model "$MODEL" \
        --layer $LAYER \
        --text "${TEST_EXAMPLES[$i]}" \
        --top-k 5

    echo ""
done

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Inference testing complete!${NC}"
else
    echo -e "${RED}✗ Inference testing failed!${NC}"
    exit 1
fi

# Step 4: Summary
echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   PIPELINE COMPLETE! 🎉${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"

echo ""
echo "Files created:"
echo "  📊 Activations: $ACTIVATION_FILE"
echo "  🧠 Best Probe:  $PROBE_FILE"
echo "  📈 Metrics:     ../../$PROBES_DIR/test_metrics.json"
echo "  📝 History:     ../../$PROBES_DIR/training_history.json"
echo ""

# Display test metrics if available
METRICS_FILE="../../$PROBES_DIR/test_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    echo -e "${BLUE}Test Performance:${NC}"

    # Extract key metrics using python
    python3 << EOF
import json
import sys

try:
    with open("$METRICS_FILE", 'r') as f:
        metrics = json.load(f)

    print(f"  Accuracy:     {metrics['accuracy']:.1%}")
    print(f"  Macro F1:     {metrics['f1_macro']:.3f}")
    print(f"  Micro F1:     {metrics['f1_micro']:.3f}")
    print(f"  Precision:    {metrics['precision_macro']:.3f}")
    print(f"  Recall:       {metrics['recall_macro']:.3f}")
except Exception as e:
    print(f"  Could not parse metrics: {e}", file=sys.stderr)
EOF

    echo ""
fi

echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review metrics: cat $METRICS_FILE"
echo "  2. Test with custom text:"
echo "     cd src/probes"
echo "     python probe_inference.py --probe $PROBE_FILE --text 'Your text here'"
echo "  3. Use in Liminal Backrooms:"
echo "     cd third_party/liminal_backrooms"
echo "     python main.py"
echo "     (Select 'Gemma 3 4B (with Probes)' in the GUI)"
echo ""

echo -e "${GREEN}All done! 🚀${NC}"