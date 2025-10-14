#!/bin/bash

# Demo script for interactive probe visualizations

echo "=========================================="
echo "Interactive Probe Visualization Demo"
echo "=========================================="
echo ""

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import textual" 2>/dev/null || echo "⚠️  Warning: textual not installed. Run: pip install textual"
python -c "import rich" 2>/dev/null || echo "⚠️  Warning: rich not installed. Run: pip install rich"
echo ""

# Example text
TEXT="After receiving feedback, I began reconsidering my approach and analyzing the problem more carefully."

echo "Demo text:"
echo "  \"$TEXT\""
echo ""
echo "=========================================="
echo ""

# Demo 1: Interactive TUI Viewer
echo "1. INTERACTIVE TUI VIEWER"
echo "   Full-screen interface with navigation"
echo ""
echo "   Controls:"
echo "   - ←/→ arrows: Navigate tokens"
echo "   - Home/End: Jump to first/last"
echo "   - Q: Quit"
echo ""
read -p "Press Enter to launch Interactive TUI..."

python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --text "$TEXT" \
    --threshold 0.1

echo ""
echo "=========================================="
echo ""

# Demo 2: Matrix Cascade Animation
echo "2. MATRIX CASCADE ANIMATION"
echo "   Matrix-style cascading visualization"
echo ""
echo "   - Tokens cascade down like Matrix rain"
echo "   - Older entries fade out"
echo "   - Top 3 activations shown per token"
echo ""
read -p "Press Enter to launch Matrix Cascade..."

python src/probes/matrix_probe_animation.py \
    --probes-dir data \
    --text "$TEXT" \
    --delay 0.4 \
    --threshold 0.1

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Try your own text:"
echo ""
echo "  # Interactive TUI"
echo "  python src/probes/interactive_probe_viewer.py \\"
echo "      --probes-dir data \\"
echo "      --text \"Your text here\""
echo ""
echo "  # Matrix Cascade"
echo "  python src/probes/matrix_probe_animation.py \\"
echo "      --probes-dir data \\"
echo "      --text \"Your text here\" \\"
echo "      --delay 0.3"
echo ""
echo "See docs/INTERACTIVE_VISUALIZATION.md for full documentation!"
echo ""
