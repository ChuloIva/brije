# Streaming Probe Visualizations - Quick Start

Three ways to visualize cognitive action probe activations in real-time!

---

## 🚀 Quick Install

```bash
# Install visualization dependencies
pip install textual rich
```

---

## 1. 🎮 Interactive TUI (Best for Exploration)

Navigate through tokens with keyboard controls and see detailed probe activations.

```bash
python src/probes/interactive_probe_viewer.py \
    --probes-dir data/probes_binary \
    --text "After analyzing the data, I reconsidered my assumptions."
```

**Controls:**
- `←` `→` Navigate tokens
- `Home` / `End` Jump to first/last
- `Q` Quit

**What you see:**
- Token stream with highlighting
- Detailed probe activations for selected token
- Heatmap grid of all probes × tokens
- Statistics and top activations

---

## 2. 🌊 Matrix Cascade (Best for Demos)

Watch tokens and activations cascade down Matrix-style.

```bash
 python src/probes/matrix_probe_animation.py \
      --probes-dir data/probes_binary \
      --text "I was in such a dark place back then, everything felt hopeless and I couldn't see any way forward. The anxiety was crushing me daily and I kept spiraling into these negative thought patterns where I'd convince myself that nothing would ever get better. I remember lying awake at night just replaying all my failures and mistakes over and over, feeling like I was trapped in this endless cycle of self-doubt and despair. But then something shifted when I started reaching out for help instead of isolating myself. I began talking to friends, seeking therapy, and slowly learning to challenge those destructive thoughts that had been controlling my life. It wasn't easy at first - there were setbacks and days when I felt like I was back at square one. But I kept pushing forward, developing new coping strategies, practicing mindfulness, and gradually rebuilding my confidence. I started setting small achievable goals and celebrating tiny victories instead of focusing on everything that was wrong. The breakthrough came when I realized I had the power to change my perspective and that my thoughts didn't define my reality. Now I'm in such a better place mentally, surrounded by supportive people, pursuing goals that actually matter to me, and I've learned that even in the darkest moments there's always hope if you're willing to take that first step toward healing." \
      --delay 0.4
```

**What you see:**
- Tokens flowing down like Matrix rain
- Top 3 activations per token
- Fading trails for visual effect
- Real-time processing animation

---

## 3. 📊 Static Display Modes (Best for Scripts)

Classic streaming output with different ASCII styles.

```bash
# Bars (default)
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your text here" \
    --display-mode bars

# Matrix style
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your text here" \
    --display-mode matrix

# Fire emojis
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your text here" \
    --display-mode fire

# Waves
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your text here" \
    --display-mode waves

# Pulse
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your text here" \
    --display-mode pulse
```

---

## 🎬 Run the Demo

```bash
./demo_interactive_visualizations.sh
```

This will show both interactive modes with example text!

---

## 📚 Full Documentation

- **Interactive modes**: `docs/INTERACTIVE_VISUALIZATION.md`
- **Display modes**: `docs/STREAMING_VISUAL_MODES.md`
- **Implementation**: `docs/STREAMING_PROBES.md`

---

## 💡 Use Cases

### Exploring Your Data
```bash
# Interactive TUI for deep analysis
python src/probes/interactive_probe_viewer.py \
    --probes-dir data/probes_binary \
    --text "After receiving feedback, I began reconsidering my approach." \
    --threshold 0.05
```

### Creating a Demo Video
```bash
# Matrix cascade for visual appeal
python src/probes/matrix_probe_animation.py \
    --probes-dir data/probes_binary \
    --text "Analyzing quarterly results revealed unexpected patterns." \
    --delay 0.5
```

### Debugging Probes
```bash
# Static display with verbose output
python src/probes/streaming_probe_inference.py \
    --probes-dir data/probes_binary \
    --text "Your test case here" \
    --display-mode bars \
    --threshold 1  # Show all activations
```

---

## 🔧 Python API

### Interactive Viewer

```python
from pathlib import Path
from streaming_probe_inference import StreamingProbeInferenceEngine
from interactive_probe_viewer import launch_interactive_viewer

engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-3-4b-it'
)

text = "Your text here"
predictions = engine.predict_streaming(text, top_k=50, show_realtime=False)

inputs = engine.tokenizer(text, return_tensors="pt")
tokens = engine.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

launch_interactive_viewer(predictions, tokens, processing_time=1.5)
```

### Matrix Cascade

```python
from pathlib import Path
from streaming_probe_inference import StreamingProbeInferenceEngine
from matrix_probe_animation import animate_simpler

engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-2-3b-it'
)

animate_simpler(engine, "Your text here", delay=0.4)
```

### Streaming Display

```python
from pathlib import Path
from streaming_probe_inference import StreamingProbeInferenceEngine

engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-2-3b-it'
)

predictions = engine.predict_streaming(
    "Your text here",
    top_k=10,
    threshold=0.1,
    show_realtime=True,
    display_mode='fire'  # or 'matrix', 'waves', 'pulse', 'bars'
)
```

---

## ⚡ Performance Tips

1. **Use smaller models** for faster visualization (gemma-2-3b-it vs gemma-3-4b-it)
2. **Adjust threshold** to show fewer/more activations (0.1 is good default)
3. **Shorten text** for testing (< 20 tokens processes quickly)
4. **Pre-compute** activations before visualizing (all modes do this automatically)

---

## 🎨 Choosing the Right Mode

| Mode | When to Use | Pros | Cons |
|------|-------------|------|------|
| **Interactive TUI** | Deep analysis, exploration | Full details, navigation | Requires keyboard input |
| **Matrix Cascade** | Demos, presentations | Eye-catching, animated | Limited detail |
| **Static Modes** | Scripts, automation | Fast, customizable | No interaction |

---

## 🐛 Troubleshooting

**Problem**: `ImportError: No module named 'textual'`
```bash
pip install textual rich
```

**Problem**: Colors look wrong
- Use a modern terminal (iTerm2, Windows Terminal, GNOME Terminal)
- Try a dark theme

**Problem**: Interactive TUI not displaying correctly
- Increase terminal size (minimum 120×30)
- Check terminal supports Unicode

**Problem**: Matrix animation too fast/slow
- Adjust `--delay` parameter (try 0.2-0.8 range)

**Problem**: Emojis showing as boxes
- Use a terminal with emoji support
- Fallback to `--display-mode bars` for ASCII-only

---

## 🎓 Next Steps

1. Try all three modes with your own text
2. Experiment with different `--threshold` values
3. Compare multiple texts to see patterns
4. Read full docs for advanced features
5. Customize colors/styles in the code

---

## 📖 Files

- `src/probes/interactive_probe_viewer.py` - Interactive TUI
- `src/probes/matrix_probe_animation.py` - Matrix cascade
- `src/probes/streaming_probe_inference.py` - Static streaming modes
- `src/probes/visualization_utils.py` - Shared utilities
- `demo_interactive_visualizations.sh` - Demo script

---

Enjoy exploring! 🚀
