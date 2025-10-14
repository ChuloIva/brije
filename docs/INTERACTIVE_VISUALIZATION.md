# Interactive Probe Visualizations

Two powerful terminal-based interactive visualizations for exploring cognitive action probe activations!

---

## 1. 🎮 Interactive TUI Viewer

Full-screen terminal UI with keyboard navigation and multiple panels.

### Features

- **Token Stream Panel**: See all tokens with highlighting
- **Probe Details Panel**: Detailed activations for selected token
- **Activation Heatmap**: Visual grid of all probes × all tokens
- **Statistics Panel**: Summary stats and top probes
- **Keyboard Navigation**: Arrow keys to explore, no mouse needed!

### Screenshot (Conceptual)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Token Stream (←/→ to navigate)                                              │
│ After receiving [feedback] , I began reconsidering my approach              │
│                   ↑ selected                                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐┌─────────────────────────────────────┐
│ Token [3]: 'feedback'               ││ Activation Heatmap (Top 10 Probes) │
│                                     ││                                     │
│ Probe               Conf    Bar     ││ Probe         0 1 2 3 4 5 6 7 8 9  │
│ Reconsidering      28.4% ██████  ✓ ││ Reconsidering · · · ● · · · ● · ·  │
│ Accepting          15.8% ████    ✓ ││ Accepting     · · · ◐ · · · · · ·  │
│ Evaluating         12.1% ███     ✓ ││ Analyzing     · · · · · · · ◐ · ·  │
│ Analyzing           9.8% ███       ││ Evaluating    · · · ○ · · · · · ·  │
│ ...                                 ││ ...                                 │
│                                     ││                                     │
└─────────────────────────────────────┘└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ Statistics                          │
│                                     │
│ Tokens: 12                          │
│ Probes: 45                          │
│ Time: 1.24s                         │
│                                     │
│ Top 5 Active:                       │
│   1. 🔥 Reconsidering               │
│   2. ⚡ Accepting                    │
│   3. 💫 Analyzing                   │
│   4. 💨 Evaluating                  │
│   5. 💨 Planning                    │
└─────────────────────────────────────┘

Q: Quit │ ←/→: Navigate │ Home/End: Jump │ R: Reload
```

### Usage

#### Command Line

```bash
# Basic usage
python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --text "After analyzing the data, I reconsidered my assumptions."

# With custom model
python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --model google/gemma-3-4b-it \
    --text "Your text here" \
    --threshold 0.05
```

#### Python API

```python
from pathlib import Path
from streaming_probe_inference import StreamingProbeInferenceEngine
from interactive_probe_viewer import launch_interactive_viewer

# Initialize engine
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data'),
    model_name='google/gemma-2-3b-it'
)

# Run inference
text = "After receiving feedback, I reconsidered my approach."
predictions = engine.predict_streaming(
    text,
    top_k=50,
    threshold=0.1,
    show_realtime=False
)

# Get tokens
inputs = engine.tokenizer(text, return_tensors="pt")
tokens = engine.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Launch interactive viewer!
launch_interactive_viewer(predictions, tokens, processing_time=1.5)
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `←` `→` | Navigate between tokens |
| `Home` | Jump to first token |
| `End` | Jump to last token |
| `R` | Reload/refresh display |
| `Q` | Quit |

### Panels Explained

#### 1. Token Stream (Top)
- Shows all tokens in the text
- **Yellow highlight**: Currently selected token
- **Cyan text**: Tokens with activations
- **Dim text**: Tokens without activations
- Use arrow keys to navigate

#### 2. Probe Details (Left)
- Shows all probe activations for the selected token
- **Sorted by confidence** (highest first)
- **Color-coded bars**: Visual intensity
- **✓ marker**: Above threshold
- **Top 15** most confident probes shown

#### 3. Activation Heatmap (Top Right)
- Grid showing all probes vs all tokens
- **Symbols**: `·○◐◑●⬤` indicate activation strength
- **Colors**: Match confidence level
- **Yellow column**: Current token position
- Helps spot patterns across the text

#### 4. Statistics (Bottom Right)
- Token/probe counts
- Processing time
- **Top 5 Active**: Most confident predictions
- Emojis show intensity: 💤💨💫⚡🔥💥🌟

---

## 2. 🌊 Matrix Cascade Animation

Matrix-style vertical cascading visualization with flowing tokens and activations.

### Features

- **Cascading display**: Tokens flow down like Matrix rain
- **Fading trails**: Older entries fade out smoothly
- **Real-time animation**: See processing happen live
- **Speed control**: Adjust animation speed
- **Top activations**: Shows top 3 probes per token

### Screenshot (Conceptual)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🌊 MATRIX CASCADE Token 8/12 (67%)                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ ▼  [  7] reconsidering  🔥 Reconsidering 45.2% | ⚡ Questioning 18.3%      ║
║                                                                              ║
║ ↓  [  6] began          💫 Planning 10.2%                                   ║
║                                                                              ║
║ ⇣  [  5] I              (none)                                              ║
║                                                                              ║
║ ↓  [  4] ,              (none)                                              ║
║                                                                              ║
║ ↓  [  3] feedback       🔥 Reconsidering 28.4% | ⚡ Accepting 15.8%        ║
║                                                                              ║
║ ↓  [  2] receiving      ⚡ Accepting 18.3% | 💫 Processing 12.1%           ║
║                                                                              ║
║ ↓  [  1] After          💨 Remembering 8.5%                                ║
║                                                                              ║
║ ↓  [  0] <bos>          💨 Metacognitive_Monitoring 5.2%                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
┌──────────────────────────────────────────────────────────────────────────────┐
│                 ▶  PLAYING  |  Speed: ████░  |  [Q] Quit                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Usage

#### Command Line

```bash
# Basic usage
python src/probes/matrix_probe_animation.py \
    --probes-dir data \
    --text "After analyzing the data, I reconsidered my assumptions."

# Adjust animation speed
python src/probes/matrix_probe_animation.py \
    --probes-dir data \
    --text "Your text here" \
    --delay 0.3  # Seconds between tokens (default: 0.4)

# Lower threshold for more activations
python src/probes/matrix_probe_animation.py \
    --probes-dir data \
    --text "Your text here" \
    --threshold 0.05
```

#### Python API

```python
from pathlib import Path
from streaming_probe_inference import StreamingProbeInferenceEngine
from matrix_probe_animation import animate_simpler

# Initialize engine
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data'),
    model_name='google/gemma-2-3b-it'
)

# Run animation
text = "After receiving feedback, I reconsidered my approach."
animate_simpler(
    engine,
    text,
    threshold=0.1,
    delay=0.4  # seconds per token
)
```

### Visual Elements

#### Indicators
- `▼` (red, bold): Latest token (just added)
- `↓` (yellow): Recent token (1-3 frames old)
- `⇣` (dim yellow): Older token (4-6 frames old)
- `↓` (dim): Very old token (fading out)

#### Emojis (Confidence Levels)
- 💤 < 10% (dormant)
- 💨 10-20% (slight)
- 💫 20-30% (weak)
- ⚡ 30-50% (moderate)
- 🔥 50-70% (strong)
- 💥 70-90% (very strong)
- 🌟 > 90% (peak)

#### Fading
Older entries gradually fade from:
1. **Bold bright white** (newest)
2. **Bright white**
3. **White**
4. **Dim white**
5. **Very dim** (about to disappear)

Then removed from view (but still in processing history).

---

## Comparison Table

| Feature | Interactive TUI | Matrix Cascade |
|---------|----------------|----------------|
| **Interaction** | Full navigation | Watch-only |
| **Exploration** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Visual Appeal** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Details** | High (all probes) | Medium (top 3) |
| **Use Case** | Deep analysis | Live demo |
| **Learning Curve** | Medium | Low |
| **Speed** | User-controlled | Auto-animated |

---

## Installation Requirements

Both visualizations require additional packages:

```bash
# For Interactive TUI
pip install textual rich

# For Matrix Cascade
pip install rich

# Optional: keyboard control (may not work in all terminals)
pip install keyboard
```

---

## Tips & Best Practices

### For Interactive TUI

1. **Wide terminal**: Use at least 120 columns for best display
2. **Navigate first**: Use arrow keys to explore before diving deep
3. **Heatmap patterns**: Look for vertical lines (consistent activations)
4. **Token context**: Select tokens with high activation to understand why
5. **Compare thresholds**: Try different threshold values to see more/fewer activations

### For Matrix Cascade

1. **Comfortable speed**: Default 0.4s works well, adjust as needed
2. **Short texts first**: Start with shorter texts to understand the display
3. **Watch patterns**: Look for activation bursts at key tokens
4. **Emoji guide**: Learn the emoji intensities to quickly gauge strength
5. **Record it**: Use terminal recording tools to save animations

---

## Common Use Cases

### 1. **Understanding Model Behavior**
Use Interactive TUI to:
- Explore which tokens trigger which cognitive actions
- Find unexpected activations
- Validate probe performance

### 2. **Live Demonstrations**
Use Matrix Cascade to:
- Show probe activations to an audience
- Create compelling visualizations
- Make videos/recordings

### 3. **Debugging Probes**
Use Interactive TUI to:
- Find tokens where probes fail
- Identify false positives
- Compare probe performance across texts

### 4. **Research Presentations**
Use Matrix Cascade to:
- Illustrate cognitive processing flow
- Show temporal dynamics
- Create eye-catching figures

---

## Troubleshooting

### Interactive TUI not displaying correctly
- **Issue**: Panels overlap or misaligned
- **Solution**: Increase terminal size (minimum 120x30 recommended)

### Matrix animation too fast/slow
- **Issue**: Hard to read
- **Solution**: Adjust `--delay` parameter (try 0.6 for slower, 0.2 for faster)

### Emojis not showing
- **Issue**: Terminal doesn't support emojis
- **Solution**: Use a modern terminal (iTerm2, Windows Terminal, GNOME Terminal)

### Keyboard controls not working (Interactive TUI)
- **Issue**: Some terminals don't support Textual keyboard input
- **Solution**: Use a fully-featured terminal emulator

### Colors look wrong
- **Issue**: Terminal color scheme conflicts
- **Solution**: Use default terminal colors or a dark theme

---

## Advanced Features

### Custom Filtering (Interactive TUI)

Modify the code to filter by probe category:

```python
# In interactive_probe_viewer.py
# Filter to only show metacognitive probes
filtered_predictions = [
    p for p in predictions
    if 'metacognitive' in p.action_name.lower()
]
```

### Export Snapshots

Save current view to file:

```python
# Add to interactive_probe_viewer.py
def save_snapshot(self, filename):
    with open(filename, 'w') as f:
        f.write(self.render_cascade().renderable)
```

### Custom Animation Speed Profiles

Create dynamic speed changes:

```python
# In matrix_probe_animation.py
delays = [0.8, 0.6, 0.4, 0.3, 0.2]  # Speed up over time
for i, delay in enumerate(delays * (len(tokens) // len(delays))):
    # ... animation code with delay
```

---

## Future Enhancements

Potential improvements (contributions welcome!):

- [ ] Mouse support for Interactive TUI
- [ ] Playback controls for Matrix (pause/resume)
- [ ] Custom color themes
- [ ] Filter by confidence threshold (live)
- [ ] Side-by-side text comparison
- [ ] Export to video/GIF
- [ ] Sound effects for high activations
- [ ] 3D perspective view
- [ ] Network graph visualization
- [ ] Integration with web dashboard

---

## Examples

### Analyze Metacognitive Text

```bash
python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --text "I'm questioning whether my approach is correct. Let me reconsider." \
    --threshold 0.05
```

### Demo for Presentation

```bash
python src/probes/matrix_probe_animation.py \
    --probes-dir data \
    --text "After analyzing the quarterly results, I began reconsidering our strategy." \
    --delay 0.5 \
    --threshold 0.1
```

### Compare Two Approaches

```bash
# Text 1: Analytical
python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --text "Examining the data reveals clear patterns in user behavior."

# Text 2: Reflective
python src/probes/interactive_probe_viewer.py \
    --probes-dir data \
    --text "After reflection, I realized my assumptions were flawed."
```

---

## Contributing

Have ideas for new visualization modes? Open an issue or PR!

Ideas we're considering:
- Audio/music visualization style
- 3D rotating token space
- Network graph with force layout
- Heatmap with clustering
- Timeline scrubber with zoom

---

## Credits

Built with:
- [Textual](https://github.com/Textualize/textual) - TUI framework
- [Rich](https://github.com/Textualize/rich) - Terminal formatting
- [nnsight](https://github.com/ndif-team/nnsight) - Model inspection

---

Enjoy exploring your probe activations! 🚀
