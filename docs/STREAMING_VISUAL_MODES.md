# Streaming Probe Visual Modes

The streaming probe inference engine supports **5 different ASCII visualization styles** for real-time output!

## Mode 1: Bars (Default)

Classic progress bar style with checkmarks:

```
================================================================================
                        STREAMING PROBE INFERENCE
                      Processing 12 tokens...
================================================================================

Token   0: '<bos>'
  ✓ ████████           Metacognitive_Monitoring  12.3% L22

Token   3: 'feedback'
  ✓ ██████████████     Reconsidering             28.4% L24
  ✓ ████████           Accepting                 15.8% L20
  ✓ ██████             Evaluating                12.1% L24

Token   7: 'reconsidering'
  ✓ ████████████████████ Reconsidering           45.2% L24
  ✓ ████████           Questioning               18.3% L26
  ✓ ██████             Metacognitive_Monitoring  14.1% L22
```

**Usage:**
```bash
python streaming_probe_inference.py --probes-dir data --display-mode bars
```

---

## Mode 2: Matrix

Matrix-style with Unicode box drawing and gradient blocks:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                  COGNITIVE ACTION MATRIX STREAM                              ║
║                          Processing 12 tokens...                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─ Token [  0] ────────────────────
│  📍 '<bos>'
│  ░▒▓█      Metacognitive_Monitoring  12.3% L22
└────────────────────────────────────────────────────────────────

┌─ Token [  3] ────────────────────
│  📍 'feedback'
│  ░▒▓█▓█    Reconsidering             28.4% L24
│  ░▒▓       Accepting                 15.8% L20
│  ░▒        Evaluating                12.1% L24
└────────────────────────────────────────────────────────────────

┌─ Token [  7] ────────────────────
│  📍 'reconsidering'
│  ░▒▓█▓█▓█▓█ Reconsidering            45.2% L24
│  ░▒▓█       Questioning              18.3% L26
│  ░▒▓        Metacognitive_Monitoring 14.1% L22
└────────────────────────────────────────────────────────────────
```

**Usage:**
```bash
python streaming_probe_inference.py --probes-dir data --display-mode matrix
```

---

## Mode 3: Fire

Flame intensity emojis showing activation "heat":

```
🔥 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔥
   NEURAL ACTIVATION STREAM - 12 tokens
🔥 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 🔥

🔸 Token   0: '<bos>'
   💨 Metacognitive_Monitoring  12.3% L22

🔸 Token   3: 'feedback'
   🔥🔥 Reconsidering             28.4% L24
   💨 Accepting                 15.8% L20
   💨 Evaluating                12.1% L24

🔸 Token   7: 'reconsidering'
   🔥🔥🔥 Reconsidering            45.2% L24
   🔥 Questioning               18.3% L26
   💨 Metacognitive_Monitoring  14.1% L22
```

**Fire Legend:**
- 💨 = 0-20% (smoke)
- 🔥 = 20-40% (small flame)
- 🔥🔥 = 40-60% (medium flame)
- 🔥🔥🔥 = 60-80% (hot flame)
- 🔥🔥🔥🔥 = 80-100% (blazing!)

**Usage:**
```bash
python streaming_probe_inference.py --probes-dir data --display-mode fire
```

---

## Mode 4: Waves

Ocean wave patterns showing activation flow:

```
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        STREAMING COGNITIVE WAVES
                              12 tokens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~ Token   0: '<bos>' ~~~
    ····· Metacognitive_Monitoring  12.3% L22

~~~ Token   3: 'feedback' ~~~
    ～～～～～ Reconsidering             28.4% L24
    ····· Accepting                 15.8% L20
    ····· Evaluating                12.1% L24

~~~ Token   7: 'reconsidering' ~~~
    ≈≈≈≈≈ Reconsidering            45.2% L24
    ····· Questioning               18.3% L26
    ····· Metacognitive_Monitoring  14.1% L22
```

**Wave Legend:**
- `·····` = 0-20% (calm)
- `～～～～～` = 20-40% (ripple)
- `≈≈≈≈≈` = 40-60% (wave)
- `∿∿∿∿∿` = 60-80% (big wave)
- `〰〰〰〰〰` = 80-100% (tsunami)

**Usage:**
```bash
python streaming_probe_inference.py --probes-dir data --display-mode waves
```

---

## Mode 5: Pulse

Heartbeat/pulse pattern with diamond symbols:

```
◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆
              ⚡ LIVE PROBE ACTIVATIONS ⚡
                  12 tokens to process
◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆

◇ [  0] '<bos>'
  ◇◆·······  Metacognitive_Monitoring  12.3% L22

◇ [  3] 'feedback'
  ◇◆◇◆◇······ Reconsidering             28.4% L24
  ◇◆·······  Accepting                 15.8% L20
  ◇◆·······  Evaluating                12.1% L24

◇ [  7] 'reconsidering'
  ◇◆◇◆◇◆◇◆◇· Reconsidering            45.2% L24
  ◇◆◇······  Questioning               18.3% L26
  ◇◆·······  Metacognitive_Monitoring  14.1% L22
```

**Pulse Pattern:**
- Alternating ◇◆ symbols = activation strength
- Dots (·) = inactive part
- Length of pattern = confidence level

**Usage:**
```bash
python streaming_probe_inference.py --probes-dir data --display-mode pulse
```

---

## Comparison Table

| Mode | Style | Best For | Visual Impact |
|------|-------|----------|---------------|
| **bars** | Classic progress bars | Clean, professional output | ⭐⭐⭐ |
| **matrix** | Unicode box drawing | Technical/hacker aesthetic | ⭐⭐⭐⭐ |
| **fire** | Emoji flames | Fun, intuitive "heat" metaphor | ⭐⭐⭐⭐⭐ |
| **waves** | Wave patterns | Smooth, flowing visualization | ⭐⭐⭐⭐ |
| **pulse** | Heartbeat diamonds | Rhythmic, medical monitor feel | ⭐⭐⭐⭐ |

---

## Python API Usage

```python
from streaming_probe_inference import StreamingProbeInferenceEngine
from pathlib import Path

engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data'),
    model_name='google/gemma-3-4b-it'
)

# Choose your favorite mode!
predictions = engine.predict_streaming(
    text="After reconsidering my approach, I began analyzing the problem.",
    top_k=5,
    threshold=0.1,
    show_realtime=True,
    display_mode='fire'  # or 'matrix', 'waves', 'pulse', 'bars'
)
```

---

## Tips

1. **Terminal Compatibility**: All modes use standard ASCII/Unicode that work in most terminals
2. **Emoji Support**: Fire mode requires emoji support (works in modern terminals)
3. **Width**: Designed for 80-character terminal width
4. **Color**: Add ANSI colors in your terminal for even better visuals!

---

## Future Modes (Ideas)

- `sparkline`: Inline sparkline graphs █▆▃▁
- `ascii-art`: Large ASCII art letters
- `retro`: Old-school BBS/ANSI art style
- `minimal`: Super compact single-line output
- `rainbow`: Colorized output with ANSI codes

Got ideas for new modes? Open an issue!