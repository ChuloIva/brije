# Streaming Probe Inference with Token-Level Tracking

## Overview

The streaming probe inference engine enables **real-time probe output** and **token-by-token activation tracking**. This allows you to see exactly which tokens trigger cognitive action probes and when.

## Features

### 1. Real-Time Output
Watch probe activations appear as each token is processed:
```
Processing 25 tokens...
================================================================================

Token 0: '<bos>'

Token 1: 'After'
  ✓ Reconsidering                12.5% (L24)

Token 2: 'receiving'

Token 3: 'feedback'
  ✓ Reconsidering                18.3% (L24)
  ✓ Metacognitive Monitoring     15.2% (L22)
...
```

### 2. Token-Level Activation Recording
Every token's activation is recorded with:
- **Token position** in sequence
- **Token text**
- **Confidence score** for each probe
- **Timestamp** when processed
- **Active status** (above threshold or not)

### 3. Activation Visualization
See how probe confidence changes across tokens:
```
================================================================================
Token-level activations for: Reconsidering
Layer: 24 | Final confidence: 45.23% | AUC: 0.892
================================================================================

 Pos | Token                | Confidence | Bar
----------------------------------------------------------------------
   0 | <bos>               |     5.12% | █
   1 | After               |    12.48% | ███
   2 | receiving           |    15.21% | ████
   3 | feedback            |    18.32% | █████ ✓
   4 | ,                   |    22.15% | ██████ ✓
   5 | I                   |    28.94% | ████████ ✓
   6 | began               |    35.67% | ██████████ ✓
   7 | reconsidering       |    45.23% | █████████████ ✓

Peak activation: 'reconsidering' (45.23%)
```

### 4. CSV Export
Export all token-level data for external analysis:
```csv
action_name,token_position,token_text,confidence,layer,timestamp,is_active
Reconsidering,0,<bos>,0.0512,24,0.001,False
Reconsidering,1,After,0.1248,24,0.003,True
Reconsidering,2,receiving,0.1521,24,0.005,True
...
```

## Usage

### Command Line

```bash
# Basic streaming with real-time output
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --text "After analyzing the data, I reconsidered my assumptions." \
    --top-k 5 \
    --threshold 0.1

# Visualize specific action
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --text "Comparing different solutions." \
    --visualize "Comparing" \
    --no-realtime

# Export to CSV
python src/probes/streaming_probe_inference.py \
    --probes-dir data \
    --text "The quarterly numbers look concerning." \
    --export-csv output/activations.csv \
    --no-realtime
```

### Python API

```python
from streaming_probe_inference import StreamingProbeInferenceEngine
from pathlib import Path

# Initialize engine
engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data'),
    model_name='google/gemma-3-4b-it',
    verbose=True
)

# Run streaming inference
text = "After receiving feedback, I reconsidered my approach."
predictions = engine.predict_streaming(
    text,
    top_k=10,
    threshold=0.1,
    show_realtime=True  # Show activations in real-time
)

# Access token-level data
for pred in predictions:
    print(f"\nAction: {pred.action_name}")
    print(f"Final confidence: {pred.confidence:.2%}")
    print(f"Peak token: '{pred.peak_activation_token}' ({pred.peak_confidence:.2%})")

    # Iterate through token activations
    for tok_act in pred.token_activations:
        print(f"  Token {tok_act.token_position}: '{tok_act.token_text}' -> {tok_act.confidence:.2%}")

# Visualize activations
engine.visualize_token_activations(predictions, action_name='Reconsidering')

# Export to CSV
engine.export_activations_csv(predictions, Path('output/activations.csv'))
```

### Jupyter Notebook

See `notebooks/Streaming_Token_Probe_Demo.ipynb` for interactive examples.

## Data Structures

### `TokenActivation`
Records a single token's probe activation:
```python
@dataclass
class TokenActivation:
    token_id: int           # Token ID from tokenizer
    token_text: str         # Human-readable token
    token_position: int     # Position in sequence
    action_name: str        # Cognitive action
    confidence: float       # Probe confidence (0-1)
    layer: int             # Layer this probe uses
    timestamp: float       # Time since processing started (seconds)
    is_active: bool        # Whether confidence >= threshold
```

### `StreamingPrediction`
Extends standard predictions with token-level data:
```python
@dataclass
class StreamingPrediction:
    action_name: str
    action_idx: int
    confidence: float
    is_active: bool
    layer: int
    auc: float
    token_activations: List[TokenActivation]  # All token activations
    peak_activation_token: Optional[str]      # Token with highest confidence
    peak_confidence: float                    # Highest confidence value
```

## Use Cases

### 1. Debugging Probes
See exactly where and when probes activate:
```python
# Find which token caused high activation
for pred in predictions:
    if pred.confidence > 0.5:
        print(f"{pred.action_name} peaked at token '{pred.peak_activation_token}'")
```

### 2. Understanding Text Processing
Identify which words/phrases trigger cognitive actions:
```python
# Find tokens that activate "Comparing"
comparing_pred = next(p for p in predictions if p.action_name == 'Comparing')
active_tokens = [tok.token_text for tok in comparing_pred.token_activations if tok.is_active]
print(f"Comparing activated at: {active_tokens}")
```

### 3. Model Analysis
Study how LLM representations encode cognitive processes:
```python
# Compare activation patterns across layers
for pred in predictions:
    activations = [tok.confidence for tok in pred.token_activations]
    print(f"{pred.action_name} (L{pred.layer}): mean={np.mean(activations):.2%}, std={np.std(activations):.2%}")
```

### 4. Research & Analysis
Export data for statistical analysis:
```python
# Export and analyze in pandas
import pandas as pd

engine.export_activations_csv(predictions, Path('data.csv'))
df = pd.read_csv('data.csv')

# Analyze activation patterns
print(df.groupby('action_name')['confidence'].describe())
print(df[df['is_active']].groupby('action_name').size())
```

## Performance Notes

- **Single Forward Pass**: All layers extracted in one go (efficient)
- **Memory**: Stores activations for all tokens × all probes
- **Real-Time Display**: Negligible overhead (just printing)
- **Large Texts**: For very long texts (>512 tokens), consider batching

## Comparison to Standard Inference

| Feature | Standard | Streaming |
|---------|----------|-----------|
| Final predictions | ✅ | ✅ |
| Real-time output | ❌ | ✅ |
| Token-level data | ❌ | ✅ |
| Peak activation tracking | ❌ | ✅ |
| CSV export | ❌ | ✅ |
| Visualization | ❌ | ✅ |
| Speed | Fast | ~Same |
| Memory | Low | Medium |

## Advanced: Activation Timeline Analysis

Track how cognitive actions evolve through text:

```python
text = "The data looks wrong. I'm questioning my initial assumptions."
predictions = engine.predict_streaming(text, top_k=5, show_realtime=False)

# Show timeline
for pos in range(len(predictions[0].token_activations)):
    token = predictions[0].token_activations[pos].token_text
    print(f"\nToken {pos}: '{token}'")

    for pred in predictions:
        tok_act = pred.token_activations[pos]
        if tok_act.is_active:
            print(f"  ✓ {pred.action_name}: {tok_act.confidence:.2%}")
```

Output:
```
Token 5: 'questioning'
  ✓ Questioning: 42.3%
  ✓ Metacognitive Monitoring: 35.1%
  ✓ Reconsidering: 28.7%

Token 8: 'assumptions'
  ✓ Reconsidering: 45.2%
  ✓ Questioning: 38.9%
```

## Files

- `src/probes/streaming_probe_inference.py` - Main implementation
- `notebooks/Streaming_Token_Probe_Demo.ipynb` - Interactive demo
- `example_streaming_probes.sh` - Command-line examples

## Next Steps

Potential enhancements:
1. **Heatmap visualization** - Visual activation patterns
2. **Interactive web UI** - Real-time probe monitoring
3. **Streaming generation** - Run probes during text generation
4. **Attention integration** - Combine with attention weights
5. **Multi-text comparison** - Side-by-side activation comparison