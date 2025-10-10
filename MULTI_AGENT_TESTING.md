# Multi-Agent Conversation System with Cognitive Action Probes

## Overview

Your project has a multi-agent conversation system located in `third_party/liminal_backrooms/` that allows multiple AI models to talk to each other, each with their own system prompts. The system now uses **Best Multi-Probe Inference** which loads each cognitive action probe from its optimal layer.

## Key Updates

### ✅ Now Using Best Multi-Probe Inference

**Before**: Used a single layer (e.g., layer 27) for all 45 probes
**After**: Each probe uses its optimal layer (ranging from layer 21-30 based on training performance)

This means:
- More accurate cognitive action predictions
- Each probe performs at its peak performance
- Layer information is included in predictions
- AUC performance metrics are available for each prediction

### File Locations

**Multi-Agent System:**
- `third_party/liminal_backrooms/main.py` - Core conversation orchestration
- `third_party/liminal_backrooms/config.py` - Configuration (models, prompts, probe settings)
- `third_party/liminal_backrooms/gemma_probes.py` - Gemma integration with probes (UPDATED)
- `third_party/liminal_backrooms/shared_utils.py` - API wrappers for different models
- `third_party/liminal_backrooms/gui.py` - GUI interface

**Probe Inference Engines:**
- `src/probes/best_multi_probe_inference.py` - **Best multi-probe engine** (each probe from optimal layer) ✓ NOW USING THIS
- `src/probes/multi_probe_inference.py` - Old single-layer engine
- `src/probes/best_probe_loader.py` - Loads best probes based on performance analysis

**Trained Probes:**
- `data/probes_binary/` - Base directory containing:
  - `layer_21/` through `layer_30/` - Probes trained on each layer
  - `per_action_layer_analysis.json` - Performance analysis for each action
  - `training_summary.json` - Overall training statistics

## Configuration

Current probe configuration in `third_party/liminal_backrooms/config.py`:

```python
# Probe configuration
ENABLE_PROBES = True
PROBE_MODE = "binary"  # Use 45 binary one-vs-rest probes
PROBES_DIR = "../../data/probes_binary"  # Base directory (parent of layer_XX dirs)
PROBE_LAYER = 27  # Not used in best multi-probe mode
PROBE_TOP_K = 5
PROBE_THRESHOLD = 0.1
```

## Available AI Models

The system supports conversations between:
- **Gemma 3 4B (with Probes)** - Local model with cognitive action detection
- **Claude 3.5 Sonnet/Haiku/Opus** - via Anthropic API
- **Gemini 2.0 Flash/Pro** - via OpenRouter
- **DeepSeek R1** - with chain-of-thought reasoning
- **o3-mini, o1, o1-mini** - via OpenRouter
- **Llama 3.1 405B** - via OpenRouter
- **Flux 1.1 Pro** - Image generation model

## System Prompt Pairs

Predefined conversation scenarios in `config.py`:

1. **Backrooms** - Free-form AI-to-AI exploration
2. **ASCII Art** - Collaborative ASCII art creation
3. **Image Model Collaboration** - AI + Image model
4. **Cognitive Roles - Analyst vs Creative** - Analytical vs creative thinking
5. **Cognitive Roles - Skeptic vs Optimist** - Critical vs positive perspectives
6. **Cognitive Roles - Metacognitive Explorer** - Self-reflective thinking

## How to Test

### Simple Test (Gemma with Probes Only)

```bash
python test_multi_agent_simple.py
```

This will:
1. Initialize Gemma 3 4B with best multi-probe inference
2. Generate a response to a test prompt
3. Show cognitive actions detected with:
   - Action name
   - Confidence score
   - Which layer the probe is from
   - AUC performance metric
   - Whether it's active (above threshold)

### Full Multi-Agent Conversation Test

```bash
python test_multi_agent_conversation.py
```

This will:
1. Run a 3-turn conversation between two Gemma instances
2. Use different system prompts (Analyst vs Creative)
3. Show cognitive actions at each turn
4. Save full conversation to `test_conversation_output.json`

### GUI Mode

```bash
cd third_party/liminal_backrooms
python main.py
```

This launches the full GUI where you can:
- Select which models to use
- Choose system prompt pairs
- Set number of turns
- View cognitive action predictions in real-time
- See conversation history

## Prediction Output Format

Each prediction now includes:

```python
{
    'action': 'analyzing',
    'confidence': 0.85,
    'is_active': True,
    'layer': 21,  # Which layer this probe is from
    'auc': 0.944  # AUC performance metric
}
```

## Layer Performance

Based on `data/probes_binary/training_summary.json`:

- **Best Layer Overall**: Layer 21 (avg AUC: 0.944)
- **Layer Range**: 21-30
- Each action has its own optimal layer
- The system automatically uses the best layer for each action

## Next Steps

1. **Test the basic integration:**
   ```bash
   python test_multi_agent_simple.py
   ```

2. **Test multi-agent conversation:**
   ```bash
   python test_multi_agent_conversation.py
   ```

3. **Launch the GUI** (requires API keys):
   - Create `.env` file in `third_party/liminal_backrooms/` (see `.env.example`)
   - Add API keys for models you want to use
   - Run `python third_party/liminal_backrooms/main.py`

## Performance Benefits

Using best multi-probe inference vs single-layer:
- ✅ Each probe operates at peak performance
- ✅ Better cognitive action detection accuracy
- ✅ Layer diversity (probes from layers 21-30)
- ✅ AUC metrics available for confidence assessment
- ✅ Single forward pass extracts all needed layers efficiently

## Troubleshooting

**If you get import errors:**
- Make sure you're running from the project root
- Check that `src/probes/` contains all probe modules
- Verify `data/probes_binary/` contains trained probes

**If models fail to load:**
- Check GPU/CPU availability
- Verify model name is correct
- Ensure sufficient VRAM/RAM

**For API-based models:**
- Create `.env` file with API keys
- Check API key permissions
- Verify internet connectivity
