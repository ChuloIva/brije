# Brije: Watching Minds Think

Ever wondered what's happening inside a language model's "mind" when it reasons, questions, or reframes a problem? Brije lets you see it happen in real-time.

## What Is This?

Imagine two AI therapists having a conversation. One plays the skeptical analyst, the other the empathetic optimist. As they talk, you can see their cognitive processes lighting up: *reconsidering*, *divergent_thinking*, *emotion_reappraisal*, *questioning*.

That's Brije. It's a window into the cognitive machinery of language models.

Using trained probes on Gemma 3 4B's internal representations, Brije detects 45 different cognitive actions as the model generates text. Think of it like an fMRI scan for AI - watching which "cognitive regions" activate during thought.

## What Can You Do With This?

### Therapy Simulations

Set up two Gemma instances with different personalities and watch them conduct therapy sessions:

- **Client AI**: Expresses concerns, demonstrates rumination, emotional patterns
- **Therapist AI**: Shows empathy, reframing, questioning techniques

As they converse, you see exactly when the therapist shifts from *accepting* to *reconsidering*, when the client moves from *ruminating* to *divergent_thinking*. You're watching the cognitive choreography of a therapeutic intervention unfold.

### Cognitive Pattern Analysis

Want to see how an AI approaches problem-solving? Give it a complex scenario and watch:

- Does it start with *analyzing* or *questioning*?
- When does *meta_awareness* kick in?
- How quickly does it move from *comparing* to *hypothesis_generation*?

You're seeing the sequence of cognitive actions, like watching someone's thought process in slow motion.

### Multi-Agent Conversations

This is where it gets interesting. Set up conversations between different AI personalities:

- **Analyst vs Creative**: Watch one AI break down problems systematically (*analyzing*, *distinguishing*, *inferring*) while the other free-associates (*divergent_thinking*, *analogical_thinking*, *imagining*)

- **Skeptic vs Optimist**: One constantly *questioning* and *critiquing*, the other *reframing* and showing *emotion_reappraisal*

- **Metacognitive Explorers**: Both AIs thinking about their thinking (*meta_awareness*, *self_questioning*, *reconsidering*)

You're not just reading their outputs - you're watching their cognitive strategies clash and evolve.

### Research & Understanding

Ever wondered:
- When does a model "realize" it needs to reconsider?
- What cognitive patterns emerge during creative vs analytical tasks?
- Can you see the difference between remembering and reasoning in the model's activations?

Brije gives you the tools to investigate these questions empirically.

## The Experience

Here's what it looks like in action:

```
AI-1 (Therapist): "I hear you saying you feel stuck. What if we looked at
                   this from a different angle?"

🧠 Cognitive Actions Detected:
   → accepting              87.3%
   → reframing             82.1%
   → questioning           45.6%
   → empathizing          38.2%
   → divergent_thinking   31.5%

AI-2 (Client): "I hadn't thought about it that way. Maybe I've been too
                focused on what went wrong..."

🧠 Cognitive Actions Detected:
   → reconsidering         91.7%
   → meta_awareness        78.4%
   → analyzing            56.3%
   → emotion_reappraisal  44.1%
   → reframing            39.8%
```

You see the moment the client starts *reconsidering* - a metacognitive shift triggered by the therapist's *reframing*. This is cognitive interaction made visible.

## How It Works

### The Science

Brije uses a technique from interpretability research where we train small "probe" classifiers on a language model's internal activations. These probes learn to detect when specific cognitive actions are happening based on the model's hidden states.

It's based on the insight that when a model is "analyzing" something, that shows up as a distinct pattern in its internal representations - different from "questioning" or "imagining".

### The Special Sauce

We append a special prompt to all text: `"\n\nThe cognitive action being demonstrated here is"`

This primes the model to encode cognitive action information explicitly in its final token representation, making the probes more accurate. It's like asking the model to think about *how* it's thinking.

### 45 Cognitive Actions

Brije detects actions across 6 categories:

**Metacognitive** (thinking about thinking)
- reconsidering, meta_awareness, self_questioning, suspending_judgment, monitoring

**Analytical** (breaking things down)
- analyzing, comparing, distinguishing, inferring, deconstructing

**Creative** (generating new ideas)
- divergent_thinking, hypothesis_generation, analogical_thinking, imagining, brainstorming

**Emotional** (managing feelings)
- emotion_reappraisal, emotion_management, accepting, empathizing, attentional_deployment

**Memory** (recall & recognition)
- remembering, recalling, recognizing, retrieving, consolidating

**Evaluative** (making judgments)
- evaluating, critiquing, assessing, judging, rating

Each gets an independent confidence score, so you can see multiple cognitive actions happening simultaneously - just like real thought.

## Getting Started

### The Easy Way: Google Colab

1. Open [`Brije_Full_Pipeline_Colab.ipynb`](./Brije_Full_Pipeline_Colab.ipynb)
2. Click Runtime → Change runtime type → GPU (A100 if available)
3. Run all cells
4. Wait ~3-4 hours while it captures activations and trains all 45 probes
5. Download your trained probes

No local GPU needed. Free with Colab.

### The Local Way

If you've got a GPU with 16GB+ VRAM:

```bash
# Install dependencies
pip install torch transformers nnsight h5py scikit-learn tqdm

# Step 1: Capture what Gemma's brain looks like (2-3 hours)
cd src/probes
python capture_activations.py \
    --dataset ../../third_party/datagen/generated_data/stratified_combined_31500.jsonl \
    --output-dir ../../data/activations \
    --model google/gemma-2-3b-it \
    --layers 27

# Step 2: Train probes to recognize cognitive actions (15 hours)
python train_binary_probes.py \
    --activations ../../data/activations/layer_27_activations.h5 \
    --output-dir ../../data/probes_binary \
    --epochs 20

# Step 3: Test it
python multi_probe_inference.py \
    --probes-dir ../../data/probes_binary \
    --text "After reconsidering my approach, I began analyzing the problem differently."
```

## Using It

### Watch Two AIs Talk

```bash
cd third_party/liminal_backrooms
python main.py
```

In the GUI:
1. Select **"Gemma 3 4B (with Probes)"** for both AI-1 and AI-2
2. Choose a prompt style:
   - **Cognitive Roles - Analyst vs Creative**: Watch different thinking styles
   - **Skeptic vs Optimist**: See conflicting cognitive strategies
   - **Metacognitive Explorers**: Both AIs thinking about thinking
3. Enter a starting prompt (or hit "Propagate" for a random one)
4. Set turns to 10-20
5. Watch the conversation unfold with cognitive actions displayed in real-time

Try therapy simulations:
```
"I feel overwhelmed by all the choices I need to make.
I keep second-guessing myself."
```

Or philosophical debates:
```
"Is it better to analyze decisions rationally or trust
your intuition? Discuss."
```

Or problem-solving:
```
"Our team is stuck on this technical problem. We've tried
the obvious solutions. What now?"
```

### Use It Programmatically

```python
from multi_probe_inference import MultiProbeInferenceEngine

# Load the probes
engine = MultiProbeInferenceEngine(
    probes_dir="data/probes_binary",
    model_name="google/gemma-2-3b-it",
    layer_idx=27
)

# Analyze any text
text = """After hearing her perspective, I started reconsidering my initial
assumptions. Maybe there was another way to look at this."""

predictions = engine.predict(text, top_k=5, threshold=0.1)

for pred in predictions:
    print(f"{pred.action_name:25} {pred.confidence:6.1%}")
```

Output:
```
reconsidering             89.3%
meta_awareness           76.8%
reframing                54.2%
accepting                41.7%
questioning              28.9%
```

### Analyze Conversation Patterns

Compare cognitive patterns between different prompts:

```python
therapeutic = "I understand this is difficult. Let's explore what might be underneath that feeling."
analytical = "Let's break this down systematically and examine each component separately."

comparison = engine.compare_texts(therapeutic, analytical)

print("Therapeutic style emphasizes:", comparison['text1_unique'][:3])
print("Analytical style emphasizes:", comparison['text2_unique'][:3])
```

## What's Happening Behind The Scenes

### The Architecture

```
Your text
    ↓
Gemma 3 4B (generating response)
    ↓
Extract layer 27 activations (2304-dimensional vector)
    ↓
Run through 45 binary probes in parallel
    ↓
Each probe outputs confidence: "Is this cognitive action happening?"
    ↓
Display top 5-10 active cognitive actions
```

Each probe is just a simple linear classifier:
```
Input (2304) → Dropout → Linear (2304→1) → Sigmoid → Confidence Score
```

But trained on 31,500 examples of cognitive actions, they become surprisingly good at detecting when specific types of thinking are happening.

### Why Binary Probes?

Instead of one big classifier that says "which of 45 actions is this?", we use 45 separate yes/no classifiers:
- "Is this analyzing?" → 87.3% confident yes
- "Is this questioning?" → 45.6% confident yes
- "Is this imagining?" → 12.1% confident no

This means multiple cognitive actions can be active simultaneously. Real thinking isn't one-dimensional - when you're analyzing something, you're often also questioning it, maybe reconsidering your assumptions too. The binary probe approach captures that cognitive richness.

## The Research Question

This project started from a simple question: **Can we see the cognitive building blocks of AI reasoning?**

Language models don't just predict the next word - they're doing something that looks suspiciously like thinking. They analyze, question, reconsider, reframe. But it all happens in a black box.

Brije opens that box. Not completely - we're still interpreting shadows on the cave wall. But now we can see *when* certain cognitive actions are likely happening, track their patterns, watch them interact.

It's a small step toward understanding not just *what* AI outputs, but *how* it thinks.

## What's Next?

Once you've got the system running, here are some experiments to try:

**Cognitive Style Transfer**
- Can you steer a model from analytical to creative thinking by showing it examples?
- Does seeing its own cognitive pattern feedback change how it responds?

**Therapeutic Intervention Analysis**
- Which cognitive actions predict successful reframing in therapy conversations?
- Can you detect when a "client" AI has a breakthrough moment?

**Creativity Measurement**
- What's the cognitive signature of creative problem-solving vs. routine analysis?
- Do creative insights correlate with specific sequences of cognitive actions?

**Bias Detection**
- Do certain prompts trigger different cognitive patterns in concerning ways?
- Can you see evidence of different "thinking styles" for different topics?

## Technical Specs

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- 16GB+ GPU VRAM (or use Colab)
- ~50GB disk space

**Performance:**
- Probe accuracy: 75-90% per action (AUC-ROC: 0.85-0.95)
- Inference speed: ~100-200ms for all 45 probes
- Training time: ~3-4 hours (activations) + ~15 hours (probes)

**Dataset:**
- 31,500 synthetic examples across 45 cognitive actions
- Based on Bloom's Taxonomy, Guilford's Structure of Intellect, and metacognitive frameworks
- Stratified sampling ensures balanced representation

## The Files

```
brije/
├── src/probes/                          # Core detection engine
│   ├── capture_activations.py          # Extract brain states from Gemma
│   ├── train_binary_probes.py          # Train the 45 cognitive detectors
│   ├── multi_probe_inference.py        # Real-time detection during generation
│   └── probe_models.py                 # The probe architectures
│
├── data/
│   ├── activations/                    # Gemma's captured brain states
│   └── probes_binary/                  # Your 45 trained cognitive detectors
│
├── third_party/
│   ├── datagen/                        # Dataset of cognitive action examples
│   └── liminal_backrooms/              # Multi-agent conversation GUI
│       └── config.py                   # Configure which probes to use
│
└── Brije_Full_Pipeline_Colab.ipynb    # One-click training in Colab
```

## License & Citation

MIT License - use it for whatever you want.

If you build something cool with this, cite it:

```bibtex
@software{brije_cognitive_probes,
  title={Brije: Watching Minds Think},
  author={Ivan Culo},
  year={2025},
  url={https://github.com/koalacrown/brije},
  note={Real-time cognitive action detection in language models}
}
```

## Acknowledgments

Built on the shoulders of giants:
- **nnsight** - for peeking inside model internals
- **Datagen** - for the cognitive actions dataset
- **Liminal Backrooms** - for the multi-agent conversation framework
- **Gemma Team** - for the base model we're probing

---

Now go watch some minds think. 🧠✨