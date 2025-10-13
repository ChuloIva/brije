# Therapeutic Conversations Analysis - Complete Output

This directory contains the complete analysis of 30 simulated therapeutic conversations with cognitive action tracking.

## 📊 Directory Structure

```
output/therapeutic_conversations/
├── checkpoint_30.json                      # Raw simulation data (30 sessions)
├── analysis/                               # Statistical analysis and visualizations
│   ├── ANALYSIS_REPORT.txt                # Human-readable summary report
│   ├── analysis_summary.json              # Structured data summary
│   ├── 01_session_statistics.png          # Demographics and session info
│   ├── 02_cognitive_actions_by_role.png   # Therapist vs client patterns
│   ├── 03_client_progression.png          # How clients change over sessions
│   ├── 04_action_emergence_decline.png    # Actions that increase/decrease
│   ├── 05_depression_vs_anxiety.png       # Cognitive signatures comparison
│   ├── 06_therapist_patterns.png          # Therapist adaptation strategies
│   ├── 07_action_confidence.png           # Confidence scores analysis
│   ├── 08_session_outcomes.png            # Session outcome indicators
│   └── 09_timeline_heatmap.png            # Actions across session timeline
└── conversation_reports/                   # Human-readable conversation transcripts
    ├── index.html                         # Master index (START HERE!)
    ├── session_000.html                   # Individual session reports...
    ├── session_001.html
    ├── ...
    ├── session_029.html
    └── conversations_full.md              # All conversations in Markdown

```

## 🚀 Quick Start

### 1. View Individual Conversations with Cognitive Actions

**Open in your browser:**
```bash
open conversation_reports/index.html
```

This provides:
- ✅ Beautiful, color-coded conversation transcripts
- ✅ All cognitive actions highlighted for each turn
- ✅ Confidence scores for each action
- ✅ Session metadata and statistics
- ✅ Easy navigation between sessions

**Cognitive Action Categories:**
- 🟣 **Purple** = Emotion-focused (perception, responding, valuing)
- 🔵 **Blue** = Cognitive (abstracting, divergent thinking, hypothesis generation)
- 🟡 **Yellow** = Metacognitive (questioning, noticing, awareness)
- 🟢 **Green** = Action-oriented (applying, reframing, response modulation)

### 2. Review Statistical Analysis

**Read the summary:**
```bash
cat analysis/ANALYSIS_REPORT.txt
```

**View visualizations:**
```bash
open analysis/
```

Each visualization answers a specific question:
1. **Session Statistics** - What were the presenting issues?
2. **Cognitive Actions by Role** - What do therapists vs clients do differently?
3. **Client Progression** - How do clients change during therapy?
4. **Emergence/Decline** - What cognitive actions increase/decrease?
5. **Depression vs Anxiety** - Different cognitive signatures?
6. **Therapist Patterns** - How do therapists adapt their approach?
7. **Action Confidence** - Which actions are most reliable?
8. **Session Outcomes** - What indicates successful therapy?
9. **Timeline Heatmap** - When do different actions occur?

## 🔍 Key Findings

### How Clients Get Better

Clients show dramatic increases in **metacognitive abilities**:

| Cognitive Action | Change (Early → Late) | Percentage |
|-----------------|----------------------|------------|
| pattern_recognition | +22 occurrences | +2200% |
| questioning | +19 occurrences | +1900% |
| divergent_thinking | +19 occurrences | +950% |
| convergent_thinking | +16 occurrences | +533% |
| abstracting | +14 occurrences | +350% |

**Translation:** Clients move from:
- 😰 **Emotional reactivity** → 🧠 **Cognitive insight**
- 🔴 **Concrete distress** → 🔵 **Abstract understanding**
- 🤔 **Confused** → 💡 **Pattern recognition**

### Depression vs Anxiety Patterns

**Depression clients:**
- More `hypothesis_generation` (why do I feel this way?)
- More `evaluating` (self-judgment)
- More `emotion_valuing` (assessing worth)

**Anxiety clients:**
- More `emotion_responding` (reacting to feelings)
- More `noticing` (hypervigilance)
- More `emotion_perception` (focused on current state)

### Most Effective Therapist Actions

1. **emotion_perception** (56 times) - Reading client emotions accurately
2. **abstracting** (48 times) - Helping see broader patterns
3. **divergent_thinking** (39 times) - Opening new perspectives
4. **emotion_valuing** (36 times) - Validating feelings
5. **emotion_characterizing** (31 times) - Naming emotions precisely

## 📈 Data Overview

- **Total Sessions:** 30
- **Total Conversation Turns:** 240
- **Therapist Turns:** 90
- **Client Turns:** 150
- **Unique Client Cognitive Actions:** 44
- **Unique Therapist Cognitive Actions:** 37

**Issue Distribution:**
- Depression: 11 sessions (36.7%)
- Both (Depression + Anxiety): 10 sessions (33.3%)
- Anxiety: 9 sessions (30.0%)

## 🛠️ How This Was Generated

These analyses were created by:

1. **Simulation Script:** `multi_agent/Therapeutic_Conversation_Simulator_100x_Colab.ipynb`
   - Generates realistic therapy conversations
   - Uses AI models with cognitive action tracking
   - Saves results to `checkpoint_30.json`

2. **Meta-Analysis Script:** `multi_agent/therapeutic_meta_analysis.py`
   - Analyzes all 30 sessions
   - Generates 9 visualizations
   - Creates statistical reports

3. **Report Generator:** `multi_agent/generate_conversation_reports.py`
   - Creates HTML/Markdown conversation transcripts
   - Color-codes cognitive actions
   - Builds navigation index

## 📝 File Formats

### checkpoint_30.json
Raw JSON data with complete conversation history and cognitive predictions:
```json
{
  "sessions_completed": 30,
  "sessions": [
    {
      "metadata": { ... },
      "conversation": [
        {
          "role": "user/assistant",
          "content": "...",
          "predictions": [ ... ]
        }
      ]
    }
  ]
}
```

### analysis_summary.json
Structured quantitative analysis results:
```json
{
  "data_overview": { ... },
  "client_progression": { ... },
  "top_therapist_actions": { ... },
  "emerging_actions": { ... }
}
```

## 🔬 Cognitive Actions Glossary

### Emotion-Focused Actions
- **emotion_perception** - Recognizing emotional states
- **emotion_responding** - Reacting to emotions
- **emotion_valuing** - Assessing emotional significance
- **emotion_characterizing** - Describing emotional qualities
- **emotion_receiving** - Accepting emotional input
- **emotion_understanding** - Comprehending emotional meaning
- **emotion_management** - Regulating emotional responses

### Cognitive Actions
- **abstracting** - Moving from specific to general
- **concretizing** - Moving from general to specific
- **divergent_thinking** - Generating multiple possibilities
- **convergent_thinking** - Integrating toward single understanding
- **pattern_recognition** - Identifying recurring themes
- **hypothesis_generation** - Forming testable theories
- **evaluating** - Making judgments/assessments

### Metacognitive Actions
- **questioning** - Inquiry and exploration
- **self_questioning** - Internal inquiry
- **cognition_awareness** - Awareness of thinking processes
- **noticing** - Directing attention

### Action-Oriented
- **applying** - Using knowledge in practice
- **situation_selection** - Choosing environments
- **response_modulation** - Adjusting responses
- **reframing** - Changing perspective

## 💡 Using This Data

### For Researchers
- Analyze conversation patterns
- Study therapeutic mechanisms
- Validate cognitive models
- Train AI systems

### For Clinicians
- Understand effective techniques
- See real-world conversation flow
- Learn cognitive intervention patterns
- Study client progression markers

### For Developers
- Train conversational AI
- Build therapy support tools
- Validate cognitive action detection
- Create adaptive systems

## 📖 Citation

If you use this data in research, please cite:

```
Therapeutic Conversation Simulations with Cognitive Action Tracking
Generated using Gemma 3 4B with Probes
Date: 2025-10-13
Sessions: 30
```

## 🤝 Questions?

For questions about:
- **Data interpretation** - See `ANALYSIS_REPORT.txt`
- **Conversation details** - Open `conversation_reports/index.html`
- **Methodology** - Review the notebook and analysis scripts

---

**Generated:** 2025-10-13
**Analysis Tools:** Python, pandas, matplotlib, seaborn
**AI Models:** Gemma 3 4B with Cognitive Probes