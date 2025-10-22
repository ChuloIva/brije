# 🧠 Comprehensive Cognitive Action Analysis Guide

## Overview

This guide describes all the advanced analyses we can perform on the positive patterns dataset using cognitive and sentiment probes. The analysis pipeline processes probe activation data to reveal deep insights about how cognitive actions relate to each other, emotional states, and therapeutic transformations.

---

## 📊 Available Analyses

### 1. **Co-occurrence Network Analysis** 🕸️

**What it reveals:**
- Which cognitive actions frequently appear together
- "Action clusters" that work as coordinated systems
- Central vs. peripheral actions in cognitive processing

**Insights:**
- **Highly connected actions** (hub nodes) might be "gateway" actions that enable many others
- **Tightly coupled pairs** suggest actions that are functionally related
- **Isolated actions** might be specialized for specific scenarios

**Example Questions:**
- Does `emotion_management` always co-occur with `metacognitive_regulation`?
- What actions form the "core" of positive thinking patterns?
- Are certain actions "loners" that rarely co-occur with others?

---

### 2. **Sentiment-Cognitive Correlation** 😊😔

**What it reveals:**
- Which actions are associated with positive vs. negative sentiment
- Actions that are "sentiment-neutral" (useful in both states)
- Emotional valence of different cognitive operations

**Insights:**
- **Positive-sentiment actions**: e.g., `reframing`, `perspective_taking`, `accepting`
- **Negative-sentiment actions**: e.g., `self_questioning`, `rumination patterns`
- **Neutral actions**: e.g., `analyzing`, `remembering`

**Example Questions:**
- Which actions improve sentiment most?
- Are metacognitive actions sentiment-neutral?
- Do emotion-focused actions correlate more strongly with sentiment than cognitive ones?

---

### 3. **Transformation Effectiveness** 🦋

**What it reveals:**
- Which actions are most common in transformation patterns (negative → positive)
- Actions that "bridge" from negative states to constructive thinking
- Therapeutic intervention targets

**Insights:**
- **High transformation ratio**: Actions that appear much more in transformation than negative patterns
- **Bridge actions**: Appear in transformation but NOT in negative patterns
- **Stable actions**: Appear across all pattern types (positive, negative, transformation)

**Example Questions:**
- What are the most powerful transformation actions?
- Which actions help people escape negative loops?
- Are some actions "too advanced" to use when very negative?

---

### 4. **Layer Activation Patterns** 🧬

**What it reveals:**
- How many neural network layers are activated by each action
- "Shallow" vs. "deep" processing actions
- Distributed vs. localized cognitive processing

**Insights:**
- **High layer count**: Distributed processing, complex integration (e.g., `analogical_thinking`)
- **Low layer count**: Localized, specialized processing (e.g., `noticing`, `remembering`)
- **Variable layer patterns**: Context-dependent actions

**Example Questions:**
- Are emotional actions processed differently than cognitive ones?
- Do transformation actions activate more layers?
- Which actions are "neurally expensive" vs. "efficient"?

---

### 5. **Cognitive Pattern Type Signatures** 🏷️

**What it reveals:**
- Which actions characterize specific mental health patterns
- Cognitive "fingerprints" of conditions (depression, anxiety, rumination, etc.)
- Pattern-specific intervention targets

**Insights:**
- **Rumination signature**: High `self_questioning`, `evaluating`, low `accepting`
- **Avolition signature**: Low `applying`, `creating`, high `emotion_receiving`
- **Healthy signature**: Balanced action diversity, high transformation actions

**Example Questions:**
- What makes rumination cognitively distinct from healthy reflection?
- Can we identify a cognitive pattern from its action signature?
- Which actions are universally present vs. pattern-specific?

---

### 6. **Action Diversity & Entropy** 🎲

**What it reveals:**
- How "diverse" or "rigid" different cognitive states are
- Cognitive flexibility vs. inflexibility
- Richness of cognitive repertoire

**Insights:**
- **High entropy**: Diverse action usage, flexible thinking
- **Low entropy**: Repetitive action usage, cognitive rigidity
- **Pattern differences**: Positive patterns may show higher diversity than negative

**Example Questions:**
- Are negative patterns more cognitively "stuck" (low diversity)?
- Do transformation patterns show increased diversity?
- Which cognitive pattern types are most rigid?

---

### 7. **Action Bridge Analysis** 🌉

**What it reveals:**
- Actions that uniquely appear in transformation but not negative patterns
- The "missing links" between negative and positive states
- Therapeutic intervention opportunities

**Insights:**
- **Key bridges**: Actions that enable the transformation process
- **Sentiment lift**: How much sentiment improves via specific bridges
- **Pattern-specific bridges**: Different bridges for different negative patterns

**Example Questions:**
- What action helps people escape rumination?
- Which bridge works for avolition but not anxiety?
- Can we predict which bridge will work based on negative pattern characteristics?

---

### 8. **Confidence Statistics** 📈

**What it reveals:**
- How "certain" the model is about different actions
- Reliability of action detection
- Clear vs. ambiguous cognitive processes

**Insights:**
- **High confidence**: Clear, unambiguous actions (e.g., `questioning`, `noticing`)
- **Low confidence**: Subtle, context-dependent actions
- **High variance**: Actions with variable expression

**Example Questions:**
- Which actions are easiest to detect?
- Are emotional actions harder to classify than cognitive ones?
- Does confidence correlate with therapeutic effectiveness?

---

### 9. **Action Clustering** 🗂️

**What it reveals:**
- Semantic groupings of actions based on usage patterns
- Hidden taxonomies beyond manual categorization
- Functional families of actions

**Insights:**
- **Emotion cluster**: All emotion-regulation actions grouped together
- **Metacognitive cluster**: Self-reflection and monitoring actions
- **Analytical cluster**: Reasoning and evaluation actions
- **Cross-category clusters**: Surprising groupings across traditional boundaries

**Example Questions:**
- Do actions cluster by taxonomy (Bloom's, emotional, etc.) or by function?
- Are there hidden action types we didn't explicitly define?
- Which actions are "hybrids" that don't fit cleanly in clusters?

---

### 10. **Dimensionality Reduction (t-SNE/PCA)** 🗺️

**What it reveals:**
- 2D "map" of the cognitive action space
- Proximities and relationships between actions
- Outliers and unique actions

**Insights:**
- **Nearby actions**: Functionally similar or frequently co-occurring
- **Distant actions**: Functionally distinct or incompatible
- **Outliers**: Unique actions unlike anything else

**Example Questions:**
- What does the "landscape" of cognitive actions look like?
- Are there distinct "regions" (emotional, cognitive, metacognitive)?
- Which actions are most unique?

---

## 🎯 Research Questions We Can Answer

### Therapeutic Research
1. **What are the most effective therapeutic actions?**
   - Use transformation effectiveness analysis
   - Identify high sentiment-lift bridges

2. **Which actions help escape rumination?**
   - Compare negative pattern signatures to transformation signatures
   - Find unique transformation actions for rumination patterns

3. **Can we predict which interventions will work?**
   - Use clustering to find action profiles
   - Match patient patterns to effective bridges

### Cognitive Science
1. **How do cognitive and emotional processes interact?**
   - Analyze co-occurrence between emotion_* and metacognitive_* actions
   - Examine sentiment correlations

2. **Are there fundamental cognitive "building blocks"?**
   - Use clustering to find irreducible action families
   - Identify hub nodes in co-occurrence network

3. **What makes thinking "flexible" vs. "rigid"?**
   - Compare action diversity between healthy and pathological patterns
   - Examine entropy differences

### Neuroscience
1. **Which cognitive processes are "expensive"?**
   - Use layer activation breadth as proxy for neural resources
   - Compare layer patterns across action types

2. **Are emotional and cognitive processes neurally distinct?**
   - Compare layer activation patterns for emotion vs. cognition actions
   - Use clustering on layer features

### Machine Learning / NLP
1. **Can we detect cognitive actions from text?**
   - Validate probe performance via confidence statistics
   - Identify ambiguous cases

2. **What makes cognitive text classification hard?**
   - Examine low-confidence actions
   - Analyze misclassifications

---

## 🔍 Deep Dive Examples

### Example 1: The "Rumination Escape" Analysis

**Goal**: Find what helps people escape ruminative loops

**Steps**:
1. Filter to analyses with `cognitive_pattern_type == "Negative self-evaluative loop"`
2. Compare negative vs transformation patterns
3. Find bridge actions (in transformation, not negative)
4. Compute sentiment lift for each bridge
5. Rank by effectiveness

**Expected insights**:
- `reframing` might be a strong bridge
- `accepting` might help break the loop
- `situation_modification` might provide escape route

---

### Example 2: The "Core Cognition" Analysis

**Goal**: What are the fundamental cognitive operations?

**Steps**:
1. Build co-occurrence network for ALL patterns
2. Compute centrality metrics (degree, betweenness, closeness)
3. Identify hub nodes (high degree)
4. Identify bridge nodes (high betweenness)

**Expected insights**:
- `emotion_management` might be a universal hub
- `metacognitive_monitoring` might bridge multiple clusters
- Specialized actions are peripheral

---

### Example 3: The "Neural Efficiency" Analysis

**Goal**: Which actions are most neurally efficient?

**Steps**:
1. Compute layer activation breadth per action
2. Compute sentiment improvement per action
3. Calculate "efficiency" = sentiment lift / layer breadth
4. Identify high-efficiency actions

**Expected insights**:
- Simple actions like `noticing` might be highly efficient
- Complex actions like `analogical_thinking` require more resources
- Emotion actions might have high impact but low cost

---

## 📁 Output Files Reference

### CSV Files (Tabular Data)
- `cooccurrence_*.csv`: Co-occurrence matrices
- `action_sentiment_correlation.csv`: Sentiment associations
- `transformation_effectiveness.csv`: Transformation power rankings
- `layer_patterns.csv`: Layer activation statistics
- `diversity_*.csv`: Entropy and diversity metrics
- `confidence_statistics.csv`: Model confidence per action
- `cluster_characteristics.csv`: Cluster profiles
- `action_statistics.csv`: Comprehensive action stats

### JSON Files (Structured Data)
- `comprehensive_report.json`: Master report with all analyses
- `clusters.json`: Clustering results

### HTML Files (Interactive Visualizations)
- `network_dashboard.html`: Master dashboard with all visualizations
- `cluster_scatter.html`: t-SNE projection of action space
- `cluster_summary.html`: Cluster statistics
- `cluster_radar.html`: Cluster profiles radar chart
- `visualization.html`: Original simple analysis (from analyze_positive_patterns.py)

---

## 🚀 Running the Analyses

### Run Everything
```bash
python src/probes/run_all_analyses.py
```

### Run Individual Analyses
```bash
# Statistical analysis
python src/probes/comprehensive_analysis.py

# Clustering
python src/probes/action_clustering.py

# Visualizations
python src/probes/visualize_networks.py
```

---

## 🎨 Visualization Types

1. **Co-occurrence Network Graph**: Interactive node-link diagram
2. **Sentiment Heatmaps**: Three-panel heatmap (mean, positive ratio, negative ratio)
3. **Transformation Flow (Sankey)**: Flow diagram showing action transitions
4. **Layer Activation Bar Chart**: Horizontal bars with error bars
5. **Cognitive Pattern Comparison**: Multi-panel bar charts
6. **Action Bridge Visualization**: Grouped bar chart showing sentiment lift
7. **Cluster Scatter Plot**: t-SNE projection with cluster colors
8. **Cluster Radar Chart**: Multi-dimensional profile comparison

---

## 💡 Future Analysis Ideas

### Advanced Network Analysis
- **Community detection**: Find natural groupings beyond clustering
- **Temporal dynamics**: If we had session sequences, track action evolution
- **Action sequences**: Markov chains of action transitions

### Causal Analysis
- **Granger causality**: Does action A predict action B?
- **Intervention analysis**: What happens when specific actions are introduced?

### Comparative Analysis
- **Cross-model comparison**: Compare different LLM probe activations
- **Cross-dataset validation**: Validate findings on other therapy corpora

### Personalization
- **Individual action profiles**: Each person's cognitive signature
- **Recommendation system**: Suggest actions based on current state

### Integration
- **Combine with symptom tracking**: Correlate actions with outcomes
- **Therapy session analysis**: Map actions to session quality ratings

---

## 📚 Theoretical Frameworks

Our analysis draws on:
- **Bloom's Taxonomy**: Hierarchical cognitive operations
- **Guilford's Structure of Intellect**: Divergent/convergent thinking
- **Gross's Process Model of Emotion Regulation**: Five families of emotion regulation
- **Mayer-Salovey Emotional Intelligence**: Four-branch model
- **CBT/DBT**: Therapeutic action frameworks
- **Network Science**: Graph theory and centrality
- **Information Theory**: Entropy and diversity

---

## ✅ Summary

This analysis pipeline provides:
1. **Statistical rigor**: Quantitative metrics for all cognitive actions
2. **Visual insights**: Interactive visualizations for exploration
3. **Therapeutic applications**: Direct relevance to mental health interventions
4. **Scientific discovery**: New insights into cognitive processes
5. **Machine learning validation**: Understand probe performance

**The key innovation**: We're not just detecting actions—we're understanding their **relationships**, **contexts**, and **transformational power**.

---

## 🤝 Contributing

To add new analyses:
1. Add functions to `comprehensive_analysis.py` or create new script
2. Update `run_all_analyses.py` to include new script
3. Document findings in this guide
4. Add visualizations to dashboard

---

**Happy analyzing! 🧠✨**