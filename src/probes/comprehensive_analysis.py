"""
Comprehensive Advanced Analysis of Positive Patterns Results
Includes network analysis, co-occurrence, sentiment correlations, and more
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import itertools
from dataclasses import dataclass

# Cognitive action descriptions from variable_pools.py
COGNITIVE_ACTIONS = {
    # Original Core Actions
    "reconsidering": "reconsidering a belief or decision",
    "reframing": "reframing a situation or perspective",
    "noticing": "noticing a pattern, feeling, or dynamic",
    "perspective_taking": "taking another's perspective or temporal view",
    "questioning": "questioning an assumption or belief",
    "abstracting": "abstracting from specifics to general patterns",
    "concretizing": "making abstract concepts concrete and specific",
    "connecting": "connecting disparate ideas or experiences",
    "distinguishing": "distinguishing between previously conflated concepts",
    "updating_beliefs": "updating mental models or beliefs",
    "suspending_judgment": "suspending judgment and staying with uncertainty",
    "pattern_recognition": "recognizing recurring patterns across situations",
    "zooming_out": "zooming out for broader context",
    "zooming_in": "zooming in on specific details",
    "analogical_thinking": "drawing analogies between domains",
    "counterfactual_reasoning": "engaging in 'what if' thinking",
    "hypothesis_generation": "generating possible explanations",
    "meta_awareness": "reflecting on one's own thinking process",
    "accepting": "accepting and letting go of control",

    # From Bloom's Taxonomy
    "remembering": "recalling relevant information or experiences",
    "understanding": "interpreting and explaining meaning",
    "applying": "using knowledge in new situations",
    "analyzing": "breaking down into components",
    "evaluating": "making judgments about value or effectiveness",
    "creating": "generating new ideas or solutions",

    # From Guilford's Structure of Intellect
    "divergent_thinking": "generating multiple creative solutions",
    "convergent_thinking": "finding the single best solution",
    "cognition_awareness": "becoming aware and comprehending",

    # Metacognitive Operations
    "metacognitive_monitoring": "tracking one's own comprehension",
    "metacognitive_regulation": "adjusting thinking strategies",
    "self_questioning": "interrogating one's own understanding",

    # Emotional/Affective Operations
    "emotional_reappraisal": "reinterpreting emotional meaning",
    "emotion_receiving": "becoming aware of emotions",
    "emotion_responding": "actively engaging with emotions",
    "emotion_valuing": "attaching worth to emotional experiences",
    "emotion_organizing": "integrating conflicting emotions",
    "emotion_characterizing": "aligning emotions with core values",
    "situation_selection": "choosing emotional contexts deliberately",
    "situation_modification": "changing circumstances to regulate emotion",
    "attentional_deployment": "directing attention for emotional regulation",
    "response_modulation": "modifying emotional expression",
    "emotion_perception": "identifying emotions in self/others",
    "emotion_facilitation": "using emotions to enhance thinking",
    "emotion_understanding": "comprehending emotional complexity",
    "emotion_management": "regulating emotions in self/others"
}


@dataclass
class Analysis:
    """Single pattern analysis record"""
    pattern_type: str
    text: str
    cognitive_pattern_name: str
    cognitive_pattern_type: str
    top_actions: List[Tuple[str, int, float]]  # (action, layer_count, confidence)
    sentiment_avg: float
    sentiment_layers: List[int]


def load_all_analyses(batches_dir: Path) -> List[Analysis]:
    """Load all batch files into Analysis objects"""
    all_analyses = []
    batch_files = sorted(batches_dir.glob("batch_*.json"))

    print(f"Loading {len(batch_files)} batch files...")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            for item in batch_data:
                all_analyses.append(Analysis(
                    pattern_type=item['pattern_type'],
                    text=item['text'],
                    cognitive_pattern_name=item['cognitive_pattern_name'],
                    cognitive_pattern_type=item['cognitive_pattern_type'],
                    top_actions=[(a[0], a[1], a[2]) for a in item['top_actions']],
                    sentiment_avg=item['sentiment_avg'],
                    sentiment_layers=item['sentiment_layers']
                ))

    print(f"Loaded {len(all_analyses)} total analyses")
    return all_analyses


def build_cooccurrence_matrix(analyses: List[Analysis], pattern_type: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Build co-occurrence matrix of cognitive actions
    Returns: (cooccurrence_df, action_frequencies)
    """
    if pattern_type:
        analyses = [a for a in analyses if a.pattern_type == pattern_type]

    # Get all unique actions
    all_actions = set()
    for analysis in analyses:
        for action, _, _ in analysis.top_actions:
            all_actions.add(action)

    actions_list = sorted(list(all_actions))
    n = len(actions_list)

    # Build co-occurrence matrix
    cooccurrence = np.zeros((n, n), dtype=int)
    action_to_idx = {action: idx for idx, action in enumerate(actions_list)}

    for analysis in analyses:
        actions_in_pattern = [action for action, _, _ in analysis.top_actions]
        # For each pair of actions in this pattern
        for action1, action2 in itertools.combinations(actions_in_pattern, 2):
            idx1, idx2 = action_to_idx[action1], action_to_idx[action2]
            cooccurrence[idx1][idx2] += 1
            cooccurrence[idx2][idx1] += 1  # Symmetric

    # Also count individual frequencies
    action_frequencies = Counter()
    for analysis in analyses:
        for action, _, _ in analysis.top_actions:
            action_frequencies[action] += 1

    df = pd.DataFrame(cooccurrence, index=actions_list, columns=actions_list)
    return df, dict(action_frequencies)


def compute_action_sentiment_correlation(analyses: List[Analysis]) -> pd.DataFrame:
    """
    Compute correlation between each action and sentiment
    """
    action_sentiments = defaultdict(list)

    for analysis in analyses:
        for action, _, _ in analysis.top_actions:
            action_sentiments[action].append(analysis.sentiment_avg)

    results = []
    for action, sentiments in action_sentiments.items():
        results.append({
            'action': action,
            'description': COGNITIVE_ACTIONS.get(action, 'Unknown'),
            'mean_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'count': len(sentiments),
            'positive_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < 0) / len(sentiments)
        })

    df = pd.DataFrame(results).sort_values('mean_sentiment', ascending=False)
    return df


def analyze_transformation_effectiveness(analyses: List[Analysis]) -> pd.DataFrame:
    """
    Which actions are most effective at transformation (appear in transformation patterns)?
    """
    transformation_actions = Counter()
    positive_actions = Counter()
    negative_actions = Counter()

    for analysis in analyses:
        for action, layer_count, confidence in analysis.top_actions:
            if analysis.pattern_type == 'transformation':
                transformation_actions[action] += 1
            elif analysis.pattern_type == 'positive':
                positive_actions[action] += 1
            elif analysis.pattern_type == 'negative':
                negative_actions[action] += 1

    # Compute transformation power: how much more common in transformation vs negative
    all_actions = set(transformation_actions.keys()) | set(positive_actions.keys()) | set(negative_actions.keys())

    results = []
    for action in all_actions:
        trans_count = transformation_actions[action]
        pos_count = positive_actions[action]
        neg_count = negative_actions[action]
        total = trans_count + pos_count + neg_count

        transformation_ratio = trans_count / total if total > 0 else 0

        results.append({
            'action': action,
            'description': COGNITIVE_ACTIONS.get(action, 'Unknown'),
            'transformation_count': trans_count,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'total_count': total,
            'transformation_ratio': transformation_ratio,
            'trans_vs_neg_ratio': trans_count / neg_count if neg_count > 0 else float('inf')
        })

    df = pd.DataFrame(results).sort_values('transformation_ratio', ascending=False)
    return df


def analyze_layer_patterns(analyses: List[Analysis]) -> pd.DataFrame:
    """
    Analyze which layers are most active for each action
    """
    action_layers = defaultdict(list)

    for analysis in analyses:
        for action, layer_count, confidence in analysis.top_actions:
            action_layers[action].append(layer_count)

    results = []
    for action, layers in action_layers.items():
        results.append({
            'action': action,
            'description': COGNITIVE_ACTIONS.get(action, 'Unknown'),
            'mean_layers': np.mean(layers),
            'std_layers': np.std(layers),
            'max_layers': max(layers),
            'min_layers': min(layers),
            'total_activations': len(layers)
        })

    df = pd.DataFrame(results).sort_values('mean_layers', ascending=False)
    return df


def analyze_cognitive_pattern_signatures(analyses: List[Analysis]) -> Dict[str, Counter]:
    """
    Which actions are characteristic of each cognitive_pattern_type?
    """
    pattern_actions = defaultdict(Counter)

    for analysis in analyses:
        for action, _, _ in analysis.top_actions:
            pattern_actions[analysis.cognitive_pattern_type][action] += 1

    return dict(pattern_actions)


def compute_action_diversity(analyses: List[Analysis], by_pattern_type: bool = True) -> pd.DataFrame:
    """
    Compute entropy/diversity metrics for each pattern type or cognitive pattern
    """
    if by_pattern_type:
        groups = defaultdict(list)
        for analysis in analyses:
            groups[analysis.pattern_type].append(analysis)
        group_key = 'pattern_type'
    else:
        groups = defaultdict(list)
        for analysis in analyses:
            groups[analysis.cognitive_pattern_type].append(analysis)
        group_key = 'cognitive_pattern_type'

    results = []
    for group_name, group_analyses in groups.items():
        # Count actions
        action_counts = Counter()
        for analysis in group_analyses:
            for action, _, _ in analysis.top_actions:
                action_counts[action] += 1

        # Compute entropy
        total = sum(action_counts.values())
        probs = np.array([count / total for count in action_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        results.append({
            group_key: group_name,
            'num_samples': len(group_analyses),
            'unique_actions': len(action_counts),
            'total_action_occurrences': total,
            'entropy': entropy,
            'mean_actions_per_sample': total / len(group_analyses),
            'most_common_action': action_counts.most_common(1)[0][0] if action_counts else None,
            'most_common_count': action_counts.most_common(1)[0][1] if action_counts else 0
        })

    df = pd.DataFrame(results).sort_values('entropy', ascending=False)
    return df


def find_action_bridges(analyses: List[Analysis]) -> List[Dict]:
    """
    Find actions that appear in negative patterns and transformation patterns
    but help bridge to positive sentiment
    """
    # Group by cognitive_pattern_name to track same scenarios
    pattern_groups = defaultdict(list)
    for analysis in analyses:
        pattern_groups[analysis.cognitive_pattern_name].append(analysis)

    bridges = []

    for pattern_name, group in pattern_groups.items():
        # Need at least negative and transformation
        neg = [a for a in group if a.pattern_type == 'negative']
        trans = [a for a in group if a.pattern_type == 'transformation']
        pos = [a for a in group if a.pattern_type == 'positive']

        if neg and trans:
            # Actions in transformation but not in negative
            neg_actions = set()
            for a in neg:
                for action, _, _ in a.top_actions:
                    neg_actions.add(action)

            trans_actions = set()
            for a in trans:
                for action, _, _ in a.top_actions:
                    trans_actions.add(action)

            unique_to_trans = trans_actions - neg_actions

            if unique_to_trans:
                avg_neg_sentiment = np.mean([a.sentiment_avg for a in neg])
                avg_trans_sentiment = np.mean([a.sentiment_avg for a in trans])
                sentiment_lift = avg_trans_sentiment - avg_neg_sentiment

                bridges.append({
                    'cognitive_pattern_name': pattern_name,
                    'cognitive_pattern_type': group[0].cognitive_pattern_type,
                    'bridge_actions': list(unique_to_trans),
                    'num_bridge_actions': len(unique_to_trans),
                    'neg_sentiment': avg_neg_sentiment,
                    'trans_sentiment': avg_trans_sentiment,
                    'sentiment_lift': sentiment_lift
                })

    return sorted(bridges, key=lambda x: x['sentiment_lift'], reverse=True)


def compute_confidence_statistics(analyses: List[Analysis]) -> pd.DataFrame:
    """
    Analyze confidence distributions per action
    """
    action_confidences = defaultdict(list)

    for analysis in analyses:
        for action, layer_count, confidence in analysis.top_actions:
            action_confidences[action].append(confidence)

    results = []
    for action, confidences in action_confidences.items():
        results.append({
            'action': action,
            'description': COGNITIVE_ACTIONS.get(action, 'Unknown'),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'median_confidence': np.median(confidences),
            'count': len(confidences)
        })

    df = pd.DataFrame(results).sort_values('mean_confidence', ascending=False)
    return df


def analyze_layer_saturation(analyses: List[Analysis]) -> Dict:
    """
    Which layers are most active overall?
    """
    layer_activations = Counter()
    layer_action_pairs = defaultdict(Counter)

    for analysis in analyses:
        for action, layer_count, confidence in analysis.top_actions:
            # Note: layer_count is the NUMBER of layers, not which layers
            # We need to use sentiment_layers for actual layer indices
            layer_activations[layer_count] += 1
            layer_action_pairs[layer_count][action] += 1

    return {
        'layer_activation_counts': dict(layer_activations),
        'layer_action_pairs': {k: dict(v) for k, v in layer_action_pairs.items()}
    }


def generate_report(analyses: List[Analysis], output_dir: Path):
    """
    Generate comprehensive analysis report with all metrics
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE COGNITIVE ACTION ANALYSIS")
    print("="*80)

    report = {}

    # 1. Co-occurrence Analysis
    print("\n1. Computing co-occurrence matrices...")
    cooccur_all, freq_all = build_cooccurrence_matrix(analyses)
    cooccur_pos, freq_pos = build_cooccurrence_matrix(analyses, 'positive')
    cooccur_neg, freq_neg = build_cooccurrence_matrix(analyses, 'negative')
    cooccur_trans, freq_trans = build_cooccurrence_matrix(analyses, 'transformation')

    report['cooccurrence'] = {
        'all': cooccur_all.to_dict(),
        'positive': cooccur_pos.to_dict(),
        'negative': cooccur_neg.to_dict(),
        'transformation': cooccur_trans.to_dict()
    }

    report['action_frequencies'] = {
        'all': freq_all,
        'positive': freq_pos,
        'negative': freq_neg,
        'transformation': freq_trans
    }

    # Save cooccurrence matrices as CSV
    cooccur_all.to_csv(output_dir / 'cooccurrence_all.csv')
    cooccur_pos.to_csv(output_dir / 'cooccurrence_positive.csv')
    cooccur_neg.to_csv(output_dir / 'cooccurrence_negative.csv')
    cooccur_trans.to_csv(output_dir / 'cooccurrence_transformation.csv')
    print("   ✓ Saved co-occurrence matrices")

    # 2. Action-Sentiment Correlation
    print("\n2. Computing action-sentiment correlations...")
    sentiment_corr = compute_action_sentiment_correlation(analyses)
    sentiment_corr.to_csv(output_dir / 'action_sentiment_correlation.csv', index=False)
    report['sentiment_correlation'] = sentiment_corr.to_dict('records')
    print("   ✓ Top positive sentiment actions:")
    for _, row in sentiment_corr.head(5).iterrows():
        print(f"      {row['action']}: {row['mean_sentiment']:.3f} ({row['description']})")

    # 3. Transformation Effectiveness
    print("\n3. Analyzing transformation effectiveness...")
    trans_eff = analyze_transformation_effectiveness(analyses)
    trans_eff.to_csv(output_dir / 'transformation_effectiveness.csv', index=False)
    report['transformation_effectiveness'] = trans_eff.to_dict('records')
    print("   ✓ Top transformation actions:")
    for _, row in trans_eff.head(5).iterrows():
        print(f"      {row['action']}: {row['transformation_ratio']:.3f} ({row['description']})")

    # 4. Layer Patterns
    print("\n4. Analyzing layer activation patterns...")
    layer_patterns = analyze_layer_patterns(analyses)
    layer_patterns.to_csv(output_dir / 'layer_patterns.csv', index=False)
    report['layer_patterns'] = layer_patterns.to_dict('records')
    print("   ✓ Most multi-layer actions:")
    for _, row in layer_patterns.head(5).iterrows():
        print(f"      {row['action']}: {row['mean_layers']:.2f} layers ({row['description']})")

    # 5. Cognitive Pattern Signatures
    print("\n5. Computing cognitive pattern signatures...")
    pattern_sigs = analyze_cognitive_pattern_signatures(analyses)
    report['cognitive_pattern_signatures'] = {k: dict(v) for k, v in pattern_sigs.items()}
    print(f"   ✓ Found {len(pattern_sigs)} unique cognitive pattern types")

    # 6. Action Diversity
    print("\n6. Computing action diversity metrics...")
    diversity_by_type = compute_action_diversity(analyses, by_pattern_type=True)
    diversity_by_cognitive = compute_action_diversity(analyses, by_pattern_type=False)
    diversity_by_type.to_csv(output_dir / 'diversity_by_pattern_type.csv', index=False)
    diversity_by_cognitive.to_csv(output_dir / 'diversity_by_cognitive_pattern.csv', index=False)
    report['diversity'] = {
        'by_pattern_type': diversity_by_type.to_dict('records'),
        'by_cognitive_pattern': diversity_by_cognitive.to_dict('records')
    }

    # 7. Action Bridges
    print("\n7. Finding action bridges (negative → transformation)...")
    bridges = find_action_bridges(analyses)
    report['action_bridges'] = bridges
    print(f"   ✓ Found {len(bridges)} bridge patterns")
    if bridges:
        print("   Top 3 bridge patterns:")
        for bridge in bridges[:3]:
            print(f"      {bridge['cognitive_pattern_name']}")
            print(f"         Sentiment lift: {bridge['sentiment_lift']:.3f}")
            print(f"         Bridge actions: {', '.join(bridge['bridge_actions'][:5])}")

    # 8. Confidence Statistics
    print("\n8. Computing confidence statistics...")
    conf_stats = compute_confidence_statistics(analyses)
    conf_stats.to_csv(output_dir / 'confidence_statistics.csv', index=False)
    report['confidence_statistics'] = conf_stats.to_dict('records')

    # 9. Layer Saturation
    print("\n9. Analyzing layer saturation...")
    layer_sat = analyze_layer_saturation(analyses)
    report['layer_saturation'] = layer_sat

    # Save full report
    with open(output_dir / 'comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - comprehensive_report.json")
    print("  - cooccurrence_*.csv")
    print("  - action_sentiment_correlation.csv")
    print("  - transformation_effectiveness.csv")
    print("  - layer_patterns.csv")
    print("  - diversity_*.csv")
    print("  - confidence_statistics.csv")

    return report


def main():
    batches_dir = Path("results/positive_patterns_analysis/batches")
    output_dir = Path("results/positive_patterns_analysis/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    analyses = load_all_analyses(batches_dir)

    # Generate report
    report = generate_report(analyses, output_dir)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal analyses: {len(analyses)}")
    print(f"  Positive: {sum(1 for a in analyses if a.pattern_type == 'positive')}")
    print(f"  Negative: {sum(1 for a in analyses if a.pattern_type == 'negative')}")
    print(f"  Transformation: {sum(1 for a in analyses if a.pattern_type == 'transformation')}")

    all_actions = set()
    for a in analyses:
        for action, _, _ in a.top_actions:
            all_actions.add(action)
    print(f"\nUnique cognitive actions observed: {len(all_actions)}")

    pattern_types = set(a.cognitive_pattern_type for a in analyses)
    print(f"Unique cognitive pattern types: {len(pattern_types)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
