"""
Comprehensive analysis of positive_patterns.jsonl using cognitive + sentiment probes
Uses single-pass inference (last token only) like universal_multi_layer_inference
"""

from gpu_utils import configure_amd_gpu
configure_amd_gpu()

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass, asdict
import sys

NNSIGHT_PATH = Path(__file__).parent.parent.parent / "third_party" / "nnsight" / "src"
sys.path.insert(0, str(NNSIGHT_PATH))

from streaming_probe_inference import StreamingProbeInferenceEngine, AggregatedPrediction


@dataclass
class PatternAnalysis:
    """Analysis results for a single pattern"""
    pattern_type: str  # 'positive', 'negative', 'transformation'
    text: str
    cognitive_pattern_name: str
    cognitive_pattern_type: str

    # Streaming inference results
    streaming_top_actions: List[Tuple[str, int, float]]  # (action, layer_count, confidence)
    streaming_sentiment_avg: float
    streaming_sentiment_layers: List[int]

    # Whole string inference results
    whole_top_actions: List[Tuple[str, int, float]]
    whole_sentiment_avg: float
    whole_sentiment_layers: List[int]

    # Comparison metrics
    action_agreement: float  # How similar are top actions?
    sentiment_diff: float  # Difference in sentiment


def load_dataset(file_path: Path) -> List[Dict]:
    """Load positive_patterns.jsonl"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_pattern_texts(entry: Dict) -> Dict[str, str]:
    """Extract positive, negative, and transformation pattern texts from entry"""
    return {
        'positive': entry.get('positive_thought_pattern', ''),
        'negative': entry.get('reference_negative_example', ''),
        'transformation': entry.get('reference_transformed_example', '')
    }


def run_inference(engine: StreamingProbeInferenceEngine,
                 text: str,
                 threshold: float,
                 streaming: bool = True) -> Tuple[List[AggregatedPrediction], List[AggregatedPrediction]]:
    """
    Run inference and return cognitive + sentiment predictions

    Returns:
        (cognitive_predictions, sentiment_predictions)
    """
    # Use single-pass inference (last token only, like universal_multi_layer_inference)
    all_predictions = predict_last_token_only(engine, text, threshold)

    # Split into cognitive and sentiment
    cognitive_preds = [p for p in all_predictions if p.action_name != "sentiment"]
    sentiment_preds = [p for p in all_predictions if p.action_name == "sentiment"]

    # Sort cognitive by layer_count then confidence
    cognitive_preds.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)

    return cognitive_preds, sentiment_preds


def predict_last_token_only(
    engine: StreamingProbeInferenceEngine,
    text: str,
    threshold: float
) -> List[AggregatedPrediction]:
    """
    Run inference on only the last token (like universal_multi_layer_inference),
    then aggregate by action across layers.

    Args:
        engine: Streaming inference engine
        text: Input text to analyze
        threshold: Confidence threshold for active predictions

    Returns:
        List of AggregatedPrediction objects aggregated by action
    """
    from collections import defaultdict
    import numpy as np

    # === COGNITIVE ACTION PROBES ===
    # Append cognitive action prompt
    cognitive_augmented_text = f"{text}\n\nThe cognitive action being demonstrated here is"

    # Extract activations from all layers for cognitive probes (single forward pass)
    cognitive_activations = {}
    with engine.model.trace(cognitive_augmented_text):
        for layer_idx in engine.layers_needed:
            hidden_states = engine.model.model.layers[layer_idx].output[0]
            # Save the LAST token representation
            cognitive_activations[layer_idx] = hidden_states[:, -1, :].save()

    # Run all cognitive probes
    cognitive_predictions = []
    with torch.no_grad():
        for (action_name, layer_idx), probe_info in engine.probes.items():
            probe = probe_info['probe']
            action_idx = engine.action_to_idx[action_name]

            # Get activation for this layer
            act = cognitive_activations[layer_idx].squeeze(0)

            # Run probe
            logits = probe(act)
            confidence = torch.sigmoid(logits).item()

            cognitive_predictions.append({
                'action_name': action_name,
                'action_idx': action_idx,
                'layer': layer_idx,
                'confidence': confidence,
                'is_active': confidence >= threshold
            })

    # === SENTIMENT PROBES ===
    sentiment_predictions = []
    if engine.include_sentiment:
        # Append sentiment prompt
        sentiment_augmented_text = f"{text}\n\nThe sentiment of this section is"

        # Extract activations from sentiment probe layers (single forward pass)
        sentiment_activations = {}
        with engine.model.trace(sentiment_augmented_text):
            for layer_idx in engine.sentiment_probes.keys():
                hidden_states = engine.model.model.layers[layer_idx].output[0]
                # Save the LAST token representation
                sentiment_activations[layer_idx] = hidden_states[:, -1, :].save()

        # Run each sentiment probe
        with torch.no_grad():
            for layer_idx, probe_info in engine.sentiment_probes.items():
                if layer_idx not in sentiment_activations:
                    continue

                probe = probe_info['probe']
                act = sentiment_activations[layer_idx].squeeze(0)

                # Run probe - sentiment probes output regression scores
                score = probe(act).item()

                sentiment_predictions.append({
                    'action_name': 'sentiment',
                    'action_idx': -1,
                    'layer': layer_idx,
                    'confidence': score,
                    'is_active': abs(score) >= threshold
                })

    # === AGGREGATE BY ACTION ===
    all_predictions = cognitive_predictions + sentiment_predictions

    # Group by action name
    action_groups = defaultdict(list)
    for pred in all_predictions:
        action_groups[pred['action_name']].append(pred)

    # Create aggregated predictions
    aggregated = []
    for action_name, preds in action_groups.items():
        is_sentiment = (action_name == "sentiment")

        # Sort by confidence to find best layer
        if is_sentiment:
            preds_sorted = sorted(preds, key=lambda x: abs(x['confidence']), reverse=True)
        else:
            preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)
        best_pred = preds_sorted[0]

        # Get all layers and confidences
        all_confidences = [p['confidence'] for p in preds]

        # Get only ACTIVE layers (above threshold)
        if is_sentiment:
            active_layers = [p['layer'] for p in preds if abs(p['confidence']) >= threshold]
            active_confidences = [p['confidence'] for p in preds if abs(p['confidence']) >= threshold]
        else:
            active_layers = [p['layer'] for p in preds if p['confidence'] >= threshold]
            active_confidences = [p['confidence'] for p in preds if p['confidence'] >= threshold]

        # Calculate aggregates
        if is_sentiment:
            max_conf = max(all_confidences, key=abs)
        else:
            max_conf = max(all_confidences)
        mean_conf = np.mean(all_confidences)

        agg_pred = AggregatedPrediction(
            action_name=action_name,
            action_idx=best_pred['action_idx'],
            layers=sorted(active_layers),
            layer_count=len(active_layers),
            max_confidence=max_conf,
            mean_confidence=mean_conf,
            best_layer=best_pred['layer'],
            is_active=(abs(max_conf) >= threshold) if is_sentiment else (max_conf >= threshold),
            layer_predictions=[],  # No individual layer predictions in single-pass mode
            peak_activation_token=None,  # No token-level tracking
            peak_confidence=max_conf
        )
        aggregated.append(agg_pred)

    # Sort by layer count (descending) then max confidence (descending)
    aggregated.sort(key=lambda x: (x.layer_count, abs(x.max_confidence)), reverse=True)

    return aggregated


def analyze_pattern(engine: StreamingProbeInferenceEngine,
                   pattern_type: str,
                   text: str,
                   entry: Dict,
                   threshold: float) -> PatternAnalysis:
    """Analyze single pattern with single-pass last-token inference"""

    # Run inference (single pass, last token only)
    cog_preds, sent_preds = run_inference(engine, text, threshold)

    # For backwards compatibility with the analysis structure, use the same results for both
    streaming_cog, streaming_sent = cog_preds, sent_preds
    whole_cog, whole_sent = cog_preds, sent_preds

    # Extract top actions from streaming
    streaming_top = [
        (p.action_name, p.layer_count, p.max_confidence)
        for p in streaming_cog[:10] if p.is_active
    ]

    # Extract top actions from whole
    whole_top = [
        (p.action_name, p.layer_count, p.max_confidence)
        for p in whole_cog[:10] if p.is_active
    ]

    # Sentiment aggregation
    streaming_sent_avg = np.mean([p.max_confidence for p in streaming_sent]) if streaming_sent else 0.0
    streaming_sent_layers = [p.best_layer for p in streaming_sent if p.is_active]

    whole_sent_avg = np.mean([p.max_confidence for p in whole_sent]) if whole_sent else 0.0
    whole_sent_layers = [p.best_layer for p in whole_sent if p.is_active]

    # Compute agreement (Jaccard similarity of top 5 actions)
    streaming_actions = set([a[0] for a in streaming_top[:5]])
    whole_actions = set([a[0] for a in whole_top[:5]])

    if len(streaming_actions) == 0 and len(whole_actions) == 0:
        action_agreement = 1.0
    else:
        action_agreement = len(streaming_actions & whole_actions) / len(streaming_actions | whole_actions) if (streaming_actions | whole_actions) else 0.0

    sentiment_diff = abs(streaming_sent_avg - whole_sent_avg)

    return PatternAnalysis(
        pattern_type=pattern_type,
        text=text[:200],  # Store first 200 chars
        cognitive_pattern_name=entry.get('cognitive_pattern_name', ''),
        cognitive_pattern_type=entry.get('cognitive_pattern_type', ''),
        streaming_top_actions=streaming_top,
        streaming_sentiment_avg=streaming_sent_avg,
        streaming_sentiment_layers=streaming_sent_layers,
        whole_top_actions=whole_top,
        whole_sentiment_avg=whole_sent_avg,
        whole_sentiment_layers=whole_sent_layers,
        action_agreement=action_agreement,
        sentiment_diff=sentiment_diff
    )


def aggregate_group_statistics(analyses: List[PatternAnalysis],
                               pattern_type: str) -> Dict:
    """Compute aggregate statistics for a pattern type group"""

    group_analyses = [a for a in analyses if a.pattern_type == pattern_type]

    if not group_analyses:
        return {}

    # Collect all streaming actions
    streaming_action_counts = Counter()
    streaming_layer_counts = defaultdict(list)

    for analysis in group_analyses:
        for action, layer_count, _ in analysis.streaming_top_actions:
            streaming_action_counts[action] += 1
            streaming_layer_counts[action].append(layer_count)

    # Collect all whole string actions
    whole_action_counts = Counter()
    whole_layer_counts = defaultdict(list)

    for analysis in group_analyses:
        for action, layer_count, _ in analysis.whole_top_actions:
            whole_action_counts[action] += 1
            whole_layer_counts[action].append(layer_count)

    # Top actions for this group
    top_streaming_actions = streaming_action_counts.most_common(15)
    top_whole_actions = whole_action_counts.most_common(15)

    # Sentiment statistics
    streaming_sentiments = [a.streaming_sentiment_avg for a in group_analyses]
    whole_sentiments = [a.whole_sentiment_avg for a in group_analyses]

    # Agreement statistics
    agreements = [a.action_agreement for a in group_analyses]
    sentiment_diffs = [a.sentiment_diff for a in group_analyses]

    return {
        'pattern_type': pattern_type,
        'num_samples': len(group_analyses),
        'top_streaming_actions': top_streaming_actions,
        'top_whole_actions': top_whole_actions,
        'avg_sentiment_streaming': float(np.mean(streaming_sentiments)),
        'std_sentiment_streaming': float(np.std(streaming_sentiments)),
        'avg_sentiment_whole': float(np.mean(whole_sentiments)),
        'std_sentiment_whole': float(np.std(whole_sentiments)),
        'avg_action_agreement': float(np.mean(agreements)),
        'avg_sentiment_diff': float(np.mean(sentiment_diffs)),
        'cognitive_pattern_types': Counter([a.cognitive_pattern_type for a in group_analyses])
    }


def compute_cross_group_differences(stats: Dict[str, Dict]) -> Dict:
    """Compare statistics across positive, negative, transformation groups"""

    comparisons = {}

    # Compare sentiment between groups
    if 'positive' in stats and 'negative' in stats:
        comparisons['sentiment_positive_vs_negative'] = {
            'streaming_diff': stats['positive']['avg_sentiment_streaming'] - stats['negative']['avg_sentiment_streaming'],
            'whole_diff': stats['positive']['avg_sentiment_whole'] - stats['negative']['avg_sentiment_whole']
        }

    if 'positive' in stats and 'transformation' in stats:
        comparisons['sentiment_positive_vs_transformation'] = {
            'streaming_diff': stats['positive']['avg_sentiment_streaming'] - stats['transformation']['avg_sentiment_streaming'],
            'whole_diff': stats['positive']['avg_sentiment_whole'] - stats['transformation']['avg_sentiment_whole']
        }

    if 'negative' in stats and 'transformation' in stats:
        comparisons['sentiment_negative_vs_transformation'] = {
            'streaming_diff': stats['negative']['avg_sentiment_streaming'] - stats['transformation']['avg_sentiment_streaming'],
            'whole_diff': stats['negative']['avg_sentiment_whole'] - stats['transformation']['avg_sentiment_whole']
        }

    # Compare action distributions
    for group1 in ['positive', 'negative', 'transformation']:
        for group2 in ['positive', 'negative', 'transformation']:
            if group1 < group2 and group1 in stats and group2 in stats:
                # Get top actions for each group
                actions1 = set([a[0] for a in stats[group1]['top_streaming_actions'][:10]])
                actions2 = set([a[0] for a in stats[group2]['top_streaming_actions'][:10]])

                unique_to_1 = actions1 - actions2
                unique_to_2 = actions2 - actions1
                shared = actions1 & actions2

                comparisons[f'actions_{group1}_vs_{group2}'] = {
                    f'unique_to_{group1}': list(unique_to_1),
                    f'unique_to_{group2}': list(unique_to_2),
                    'shared': list(shared),
                    'overlap_ratio': len(shared) / len(actions1 | actions2) if (actions1 | actions2) else 0.0
                }

    return comparisons


def create_html_visualization(stats: Dict[str, Dict],
                             comparisons: Dict,
                             analyses: List[PatternAnalysis],
                             output_path: Path):
    """Create interactive HTML visualization with all analyses"""

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Positive Patterns Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #764ba2;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h3 {{
            color: #667eea;
            margin-top: 25px;
        }}
        .summary {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-card h4 {{
            margin: 0 0 10px 0;
            color: #764ba2;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .comparison-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table tr:hover {{
            background: #f5f7fa;
        }}
        .chart {{
            margin: 30px 0;
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        .transformation {{ color: #f59e0b; }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }}
        .tab {{
            padding: 10px 20px;
            background: #f5f7fa;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s;
        }}
        .tab.active {{
            background: #667eea;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 Cognitive & Sentiment Pattern Analysis</h1>
    <p style="text-align: center; color: #666; font-size: 1.1em;">
        Comprehensive analysis of {sum(s['num_samples'] for s in stats.values())} patterns across
        {len(stats)} categories using cognitive + sentiment probes
    </p>
"""

    # Summary statistics
    html += """
    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stat-grid">
"""

    for pattern_type, stat in stats.items():
        sentiment_emoji = "😊" if stat['avg_sentiment_streaming'] > 0 else "😔"
        html += f"""
            <div class="stat-card">
                <h4>{pattern_type.upper()}</h4>
                <div class="value {pattern_type}">{stat['num_samples']}</div>
                <p style="margin: 5px 0; color: #666;">
                    {sentiment_emoji} Sentiment: {stat['avg_sentiment_streaming']:.3f}
                </p>
                <p style="margin: 5px 0; color: #666;">
                    Agreement: {stat['avg_action_agreement']:.1%}
                </p>
            </div>
"""

    html += """
        </div>
    </div>
"""

    # Tabs for different analyses
    html += """
    <div class="tabs">
        <button class="tab active" onclick="showTab('sentiment')">Sentiment Analysis</button>
        <button class="tab" onclick="showTab('actions')">Top Actions</button>
        <button class="tab" onclick="showTab('comparisons')">Cross-Group Comparisons</button>
        <button class="tab" onclick="showTab('inference')">Streaming vs Whole</button>
    </div>
"""

    # TAB 1: Sentiment Analysis
    html += """
    <div id="sentiment" class="tab-content active">
        <h2>😊😔 Sentiment Analysis</h2>
"""

    # Sentiment bar chart data
    pattern_types = list(stats.keys())
    streaming_sentiments = [stats[pt]['avg_sentiment_streaming'] for pt in pattern_types]
    whole_sentiments = [stats[pt]['avg_sentiment_whole'] for pt in pattern_types]

    html += f"""
        <div class="chart" id="sentiment-chart"></div>
        <script>
            var data = [
                {{
                    x: {json.dumps(pattern_types)},
                    y: {json.dumps(streaming_sentiments)},
                    name: 'Streaming Inference',
                    type: 'bar',
                    marker: {{color: '#667eea'}}
                }},
                {{
                    x: {json.dumps(pattern_types)},
                    y: {json.dumps(whole_sentiments)},
                    name: 'Whole String Inference',
                    type: 'bar',
                    marker: {{color: '#764ba2'}}
                }}
            ];
            var layout = {{
                title: 'Average Sentiment by Pattern Type',
                barmode: 'group',
                yaxis: {{title: 'Sentiment Score'}},
                xaxis: {{title: 'Pattern Type'}}
            }};
            Plotly.newPlot('sentiment-chart', data, layout);
        </script>
"""

    # Sentiment distribution violin plot
    all_streaming_sents = []
    all_whole_sents = []
    all_types = []

    for pt in pattern_types:
        pt_analyses = [a for a in analyses if a.pattern_type == pt]
        for a in pt_analyses:
            all_streaming_sents.append(a.streaming_sentiment_avg)
            all_whole_sents.append(a.whole_sentiment_avg)
            all_types.append(pt)

    html += f"""
        <div class="chart" id="sentiment-violin"></div>
        <script>
            var violinData = [];
"""

    for pt in pattern_types:
        pt_streaming = [analyses[i].streaming_sentiment_avg for i in range(len(analyses)) if analyses[i].pattern_type == pt]
        html += f"""
            violinData.push({{
                y: {json.dumps(pt_streaming)},
                type: 'violin',
                name: '{pt}',
                box: {{visible: true}},
                meanline: {{visible: true}}
            }});
"""

    html += """
            var violinLayout = {
                title: 'Sentiment Distribution by Pattern Type (Streaming)',
                yaxis: {title: 'Sentiment Score'}
            };
            Plotly.newPlot('sentiment-violin', violinData, violinLayout);
        </script>
    </div>
"""

    # TAB 2: Top Actions
    html += """
    <div id="actions" class="tab-content">
        <h2>🎯 Top Cognitive Actions by Pattern Type</h2>
"""

    for pattern_type in pattern_types:
        stat = stats[pattern_type]
        top_actions = stat['top_streaming_actions'][:15]

        actions_list = [a[0] for a in top_actions]
        counts_list = [a[1] for a in top_actions]

        html += f"""
        <h3 class="{pattern_type}">{pattern_type.upper()} Patterns</h3>
        <div class="chart" id="actions-{pattern_type}"></div>
        <script>
            var data_{pattern_type} = [{{
                x: {json.dumps(counts_list)},
                y: {json.dumps(actions_list)},
                type: 'bar',
                orientation: 'h',
                marker: {{color: '#667eea'}}
            }}];
            var layout_{pattern_type} = {{
                title: 'Top 15 Actions in {pattern_type.title()} Patterns',
                xaxis: {{title: 'Frequency'}},
                yaxis: {{title: 'Action', automargin: true}},
                height: 500
            }};
            Plotly.newPlot('actions-{pattern_type}', data_{pattern_type}, layout_{pattern_type});
        </script>
"""

    html += """
    </div>
"""

    # TAB 3: Comparisons
    html += """
    <div id="comparisons" class="tab-content">
        <h2>⚖️ Cross-Group Comparisons</h2>
"""

    # Sentiment comparisons
    html += """
        <h3>Sentiment Differences</h3>
        <table class="comparison-table">
            <tr>
                <th>Comparison</th>
                <th>Streaming Difference</th>
                <th>Whole String Difference</th>
            </tr>
"""

    for key, value in comparisons.items():
        if 'sentiment' in key:
            comparison_name = key.replace('sentiment_', '').replace('_', ' vs ').upper()
            html += f"""
            <tr>
                <td>{comparison_name}</td>
                <td>{value['streaming_diff']:.3f}</td>
                <td>{value['whole_diff']:.3f}</td>
            </tr>
"""

    html += """
        </table>
"""

    # Action comparisons
    html += """
        <h3>Unique & Shared Actions Between Groups</h3>
"""

    for key, value in comparisons.items():
        if 'actions' in key:
            comparison_name = key.replace('actions_', '').replace('_', ' vs ').upper()
            html += f"""
        <div style="background: #f5f7fa; padding: 15px; margin: 15px 0; border-radius: 10px;">
            <h4>{comparison_name}</h4>
            <p><strong>Overlap Ratio:</strong> {value['overlap_ratio']:.1%}</p>
            <p><strong>Shared Actions ({len(value['shared'])}):</strong> {', '.join(value['shared'][:10])}</p>
"""

            groups = comparison_name.split(' VS ')
            if len(groups) == 2:
                g0 = groups[0].lower()
                g1 = groups[1].lower()
                unique_0 = value.get(f'unique_to_{g0}', [])
                unique_1 = value.get(f'unique_to_{g1}', [])
                html += f"""
            <p><strong>Unique to {groups[0]} ({len(unique_0)}):</strong>
                {', '.join(list(unique_0)[:10])}</p>
            <p><strong>Unique to {groups[1]} ({len(unique_1)}):</strong>
                {', '.join(list(unique_1)[:10])}</p>
"""

            html += """
        </div>
"""

    html += """
    </div>
"""

    # TAB 4: Streaming vs Whole
    html += """
    <div id="inference" class="tab-content">
        <h2>🔄 Streaming vs Whole String Inference</h2>
"""

    # Agreement scatter
    agreements_by_type = {pt: [] for pt in pattern_types}
    sentiment_diffs_by_type = {pt: [] for pt in pattern_types}

    for a in analyses:
        agreements_by_type[a.pattern_type].append(a.action_agreement)
        sentiment_diffs_by_type[a.pattern_type].append(a.sentiment_diff)

    html += f"""
        <div class="chart" id="agreement-scatter"></div>
        <script>
            var scatterData = [];
"""

    for pt in pattern_types:
        agreements = agreements_by_type[pt]
        sent_diffs = sentiment_diffs_by_type[pt]

        html += f"""
            scatterData.push({{
                x: {json.dumps(agreements)},
                y: {json.dumps(sent_diffs)},
                mode: 'markers',
                type: 'scatter',
                name: '{pt}',
                marker: {{size: 8, opacity: 0.6}}
            }});
"""

    html += """
            var scatterLayout = {
                title: 'Action Agreement vs Sentiment Difference',
                xaxis: {title: 'Action Agreement (Jaccard Similarity)'},
                yaxis: {title: 'Sentiment Difference (Abs)'},
                hovermode: 'closest'
            };
            Plotly.newPlot('agreement-scatter', scatterData, scatterLayout);
        </script>

        <div class="stat-grid">
"""

    for pt in pattern_types:
        avg_agreement = np.mean(agreements_by_type[pt])
        avg_diff = np.mean(sentiment_diffs_by_type[pt])

        html += f"""
            <div class="stat-card">
                <h4>{pt.upper()}</h4>
                <p>Avg Agreement: <span class="value" style="font-size: 1.5em;">{avg_agreement:.1%}</span></p>
                <p>Avg Sentiment Diff: <span class="value" style="font-size: 1.5em;">{avg_diff:.3f}</span></p>
            </div>
"""

    html += """
        </div>
    </div>
"""

    # JavaScript for tabs
    html += """
    <script>
        function showTab(tabName) {
            // Hide all tabs
            var tabs = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }

            // Remove active from buttons
            var buttons = document.getElementsByClassName('tab');
            for (var i = 0; i < buttons.length; i++) {
                buttons[i].classList.remove('active');
            }

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
"""

    html += """
</div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Saved interactive visualization to {output_path}")


def main():
    """Main analysis pipeline"""

    # Configuration
    data_path = Path("data/positive_patterns.jsonl")
    probes_base_dir = Path("data/probes_binary")
    sentiment_probes_dir = Path("data/sentiment")
    model_name = "google/gemma-3-4b-it"
    threshold = 0.5
    output_dir = Path("results/positive_patterns_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("POSITIVE PATTERNS COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Dataset: {data_path}")
    print(f"Probes: {probes_base_dir}")
    print(f"Sentiment probes: {sentiment_probes_dir}")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print("="*80)

    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = load_dataset(data_path)
    print(f"   Loaded {len(dataset)} entries")

    # Initialize inference engine with both cognitive and sentiment probes
    print("\n🔧 Initializing inference engine...")
    engine = StreamingProbeInferenceEngine(
        probes_base_dir=probes_base_dir,
        model_name=model_name,
        sentiment_probes_dir=sentiment_probes_dir,
        include_sentiment=True,
        layer_range=(15, 30),  # Cognitive probes
        verbose=True
    )

    # Analyze all patterns
    print("\n🔬 Analyzing patterns...")
    print(f"   Total entries to process: {len(dataset)}")
    print(f"   Each entry has 3 patterns: positive, negative, transformation")
    print(f"   Expected total analyses: ~{len(dataset) * 3}")
    print()
    all_analyses = []

    for idx, entry in enumerate(dataset):
        if idx % 10 == 0:
            print(f"   [{idx}/{len(dataset)}] Processing entry {idx} ({idx/len(dataset)*100:.1f}%) - Completed: {len(all_analyses)} analyses")

        # Extract pattern texts
        patterns = extract_pattern_texts(entry)

        # Analyze each pattern type
        for pattern_type, text in patterns.items():
            if not text.strip():
                continue

            try:
                if idx < 5 or idx % 50 == 0:  # Extra verbose for first few and every 50
                    print(f"      Analyzing {pattern_type} ({len(text)} chars)...")

                analysis = analyze_pattern(
                    engine,
                    pattern_type,
                    text,
                    entry,
                    threshold
                )
                all_analyses.append(analysis)

                if idx < 5 or idx % 50 == 0:
                    print(f"         ✓ Top action: {analysis.streaming_top_actions[0][0] if analysis.streaming_top_actions else 'none'}")
            except Exception as e:
                print(f"   ⚠ Warning: Failed to analyze {pattern_type} pattern #{idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n✓ Analyzed {len(all_analyses)} patterns")

    # Compute group statistics
    print("\n📊 Computing group statistics...")
    pattern_types = ['positive', 'negative', 'transformation']
    group_stats = {}

    for pt in pattern_types:
        stats = aggregate_group_statistics(all_analyses, pt)
        if stats:
            group_stats[pt] = stats
            print(f"   {pt.upper()}: {stats['num_samples']} samples")

    # Compute cross-group comparisons
    print("\n⚖️ Computing cross-group comparisons...")
    comparisons = compute_cross_group_differences(group_stats)

    # Save results to JSON
    results_json = output_dir / "analysis_results.json"
    print(f"\n💾 Saving results to {results_json}...")

    with open(results_json, 'w') as f:
        json.dump({
            'statistics': group_stats,
            'comparisons': comparisons,
            'analyses': [asdict(a) for a in all_analyses]
        }, f, indent=2)

    print(f"✓ Saved results")

    # Create HTML visualization
    html_output = output_dir / "visualization.html"
    print(f"\n🎨 Creating interactive visualization...")
    create_html_visualization(group_stats, comparisons, all_analyses, html_output)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"📈 Visualization: {html_output}")
    print(f"📋 JSON data: {results_json}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for pt, stats in group_stats.items():
        print(f"\n{pt.upper()} Patterns ({stats['num_samples']} samples):")
        print(f"  Sentiment (streaming): {stats['avg_sentiment_streaming']:.3f} ± {stats['std_sentiment_streaming']:.3f}")
        print(f"  Sentiment (whole): {stats['avg_sentiment_whole']:.3f} ± {stats['std_sentiment_whole']:.3f}")
        print(f"  Action agreement: {stats['avg_action_agreement']:.1%}")
        print(f"  Top 5 actions:")
        for i, (action, count) in enumerate(stats['top_streaming_actions'][:5], 1):
            print(f"    {i}. {action}: {count}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
