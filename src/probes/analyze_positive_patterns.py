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

    # Inference results
    top_actions: List[Tuple[str, int, float]]  # (action, layer_count, confidence)
    sentiment_avg: float
    sentiment_layers: List[int]


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

    # Clear cognitive activations from GPU memory
    del cognitive_activations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete

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

        # Clear sentiment activations from GPU memory
        del sentiment_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete

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

    # Extract top actions
    top_actions = [
        (p.action_name, p.layer_count, p.max_confidence)
        for p in cog_preds[:10] if p.is_active
    ]

    # Sentiment aggregation
    sentiment_avg = np.mean([p.max_confidence for p in sent_preds]) if sent_preds else 0.0
    sentiment_layers = [p.best_layer for p in sent_preds if p.is_active]

    # Clean up predictions to free memory
    del cog_preds, sent_preds

    return PatternAnalysis(
        pattern_type=pattern_type,
        text=text[:200],  # Store first 200 chars
        cognitive_pattern_name=entry.get('cognitive_pattern_name', ''),
        cognitive_pattern_type=entry.get('cognitive_pattern_type', ''),
        top_actions=top_actions,
        sentiment_avg=sentiment_avg,
        sentiment_layers=sentiment_layers
    )


def aggregate_group_statistics(analyses: List[PatternAnalysis],
                               pattern_type: str) -> Dict:
    """Compute aggregate statistics for a pattern type group"""

    group_analyses = [a for a in analyses if a.pattern_type == pattern_type]

    if not group_analyses:
        return {}

    # Collect all actions
    action_counts = Counter()
    layer_counts = defaultdict(list)

    for analysis in group_analyses:
        for action, layer_count, _ in analysis.top_actions:
            action_counts[action] += 1
            layer_counts[action].append(layer_count)

    # Top actions for this group
    top_actions = action_counts.most_common(15)

    # Sentiment statistics
    sentiments = [a.sentiment_avg for a in group_analyses]

    return {
        'pattern_type': pattern_type,
        'num_samples': len(group_analyses),
        'top_actions': top_actions,
        'avg_sentiment': float(np.mean(sentiments)),
        'std_sentiment': float(np.std(sentiments)),
        'cognitive_pattern_types': Counter([a.cognitive_pattern_type for a in group_analyses])
    }


def compute_cross_group_differences(stats: Dict[str, Dict]) -> Dict:
    """Compare statistics across positive, negative, transformation groups"""

    comparisons = {}

    # Compare sentiment between groups
    if 'positive' in stats and 'negative' in stats:
        comparisons['sentiment_positive_vs_negative'] = {
            'diff': stats['positive']['avg_sentiment'] - stats['negative']['avg_sentiment']
        }

    if 'positive' in stats and 'transformation' in stats:
        comparisons['sentiment_positive_vs_transformation'] = {
            'diff': stats['positive']['avg_sentiment'] - stats['transformation']['avg_sentiment']
        }

    if 'negative' in stats and 'transformation' in stats:
        comparisons['sentiment_negative_vs_transformation'] = {
            'diff': stats['negative']['avg_sentiment'] - stats['transformation']['avg_sentiment']
        }

    # Compare action distributions
    for group1 in ['positive', 'negative', 'transformation']:
        for group2 in ['positive', 'negative', 'transformation']:
            if group1 < group2 and group1 in stats and group2 in stats:
                # Get top actions for each group
                actions1 = set([a[0] for a in stats[group1]['top_actions'][:10]])
                actions2 = set([a[0] for a in stats[group2]['top_actions'][:10]])

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
        sentiment_emoji = "😊" if stat['avg_sentiment'] > 0 else "😔"
        html += f"""
            <div class="stat-card">
                <h4>{pattern_type.upper()}</h4>
                <div class="value {pattern_type}">{stat['num_samples']}</div>
                <p style="margin: 5px 0; color: #666;">
                    {sentiment_emoji} Sentiment: {stat['avg_sentiment']:.3f}
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
    </div>
"""

    # TAB 1: Sentiment Analysis
    html += """
    <div id="sentiment" class="tab-content active">
        <h2>😊😔 Sentiment Analysis</h2>
"""

    # Sentiment bar chart data
    pattern_types = list(stats.keys())
    sentiments = [stats[pt]['avg_sentiment'] for pt in pattern_types]

    html += f"""
        <div class="chart" id="sentiment-chart"></div>
        <script>
            var data = [
                {{
                    x: {json.dumps(pattern_types)},
                    y: {json.dumps(sentiments)},
                    type: 'bar',
                    marker: {{color: '#667eea'}}
                }}
            ];
            var layout = {{
                title: 'Average Sentiment by Pattern Type',
                yaxis: {{title: 'Sentiment Score'}},
                xaxis: {{title: 'Pattern Type'}}
            }};
            Plotly.newPlot('sentiment-chart', data, layout);
        </script>
"""

    # Sentiment distribution violin plot
    html += f"""
        <div class="chart" id="sentiment-violin"></div>
        <script>
            var violinData = [];
"""

    for pt in pattern_types:
        pt_sentiments = [analyses[i].sentiment_avg for i in range(len(analyses)) if analyses[i].pattern_type == pt]
        html += f"""
            violinData.push({{
                y: {json.dumps(pt_sentiments)},
                type: 'violin',
                name: '{pt}',
                box: {{visible: true}},
                meanline: {{visible: true}}
            }});
"""

    html += """
            var violinLayout = {
                title: 'Sentiment Distribution by Pattern Type',
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
        top_actions = stat['top_actions'][:15]

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
                <th>Difference</th>
            </tr>
"""

    for key, value in comparisons.items():
        if 'sentiment' in key:
            comparison_name = key.replace('sentiment_', '').replace('_', ' vs ').upper()
            html += f"""
            <tr>
                <td>{comparison_name}</td>
                <td>{value['diff']:.3f}</td>
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


def save_batch_results(batch_analyses: List[PatternAnalysis], batch_num: int, output_dir: Path):
    """Save a batch of analyses to disk"""
    batch_file = output_dir / f"batch_{batch_num:04d}.json"
    with open(batch_file, 'w') as f:
        json.dump([asdict(a) for a in batch_analyses], f, indent=2)
    print(f"      💾 Saved batch {batch_num} ({len(batch_analyses)} analyses) to {batch_file.name}")


def load_all_batch_results(output_dir: Path) -> List[PatternAnalysis]:
    """Load all batch results from disk"""
    all_analyses = []
    batch_files = sorted(output_dir.glob("batch_*.json"))

    print(f"\n📂 Loading {len(batch_files)} batch files...")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            for analysis_dict in batch_data:
                all_analyses.append(PatternAnalysis(**analysis_dict))

    print(f"   ✓ Loaded {len(all_analyses)} total analyses")
    return all_analyses


def main():
    """Main analysis pipeline"""

    # Configuration
    data_path = Path("data/positive_patterns.jsonl")
    probes_base_dir = Path("data/probes_binary")
    sentiment_probes_dir = Path("data/sentiment")
    model_name = "google/gemma-3-4b-it"
    threshold = 0.5
    batch_size = 10  # Save results every 10 patterns
    memory_cleanup_interval = 5  # Clear GPU cache every 5 patterns
    output_dir = Path("results/positive_patterns_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create batches subdirectory
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(exist_ok=True)

    print("="*80)
    print("POSITIVE PATTERNS COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Dataset: {data_path}")
    print(f"Probes: {probes_base_dir}")
    print(f"Sentiment probes: {sentiment_probes_dir}")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Batch size: {batch_size} patterns")
    print(f"Memory cleanup interval: {memory_cleanup_interval} patterns")
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

    # Analyze all patterns with batch saving
    print("\n🔬 Analyzing patterns...")
    print(f"   Total entries to process: {len(dataset)}")
    print(f"   Each entry has 3 patterns: positive, negative, transformation")
    print(f"   Expected total analyses: ~{len(dataset) * 3}")
    print(f"   Saving results every {batch_size} patterns")
    print()

    batch_analyses = []
    batch_num = 0
    total_analyzed = 0

    for idx, entry in enumerate(dataset):
        if idx % 10 == 0:
            print(f"   [{idx}/{len(dataset)}] Processing entry {idx} ({idx/len(dataset)*100:.1f}%) - Analyzed: {total_analyzed} patterns")

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
                batch_analyses.append(analysis)
                total_analyzed += 1

                if idx < 5 or idx % 50 == 0:
                    print(f"         ✓ Top action: {analysis.top_actions[0][0] if analysis.top_actions else 'none'}")

                # CRITICAL: Clear cache after EACH example to prevent OOMs
                # The model.trace() calls create computational graphs that accumulate
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Save batch when it reaches batch_size
                if len(batch_analyses) >= batch_size:
                    save_batch_results(batch_analyses, batch_num, batches_dir)
                    batch_num += 1
                    batch_analyses = []  # Clear batch from memory

            except Exception as e:
                print(f"   ⚠ Warning: Failed to analyze {pattern_type} pattern #{idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save remaining analyses in final batch
    if batch_analyses:
        save_batch_results(batch_analyses, batch_num, batches_dir)
        batch_num += 1

    print(f"\n✓ Analyzed {total_analyzed} patterns in {batch_num} batches")

    # Final memory cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load all batch results for aggregation
    print("\n📊 Loading all batches for aggregation...")
    all_analyses = load_all_batch_results(batches_dir)

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
        print(f"  Sentiment: {stats['avg_sentiment']:.3f} ± {stats['std_sentiment']:.3f}")
        print(f"  Top 5 actions:")
        for i, (action, count) in enumerate(stats['top_actions'][:5], 1):
            print(f"    {i}. {action}: {count}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
