#!/usr/bin/env python3
"""
Create an interactive HTML transcript with streaming probe annotations.

This script processes a therapy transcript CSV and generates an interactive HTML
where each token is color-coded by sentiment and includes hover tooltips showing
cognitive action predictions.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import sys
from tqdm import tqdm

# Add src and parent directory to path
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / 'src'))
sys.path.insert(0, str(base_path / 'src' / 'probes'))

from probes.streaming_probe_inference import StreamingProbeInferenceEngine
from probes.action_categories import ACTION_TO_CATEGORY, get_action_category


def load_transcript(csv_path: Path) -> pd.DataFrame:
    """Load transcript from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def get_category_color(action_name: str) -> str:
    """Get color for a cognitive action based on its category."""
    category = get_action_category(action_name)
    colors = {
        'metacognitive': '#3b82f6',  # Blue
        'analytical': '#22c55e',      # Green
        'creative': '#a855f7',        # Purple
        'emotional': '#ef4444',       # Red
        'memory': '#f59e0b',          # Amber
    }
    return colors.get(category, '#6b7280')  # Gray fallback


def get_sentiment_color(sentiment_score: float) -> str:
    """
    Get color based on normalized sentiment score (z-score).
    Negative = red tones, Positive = green tones
    """
    # Clamp to reasonable range for visualization
    clamped = max(-2.5, min(2.5, sentiment_score))

    if clamped < -1.5:
        return '#dc2626'  # Strong negative - red-700
    elif clamped < -0.5:
        return '#f87171'  # Mild negative - red-400
    elif clamped < 0.5:
        return '#d1d5db'  # Neutral - gray-300
    elif clamped < 1.5:
        return '#86efac'  # Mild positive - green-300
    else:
        return '#22c55e'  # Strong positive - green-500


def process_transcript_with_probes(
    df: pd.DataFrame,
    engine: StreamingProbeInferenceEngine,
    cognitive_threshold: float = 0.1,
    top_k_actions: int = 5
) -> List[Dict[str, Any]]:
    """
    Process each utterance with streaming probe inference.

    Returns a list of utterances with token-level annotations.
    """
    annotated_utterances = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing utterances"):
        text = row['text']
        speaker = row['speaker']
        speaker_name = row['speaker_name']
        turn_number = row['turn_number']

        # Run streaming inference
        predictions = engine.predict_streaming(
            text=text,
            top_k=50,  # Get all predictions
            display_mode=None  # No console output
        )

        # Extract token-level data
        token_data = []

        # Build a map of token_position -> activations
        token_map = {}

        for pred in predictions:
            for tok_act in pred.token_activations:
                pos = tok_act.token_position
                if pos not in token_map:
                    token_map[pos] = {
                        'token_text': tok_act.token_text,
                        'token_id': tok_act.token_id,
                        'position': pos,
                        'cognitive_actions': [],
                        'sentiment_scores': []
                    }

                if tok_act.probe_type == 'cognitive':
                    if tok_act.confidence >= cognitive_threshold:
                        token_map[pos]['cognitive_actions'].append({
                            'action': tok_act.action_name,
                            'confidence': tok_act.confidence,
                            'layer': tok_act.layer,
                            'category': get_category_name(tok_act.action_name)
                        })
                elif tok_act.probe_type == 'sentiment':
                    token_map[pos]['sentiment_scores'].append({
                        'score': tok_act.confidence,
                        'layer': tok_act.layer
                    })

        # Process each token
        for pos in sorted(token_map.keys()):
            tok = token_map[pos]

            # Aggregate cognitive actions (top-k by confidence)
            cognitive = sorted(
                tok['cognitive_actions'],
                key=lambda x: x['confidence'],
                reverse=True
            )[:top_k_actions]

            # Average sentiment score across layers
            avg_sentiment = 0.0
            if tok['sentiment_scores']:
                avg_sentiment = sum(s['score'] for s in tok['sentiment_scores']) / len(tok['sentiment_scores'])

            token_data.append({
                'text': tok['token_text'],
                'position': pos,
                'sentiment_score': avg_sentiment,
                'sentiment_color': get_sentiment_color(avg_sentiment),
                'cognitive_actions': cognitive
            })

        annotated_utterances.append({
            'turn_number': turn_number,
            'speaker': speaker,
            'speaker_name': speaker_name,
            'original_text': text,
            'tokens': token_data
        })

    return annotated_utterances


def get_category_name(action_name: str) -> str:
    """Get the category name for an action."""
    return get_action_category(action_name)


def generate_html(
    annotated_utterances: List[Dict[str, Any]],
    output_path: Path,
    session_name: str
):
    """Generate interactive HTML with embedded probe data."""

    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Therapy Transcript: {session_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .legend {{
            background: #f9fafb;
            padding: 20px 30px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .legend h3 {{
            color: #1f2937;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}

        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .legend-color {{
            width: 30px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}

        .legend-label {{
            font-size: 0.9em;
            color: #4b5563;
        }}

        .transcript {{
            padding: 30px;
            max-height: calc(100vh - 400px);
            overflow-y: auto;
        }}

        .utterance {{
            margin-bottom: 25px;
            padding: 20px;
            background: #f9fafb;
            border-radius: 12px;
            border-left: 4px solid #e5e7eb;
            transition: all 0.2s;
        }}

        .utterance:hover {{
            background: #f3f4f6;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }}

        .utterance.therapist {{
            border-left-color: #3b82f6;
        }}

        .utterance.client {{
            border-left-color: #10b981;
        }}

        .speaker-label {{
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .turn-number {{
            font-size: 0.85em;
            color: #6b7280;
            background: #e5e7eb;
            padding: 2px 8px;
            border-radius: 12px;
        }}

        .text-content {{
            line-height: 1.8;
            font-size: 1.05em;
        }}

        .token {{
            cursor: pointer;
            padding: 3px 1px;
            border-radius: 3px;
            transition: all 0.15s;
            position: relative;
            display: inline-block;
        }}

        .token:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            z-index: 10;
        }}

        .tooltip {{
            display: none;
            position: absolute;
            background: white;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            min-width: 300px;
            max-width: 400px;
            font-size: 0.9em;
        }}

        .tooltip.active {{
            display: block;
        }}

        .tooltip-header {{
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .tooltip-section {{
            margin-bottom: 12px;
        }}

        .tooltip-section-title {{
            font-size: 0.85em;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}

        .sentiment-bar {{
            height: 20px;
            background: linear-gradient(to right, #dc2626, #d1d5db, #22c55e);
            border-radius: 10px;
            position: relative;
            margin-bottom: 5px;
        }}

        .sentiment-marker {{
            position: absolute;
            width: 4px;
            height: 100%;
            background: #1f2937;
            border-radius: 2px;
            top: 0;
        }}

        .sentiment-value {{
            font-size: 0.85em;
            color: #4b5563;
            text-align: center;
        }}

        .action-list {{
            list-style: none;
        }}

        .action-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            margin-bottom: 4px;
            background: #f9fafb;
            border-radius: 6px;
            border-left: 3px solid;
        }}

        .action-name {{
            font-weight: 500;
            color: #1f2937;
            font-size: 0.9em;
        }}

        .action-confidence {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .confidence-bar {{
            width: 60px;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            background: currentColor;
            border-radius: 4px;
        }}

        .confidence-value {{
            font-size: 0.85em;
            color: #6b7280;
            min-width: 40px;
            text-align: right;
        }}

        .category-badge {{
            display: inline-block;
            font-size: 0.75em;
            padding: 2px 6px;
            border-radius: 4px;
            background: #e5e7eb;
            color: #4b5563;
            margin-left: 6px;
        }}

        .layer-info {{
            font-size: 0.75em;
            color: #9ca3af;
            margin-left: 4px;
        }}

        /* Category colors */
        .metacognitive {{ color: #3b82f6; }}
        .analytical {{ color: #22c55e; }}
        .creative {{ color: #a855f7; }}
        .emotional {{ color: #ef4444; }}
        .memory {{ color: #f59e0b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{session_name}</h1>
            <p>Interactive Therapy Transcript with Cognitive & Sentiment Analysis</p>
        </div>

        <div class="legend">
            <h3>Sentiment Color Guide</h3>
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #dc2626;"></div>
                    <span class="legend-label">Strong Negative</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f87171;"></div>
                    <span class="legend-label">Mild Negative</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #d1d5db;"></div>
                    <span class="legend-label">Neutral</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #86efac;"></div>
                    <span class="legend-label">Mild Positive</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #22c55e;"></div>
                    <span class="legend-label">Strong Positive</span>
                </div>
            </div>
        </div>

        <div class="transcript">
"""

    # Add each utterance
    for utt in annotated_utterances:
        speaker_class = utt['speaker'].lower()
        html += f"""
            <div class="utterance {speaker_class}">
                <div class="speaker-label">
                    {utt['speaker_name']}
                    <span class="turn-number">Turn {utt['turn_number']}</span>
                </div>
                <div class="text-content">
"""

        # Add each token
        for token in utt['tokens']:
            # Prepare cognitive actions data
            actions_json = json.dumps(token['cognitive_actions'])
            sentiment = token['sentiment_score']

            html += f"""<span class="token" style="background-color: {token['sentiment_color']};" data-actions='{actions_json}' data-sentiment="{sentiment:.3f}">{token['text']}</span>"""

        html += """
                </div>
            </div>
"""

    # Close transcript div and add tooltip
    html += """
        </div>
    </div>

    <div class="tooltip" id="tooltip">
        <div class="tooltip-content"></div>
    </div>

    <script>
        const tooltip = document.getElementById('tooltip');
        const tooltipContent = tooltip.querySelector('.tooltip-content');
        const tokens = document.querySelectorAll('.token');

        // Category colors
        const categoryColors = {
            'metacognitive': '#3b82f6',
            'analytical': '#22c55e',
            'creative': '#a855f7',
            'emotional': '#ef4444',
            'memory': '#f59e0b'
        };

        tokens.forEach(token => {
            token.addEventListener('mouseenter', (e) => {
                const actions = JSON.parse(token.dataset.actions);
                const sentiment = parseFloat(token.dataset.sentiment);

                // Build tooltip content
                let html = '<div class="tooltip-header">Token: "' + token.textContent + '"</div>';

                // Sentiment section
                html += '<div class="tooltip-section">';
                html += '<div class="tooltip-section-title">Sentiment Analysis</div>';
                html += '<div class="sentiment-bar">';

                // Calculate marker position (0 = left/-2.5, 50% = center/0, 100% = right/2.5)
                const markerPos = ((sentiment + 2.5) / 5.0) * 100;
                html += '<div class="sentiment-marker" style="left: ' + markerPos + '%;"></div>';
                html += '</div>';
                html += '<div class="sentiment-value">Score: ' + sentiment.toFixed(3) + '</div>';
                html += '</div>';

                // Cognitive actions section
                if (actions.length > 0) {
                    html += '<div class="tooltip-section">';
                    html += '<div class="tooltip-section-title">Top Cognitive Actions</div>';
                    html += '<ul class="action-list">';

                    actions.forEach(action => {
                        const color = categoryColors[action.category] || '#6b7280';
                        const confidence = (action.confidence * 100).toFixed(1);

                        html += '<div class="action-item" style="border-left-color: ' + color + ';">';
                        html += '<span class="action-name">';
                        html += action.action.replace(/_/g, ' ');
                        html += '<span class="layer-info">L' + action.layer + '</span>';
                        html += '</span>';
                        html += '<div class="action-confidence">';
                        html += '<div class="confidence-bar">';
                        html += '<div class="confidence-fill" style="width: ' + confidence + '%; background: ' + color + ';"></div>';
                        html += '</div>';
                        html += '<span class="confidence-value">' + confidence + '%</span>';
                        html += '</div>';
                        html += '</div>';
                    });

                    html += '</ul>';
                    html += '</div>';
                } else {
                    html += '<div class="tooltip-section">';
                    html += '<div class="tooltip-section-title">Cognitive Actions</div>';
                    html += '<p style="color: #9ca3af; font-size: 0.9em;">No significant cognitive actions detected</p>';
                    html += '</div>';
                }

                tooltipContent.innerHTML = html;

                // Position tooltip
                const rect = token.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();

                let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
                let top = rect.top - tooltipRect.height - 10;

                // Keep tooltip in viewport
                if (left < 10) left = 10;
                if (left + tooltipRect.width > window.innerWidth - 10) {
                    left = window.innerWidth - tooltipRect.width - 10;
                }

                // If tooltip goes above viewport, show below token
                if (top < 10) {
                    top = rect.bottom + 10;
                }

                tooltip.style.left = left + 'px';
                tooltip.style.top = top + window.scrollY + 'px';
                tooltip.classList.add('active');
            });

            token.addEventListener('mouseleave', () => {
                tooltip.classList.remove('active');
            });
        });

        // Hide tooltip when scrolling
        document.querySelector('.transcript').addEventListener('scroll', () => {
            tooltip.classList.remove('active');
        });
    </script>
</body>
</html>
"""

    # Write to file
    output_path.write_text(html)
    print(f"\n✓ Interactive HTML saved to: {output_path}")


def main():
    """Main execution function."""
    # Paths
    base_dir = Path(__file__).parent
    csv_path = base_dir / "output/carl_rogers_analysis/Kathy_session.csv"
    output_path = base_dir / "output/carl_rogers_analysis/Kathy_session_interactive.html"

    print("=" * 80)
    print("Interactive Therapy Transcript Generator")
    print("=" * 80)

    # Load transcript
    print(f"\n📄 Loading transcript from: {csv_path}")
    df = load_transcript(csv_path)
    print(f"   Loaded {len(df)} utterances")

    # Initialize streaming probe engine
    print("\n🧠 Initializing streaming probe engine...")
    print("   - Sentiment probes: layers 1-11 (default)")
    print("   - Cognitive probes: layers 15-30")

    engine = StreamingProbeInferenceEngine(
        probes_base_dir=base_dir / 'data' / 'probes_binary',
        model_name='google/gemma-3-4b-it',
        layer_range=(15, 30),  # Cognitive probes from layer 15 onwards
        sentiment_probes_dir=base_dir / 'data' / 'sentiment',
        include_sentiment=True
    )

    print("   ✓ Engine initialized")

    # Process transcript
    print("\n🔍 Processing transcript with streaming inference...")
    annotated_utterances = process_transcript_with_probes(
        df=df,
        engine=engine,
        cognitive_threshold=0.1,
        top_k_actions=5
    )

    # Generate HTML
    print("\n🎨 Generating interactive HTML...")
    generate_html(
        annotated_utterances=annotated_utterances,
        output_path=output_path,
        session_name="Kathy Session - Carl Rogers (1975)"
    )

    print("\n" + "=" * 80)
    print("✨ Complete! Open the HTML file in your browser to explore.")
    print("=" * 80)


if __name__ == "__main__":
    main()
