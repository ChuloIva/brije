#!/usr/bin/env python3
"""
Parse interactive HTML transcripts and aggregate cognitive action predictions.
Similar to the output format in output_example_3.txt
"""

import json
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import argparse
from pathlib import Path


def extract_utterances_from_html(html_content):
    """Extract utterances and their cognitive actions from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    utterances = []
    utterance_divs = soup.find_all('div', class_='utterance')
    
    for idx, utt_div in enumerate(utterance_divs, 1):
        # Get speaker info
        speaker_label = utt_div.find('div', class_='speaker-label')
        if speaker_label:
            speaker_text = speaker_label.get_text(strip=True)
            # Remove turn number
            turn_span = speaker_label.find('span', class_='turn-number')
            if turn_span:
                speaker = speaker_text.replace(turn_span.get_text(strip=True), '').strip()
            else:
                speaker = speaker_text
        else:
            speaker = "Unknown"
        
        # Get text content
        text_content_div = utt_div.find('div', class_='text-content')
        if not text_content_div:
            continue
            
        # Extract plain text (removing <bos> tokens and special characters)
        text_tokens = []
        all_actions = []
        all_sentiments = []
        
        for token_span in text_content_div.find_all('span', class_='token'):
            token_text = token_span.get_text()
            # Skip <bos> tokens
            if token_text != '<bos>':
                # Clean up token (remove ▁ prefix used for spaces)
                clean_token = token_text.replace('▁', ' ')
                text_tokens.append(clean_token)
            
            # Extract actions from data-actions attribute
            actions_attr = token_span.get('data-actions', '[]')
            try:
                actions = json.loads(actions_attr)
                all_actions.extend(actions)
            except json.JSONDecodeError:
                continue
            
            # Extract sentiment from data-sentiment attribute
            sentiment_attr = token_span.get('data-sentiment', '0.000')
            try:
                sentiment = float(sentiment_attr)
                all_sentiments.append(sentiment)
            except (ValueError, TypeError):
                continue
        
        # Combine text tokens
        full_text = ''.join(text_tokens).strip()
        
        # Calculate sentiment statistics
        sentiment_stats = None
        if all_sentiments:
            sentiment_stats = {
                'mean': sum(all_sentiments) / len(all_sentiments),
                'min': min(all_sentiments),
                'max': max(all_sentiments),
                'values': all_sentiments
            }
        
        utterances.append({
            'index': idx,
            'speaker': speaker,
            'text': full_text,
            'actions': all_actions,
            'sentiment': sentiment_stats
        })
    
    return utterances


def aggregate_actions(actions):
    """Aggregate actions by type, layers, count, and max confidence."""
    # Group by action name
    action_groups = defaultdict(lambda: {
        'layers': set(),
        'confidences': [],
        'count': 0
    })
    
    for action in actions:
        action_name = action['action']
        layer = action['layer']
        confidence = action['confidence']
        
        action_groups[action_name]['layers'].add(layer)
        action_groups[action_name]['confidences'].append(confidence)
        action_groups[action_name]['count'] += 1
    
    # Convert to sorted list
    aggregated = []
    for action_name, data in action_groups.items():
        aggregated.append({
            'action': action_name,
            'layers': sorted(data['layers']),
            'count': data['count'],
            'max_confidence': max(data['confidences'])
        })
    
    # Sort by count (descending), then by max_confidence (descending)
    aggregated.sort(key=lambda x: (x['count'], x['max_confidence']), reverse=True)
    
    return aggregated


def format_layers(layers, max_width=20):
    """Format layer numbers to fit in a fixed width."""
    layer_str = ", ".join(str(l) for l in layers)
    # Don't truncate - just return the full layer string
    return f"({layer_str})"


def print_utterance_analysis(utterance, show_all=False, threshold=0.5, file=None):
    """Print analysis for a single utterance in the output_example_3.txt format."""
    idx = utterance['index']
    speaker = utterance['speaker']
    text = utterance['text']
    sentiment = utterance.get('sentiment')
    
    print(f"\n[{idx}] {speaker}:", file=file)
    print(f'"{text}"', file=file)
    print("-" * 80, file=file)
    
    # Print sentiment information if available
    if sentiment:
        print(f"Sentiment: Mean={sentiment['mean']:+.3f}, Min={sentiment['min']:+.3f}, Max={sentiment['max']:+.3f}", file=file)
    
    aggregated = aggregate_actions(utterance['actions'])
    
    if not aggregated:
        print("  No cognitive actions detected", file=file)
        return
    
    print("Predictions grouped by action:", file=file)
    
    for i, action_data in enumerate(aggregated, 1):
        action = action_data['action']
        layers = action_data['layers']
        count = action_data['count']
        max_conf = action_data['max_confidence']
        
        # Skip low confidence predictions unless show_all is True
        if not show_all and max_conf < threshold:
            continue
        
        # Format layers to fit width
        layer_str = ", ".join(str(l) for l in layers)
        # Pad to width of 20 chars
        if len(layer_str) <= 18:
            layer_str = layer_str + " " * (18 - len(layer_str))
        
        # Check if passes threshold
        passes = "✓" if max_conf >= threshold else " "
        
        # Format action name with padding
        action_display = action.replace('_', ' ')
        
        print(f"  {passes} {i:2d}. {action_display:30s}  (Layers {layer_str})  Count: {count:2d}  Max: {max_conf:.4f}", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Parse interactive HTML transcripts and aggregate cognitive action predictions"
    )
    parser.add_argument(
        'html_file',
        type=str,
        help='Path to the interactive HTML file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for displaying actions (default: 0.5)'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all predictions regardless of confidence'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: <input_basename>_analysis.txt)'
    )
    parser.add_argument(
        '--max-utterances',
        type=int,
        help='Maximum number of utterances to process'
    )
    parser.add_argument(
        '--no-file',
        action='store_true',
        help='Print to stdout instead of file'
    )
    
    args = parser.parse_args()
    
    # Read HTML file
    html_path = Path(args.html_file)
    if not html_path.exists():
        print(f"Error: File not found: {args.html_file}")
        return 1
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract utterances
    utterances = extract_utterances_from_html(html_content)
    
    if not utterances:
        print("No utterances found in HTML file")
        return 1
    
    # Limit utterances if requested
    if args.max_utterances:
        utterances = utterances[:args.max_utterances]
    
    # Determine output file
    output_file = None
    if not args.no_file:
        if args.output:
            output_path = Path(args.output)
        else:
            # Create default output filename
            output_path = html_path.parent / f"{html_path.stem}_analysis.txt"
        
        output_file = open(output_path, 'w', encoding='utf-8')
        print(f"Writing analysis to: {output_path}")
    
    try:
        # Print header
        print(f"Processing {len(utterances)} utterances from: {html_path.name}", file=output_file)
        print(file=output_file)
        print("=" * 80, file=output_file)
        
        # Process each utterance
        for utterance in utterances:
            print_utterance_analysis(utterance, show_all=args.show_all, threshold=args.threshold, file=output_file)
        
        print(file=output_file)
        print("=" * 80, file=output_file)
        print(f"\nProcessing complete. Analyzed {len(utterances)} utterances.", file=output_file)
        
        if output_file:
            print(f"\nAnalysis saved to: {output_path}")
    
    finally:
        if output_file:
            output_file.close()
    
    return 0


if __name__ == '__main__':
    exit(main())

