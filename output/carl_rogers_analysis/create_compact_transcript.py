import json
from collections import defaultdict, Counter
import statistics

def format_processes_compact(action_layer_details):
    """Format processes as a compact table-like list."""
    if not action_layer_details:
        return "  (No active processes detected)"

    lines = []
    for idx, (process_name, layers) in enumerate(sorted(action_layer_details.items()), 1):
        # Format process name
        readable_name = process_name.replace('_', ' ')

        # Get layer numbers and max confidence
        layer_nums = sorted([l['layer'] for l in layers])
        max_conf = max([l['confidence'] for l in layers])
        count = len(layers)

        # Format layer list (pad to show alignment)
        layer_str = ", ".join([str(l) for l in layer_nums])

        # Create formatted line
        line = f"  ✓ {idx}. {readable_name:<30} (Layers {layer_str:<20})  Count: {count:2}  Max: {max_conf:.4f}"
        lines.append(line)

    return "\n".join(lines)

def collect_statistics(sessions):
    """Collect statistics about processes, layers, and confidences."""
    stats = {
        'process_counts': Counter(),
        'process_confidences': defaultdict(list),
        'layer_usage': defaultdict(int),
        'process_layer_mapping': defaultdict(lambda: defaultdict(int)),
        'total_utterances': 0,
        'utterances_by_speaker': Counter()
    }

    for session in sessions:
        for utterance in session['utterances']:
            stats['total_utterances'] += 1
            stats['utterances_by_speaker'][utterance['speaker']] += 1

            # Process layer details
            if 'action_layer_details' in utterance:
                for process_name, layers in utterance['action_layer_details'].items():
                    stats['process_counts'][process_name] += len(layers)

                    for layer_info in layers:
                        layer_num = layer_info['layer']
                        confidence = layer_info['confidence']

                        stats['layer_usage'][layer_num] += 1
                        stats['process_confidences'][process_name].append(confidence)
                        stats['process_layer_mapping'][process_name][layer_num] += 1

    return stats

def create_statistics_summary(stats):
    """Create a formatted statistics summary."""
    summary = []

    summary.append("=" * 80 + "\n")
    summary.append("STATISTICS SUMMARY\n")
    summary.append("=" * 80 + "\n\n")

    summary.append(f"Total Utterances:      {stats['total_utterances']}\n")
    summary.append(f"Therapist Utterances:  {stats['utterances_by_speaker']['therapist']}\n")
    summary.append(f"Client Utterances:     {stats['utterances_by_speaker']['client']}\n\n")

    # Top processes
    summary.append("-" * 80 + "\n")
    summary.append("TOP 20 MOST FREQUENT PROCESSES\n")
    summary.append("-" * 80 + "\n\n")

    for idx, (process, count) in enumerate(stats['process_counts'].most_common(20), 1):
        readable_name = process.replace('_', ' ')

        # Average confidence
        if process in stats['process_confidences']:
            avg_conf = statistics.mean(stats['process_confidences'][process])
            max_conf = max(stats['process_confidences'][process])
            min_conf = min(stats['process_confidences'][process])
        else:
            avg_conf = max_conf = min_conf = 0

        # Most common layers
        if process in stats['process_layer_mapping']:
            top_layers = sorted(
                stats['process_layer_mapping'][process].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            layers_str = ", ".join([str(l) for l, _ in top_layers])
        else:
            layers_str = "N/A"

        summary.append(f"  {idx:2}. {readable_name:<30} ")
        summary.append(f"Count: {count:4}  Avg: {avg_conf:.3f}  Max: {max_conf:.3f}  ")
        summary.append(f"Layers: {layers_str}\n")

    # Layer usage
    summary.append("\n" + "-" * 80 + "\n")
    summary.append("LAYER USAGE DISTRIBUTION\n")
    summary.append("-" * 80 + "\n\n")

    for layer in sorted(stats['layer_usage'].keys()):
        count = stats['layer_usage'][layer]
        bar_length = int(count / 100)
        bar = "█" * bar_length
        summary.append(f"  Layer {layer:2}:  {count:5} activations  {bar}\n")

    summary.append("\n" + "=" * 80 + "\n\n")

    return "".join(summary)

def create_compact_transcript(json_path, output_path):
    """Convert annotated JSON to compact readable format."""

    with open(json_path, 'r') as f:
        sessions = json.load(f)

    # Collect statistics
    print("Collecting statistics...")
    stats = collect_statistics(sessions)

    with open(output_path, 'w') as out:
        out.write("CARL ROGERS THERAPY SESSIONS - ANNOTATED TRANSCRIPTS\n")
        out.write("=" * 80 + "\n\n")
        out.write("Cognitive and emotional processes shown with layers and confidence scores.\n\n")

        # Write statistics summary
        out.write(create_statistics_summary(stats))
        out.write("\n" + "=" * 80 + "\n")
        out.write("TRANSCRIPTS\n")
        out.write("=" * 80 + "\n\n")

        # Write sessions
        for session_idx, session in enumerate(sessions, 1):
            # Write session header
            session_title = session.get('title', session.get('session_name', f'Session {session_idx}'))
            out.write("\n" + "=" * 80 + "\n")
            out.write(f"SESSION: {session_title}\n")
            out.write("=" * 80 + "\n\n")

            # Write utterances
            for utt_idx, utterance in enumerate(session['utterances'], 1):
                speaker = utterance['speaker'].upper()
                text = utterance['text']

                # Format the utterance
                out.write(f"[{utt_idx}] {speaker}:\n")
                out.write(f"{text}\n\n")

                # Get active processes
                action_layer_details = utterance.get('action_layer_details', {})
                processes_formatted = format_processes_compact(action_layer_details)
                out.write(f"{processes_formatted}\n\n")
                out.write("-" * 80 + "\n\n")

    print(f"✓ Created compact transcript: {output_path}")
    print(f"  Total sessions: {len(sessions)}")
    total_utterances = sum(len(s['utterances']) for s in sessions)
    print(f"  Total utterances: {total_utterances}")
    print(f"  Unique processes detected: {len(stats['process_counts'])}")
    print(f"  Layers used: {min(stats['layer_usage'].keys())} - {max(stats['layer_usage'].keys())}")

if __name__ == "__main__":
    json_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/annotated_transcripts.json"
    output_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/compact_transcript.txt"

    create_compact_transcript(json_path, output_path)