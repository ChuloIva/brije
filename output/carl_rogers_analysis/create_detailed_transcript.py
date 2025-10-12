import json
from collections import defaultdict, Counter
import statistics

def format_active_processes_detailed(predictions, action_layer_details):
    """Extract and format active cognitive/emotional processes with layer and confidence info."""
    active = []
    for process_name, process_data in predictions.items():
        if process_data.get('is_active', False):
            # Format the process name to be more readable
            readable_name = process_name.replace('_', ' ').title()

            # Get layer details if available
            if process_name in action_layer_details:
                layers = action_layer_details[process_name]
                # Get the best layer (highest confidence)
                best_layer = max(layers, key=lambda x: x['confidence'])
                layer_num = best_layer['layer']
                confidence = best_layer['confidence']

                # Format with layer and confidence
                active.append(f"{readable_name} [L{layer_num}, {confidence:.2f}]")
            else:
                active.append(readable_name)

    return active

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

            # Process predictions
            for process_name, process_data in utterance['predictions'].items():
                if process_data.get('is_active', False):
                    stats['process_counts'][process_name] += 1

            # Process layer details
            if 'action_layer_details' in utterance:
                for process_name, layers in utterance['action_layer_details'].items():
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

    summary.append("# Statistics Summary\n")
    summary.append(f"**Total Utterances:** {stats['total_utterances']}\n")
    summary.append(f"**Therapist Utterances:** {stats['utterances_by_speaker']['therapist']}\n")
    summary.append(f"**Client Utterances:** {stats['utterances_by_speaker']['client']}\n\n")

    # Top processes
    summary.append("## Top 15 Most Frequent Processes\n\n")
    summary.append("| Process | Count | Avg Confidence | Most Common Layers |\n")
    summary.append("|---------|-------|----------------|--------------------|\n")

    for process, count in stats['process_counts'].most_common(15):
        readable_name = process.replace('_', ' ').title()

        # Average confidence
        if process in stats['process_confidences']:
            avg_conf = statistics.mean(stats['process_confidences'][process])
            avg_conf_str = f"{avg_conf:.3f}"
        else:
            avg_conf_str = "N/A"

        # Most common layers
        if process in stats['process_layer_mapping']:
            top_layers = sorted(
                stats['process_layer_mapping'][process].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            layers_str = ", ".join([f"L{l}" for l, _ in top_layers])
        else:
            layers_str = "N/A"

        summary.append(f"| {readable_name} | {count} | {avg_conf_str} | {layers_str} |\n")

    # Layer usage
    summary.append("\n## Layer Usage Distribution\n\n")
    summary.append("| Layer | Activations |\n")
    summary.append("|-------|-------------|\n")

    for layer in sorted(stats['layer_usage'].keys()):
        count = stats['layer_usage'][layer]
        summary.append(f"| Layer {layer} | {count} |\n")

    # Process confidence details
    summary.append("\n## Process Confidence Statistics\n\n")
    summary.append("| Process | Min | Max | Mean | Median | Std Dev |\n")
    summary.append("|---------|-----|-----|------|--------|----------|\n")

    for process in sorted(stats['process_counts'].keys(), key=lambda x: stats['process_counts'][x], reverse=True)[:10]:
        if process in stats['process_confidences'] and stats['process_confidences'][process]:
            confs = stats['process_confidences'][process]
            readable_name = process.replace('_', ' ').title()

            min_conf = min(confs)
            max_conf = max(confs)
            mean_conf = statistics.mean(confs)
            median_conf = statistics.median(confs)
            std_conf = statistics.stdev(confs) if len(confs) > 1 else 0

            summary.append(f"| {readable_name} | {min_conf:.3f} | {max_conf:.3f} | {mean_conf:.3f} | {median_conf:.3f} | {std_conf:.3f} |\n")

    return "".join(summary)

def create_detailed_transcript(json_path, output_path):
    """Convert annotated JSON to detailed readable markdown."""

    with open(json_path, 'r') as f:
        sessions = json.load(f)

    # Collect statistics
    print("Collecting statistics...")
    stats = collect_statistics(sessions)

    with open(output_path, 'w') as out:
        out.write("# Carl Rogers Therapy Sessions - Detailed Annotated Transcripts\n\n")
        out.write("_Cognitive and emotional processes are shown with their layer number [L#] and confidence score._\n\n")

        # Write statistics summary
        out.write(create_statistics_summary(stats))
        out.write("\n---\n\n")

        # Write sessions
        for session_idx, session in enumerate(sessions, 1):
            # Write session header
            session_title = session.get('title', session.get('session_name', f'Session {session_idx}'))
            out.write(f"## {session_title}\n\n")
            out.write(f"**Session ID:** {session.get('transcript_id', 'Unknown')}\n\n")
            out.write("---\n\n")

            # Write utterances
            for utterance in session['utterances']:
                speaker = utterance['speaker'].title()
                text = utterance['text']

                # Get active processes with details
                action_layer_details = utterance.get('action_layer_details', {})
                active_processes = format_active_processes_detailed(
                    utterance['predictions'],
                    action_layer_details
                )

                # Format the utterance
                out.write(f"**{speaker}:** {text}\n\n")

                if active_processes:
                    processes_str = ", ".join(active_processes)
                    out.write(f"_({processes_str})_\n\n")
                else:
                    out.write(f"_(No active processes detected)_\n\n")

                out.write("---\n\n")

            # Add spacing between sessions
            out.write("\n\n")

    print(f"✓ Created detailed transcript: {output_path}")
    print(f"  Total sessions: {len(sessions)}")
    total_utterances = sum(len(s['utterances']) for s in sessions)
    print(f"  Total utterances: {total_utterances}")
    print(f"  Unique processes detected: {len(stats['process_counts'])}")
    print(f"  Layers used: {min(stats['layer_usage'].keys())} - {max(stats['layer_usage'].keys())}")

if __name__ == "__main__":
    json_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/annotated_transcripts.json"
    output_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/detailed_transcript.md"

    create_detailed_transcript(json_path, output_path)