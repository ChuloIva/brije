import json

def format_active_processes(predictions):
    """Extract and format active cognitive/emotional processes."""
    active = []
    for process_name, process_data in predictions.items():
        if process_data.get('is_active', False):
            # Format the process name to be more readable
            readable_name = process_name.replace('_', ' ').title()
            active.append(readable_name)
    return active

def create_readable_transcript(json_path, output_path):
    """Convert annotated JSON to readable markdown."""

    with open(json_path, 'r') as f:
        sessions = json.load(f)

    with open(output_path, 'w') as out:
        out.write("# Carl Rogers Therapy Sessions - Annotated Transcripts\n\n")
        out.write("_Cognitive and emotional processes are shown in parentheses after each utterance._\n\n")
        out.write("---\n\n")

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

                # Get active processes
                active_processes = format_active_processes(utterance['predictions'])

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

    print(f"✓ Created readable transcript: {output_path}")
    print(f"  Total sessions: {len(sessions)}")
    total_utterances = sum(len(s['utterances']) for s in sessions)
    print(f"  Total utterances: {total_utterances}")

if __name__ == "__main__":
    json_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/annotated_transcripts.json"
    output_path = "/Users/ivanculo/Desktop/Projects/ment_helth/brije/output/carl_rogers_analysis/readable_transcript.md"

    create_readable_transcript(json_path, output_path)