import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple

def parse_transcripts(input_file: str, output_dir: str = "output/carl_rogers_analysis"):
    """
    Parse Carl Rogers therapy transcripts into structured datasets.

    The transcripts use different coding schemes:
    - Gloria session: T = Therapist, C = Client
    - Sylvia sessions: C = Carl Rogers (Therapist), S = Sylvia (Client)
    - Kathy session: Pattern needs to be determined
    - Dione sessions: Pattern needs to be determined
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Define session markers and their configurations
    sessions_config = [
        {
            'name': 'Gloria',
            'start_markers': ['GLORIA', 'THERAPIST: DR. CARL ROGERS'],
            'end_markers': ['Sylvia 4th Interview', 'Rogers\' Transcripts, Volume 12, Sylvia'],
            'therapist_code': 'T',
            'client_code': 'C',
            'client_name': 'Gloria',
            'year': 1965
        },
        {
            'name': 'Sylvia_4th',
            'start_markers': ['Carl Rogers 4th Interview with Sylvia'],
            'end_markers': ['"Struggle for Acceptance"', 'Session 5 with Sylvia'],
            'therapist_code': 'C',  # Carl Rogers
            'client_code': 'S',      # Sylvia
            'client_name': 'Sylvia',
            'year': 1975
        },
        {
            'name': 'Sylvia_5th',
            'start_markers': ['"Struggle for Acceptance"', 'Session 5 with Sylvia'],
            'end_markers': ['END OF SESSION', 'Kathy Interview by Carl Rogers'],
            'therapist_code': 'C',
            'client_code': 'S',
            'client_name': 'Sylvia',
            'year': 1975
        },
        {
            'name': 'Kathy',
            'start_markers': ['Kathy Interview by Carl Rogers', 'Carl Rogers\' Interview with Kathy'],
            'end_markers': ['Dione 1st Filmed Interview', 'Carl Rogers 1st Interview with Dione'],
            'therapist_code': 'T',
            'client_code': 'C',
            'client_name': 'Kathy',
            'year': 1975
        },
        {
            'name': 'Dione_1st',
            'start_markers': ['Carl Rogers\' First Session with Dione'],
            'end_markers': ['Carl Rogers Counsels an Individual on Anger'],
            'therapist_code': 'T',
            'client_code': 'C',
            'client_name': 'Dione',
            'year': 1977
        },
        {
            'name': 'Dione_2nd',
            'start_markers': ['Carl Rogers Counsels an Individual on Anger and Hurt'],
            'end_markers': ['FINAL COMMENT BY ROGERS', 'END_OF_FILE'],
            'therapist_code': 'T',
            'client_code': 'C',
            'client_name': 'Dione',
            'year': 1977
        }
    ]

    all_sessions_data = []

    for session_config in sessions_config:
        print(f"\nProcessing {session_config['name']} session...")
        session_data = extract_session(lines, session_config)

        if session_data:
            all_sessions_data.append(session_data)

            # Save individual session
            save_session_data(session_data, output_path)
            print(f"  Found {len(session_data['utterances'])} utterances")
        else:
            print(f"  No utterances found")

    # Save combined dataset
    save_combined_dataset(all_sessions_data, output_path)

    return all_sessions_data

def extract_session(lines: List[str], config: Dict) -> Dict:
    """Extract a single therapy session based on configuration."""

    # Find session boundaries
    start_idx = None
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if start_idx is None:
            for marker in config['start_markers']:
                if marker in line:
                    start_idx = i
                    break
        elif start_idx is not None:
            for marker in config['end_markers']:
                if marker in line:
                    end_idx = i
                    break
            if end_idx < len(lines):
                break

    if start_idx is None:
        return None

    # Extract utterances
    utterances = []
    session_lines = lines[start_idx:end_idx]

    # Pattern to match speaker codes like T1, C2, S3, etc.
    # More flexible pattern that handles various formats including missing colons
    # Matches: T1: text, T1 text, T1(continued): text, etc.
    pattern = re.compile(r'^([' + config['therapist_code'] + config['client_code'] + r'])\s*(\d+)(\s*\(continued\))?:?\s+(.*)', re.IGNORECASE)

    # Pattern to match commentary that should be skipped
    commentary_pattern = re.compile(r'^\[.*\]$|^[CS] Commentary|^Rogers Commentary|^Rogers\' Transcripts|^Carl Commentary|^Sylvia Commentary|^Carl:|^Sylvia:|^Kathy:|^Dione:|^\[Source:')

    current_utterance = None
    in_commentary = False

    for line in session_lines:
        original_line = line
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for commentary blocks
        if line.startswith('['):
            in_commentary = True
        if in_commentary and line.endswith(']'):
            in_commentary = False
            continue
        if in_commentary:
            continue

        # Skip known commentary and header patterns
        if commentary_pattern.search(line):
            continue

        # Skip page markers and other metadata
        if 'Rogers\' Transcripts' in line or 'page' in line.lower():
            continue

        # Try to match speaker pattern at the start of line
        match = pattern.match(line)

        if match:
            # Save previous utterance if exists
            if current_utterance:
                utterances.append(current_utterance)

            speaker_code = match.group(1)
            turn_number = int(match.group(2))
            text = match.group(4).strip()

            # Determine speaker role
            if speaker_code == config['therapist_code']:
                speaker = 'therapist'
                speaker_name = 'Carl Rogers'
            else:
                speaker = 'client'
                speaker_name = config['client_name']

            current_utterance = {
                'session': config['name'],
                'turn_number': turn_number,
                'speaker': speaker,
                'speaker_name': speaker_name,
                'speaker_code': speaker_code,
                'text': text,
                'year': config['year']
            }
        elif current_utterance:
            # Continuation of previous utterance
            # But we need to be careful - if the line contains a NEW speaker code at the START,
            # we should NOT append it. Check if line starts with a speaker pattern
            if not pattern.match(line):
                # Also skip standalone stage directions/gestures
                if not (line.startswith('(') and line.endswith(')')):
                    # Avoid appending lines that look like metadata
                    if not ('Transcripts' in line or 'Interview' in line or line.startswith('Rogers:')):
                        current_utterance['text'] += ' ' + line

    # Don't forget the last utterance
    if current_utterance:
        utterances.append(current_utterance)

    # Clean up utterances
    for utterance in utterances:
        # Remove inline annotations like (T: Mhm) or (C: Yes)
        utterance['text'] = re.sub(r'\([TCScR]:\s*[^)]+\)', '', utterance['text'])
        # Remove gesture/action annotations
        utterance['text'] = re.sub(r'\([^)]*\)', '', utterance['text'])

        # Remove metadata and commentary that may have leaked in
        # Remove anything after "This transcript is available"
        if 'This transcript is available' in utterance['text']:
            utterance['text'] = utterance['text'].split('This transcript is available')[0]
        # Remove anything after "Rogers' Transcripts, Volume"
        if 'Rogers\' Transcripts' in utterance['text']:
            utterance['text'] = utterance['text'].split('Rogers\' Transcripts')[0]
        # Remove Carl/Sylvia commentary that leaked in
        for commentary_marker in ['much place as bad feelings', 'Carl Commentary', 'Sylvia Commentary', 'C Commentary', 'S Commentary']:
            if commentary_marker in utterance['text']:
                utterance['text'] = utterance['text'].split(commentary_marker)[0]

        # Clean up extra whitespace
        utterance['text'] = ' '.join(utterance['text'].split())
        utterance['text'] = utterance['text'].strip()

    # Filter out empty utterances
    utterances = [u for u in utterances if u['text']]

    return {
        'session_name': config['name'],
        'client_name': config['client_name'],
        'year': config['year'],
        'utterances': utterances
    }

def save_session_data(session_data: Dict, output_path: Path):
    """Save session data in multiple formats."""

    session_name = session_data['session_name']

    # Save as JSON
    json_file = output_path / f"{session_name}_session.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    # Save as CSV
    csv_file = output_path / f"{session_name}_session.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if session_data['utterances']:
            fieldnames = ['session', 'turn_number', 'speaker', 'speaker_name', 'text', 'year']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for utterance in session_data['utterances']:
                writer.writerow({
                    'session': utterance['session'],
                    'turn_number': utterance['turn_number'],
                    'speaker': utterance['speaker'],
                    'speaker_name': utterance['speaker_name'],
                    'text': utterance['text'],
                    'year': utterance['year']
                })

def save_combined_dataset(all_sessions: List[Dict], output_path: Path):
    """Save combined dataset from all sessions."""

    # Combine all utterances
    all_utterances = []
    for session in all_sessions:
        all_utterances.extend(session['utterances'])

    # Save combined JSON
    combined_data = {
        'total_sessions': len(all_sessions),
        'total_utterances': len(all_utterances),
        'sessions': all_sessions
    }

    json_file = output_path / "all_sessions_combined.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    # Save combined CSV
    csv_file = output_path / "all_sessions_combined.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['session', 'turn_number', 'speaker', 'speaker_name', 'text', 'year']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in all_utterances:
            writer.writerow({
                'session': utterance['session'],
                'turn_number': utterance['turn_number'],
                'speaker': utterance['speaker'],
                'speaker_name': utterance['speaker_name'],
                'text': utterance['text'],
                'year': utterance['year']
            })

    # Print summary
    print("\n" + "="*60)
    print("PARSING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total sessions processed: {len(all_sessions)}")
    print(f"Total utterances extracted: {len(all_utterances)}")
    print("\nBreakdown by session:")
    for session in all_sessions:
        print(f"  {session['session_name']:15s} ({session['year']}): {len(session['utterances']):3d} utterances")

    therapist_count = sum(1 for u in all_utterances if u['speaker'] == 'therapist')
    client_count = sum(1 for u in all_utterances if u['speaker'] == 'client')
    print(f"\nSpeaker breakdown:")
    print(f"  Therapist (Carl Rogers): {therapist_count} utterances")
    print(f"  Clients: {client_count} utterances")

    print(f"\nFiles saved to: {output_path}")
    print("  - Individual session JSON files")
    print("  - Individual session CSV files")
    print("  - all_sessions_combined.json")
    print("  - all_sessions_combined.csv")

if __name__ == "__main__":
    input_file = "output/carl_rogers_analysis/raw_transcripts.txt"
    parse_transcripts(input_file)
