"""
Parse Carl Rogers therapy transcripts from raw text file and save as structured JSON.

This script extracts therapy sessions from the raw transcripts text file,
identifying therapist (Rogers) and client utterances.

Usage:
    python parse_carl_rogers_pdf.py

Output:
    - output/carl_rogers_analysis/parsed_transcripts.json
"""

import re
import json
from pathlib import Path


def clean_text(text):
    """Remove stage directions, pauses, and clean text"""
    # Remove content in parentheses (stage directions, pauses)
    text = re.sub(r'\([^)]*\)', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_transcripts(text):
    """
    Parse Carl Rogers transcripts with various formats:
    - T1, T2, C1, C2 (T=Therapist/Rogers, C=Client/Gloria)
    - S1, S2, C1, C2 (S=Client/Sylvia, C=Counselor/Rogers)
    - Other similar patterns
    """
    lines = text.split('\n')

    transcripts = []
    current_transcript = None
    current_utterance_buffer = []
    current_speaker = None
    current_utterance_num = None

    # Pattern for utterance markers: T1, T2, C3, S1, etc.
    # More flexible to handle different formats
    utterance_pattern = r'^([A-Z])(\d+)\s*[:.]\s*(.*)'

    # Patterns for new session/interview headers
    session_patterns = [
        r'(?:Session|Interview|Transcript|Case)\s+(?:of\s+)?([^\n]+)',
        r'Rogers.*(?:Interview|Session|Transcript)',
        r'Gloria.*(?:Interview|Session|Transcript)',
        r'Sylvia.*(?:Interview|Session|Transcript)',
        r'^[A-Z\s]+INTERVIEW',
    ]

    # Track which letters represent which speaker in current transcript
    # Will be inferred from context
    therapist_codes = set()
    client_codes = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for new session header
        is_new_session = False
        for pattern in session_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Save previous utterance if exists
                if current_transcript and current_utterance_buffer and current_speaker:
                    full_text = ' '.join(current_utterance_buffer)
                    full_text = clean_text(full_text)
                    if len(full_text) > 10:  # Only save substantial utterances
                        current_transcript['utterances'].append({
                            'utterance_id': current_utterance_num,
                            'speaker': current_speaker,
                            'text': full_text
                        })

                # Save previous transcript if exists
                if current_transcript and len(current_transcript['utterances']) > 0:
                    transcripts.append(current_transcript)

                # Start new transcript
                current_transcript = {
                    'transcript_id': len(transcripts) + 1,
                    'title': line,
                    'utterances': []
                }
                current_utterance_buffer = []
                current_speaker = None
                current_utterance_num = None
                # Reset speaker code mappings
                therapist_codes = set()
                client_codes = set()
                is_new_session = True
                break

        if is_new_session:
            continue

        # Check for utterance marker (T1, C2, S1, etc.)
        match = re.match(utterance_pattern, line)

        if match:
            # Save previous utterance if exists
            if current_transcript and current_utterance_buffer and current_speaker:
                full_text = ' '.join(current_utterance_buffer)
                full_text = clean_text(full_text)
                if len(full_text) > 10:  # Only save substantial utterances
                    current_transcript['utterances'].append({
                        'utterance_id': current_utterance_num,
                        'speaker': current_speaker,
                        'text': full_text
                    })

            # Start new utterance
            speaker_code = match.group(1)  # Letter (T, C, S, etc.)
            utterance_num = int(match.group(2))  # Number
            utterance_text = match.group(3).strip()  # Rest of the line

            # Determine speaker based on code
            # Common patterns:
            # - T = Therapist (Rogers)
            # - C = Client (Gloria, Jan, etc.) OR Counselor (Rogers)
            # - S = Client (Sylvia)
            # - R = Rogers

            # Heuristic: T and R are always therapist
            # If we see both C and S, C is likely therapist (Counselor)
            # Otherwise C is client

            if speaker_code in ['T', 'R']:
                current_speaker = 'therapist'
                therapist_codes.add(speaker_code)
            elif speaker_code == 'S':
                current_speaker = 'client'
                client_codes.add(speaker_code)
            elif speaker_code == 'C':
                # Context-dependent
                if 'S' in client_codes or 'T' in therapist_codes or 'R' in therapist_codes:
                    # If we've seen S as client, or T/R as therapist, then C is therapist
                    current_speaker = 'therapist'
                    therapist_codes.add(speaker_code)
                else:
                    # Default: C is client (most common in Gloria interview)
                    current_speaker = 'client'
                    client_codes.add(speaker_code)
            else:
                # Other letters - try to infer
                # First speaker is usually therapist
                if not therapist_codes and not client_codes:
                    current_speaker = 'therapist'
                    therapist_codes.add(speaker_code)
                elif speaker_code in therapist_codes:
                    current_speaker = 'therapist'
                elif speaker_code in client_codes:
                    current_speaker = 'client'
                else:
                    # Alternate assumption
                    current_speaker = 'client'
                    client_codes.add(speaker_code)

            current_utterance_num = utterance_num
            current_utterance_buffer = [utterance_text] if utterance_text else []

        else:
            # Continuation of current utterance
            if current_utterance_buffer is not None:
                current_utterance_buffer.append(line)

    # Save last utterance and transcript
    if current_transcript and current_utterance_buffer and current_speaker:
        full_text = ' '.join(current_utterance_buffer)
        full_text = clean_text(full_text)
        if len(full_text) > 10:
            current_transcript['utterances'].append({
                'utterance_id': current_utterance_num,
                'speaker': current_speaker,
                'text': full_text
            })

    if current_transcript and len(current_transcript['utterances']) > 0:
        transcripts.append(current_transcript)

    return transcripts


def main():
    text_path = 'output/carl_rogers_analysis/raw_transcripts.txt'
    output_dir = Path('output/carl_rogers_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CARL ROGERS TRANSCRIPT PARSER")
    print("="*80)

    print(f"\n📄 Reading text from: {text_path}")

    # Read text from file
    with open(text_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    print(f"✅ Loaded {len(full_text):,} characters")

    # Parse transcripts
    print("\n🔍 Parsing transcripts...")
    transcripts = parse_transcripts(full_text)

    print(f"✅ Parsed {len(transcripts)} therapy sessions")

    # Statistics
    total_utterances = sum(len(t['utterances']) for t in transcripts)
    therapist_utterances = sum(
        sum(1 for u in t['utterances'] if u['speaker'] == 'therapist')
        for t in transcripts
    )
    client_utterances = sum(
        sum(1 for u in t['utterances'] if u['speaker'] == 'client')
        for t in transcripts
    )

    print(f"\n📊 Statistics:")
    print(f"   Total utterances: {total_utterances}")
    print(f"   Therapist (Rogers) utterances: {therapist_utterances}")
    print(f"   Client utterances: {client_utterances}")

    # Show distribution
    print(f"\n📋 Utterances per transcript:")
    for t in transcripts[:10]:
        t_count = sum(1 for u in t['utterances'] if u['speaker'] == 'therapist')
        c_count = sum(1 for u in t['utterances'] if u['speaker'] == 'client')
        title_preview = t['title'][:50] + "..." if len(t['title']) > 50 else t['title']
        print(f"   {t['transcript_id']:2d}. {title_preview:50s} (T:{t_count:3d}, C:{c_count:3d})")

    if len(transcripts) > 10:
        print(f"   ... and {len(transcripts) - 10} more transcripts")

    # Show sample utterances
    print(f"\n📝 Sample from first transcript:")
    if transcripts:
        first_transcript = transcripts[0]
        print(f"   Title: {first_transcript['title']}")
        print(f"   Utterances:")
        for utt in first_transcript['utterances'][:5]:
            speaker_label = "[THERAPIST]" if utt['speaker'] == 'therapist' else "[CLIENT]   "
            text_preview = utt['text'][:80] + "..." if len(utt['text']) > 80 else utt['text']
            print(f"      {speaker_label} {text_preview}")

    # Save to JSON
    output_file = output_dir / 'parsed_transcripts.json'
    print(f"\n💾 Saving to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(transcripts)} transcripts")

    # Also save statistics
    stats_file = output_dir / 'parsing_statistics.json'
    stats = {
        'total_transcripts': len(transcripts),
        'total_utterances': total_utterances,
        'therapist_utterances': therapist_utterances,
        'client_utterances': client_utterances,
        'transcripts': [
            {
                'transcript_id': t['transcript_id'],
                'title': t['title'],
                'total_utterances': len(t['utterances']),
                'therapist_utterances': sum(1 for u in t['utterances'] if u['speaker'] == 'therapist'),
                'client_utterances': sum(1 for u in t['utterances'] if u['speaker'] == 'client')
            }
            for t in transcripts
        ]
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"📊 Saved statistics to: {stats_file}")

    print("\n" + "="*80)
    print("PARSING COMPLETE!")
    print("="*80)
    print(f"\n✅ You can now use the notebook with the parsed transcripts")
    print(f"   File: {output_file}")


if __name__ == "__main__":
    main()
