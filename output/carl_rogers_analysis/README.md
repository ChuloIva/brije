# Carl Rogers Therapy Sessions Dataset

This dataset contains parsed transcripts from six therapy sessions conducted by Carl Rogers, the founder of person-centered therapy. The transcripts have been structured into a clean, machine-readable format.

## Dataset Overview

- **Total Sessions**: 6
- **Total Utterances**: 2,118
- **Years Covered**: 1965-1977
- **Clients**: Gloria, Sylvia (2 sessions), Kathy, Dione (2 sessions)

## Session Breakdown

| Session | Year | Client | Utterances | Description |
|---------|------|--------|------------|-------------|
| Gloria | 1965 | Gloria | 860 | Famous filmed interview demonstrating Rogers' approach |
| Sylvia_4th | 1975 | Sylvia | 106 | Fourth interview with Sylvia |
| Sylvia_5th | 1975 | Sylvia | 124 | Fifth interview with Sylvia ("Struggle for Acceptance") |
| Kathy | 1975 | Kathy | 605 | Interview exploring loneliness and relationships |
| Dione_1st | 1977 | Dione | 124 | First session with Dione |
| Dione_2nd | 1977 | Dione | 299 | Second session on anger and hurt |

## Speaker Breakdown

- **Therapist (Carl Rogers)**: 977 utterances
- **Clients (All)**: 1,141 utterances

## File Formats

### Individual Session Files
Each session is available in both JSON and CSV formats:
- `{session_name}_session.json` - Structured JSON with metadata
- `{session_name}_session.csv` - Flat CSV format for easy analysis

### Combined Dataset
- `all_sessions_combined.json` - All sessions in a single JSON file with metadata
- `all_sessions_combined.csv` - All utterances in a single CSV file

## Data Structure

### CSV Format
Each row contains:
- `session`: Session identifier (e.g., "Gloria", "Sylvia_4th")
- `turn_number`: Sequential number of the utterance within the session
- `speaker`: Role ("therapist" or "client")
- `speaker_name`: Name of the speaker ("Carl Rogers" or client name)
- `text`: The cleaned utterance text
- `year`: Year of the session

### JSON Format
```json
{
  "session_name": "Gloria",
  "client_name": "Gloria",
  "year": 1965,
  "utterances": [
    {
      "session": "Gloria",
      "turn_number": 1,
      "speaker": "therapist",
      "speaker_name": "Carl Rogers",
      "speaker_code": "T",
      "text": "Good morning. I'm Dr. Rogers, you must be Gloria.",
      "year": 1965
    }
  ]
}
```

## Coding Schemes

The original transcripts used different speaker coding schemes:
- **Gloria & Kathy sessions**: T = Therapist (Rogers), C = Client
- **Sylvia sessions**: C = Carl Rogers (Therapist), S = Sylvia (Client)
- **Dione sessions**: T = Therapist (Rogers), C = Client (Dione)

All data has been normalized to use consistent `speaker` and `speaker_name` fields.

## Data Cleaning

The following cleaning steps were applied:
1. Removed inline annotations (e.g., "(T: Mhm)", "(C: Yes)")
2. Removed gesture/action descriptions (e.g., "(smiles)", "(pause)")
3. Removed commentary blocks from Rogers and clients
4. Removed page markers and metadata
5. Cleaned up extra whitespace
6. Filtered out empty utterances

## Source

These transcripts were originally published in:
> Brodley, B. T., & Lietaer, G. (Eds.). (Year). Transcripts of Carl Rogers' Therapy Sessions, Volume 12.

Available for purposes of research, study and teaching. Not to be sold.

## Usage Examples

### Python (pandas)
```python
import pandas as pd

# Load all sessions
df = pd.read_csv('all_sessions_combined.csv')

# Filter therapist utterances
therapist = df[df['speaker'] == 'therapist']

# Get Gloria session only
gloria = df[df['session'] == 'Gloria']
```

### Python (json)
```python
import json

with open('all_sessions_combined.json', 'r') as f:
    data = json.load(f)

print(f"Total sessions: {data['total_sessions']}")
print(f"Total utterances: {data['total_utterances']}")

# Access individual session
gloria = data['sessions'][0]
```

## Notes

- Some utterances may contain residual formatting artifacts despite cleaning efforts
- Turn numbers restart for each session
- The Sylvia 4th and 5th sessions were conducted on consecutive days
- Dione is presented as a male client dealing with illness and personal challenges
- Original line breaks within utterances have been converted to spaces

## Generated

This dataset was generated using the `parse_carl_rogers_transcripts.py` script on 2025-10-12.
