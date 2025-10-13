"""
Generate Human-Readable Conversation Reports
==============================================
Creates formatted HTML and Markdown reports of all therapeutic conversations
with cognitive actions annotated for each turn.
"""

import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "output/therapeutic_conversations/checkpoint_30.json"
OUTPUT_DIR = "output/therapeutic_conversations/conversation_reports"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING CONVERSATION REPORTS")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📥 Loading data...")
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

sessions = data['sessions']
num_sessions = len(sessions)
print(f"   ✅ Loaded {num_sessions} sessions")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_active_actions(predictions):
    """Extract active cognitive actions with confidence scores"""
    active = []
    for pred in predictions:
        if pred.get('is_active', False):
            active.append({
                'action': pred['action'],
                'confidence': pred.get('max_confidence', 0),
                'layers': pred.get('layers', [])
            })
    # Sort by confidence
    active.sort(key=lambda x: x['confidence'], reverse=True)
    return active

def role_to_speaker(role):
    """Convert role to readable speaker name"""
    if 'therapist' in role.lower():
        return "Therapist"
    elif 'conclusion' in role.lower():
        return "Client (Concluding)"
    elif 'client' in role.lower():
        return "Client"
    else:
        return role.capitalize()

def get_action_color(action):
    """Assign colors to different action categories"""
    emotion_actions = ['emotion_perception', 'emotion_responding', 'emotion_valuing',
                       'emotion_characterizing', 'emotion_receiving', 'emotion_understanding',
                       'emotion_management']
    cognitive_actions = ['abstracting', 'concretizing', 'divergent_thinking', 'convergent_thinking',
                         'pattern_recognition', 'hypothesis_generation', 'evaluating']
    metacognitive_actions = ['questioning', 'self_questioning', 'cognition_awareness', 'noticing']
    action_oriented = ['applying', 'situation_selection', 'response_modulation', 'reframing']

    if action in emotion_actions:
        return '#E8B4F5'  # Purple
    elif action in cognitive_actions:
        return '#AED9E0'  # Blue
    elif action in metacognitive_actions:
        return '#FAE29C'  # Yellow
    elif action in action_oriented:
        return '#B8E0AC'  # Green
    else:
        return '#E0E0E0'  # Gray

# ============================================================================
# GENERATE HTML REPORTS (INDIVIDUAL)
# ============================================================================

print("\n📝 Generating individual HTML reports...")

html_template_header = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Session {session_id} - Therapeutic Conversation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .metadata {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metadata-item {{
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 3px solid #667eea;
            border-radius: 4px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #555;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metadata-value {{
            color: #333;
            margin-top: 5px;
        }}
        .conversation {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .turn {{
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .turn:hover {{
            transform: translateX(5px);
        }}
        .turn.therapist {{
            background: linear-gradient(to right, #e3f2fd, #f5f5f5);
            border-left: 5px solid #2196F3;
        }}
        .turn.client {{
            background: linear-gradient(to right, #f3e5f5, #f5f5f5);
            border-left: 5px solid #9C27B0;
        }}
        .turn.client-conclusion {{
            background: linear-gradient(to right, #e8f5e9, #f5f5f5);
            border-left: 5px solid #4CAF50;
        }}
        .speaker {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #333;
        }}
        .turn-number {{
            display: inline-block;
            background-color: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        .content {{
            margin: 15px 0;
            color: #444;
            font-size: 1.05em;
            line-height: 1.8;
        }}
        .cognitive-actions {{
            margin-top: 15px;
        }}
        .actions-label {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .action-tag {{
            display: inline-block;
            padding: 6px 12px;
            margin: 4px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .action-tag:hover {{
            transform: scale(1.05);
        }}
        .confidence {{
            opacity: 0.7;
            font-size: 0.85em;
            margin-left: 4px;
        }}
        .legend {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            padding: 8px;
            border-radius: 4px;
            background-color: #f9f9f9;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            margin-right: 10px;
        }}
        .no-actions {{
            color: #999;
            font-style: italic;
        }}
        .stats-summary {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Session {session_id}: Therapeutic Conversation</h1>
        <p>Primary Issue: <strong>{primary_issue}</strong></p>
    </div>
"""

html_template_footer = """
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #999; font-size: 0.9em;">
        <p>Generated by therapeutic_meta_analysis.py on {timestamp}</p>
    </div>
</body>
</html>
"""

for session in sessions:
    session_id = session['metadata']['session_id']
    meta = session['metadata']
    conversation = session['conversation']

    # Count cognitive actions
    total_actions = 0
    unique_actions = set()

    html_content = html_template_header.format(
        session_id=session_id,
        primary_issue=meta['primary_issue'].upper()
    )

    # Metadata section
    html_content += """
    <div class="metadata">
        <h2>Session Metadata</h2>
        <div class="metadata-grid">
    """

    metadata_items = [
        ("Primary Issue", meta['primary_issue'].capitalize()),
        ("Primary Symptoms", ", ".join(meta['primary_symptoms'])),
        ("Cognitive Distortion", meta['cognitive_distortion']),
        ("Emotional Presentation", meta['emotional_presentation']),
        ("Symptom Duration", meta['symptom_duration']),
        ("Trigger Context", meta['trigger_context']),
        ("Support Level", meta['support_level']),
        ("Therapy Goal", meta['therapy_goal']),
    ]

    for label, value in metadata_items:
        html_content += f"""
            <div class="metadata-item">
                <div class="metadata-label">{label}</div>
                <div class="metadata-value">{value}</div>
            </div>
        """

    html_content += """
        </div>
    </div>
    """

    # Legend
    html_content += """
    <div class="legend">
        <h3>Cognitive Action Categories</h3>
        <div class="legend-grid">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #E8B4F5;"></div>
                <span>Emotion-focused</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #AED9E0;"></div>
                <span>Cognitive</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FAE29C;"></div>
                <span>Metacognitive</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #B8E0AC;"></div>
                <span>Action-oriented</span>
            </div>
        </div>
    </div>
    """

    # Conversation section
    html_content += """
    <div class="conversation">
        <h2 style="color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Conversation</h2>
    """

    for turn_idx, turn in enumerate(conversation):
        # Determine role
        if turn_idx == 0:
            role = "client"
            speaker = "Client (Opening)"
        elif turn.get('ai_name') == 'Therapist':
            role = "therapist"
            speaker = "Therapist"
        elif turn.get('ai_name') == 'Client (Conclusion)':
            role = "client-conclusion"
            speaker = "Client (Concluding)"
        elif turn.get('ai_name') == 'Client':
            role = "client"
            speaker = "Client"
        else:
            role = "unknown"
            speaker = "Unknown"

        content = turn.get('content', '').strip()
        predictions = turn.get('predictions', [])
        active_actions = extract_active_actions(predictions)

        # Update stats
        total_actions += len(active_actions)
        for action in active_actions:
            unique_actions.add(action['action'])

        html_content += f"""
        <div class="turn {role}">
            <div class="speaker">
                <span class="turn-number">Turn {turn_idx + 1}</span>
                {speaker}
            </div>
            <div class="content">{content}</div>
        """

        if active_actions:
            html_content += """
            <div class="cognitive-actions">
                <div class="actions-label">Cognitive Actions:</div>
            """
            for action_data in active_actions:
                action = action_data['action']
                confidence = action_data['confidence']
                color = get_action_color(action)
                html_content += f"""
                <span class="action-tag" style="background-color: {color};">
                    {action.replace('_', ' ')}
                    <span class="confidence">({confidence:.2f})</span>
                </span>
                """
            html_content += """
            </div>
            """
        else:
            html_content += """
            <div class="cognitive-actions">
                <div class="no-actions">No active cognitive actions detected</div>
            </div>
            """

        html_content += """
        </div>
        """

    html_content += """
    </div>
    """

    # Add stats summary at the end
    html_content += f"""
    <div class="stats-summary">
        <div class="stat-box">
            <div class="stat-number">{len(conversation)}</div>
            <div class="stat-label">Total Turns</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_actions}</div>
            <div class="stat-label">Total Cognitive Actions</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{len(unique_actions)}</div>
            <div class="stat-label">Unique Actions</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{total_actions / len(conversation):.1f}</div>
            <div class="stat-label">Avg Actions/Turn</div>
        </div>
    </div>
    """

    html_content += html_template_footer.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Save HTML file
    output_file = f"{OUTPUT_DIR}/session_{session_id:03d}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"   ✅ Session {session_id:03d} saved")

# ============================================================================
# GENERATE MASTER INDEX
# ============================================================================

print("\n📑 Generating master index...")

index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Therapeutic Conversations - Master Index</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .summary {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-number {
            font-size: 3em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }
        .sessions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .session-card {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        .session-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .session-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        .session-issue {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .issue-depression {
            background-color: #E8B4F5;
            color: #4A148C;
        }
        .issue-anxiety {
            background-color: #FFCDD2;
            color: #B71C1C;
        }
        .issue-both {
            background-color: #FFE082;
            color: #F57F17;
        }
        .session-meta {
            color: #666;
            font-size: 0.9em;
            line-height: 1.6;
        }
        .view-link {
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background-color: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        .view-link:hover {
            background-color: #764ba2;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Therapeutic Conversations</h1>
        <p>Meta-Analysis of 30 Simulated Therapy Sessions</p>
    </div>

    <div class="summary">
        <h2 style="color: #667eea;">Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">30</div>
                <div class="stat-label">Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">240</div>
                <div class="stat-label">Total Turns</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">11</div>
                <div class="stat-label">Depression</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">9</div>
                <div class="stat-label">Anxiety</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">10</div>
                <div class="stat-label">Both</div>
            </div>
        </div>
    </div>

    <h2 style="color: #333; margin-bottom: 20px;">All Sessions</h2>
    <div class="sessions-grid">
"""

for session in sessions:
    sid = session['metadata']['session_id']
    issue = session['metadata']['primary_issue']
    distortion = session['metadata']['cognitive_distortion']
    goal = session['metadata']['therapy_goal']
    turns = len(session['conversation'])

    issue_class = f"issue-{issue}"

    index_html += f"""
        <div class="session-card">
            <div class="session-title">Session {sid:03d}</div>
            <div class="session-issue {issue_class}">{issue.upper()}</div>
            <div class="session-meta">
                <strong>Turns:</strong> {turns}<br>
                <strong>Distortion:</strong> {distortion}<br>
                <strong>Goal:</strong> {goal}
            </div>
            <a href="session_{sid:03d}.html" class="view-link">View Session →</a>
        </div>
    """

index_html += """
    </div>
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #999;">
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
</body>
</html>
"""

with open(f"{OUTPUT_DIR}/index.html", 'w', encoding='utf-8') as f:
    f.write(index_html)

print(f"   ✅ Master index saved")

# ============================================================================
# GENERATE MARKDOWN SUMMARY
# ============================================================================

print("\n📄 Generating markdown summary...")

md_content = f"""# Therapeutic Conversations Analysis

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

- **Total Sessions:** {num_sessions}
- **Total Conversation Turns:** {sum(len(s['conversation']) for s in sessions)}
- **Depression Cases:** {sum(1 for s in sessions if s['metadata']['primary_issue'] == 'depression')}
- **Anxiety Cases:** {sum(1 for s in sessions if s['metadata']['primary_issue'] == 'anxiety')}
- **Both:** {sum(1 for s in sessions if s['metadata']['primary_issue'] == 'both')}

---

## Sessions

"""

for session in sessions:
    meta = session['metadata']
    conversation = session['conversation']

    md_content += f"""
### Session {meta['session_id']:03d} - {meta['primary_issue'].upper()}

**Metadata:**
- **Primary Symptoms:** {', '.join(meta['primary_symptoms'])}
- **Cognitive Distortion:** {meta['cognitive_distortion']}
- **Emotional Presentation:** {meta['emotional_presentation']}
- **Therapy Goal:** {meta['therapy_goal']}
- **Turns:** {len(conversation)}

**Conversation:**

"""

    for turn_idx, turn in enumerate(conversation):
        if turn_idx == 0:
            speaker = "**Client (Opening):**"
        elif turn.get('ai_name') == 'Therapist':
            speaker = "**Therapist:**"
        elif turn.get('ai_name') == 'Client (Conclusion)':
            speaker = "**Client (Concluding):**"
        elif turn.get('ai_name') == 'Client':
            speaker = "**Client:**"
        else:
            speaker = "**Unknown:**"

        content = turn.get('content', '').strip()
        predictions = turn.get('predictions', [])
        active_actions = extract_active_actions(predictions)

        md_content += f"{turn_idx + 1}. {speaker}\n"
        md_content += f"   {content}\n"

        if active_actions:
            actions_str = ", ".join([f"`{a['action']}` ({a['confidence']:.2f})" for a in active_actions])
            md_content += f"   \n   *Cognitive Actions:* {actions_str}\n"

        md_content += "\n"

    md_content += "---\n\n"

with open(f"{OUTPUT_DIR}/conversations_full.md", 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"   ✅ Markdown summary saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✨ CONVERSATION REPORTS COMPLETE!")
print("="*80)
print(f"\n📁 All reports saved to: {OUTPUT_DIR}/")
print(f"\n📊 Generated:")
print(f"   • {num_sessions} individual HTML session reports")
print(f"   • 1 master index (index.html)")
print(f"   • 1 complete markdown file (conversations_full.md)")
print(f"\n💡 Open {OUTPUT_DIR}/index.html in your browser to navigate all sessions!")
print("\n" + "="*80)
