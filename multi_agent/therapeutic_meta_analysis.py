"""
Therapeutic Conversation Meta-Analysis
=======================================
Comprehensive analysis of therapeutic sessions to identify:
- Cognitive action patterns across therapy sessions
- How clients improve over conversation turns
- Differences between depression and anxiety presentations
- Therapeutic techniques and their effects
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "output/therapeutic_conversations/checkpoint_30.json"
OUTPUT_DIR = "output/therapeutic_conversations/analysis"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("THERAPEUTIC CONVERSATION META-ANALYSIS")
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
# EXTRACT STRUCTURED DATA
# ============================================================================

print("\n🔍 Extracting structured data...")

def extract_cognitive_actions(predictions):
    """Extract active cognitive actions from predictions"""
    active_actions = []
    for pred in predictions:
        if pred.get('is_active', False):
            active_actions.append({
                'action': pred['action'],
                'count': pred.get('count', 1),
                'max_confidence': pred.get('max_confidence', 0),
                'layers': pred.get('layers', [])
            })
    return active_actions

# Build comprehensive dataframe
rows = []
for session in sessions:
    meta = session['metadata']
    conv = session['conversation']

    for turn_idx, turn in enumerate(conv):
        # Determine role (skip first turn which is opening)
        if turn_idx == 0:
            role = "client_opening"
        elif turn.get('ai_name') == 'Therapist':
            role = "therapist"
        elif turn.get('ai_name') == 'Client':
            role = "client"
        elif turn.get('ai_name') == 'Client (Conclusion)':
            role = "client_conclusion"
        else:
            role = "unknown"

        # Extract cognitive actions
        predictions = turn.get('predictions', [])
        active_actions = extract_cognitive_actions(predictions)

        # Count words
        content = turn.get('content', '')
        word_count = len(content.split())

        # Base row
        row = {
            'session_id': meta['session_id'],
            'turn_idx': turn_idx,
            'role': role,
            'primary_issue': meta['primary_issue'],
            'trigger_context': meta['trigger_context'],
            'symptom_duration': meta['symptom_duration'],
            'emotional_presentation': meta['emotional_presentation'],
            'cognitive_distortion': meta['cognitive_distortion'],
            'support_level': meta['support_level'],
            'therapy_goal': meta['therapy_goal'],
            'word_count': word_count,
            'content': content,
            'num_active_actions': len(active_actions),
            'active_actions': [a['action'] for a in active_actions],
            'action_confidences': {a['action']: a['max_confidence'] for a in active_actions}
        }

        rows.append(row)

df = pd.DataFrame(rows)
print(f"   ✅ Created dataframe with {len(df)} conversation turns")
print(f"   • Sessions: {df['session_id'].nunique()}")
print(f"   • Therapist turns: {len(df[df['role'] == 'therapist'])}")
print(f"   • Client turns: {len(df[df['role'].str.contains('client')])}")

# ============================================================================
# 1. BASIC SESSION STATISTICS
# ============================================================================

print("\n" + "="*80)
print("1️⃣  BASIC SESSION STATISTICS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1.1 Primary issue distribution
ax = axes[0, 0]
issue_counts = df.groupby('session_id')['primary_issue'].first().value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
ax.pie(issue_counts.values, labels=issue_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90)
ax.set_title('Primary Issue Distribution', fontweight='bold', fontsize=12)

# 1.2 Trigger contexts
ax = axes[0, 1]
trigger_data = df.groupby('session_id')['trigger_context'].first().value_counts().head(8)
ax.barh(range(len(trigger_data)), trigger_data.values, color='steelblue', alpha=0.7)
ax.set_yticks(range(len(trigger_data)))
ax.set_yticklabels([t[:35] + '...' if len(t) > 35 else t for t in trigger_data.index], fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Top Trigger Contexts', fontweight='bold', fontsize=12)
ax.invert_yaxis()

# 1.3 Symptom duration
ax = axes[0, 2]
duration_data = df.groupby('session_id')['symptom_duration'].first().value_counts()
ax.bar(range(len(duration_data)), duration_data.values, color='coral', alpha=0.7)
ax.set_xticks(range(len(duration_data)))
ax.set_xticklabels(duration_data.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Frequency')
ax.set_title('Symptom Duration', fontweight='bold', fontsize=12)

# 1.4 Cognitive distortions
ax = axes[1, 0]
distortion_data = df.groupby('session_id')['cognitive_distortion'].first().value_counts().head(8)
ax.barh(range(len(distortion_data)), distortion_data.values, color='orange', alpha=0.7)
ax.set_yticks(range(len(distortion_data)))
ax.set_yticklabels([d[:35] + '...' if len(d) > 35 else d for d in distortion_data.index], fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Cognitive Distortions', fontweight='bold', fontsize=12)
ax.invert_yaxis()

# 1.5 Emotional presentation
ax = axes[1, 1]
emotion_data = df.groupby('session_id')['emotional_presentation'].first().value_counts()
ax.bar(range(len(emotion_data)), emotion_data.values, color='purple', alpha=0.7)
ax.set_xticks(range(len(emotion_data)))
ax.set_xticklabels(emotion_data.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Frequency')
ax.set_title('Emotional Presentations', fontweight='bold', fontsize=12)

# 1.6 Support levels
ax = axes[1, 2]
support_data = df.groupby('session_id')['support_level'].first().value_counts()
ax.barh(range(len(support_data)), support_data.values, color='teal', alpha=0.7)
ax.set_yticks(range(len(support_data)))
ax.set_yticklabels([s[:35] + '...' if len(s) > 35 else s for s in support_data.index], fontsize=9)
ax.set_xlabel('Frequency')
ax.set_title('Support Levels', fontweight='bold', fontsize=12)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_session_statistics.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 01_session_statistics.png")
plt.close()

# ============================================================================
# 2. COGNITIVE ACTIONS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2️⃣  COGNITIVE ACTIONS ANALYSIS")
print("="*80)

# Aggregate all cognitive actions by role
therapist_actions = Counter()
client_actions = Counter()

for _, row in df.iterrows():
    role = row['role']
    actions = row['active_actions']

    for action in actions:
        if role == 'therapist':
            therapist_actions[action] += 1
        elif 'client' in role:
            client_actions[action] += 1

print(f"   • Therapist unique actions: {len(therapist_actions)}")
print(f"   • Client unique actions: {len(client_actions)}")

# Top actions comparison
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# 2.1 Therapist top actions
ax = axes[0]
top_therapist = dict(therapist_actions.most_common(20))
if top_therapist:
    actions = list(top_therapist.keys())
    counts = list(top_therapist.values())
    ax.barh(range(len(actions)), counts, color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=10)
    ax.set_xlabel('Frequency Across All Sessions', fontsize=11)
    ax.set_title('Top 20 Therapist Cognitive Actions', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No therapist action data', ha='center', va='center')
    ax.set_title('Top 20 Therapist Cognitive Actions', fontweight='bold', fontsize=14)

# 2.2 Client top actions
ax = axes[1]
top_client = dict(client_actions.most_common(20))
if top_client:
    actions = list(top_client.keys())
    counts = list(top_client.values())
    ax.barh(range(len(actions)), counts, color='coral', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=10)
    ax.set_xlabel('Frequency Across All Sessions', fontsize=11)
    ax.set_title('Top 20 Client Cognitive Actions', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No client action data', ha='center', va='center')
    ax.set_title('Top 20 Client Cognitive Actions', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_cognitive_actions_by_role.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 02_cognitive_actions_by_role.png")
plt.close()

# ============================================================================
# 3. PROGRESSION ANALYSIS: HOW CLIENTS IMPROVE
# ============================================================================

print("\n" + "="*80)
print("3️⃣  CLIENT PROGRESSION ANALYSIS")
print("="*80)

# Track client cognitive actions across conversation turns
client_df = df[df['role'].str.contains('client')].copy()

# Compute turn position (normalize to 0-1)
for session_id in client_df['session_id'].unique():
    mask = client_df['session_id'] == session_id
    session_turns = client_df[mask]
    max_turn = session_turns['turn_idx'].max()
    client_df.loc[mask, 'normalized_position'] = session_turns['turn_idx'] / max_turn

# Group by position bins
client_df['position_bin'] = pd.cut(client_df['normalized_position'],
                                     bins=[0, 0.33, 0.66, 1.0],
                                     labels=['Early', 'Middle', 'Late'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 3.1 Number of active cognitive actions over time
ax = axes[0, 0]
progression_stats = client_df.groupby('position_bin')['num_active_actions'].agg(['mean', 'std']).reset_index()
x_pos = range(len(progression_stats))
ax.bar(x_pos, progression_stats['mean'], yerr=progression_stats['std'],
       color='coral', alpha=0.7, capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(progression_stats['position_bin'])
ax.set_ylabel('Avg Number of Active Cognitive Actions')
ax.set_title('Client Cognitive Activity Across Session', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 3.2 Word count over time (engagement)
ax = axes[0, 1]
word_progression = client_df.groupby('position_bin')['word_count'].agg(['mean', 'std']).reset_index()
ax.bar(x_pos, word_progression['mean'], yerr=word_progression['std'],
       color='steelblue', alpha=0.7, capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels(word_progression['position_bin'])
ax.set_ylabel('Avg Word Count')
ax.set_title('Client Engagement (Word Count) Across Session', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 3.3 Top actions in early vs late stage
early_actions = Counter()
late_actions = Counter()

for _, row in client_df.iterrows():
    if row['position_bin'] == 'Early':
        for action in row['active_actions']:
            early_actions[action] += 1
    elif row['position_bin'] == 'Late':
        for action in row['active_actions']:
            late_actions[action] += 1

ax = axes[1, 0]
top_early = dict(early_actions.most_common(10))
if top_early:
    actions = list(top_early.keys())
    counts = list(top_early.values())
    ax.barh(range(len(actions)), counts, color='#FFB6B6', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Top Client Actions - EARLY Session', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

ax = axes[1, 1]
top_late = dict(late_actions.most_common(10))
if top_late:
    actions = list(top_late.keys())
    counts = list(top_late.values())
    ax.barh(range(len(actions)), counts, color='#6BCF7F', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Top Client Actions - LATE Session', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_client_progression.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 03_client_progression.png")
plt.close()

# ============================================================================
# 4. EMERGENCE AND DECLINE OF COGNITIVE ACTIONS
# ============================================================================

print("\n" + "="*80)
print("4️⃣  COGNITIVE ACTION EMERGENCE PATTERNS")
print("="*80)

# Calculate change in action frequency from early to late
action_changes = {}
all_actions = set(early_actions.keys()) | set(late_actions.keys())

for action in all_actions:
    early_count = early_actions.get(action, 0)
    late_count = late_actions.get(action, 0)
    total = early_count + late_count

    if total > 0:
        # Calculate percentage change
        if early_count > 0:
            pct_change = ((late_count - early_count) / early_count) * 100
        else:
            pct_change = 100 if late_count > 0 else 0

        action_changes[action] = {
            'early': early_count,
            'late': late_count,
            'change': late_count - early_count,
            'pct_change': pct_change,
            'total': total
        }

# Sort by absolute change
sorted_changes = sorted(action_changes.items(),
                        key=lambda x: abs(x[1]['change']),
                        reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# 4.1 Actions that INCREASE (emergence)
ax = axes[0]
increasing = [(action, data['change']) for action, data in sorted_changes
              if data['change'] > 0][:15]
if increasing:
    actions, changes = zip(*increasing)
    colors_inc = ['#6BCF7F' if c > 0 else '#FFB6B6' for c in changes]
    ax.barh(range(len(actions)), changes, color=colors_inc, alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=10)
    ax.set_xlabel('Change in Frequency (Early → Late)', fontsize=11)
    ax.set_title('Cognitive Actions that EMERGE (Increase)', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No emerging actions found', ha='center', va='center')

# 4.2 Actions that DECREASE (decline)
ax = axes[1]
decreasing = [(action, data['change']) for action, data in sorted_changes
              if data['change'] < 0][:15]
if decreasing:
    actions, changes = zip(*decreasing)
    colors_dec = ['#FFB6B6' for _ in changes]
    ax.barh(range(len(actions)), changes, color=colors_dec, alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=10)
    ax.set_xlabel('Change in Frequency (Early → Late)', fontsize=11)
    ax.set_title('Cognitive Actions that DECLINE (Decrease)', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No declining actions found', ha='center', va='center')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_action_emergence_decline.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 04_action_emergence_decline.png")
plt.close()

# ============================================================================
# 5. DEPRESSION VS ANXIETY COMPARISON
# ============================================================================

print("\n" + "="*80)
print("5️⃣  DEPRESSION vs ANXIETY COGNITIVE PATTERNS")
print("="*80)

# Separate by primary issue
depression_df = df[df['primary_issue'] == 'depression'].copy()
anxiety_df = df[df['primary_issue'] == 'anxiety'].copy()
both_df = df[df['primary_issue'] == 'both'].copy()

# Client turns only
dep_client = depression_df[depression_df['role'].str.contains('client')]
anx_client = anxiety_df[anxiety_df['role'].str.contains('client')]

# Aggregate actions
dep_actions = Counter()
anx_actions = Counter()

for _, row in dep_client.iterrows():
    for action in row['active_actions']:
        dep_actions[action] += 1

for _, row in anx_client.iterrows():
    for action in row['active_actions']:
        anx_actions[action] += 1

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 5.1 Depression client actions
ax = axes[0, 0]
top_dep = dict(dep_actions.most_common(15))
if top_dep:
    actions = list(top_dep.keys())
    counts = list(top_dep.values())
    ax.barh(range(len(actions)), counts, color='#7B68EE', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Depression Clients - Top Cognitive Actions', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No depression client data', ha='center', va='center')

# 5.2 Anxiety client actions
ax = axes[0, 1]
top_anx = dict(anx_actions.most_common(15))
if top_anx:
    actions = list(top_anx.keys())
    counts = list(top_anx.values())
    ax.barh(range(len(actions)), counts, color='#FF6B6B', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Anxiety Clients - Top Cognitive Actions', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No anxiety client data', ha='center', va='center')

# 5.3 Unique to depression
ax = axes[1, 0]
dep_unique = set(dep_actions.keys()) - set(anx_actions.keys())
if dep_unique:
    unique_counts = {action: dep_actions[action] for action in dep_unique}
    sorted_unique = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_unique:
        actions, counts = zip(*sorted_unique)
        ax.barh(range(len(actions)), counts, color='#7B68EE', alpha=0.6)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=9)
        ax.set_xlabel('Frequency')
        ax.set_title('Cognitive Actions UNIQUE to Depression', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No unique depression actions', ha='center', va='center')
else:
    ax.text(0.5, 0.5, 'No unique depression actions', ha='center', va='center')

# 5.4 Unique to anxiety
ax = axes[1, 1]
anx_unique = set(anx_actions.keys()) - set(dep_actions.keys())
if anx_unique:
    unique_counts = {action: anx_actions[action] for action in anx_unique}
    sorted_unique = sorted(unique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_unique:
        actions, counts = zip(*sorted_unique)
        ax.barh(range(len(actions)), counts, color='#FF6B6B', alpha=0.6)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=9)
        ax.set_xlabel('Frequency')
        ax.set_title('Cognitive Actions UNIQUE to Anxiety', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.text(0.5, 0.5, 'No unique anxiety actions', ha='center', va='center')
else:
    ax.text(0.5, 0.5, 'No unique anxiety actions', ha='center', va='center')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_depression_vs_anxiety.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 05_depression_vs_anxiety.png")
plt.close()

# ============================================================================
# 6. THERAPIST RESPONSE PATTERNS
# ============================================================================

print("\n" + "="*80)
print("6️⃣  THERAPIST RESPONSE PATTERNS")
print("="*80)

therapist_df = df[df['role'] == 'therapist'].copy()

# Normalize turn position for therapists
for session_id in therapist_df['session_id'].unique():
    mask = therapist_df['session_id'] == session_id
    session_turns = therapist_df[mask]
    max_turn = session_turns['turn_idx'].max()
    therapist_df.loc[mask, 'normalized_position'] = session_turns['turn_idx'] / max_turn

therapist_df['position_bin'] = pd.cut(therapist_df['normalized_position'],
                                       bins=[0, 0.33, 0.66, 1.0],
                                       labels=['Early', 'Middle', 'Late'])

# Actions by phase
early_therapist = Counter()
late_therapist = Counter()

for _, row in therapist_df.iterrows():
    if row['position_bin'] == 'Early':
        for action in row['active_actions']:
            early_therapist[action] += 1
    elif row['position_bin'] == 'Late':
        for action in row['active_actions']:
            late_therapist[action] += 1

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 6.1 Early therapist actions
ax = axes[0, 0]
top_early_t = dict(early_therapist.most_common(12))
if top_early_t:
    actions = list(top_early_t.keys())
    counts = list(top_early_t.values())
    ax.barh(range(len(actions)), counts, color='#5DADE2', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Therapist Actions - EARLY Session', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

# 6.2 Late therapist actions
ax = axes[0, 1]
top_late_t = dict(late_therapist.most_common(12))
if top_late_t:
    actions = list(top_late_t.keys())
    counts = list(top_late_t.values())
    ax.barh(range(len(actions)), counts, color='#58D68D', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Therapist Actions - LATE Session', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

# 6.3 Therapist adaptation: response to depression
ax = axes[1, 0]
dep_therapist = df[(df['primary_issue'] == 'depression') & (df['role'] == 'therapist')]
dep_t_actions = Counter()
for _, row in dep_therapist.iterrows():
    for action in row['active_actions']:
        dep_t_actions[action] += 1
top_dep_t = dict(dep_t_actions.most_common(12))
if top_dep_t:
    actions = list(top_dep_t.keys())
    counts = list(top_dep_t.values())
    ax.barh(range(len(actions)), counts, color='#AF7AC5', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Therapist Responding to DEPRESSION', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

# 6.4 Therapist adaptation: response to anxiety
ax = axes[1, 1]
anx_therapist = df[(df['primary_issue'] == 'anxiety') & (df['role'] == 'therapist')]
anx_t_actions = Counter()
for _, row in anx_therapist.iterrows():
    for action in row['active_actions']:
        anx_t_actions[action] += 1
top_anx_t = dict(anx_t_actions.most_common(12))
if top_anx_t:
    actions = list(top_anx_t.keys())
    counts = list(top_anx_t.values())
    ax.barh(range(len(actions)), counts, color='#EC7063', alpha=0.8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=9)
    ax.set_xlabel('Frequency')
    ax.set_title('Therapist Responding to ANXIETY', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_therapist_patterns.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 06_therapist_patterns.png")
plt.close()

# ============================================================================
# 7. ACTION CONFIDENCE AND LAYER ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("7️⃣  COGNITIVE ACTION CONFIDENCE ANALYSIS")
print("="*80)

# Extract confidence scores for top actions
action_confidences = defaultdict(list)

for _, row in df.iterrows():
    for action, conf in row['action_confidences'].items():
        action_confidences[action].append(conf)

# Calculate statistics
confidence_stats = {}
for action, confs in action_confidences.items():
    confidence_stats[action] = {
        'mean': np.mean(confs),
        'std': np.std(confs),
        'min': np.min(confs),
        'max': np.max(confs),
        'count': len(confs)
    }

# Sort by frequency
sorted_by_freq = sorted(confidence_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# 7.1 Mean confidence by action
ax = axes[0]
actions_list = [item[0] for item in sorted_by_freq]
means = [item[1]['mean'] for item in sorted_by_freq]
stds = [item[1]['std'] for item in sorted_by_freq]

ax.barh(range(len(actions_list)), means, xerr=stds, color='teal', alpha=0.7, capsize=3)
ax.set_yticks(range(len(actions_list)))
ax.set_yticklabels(actions_list, fontsize=9)
ax.set_xlabel('Mean Confidence Score', fontsize=11)
ax.set_title('Top 20 Actions - Mean Confidence', fontweight='bold', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 7.2 Confidence vs frequency scatter
ax = axes[1]
all_actions_data = [(action, stats['mean'], stats['count'])
                    for action, stats in confidence_stats.items()]
means_scatter = [item[1] for item in all_actions_data]
counts_scatter = [item[2] for item in all_actions_data]

ax.scatter(counts_scatter, means_scatter, alpha=0.6, s=50, color='purple')
ax.set_xlabel('Frequency (# of occurrences)', fontsize=11)
ax.set_ylabel('Mean Confidence Score', fontsize=11)
ax.set_title('Cognitive Action Confidence vs Frequency', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3)

# Annotate top actions
for item in sorted_by_freq[:10]:
    action = item[0]
    mean = item[1]['mean']
    count = item[1]['count']
    ax.annotate(action, (count, mean), fontsize=7, alpha=0.7,
                xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_action_confidence.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 07_action_confidence.png")
plt.close()

# ============================================================================
# 8. SESSION OUTCOME INDICATORS
# ============================================================================

print("\n" + "="*80)
print("8️⃣  SESSION OUTCOME INDICATORS")
print("="*80)

# Analyze final client conclusions
conclusions = df[df['role'] == 'client_conclusion'].copy()

if len(conclusions) > 0:
    # Analyze actions in conclusions
    conclusion_actions = Counter()
    for _, row in conclusions.iterrows():
        for action in row['active_actions']:
            conclusion_actions[action] += 1

    # Compare to overall client actions
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 8.1 Most common actions in conclusions
    ax = axes[0, 0]
    top_conclusion = dict(conclusion_actions.most_common(15))
    if top_conclusion:
        actions = list(top_conclusion.keys())
        counts = list(top_conclusion.values())
        ax.barh(range(len(actions)), counts, color='#52BE80', alpha=0.8)
        ax.set_yticks(range(len(actions)))
        ax.set_yticklabels(actions, fontsize=9)
        ax.set_xlabel('Frequency')
        ax.set_title('Cognitive Actions in Client CONCLUSIONS', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    # 8.2 Number of actions in conclusion by issue type
    ax = axes[0, 1]
    conclusion_by_issue = conclusions.groupby('primary_issue')['num_active_actions'].agg(['mean', 'std']).reset_index()
    x_pos = range(len(conclusion_by_issue))
    ax.bar(x_pos, conclusion_by_issue['mean'], yerr=conclusion_by_issue['std'],
           color=['#FF6B6B', '#4ECDC4', '#95E1D3'][:len(conclusion_by_issue)],
           alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conclusion_by_issue['primary_issue'])
    ax.set_ylabel('Avg Number of Active Actions')
    ax.set_title('Cognitive Activity in Conclusions by Issue Type', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 8.3 Word count in conclusions
    ax = axes[1, 0]
    conclusion_words = conclusions.groupby('primary_issue')['word_count'].agg(['mean', 'std']).reset_index()
    ax.bar(x_pos, conclusion_words['mean'], yerr=conclusion_words['std'],
           color=['#FF6B6B', '#4ECDC4', '#95E1D3'][:len(conclusion_words)],
           alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conclusion_words['primary_issue'])
    ax.set_ylabel('Avg Word Count')
    ax.set_title('Conclusion Length by Issue Type', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 8.4 Session trajectory: first client turn vs conclusion
    ax = axes[1, 1]
    first_client = df[df['turn_idx'] == 0]

    first_actions_avg = first_client['num_active_actions'].mean()
    conclusion_actions_avg = conclusions['num_active_actions'].mean()

    categories = ['First Turn', 'Conclusion']
    values = [first_actions_avg, conclusion_actions_avg]
    colors_traj = ['#FFB6B6', '#6BCF7F']

    ax.bar(categories, values, color=colors_traj, alpha=0.7, width=0.5)
    ax.set_ylabel('Avg Number of Active Cognitive Actions')
    ax.set_title('Client Cognitive Activity: First Turn vs Conclusion', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage change annotation
    pct_change = ((conclusion_actions_avg - first_actions_avg) / first_actions_avg) * 100
    ax.text(1, conclusion_actions_avg, f'+{pct_change:.1f}%' if pct_change > 0 else f'{pct_change:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color='green' if pct_change > 0 else 'red')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_session_outcomes.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: 08_session_outcomes.png")
    plt.close()
else:
    print("   ⚠️  No conclusion data found")

# ============================================================================
# 9. HEATMAP: ACTIONS ACROSS SESSION TIMELINE
# ============================================================================

print("\n" + "="*80)
print("9️⃣  SESSION TIMELINE HEATMAP")
print("="*80)

# Get top 20 most common client actions
top_20_client_actions = [action for action, _ in client_actions.most_common(20)]

# Create matrix: actions x position bins
client_timeline_df = df[df['role'].str.contains('client')].copy()

# Normalize turn position for timeline
for session_id in client_timeline_df['session_id'].unique():
    mask = client_timeline_df['session_id'] == session_id
    session_turns = client_timeline_df[mask]
    max_turn = session_turns['turn_idx'].max()
    if max_turn > 0:
        client_timeline_df.loc[mask, 'normalized_position'] = session_turns['turn_idx'] / max_turn
    else:
        client_timeline_df.loc[mask, 'normalized_position'] = 0

client_timeline_df['position_bin'] = pd.cut(client_timeline_df['normalized_position'],
                                             bins=[0, 0.25, 0.5, 0.75, 1.0],
                                             labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Build heatmap matrix
heatmap_data = []
for action in top_20_client_actions:
    row = []
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        quarter_df = client_timeline_df[client_timeline_df['position_bin'] == quarter]
        count = sum(1 for actions in quarter_df['active_actions'] if action in actions)
        row.append(count)
    heatmap_data.append(row)

heatmap_matrix = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(heatmap_matrix,
            xticklabels=['Q1 (Early)', 'Q2', 'Q3', 'Q4 (Late)'],
            yticklabels=top_20_client_actions,
            cmap='YlOrRd',
            annot=True,
            fmt='d',
            cbar_kws={'label': 'Frequency'},
            ax=ax)
ax.set_title('Client Cognitive Actions Across Session Timeline', fontweight='bold', fontsize=14)
ax.set_xlabel('Session Phase', fontsize=12)
ax.set_ylabel('Cognitive Action', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_timeline_heatmap.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: 09_timeline_heatmap.png")
plt.close()

# ============================================================================
# 10. GENERATE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("🔟  GENERATING SUMMARY REPORT")
print("="*80)

report = f"""
THERAPEUTIC CONVERSATION META-ANALYSIS REPORT
==============================================

Data Overview
-------------
• Total Sessions: {num_sessions}
• Total Conversation Turns: {len(df)}
• Therapist Turns: {len(df[df['role'] == 'therapist'])}
• Client Turns: {len(df[df['role'].str.contains('client')])}

Issue Distribution
------------------
"""

for issue, count in df.groupby('session_id')['primary_issue'].first().value_counts().items():
    pct = (count / num_sessions) * 100
    report += f"• {issue.capitalize()}: {count} sessions ({pct:.1f}%)\n"

report += f"""

Top Cognitive Distortions
--------------------------
"""
for distortion, count in df.groupby('session_id')['cognitive_distortion'].first().value_counts().head(5).items():
    report += f"• {distortion}: {count} sessions\n"

report += f"""

Key Findings: Client Progression
---------------------------------
"""

# Early vs late statistics
early_stats = client_df[client_df['position_bin'] == 'Early']['num_active_actions'].mean()
late_stats = client_df[client_df['position_bin'] == 'Late']['num_active_actions'].mean()
change_pct = ((late_stats - early_stats) / early_stats) * 100

report += f"• Cognitive Activity Change: {change_pct:+.1f}% (Early: {early_stats:.2f} → Late: {late_stats:.2f})\n"

report += f"\nTop Emerging Actions (increase from early to late):\n"
for action, data in sorted_changes[:5]:
    if data['change'] > 0:
        report += f"  - {action}: +{data['change']} occurrences ({data['pct_change']:+.1f}%)\n"

report += f"\nTop Declining Actions (decrease from early to late):\n"
for action, data in sorted_changes[:5]:
    if data['change'] < 0:
        report += f"  - {action}: {data['change']} occurrences ({data['pct_change']:.1f}%)\n"

report += f"""

Therapist Response Patterns
----------------------------
"""
report += f"• Top Therapist Cognitive Actions:\n"
for action, count in therapist_actions.most_common(5):
    report += f"  - {action}: {count} times\n"

report += f"""

Depression vs Anxiety Signatures
---------------------------------
"""

if dep_actions and anx_actions:
    report += f"• Depression - Most Common Actions:\n"
    for action, count in dep_actions.most_common(5):
        report += f"  - {action}: {count} times\n"

    report += f"\n• Anxiety - Most Common Actions:\n"
    for action, count in anx_actions.most_common(5):
        report += f"  - {action}: {count} times\n"

    dep_unique_set = set(dep_actions.keys()) - set(anx_actions.keys())
    anx_unique_set = set(anx_actions.keys()) - set(dep_actions.keys())

    report += f"\n• Actions Unique to Depression: {len(dep_unique_set)}\n"
    report += f"• Actions Unique to Anxiety: {len(anx_unique_set)}\n"

report += f"""

Session Outcome Indicators
---------------------------
"""

if len(conclusions) > 0:
    report += f"• Sessions with Conclusions: {len(conclusions)}\n"
    report += f"• Avg Cognitive Actions in Conclusions: {conclusions['num_active_actions'].mean():.2f}\n"
    report += f"• Avg Word Count in Conclusions: {conclusions['word_count'].mean():.1f}\n"

    report += f"\n• Top Actions in Client Conclusions:\n"
    for action, count in conclusion_actions.most_common(5):
        report += f"  - {action}: {count} times\n"

report += f"""

Generated Visualizations
------------------------
• 01_session_statistics.png - Distribution of issues, triggers, symptoms
• 02_cognitive_actions_by_role.png - Therapist vs client action patterns
• 03_client_progression.png - How clients change across session
• 04_action_emergence_decline.png - Actions that increase/decrease
• 05_depression_vs_anxiety.png - Cognitive signatures comparison
• 06_therapist_patterns.png - Therapist adaptation strategies
• 07_action_confidence.png - Confidence scores and reliability
• 08_session_outcomes.png - Indicators of therapeutic progress
• 09_timeline_heatmap.png - Action frequency across session phases

---
Generated by therapeutic_meta_analysis.py
"""

# Save report
report_file = f"{OUTPUT_DIR}/ANALYSIS_REPORT.txt"
with open(report_file, 'w') as f:
    f.write(report)

print(f"   ✅ Saved: ANALYSIS_REPORT.txt")

# Also save structured JSON summary
summary_json = {
    'data_overview': {
        'total_sessions': num_sessions,
        'total_turns': len(df),
        'therapist_turns': len(df[df['role'] == 'therapist']),
        'client_turns': len(df[df['role'].str.contains('client')])
    },
    'issue_distribution': {k: int(v) for k, v in df.groupby('session_id')['primary_issue'].first().value_counts().items()},
    'client_progression': {
        'early_avg_actions': float(early_stats),
        'late_avg_actions': float(late_stats),
        'change_percent': float(change_pct)
    },
    'top_therapist_actions': dict(therapist_actions.most_common(20)),
    'top_client_actions': dict(client_actions.most_common(20)),
    'emerging_actions': {action: data['change'] for action, data in sorted_changes[:10] if data['change'] > 0},
    'declining_actions': {action: data['change'] for action, data in sorted_changes[:10] if data['change'] < 0}
}

json_file = f"{OUTPUT_DIR}/analysis_summary.json"
with open(json_file, 'w') as f:
    json.dump(summary_json, f, indent=2)

print(f"   ✅ Saved: analysis_summary.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✨ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📊 Generated {9} visualizations + 2 reports")
print(f"📁 All outputs saved to: {OUTPUT_DIR}/")
print("\nKey Insights:")
print(f"  • Analyzed {num_sessions} therapeutic sessions")
print(f"  • Tracked {len(set(client_actions.keys()))} unique client cognitive actions")
print(f"  • Tracked {len(set(therapist_actions.keys()))} unique therapist cognitive actions")
print(f"  • Client cognitive activity change: {change_pct:+.1f}% (early to late)")

if len(conclusions) > 0:
    first_avg = df[df['turn_idx'] == 0]['num_active_actions'].mean()
    conclusion_avg = conclusions['num_active_actions'].mean()
    overall_change = ((conclusion_avg - first_avg) / first_avg) * 100
    print(f"  • Overall improvement (first turn → conclusion): {overall_change:+.1f}%")

print("\n" + "="*80)
