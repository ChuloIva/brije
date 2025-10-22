"""
Advanced Network Visualizations for Cognitive Action Analysis
Creates interactive network graphs, heatmaps, and flow diagrams
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx


def load_comprehensive_report(report_path: Path) -> dict:
    """Load the comprehensive report JSON"""
    with open(report_path, 'r') as f:
        return json.load(f)


def create_cooccurrence_network(cooccur_df: pd.DataFrame,
                                threshold: int = 5,
                                top_n: int = 30,
                                title: str = "Action Co-occurrence Network") -> go.Figure:
    """
    Create interactive network graph from co-occurrence matrix
    """
    # Load cooccurrence data
    if isinstance(cooccur_df, str):
        cooccur_df = pd.read_csv(cooccur_df, index_col=0)

    # Filter to top N most frequent actions
    action_totals = cooccur_df.sum(axis=1).sort_values(ascending=False)
    top_actions = action_totals.head(top_n).index.tolist()

    # Filter matrix
    cooccur_filtered = cooccur_df.loc[top_actions, top_actions]

    # Create NetworkX graph
    G = nx.Graph()

    # Add edges with weights
    for i, action1 in enumerate(top_actions):
        for j, action2 in enumerate(top_actions):
            if i < j:  # Only upper triangle
                weight = cooccur_filtered.loc[action1, action2]
                if weight >= threshold:
                    G.add_edge(action1, action2, weight=weight)

    # Compute node sizes based on degree
    node_degrees = dict(G.degree())

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge traces
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']

        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=weight/5, color='rgba(125,125,125,0.3)'),
            hoverinfo='none',
            showlegend=False
        ))

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Connections: {node_degrees[node]}")
        node_size.append(20 + node_degrees[node] * 3)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node.replace('_', ' ') for node in G.nodes()],
        textposition='top center',
        textfont=dict(size=9),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node<br>Degree"),
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])

    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240,240,240,0.9)',
        width=1200,
        height=900
    )

    return fig


def create_sentiment_heatmap(action_sentiment_df: pd.DataFrame, top_n: int = 40) -> go.Figure:
    """
    Create heatmap of actions by sentiment correlation
    """
    if isinstance(action_sentiment_df, (str, Path)):
        action_sentiment_df = pd.read_csv(action_sentiment_df)

    # Sort by mean sentiment and take top N
    df_sorted = action_sentiment_df.sort_values('mean_sentiment', ascending=False).head(top_n)

    # Prepare data for heatmap
    actions = df_sorted['action'].tolist()
    sentiments = df_sorted['mean_sentiment'].tolist()
    counts = df_sorted['count'].tolist()
    pos_ratios = df_sorted['positive_ratio'].tolist()
    neg_ratios = df_sorted['negative_ratio'].tolist()

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Mean Sentiment', 'Positive Ratio', 'Negative Ratio'),
        horizontal_spacing=0.1
    )

    # Mean sentiment
    fig.add_trace(
        go.Heatmap(
            z=[[s] for s in sentiments],
            y=actions,
            x=[''],
            colorscale='RdYlGn',
            zmid=0,
            showscale=True,
            colorbar=dict(x=0.3),
            hovertemplate='%{y}<br>Sentiment: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Positive ratio
    fig.add_trace(
        go.Heatmap(
            z=[[r] for r in pos_ratios],
            y=actions,
            x=[''],
            colorscale='Greens',
            showscale=True,
            colorbar=dict(x=0.62),
            hovertemplate='%{y}<br>Positive: %{z:.1%}<extra></extra>'
        ),
        row=1, col=2
    )

    # Negative ratio
    fig.add_trace(
        go.Heatmap(
            z=[[r] for r in neg_ratios],
            y=actions,
            x=[''],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(x=0.95),
            hovertemplate='%{y}<br>Negative: %{z:.1%}<extra></extra>'
        ),
        row=1, col=3
    )

    fig.update_layout(
        title='Cognitive Actions by Sentiment Association',
        title_x=0.5,
        height=1200,
        width=1400
    )

    # Update y-axes to show full action names
    for i in range(1, 4):
        fig.update_yaxes(tickfont=dict(size=9), row=1, col=i)

    return fig


def create_transformation_flow(trans_eff_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Create Sankey diagram showing flow from negative -> transformation -> positive
    """
    if isinstance(trans_eff_df, (str, Path)):
        trans_eff_df = pd.read_csv(trans_eff_df)

    # Filter to actions with significant counts
    df_filtered = trans_eff_df[trans_eff_df['total_count'] >= 5].head(top_n)

    # Create Sankey data
    labels = []
    sources = []
    targets = []
    values = []
    colors = []

    # Create node labels: Negative -> Action -> Transformation/Positive
    action_nodes = {}
    node_idx = 0

    # Add pattern type nodes
    neg_idx = node_idx
    labels.append('NEGATIVE')
    colors.append('rgba(239, 68, 68, 0.8)')  # Red
    node_idx += 1

    trans_idx = node_idx
    labels.append('TRANSFORMATION')
    colors.append('rgba(245, 158, 11, 0.8)')  # Orange
    node_idx += 1

    pos_idx = node_idx
    labels.append('POSITIVE')
    colors.append('rgba(16, 185, 129, 0.8)')  # Green
    node_idx += 1

    # Add action nodes
    for _, row in df_filtered.iterrows():
        action = row['action']
        action_nodes[action] = node_idx
        labels.append(action.replace('_', ' ').title())
        colors.append('rgba(102, 126, 234, 0.6)')  # Purple
        node_idx += 1

    # Create flows
    for _, row in df_filtered.iterrows():
        action = row['action']
        action_idx = action_nodes[action]

        # Negative -> Action
        if row['negative_count'] > 0:
            sources.append(neg_idx)
            targets.append(action_idx)
            values.append(row['negative_count'])

        # Action -> Transformation
        if row['transformation_count'] > 0:
            sources.append(action_idx)
            targets.append(trans_idx)
            values.append(row['transformation_count'])

        # Action -> Positive
        if row['positive_count'] > 0:
            sources.append(action_idx)
            targets.append(pos_idx)
            values.append(row['positive_count'])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=2),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(200, 200, 200, 0.3)'
        )
    )])

    fig.update_layout(
        title='Cognitive Action Flow Across Pattern Types',
        title_x=0.5,
        font=dict(size=11),
        height=900,
        width=1400
    )

    return fig


def create_layer_activation_plot(layer_patterns_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """
    Create plot showing layer activation patterns
    """
    if isinstance(layer_patterns_df, (str, Path)):
        layer_patterns_df = pd.read_csv(layer_patterns_df)

    df_sorted = layer_patterns_df.sort_values('mean_layers', ascending=False).head(top_n)

    fig = go.Figure()

    # Add error bars
    fig.add_trace(go.Bar(
        y=df_sorted['action'],
        x=df_sorted['mean_layers'],
        orientation='h',
        error_x=dict(
            type='data',
            array=df_sorted['std_layers'],
            visible=True
        ),
        marker=dict(
            color=df_sorted['mean_layers'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Mean<br>Layers")
        ),
        hovertemplate='<b>%{y}</b><br>Mean layers: %{x:.2f}<br>Total activations: %{customdata}<extra></extra>',
        customdata=df_sorted['total_activations']
    ))

    fig.update_layout(
        title='Layer Activation Breadth by Action (Mean ± Std)',
        title_x=0.5,
        xaxis_title='Mean Number of Active Layers',
        yaxis_title='',
        height=900,
        width=1200,
        yaxis=dict(tickfont=dict(size=9))
    )

    return fig


def create_cognitive_pattern_comparison(report: dict, top_n: int = 10) -> go.Figure:
    """
    Compare top actions across different cognitive pattern types
    """
    pattern_sigs = report['cognitive_pattern_signatures']

    # Get top cognitive patterns by total actions
    pattern_totals = {k: sum(v.values()) for k, v in pattern_sigs.items()}
    top_patterns = sorted(pattern_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Create subplots
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[p[0] for p in top_patterns],
        vertical_spacing=0.08,
        horizontal_spacing=0.15
    )

    for idx, (pattern_name, total) in enumerate(top_patterns):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        actions = pattern_sigs[pattern_name]
        top_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:10]

        action_names = [a[0] for a in top_actions]
        action_counts = [a[1] for a in top_actions]

        fig.add_trace(
            go.Bar(
                y=action_names,
                x=action_counts,
                orientation='h',
                marker=dict(color='rgba(102, 126, 234, 0.7)'),
                showlegend=False,
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text='Count', row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=8), row=row, col=col)

    fig.update_layout(
        title='Top Actions by Cognitive Pattern Type',
        title_x=0.5,
        height=1600,
        width=1400,
        showlegend=False
    )

    return fig


def create_action_bridge_visualization(bridges: list, top_n: int = 15) -> go.Figure:
    """
    Visualize action bridges (negative -> transformation improvements)
    """
    # Take top N by sentiment lift
    top_bridges = sorted(bridges, key=lambda x: x['sentiment_lift'], reverse=True)[:top_n]

    fig = go.Figure()

    pattern_names = [b['cognitive_pattern_name'][:40] + '...' if len(b['cognitive_pattern_name']) > 40
                     else b['cognitive_pattern_name'] for b in top_bridges]
    sentiment_lifts = [b['sentiment_lift'] for b in top_bridges]
    neg_sentiments = [b['neg_sentiment'] for b in top_bridges]
    trans_sentiments = [b['trans_sentiment'] for b in top_bridges]
    bridge_counts = [b['num_bridge_actions'] for b in top_bridges]

    # Create grouped bar chart
    fig.add_trace(go.Bar(
        name='Negative Sentiment',
        y=pattern_names,
        x=neg_sentiments,
        orientation='h',
        marker=dict(color='rgba(239, 68, 68, 0.7)'),
        hovertemplate='<b>%{y}</b><br>Negative: %{x:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Transformation Sentiment',
        y=pattern_names,
        x=trans_sentiments,
        orientation='h',
        marker=dict(color='rgba(16, 185, 129, 0.7)'),
        hovertemplate='<b>%{y}</b><br>Transformation: %{x:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='Sentiment Lift via Action Bridges (Negative → Transformation)',
        title_x=0.5,
        xaxis_title='Sentiment Score',
        barmode='group',
        height=700,
        width=1400,
        yaxis=dict(tickfont=dict(size=9)),
        legend=dict(x=0.7, y=0.95)
    )

    return fig


def create_comprehensive_dashboard(analysis_dir: Path, output_path: Path):
    """
    Create comprehensive HTML dashboard with all visualizations
    """
    print("Creating comprehensive visualization dashboard...")

    # Load data
    report = load_comprehensive_report(analysis_dir / 'comprehensive_report.json')

    # Create all figures
    print("  1. Co-occurrence network...")
    cooccur_df = pd.read_csv(analysis_dir / 'cooccurrence_all.csv', index_col=0)
    fig_network = create_cooccurrence_network(cooccur_df, threshold=3, top_n=35)

    print("  2. Sentiment heatmap...")
    fig_sentiment = create_sentiment_heatmap(analysis_dir / 'action_sentiment_correlation.csv', top_n=40)

    print("  3. Transformation flow...")
    fig_flow = create_transformation_flow(analysis_dir / 'transformation_effectiveness.csv', top_n=25)

    print("  4. Layer activation plot...")
    fig_layers = create_layer_activation_plot(analysis_dir / 'layer_patterns.csv', top_n=35)

    print("  5. Cognitive pattern comparison...")
    fig_patterns = create_cognitive_pattern_comparison(report, top_n=10)

    print("  6. Action bridges...")
    fig_bridges = create_action_bridge_visualization(report['action_bridges'], top_n=15)

    # Create HTML with all figures
    html_parts = []
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comprehensive Cognitive Action Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .section {
            margin: 40px 0;
            padding: 20px;
            background: #f5f7fa;
            border-radius: 10px;
        }
        h2 {
            color: #764ba2;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .description {
            color: #666;
            margin: 15px 0;
            line-height: 1.6;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 Comprehensive Cognitive Action Network Analysis</h1>
    <p style="text-align: center; color: #666; font-size: 1.1em;">
        Advanced visualizations of cognitive action patterns, co-occurrences, and transformations
    </p>
""")

    # Add each figure
    sections = [
        ("Action Co-occurrence Network", fig_network,
         "This network shows which cognitive actions frequently appear together. Node size represents connectivity, and edge thickness represents co-occurrence frequency."),
        ("Sentiment Association Heatmap", fig_sentiment,
         "Actions sorted by their average sentiment association. Shows which actions correlate with positive vs negative emotional states."),
        ("Transformation Flow", fig_flow,
         "Sankey diagram showing how cognitive actions flow from negative patterns through transformation to positive outcomes."),
        ("Layer Activation Breadth", fig_layers,
         "Shows how many neural network layers are activated by each action. Higher breadth suggests more distributed processing."),
        ("Cognitive Pattern Signatures", fig_patterns,
         "Top actions characteristic of each cognitive pattern type (e.g., rumination, avolition, etc.)."),
        ("Action Bridges", fig_bridges,
         "Actions that appear in transformation patterns and drive sentiment improvement from negative to positive.")
    ]

    for title, fig, description in sections:
        html_parts.append(f"""
    <div class="section">
        <h2>{title}</h2>
        <p class="description">{description}</p>
        <div>
            {fig.to_html(include_plotlyjs=False, full_html=False, div_id=title.replace(' ', '_').lower())}
        </div>
    </div>
""")

    html_parts.append("""
</div>
</body>
</html>
""")

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"\n✓ Dashboard saved to: {output_path}")


def main():
    analysis_dir = Path("results/positive_patterns_analysis/comprehensive_analysis")
    output_path = Path("results/positive_patterns_analysis/network_dashboard.html")

    create_comprehensive_dashboard(analysis_dir, output_path)

    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOpen {output_path} in your browser to view the interactive dashboard")


if __name__ == "__main__":
    main()
