"""
Action Clustering and Dimensionality Reduction Analysis
Uses embeddings to find semantic clusters of cognitive actions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple


def load_analyses(batches_dir: Path) -> List[dict]:
    """Load all batch files"""
    all_analyses = []
    batch_files = sorted(batches_dir.glob("batch_*.json"))

    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_analyses.extend(batch_data)

    return all_analyses


def create_action_context_vectors(analyses: List[dict]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Create feature vectors for each action based on:
    - Co-occurrence patterns
    - Sentiment associations
    - Layer activation patterns
    - Pattern type distributions
    """
    # Collect statistics for each action
    action_stats = defaultdict(lambda: {
        'cooccurrences': Counter(),
        'sentiments': [],
        'layers': [],
        'pattern_types': Counter(),
        'cognitive_patterns': Counter(),
        'confidences': []
    })

    # Collect co-occurrences
    for analysis in analyses:
        actions_in_pattern = [action for action, _, _ in analysis['top_actions']]

        for action, layer_count, confidence in analysis['top_actions']:
            stats = action_stats[action]

            # Track co-occurring actions
            for other_action in actions_in_pattern:
                if other_action != action:
                    stats['cooccurrences'][other_action] += 1

            # Track sentiment
            stats['sentiments'].append(analysis['sentiment_avg'])

            # Track layers
            stats['layers'].append(layer_count)

            # Track pattern type
            stats['pattern_types'][analysis['pattern_type']] += 1

            # Track cognitive pattern type
            stats['cognitive_patterns'][analysis['cognitive_pattern_type']] += 1

            # Track confidence
            stats['confidences'].append(confidence)

    # Get all actions
    all_actions = sorted(list(action_stats.keys()))

    # Create feature vectors
    feature_vectors = {}

    for action in all_actions:
        stats = action_stats[action]

        features = []

        # 1. Sentiment features (2 features)
        features.append(np.mean(stats['sentiments']))
        features.append(np.std(stats['sentiments']))

        # 2. Layer features (3 features)
        features.append(np.mean(stats['layers']))
        features.append(np.std(stats['layers']))
        features.append(np.max(stats['layers']) if stats['layers'] else 0)

        # 3. Confidence features (2 features)
        features.append(np.mean(stats['confidences']))
        features.append(np.std(stats['confidences']))

        # 4. Pattern type distribution (3 features)
        total_patterns = sum(stats['pattern_types'].values())
        features.append(stats['pattern_types']['positive'] / total_patterns if total_patterns > 0 else 0)
        features.append(stats['pattern_types']['negative'] / total_patterns if total_patterns > 0 else 0)
        features.append(stats['pattern_types']['transformation'] / total_patterns if total_patterns > 0 else 0)

        # 5. Co-occurrence features (top 20 most common co-occurrences)
        top_cooccurs = stats['cooccurrences'].most_common(20)
        cooccur_vector = [0] * len(all_actions)
        for cooccur_action, count in top_cooccurs:
            if cooccur_action in all_actions:
                idx = all_actions.index(cooccur_action)
                cooccur_vector[idx] = count

        # Normalize co-occurrence vector
        cooccur_sum = sum(cooccur_vector)
        if cooccur_sum > 0:
            cooccur_vector = [c / cooccur_sum for c in cooccur_vector]

        features.extend(cooccur_vector)

        feature_vectors[action] = np.array(features)

    return feature_vectors, all_actions


def perform_clustering(feature_matrix: np.ndarray,
                      action_names: List[str],
                      n_clusters: int = 6) -> Tuple[np.ndarray, Dict]:
    """
    Perform K-means clustering on action feature vectors
    """
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # Compute silhouette score
    silhouette = silhouette_score(feature_matrix, cluster_labels)

    # Group actions by cluster
    clusters = defaultdict(list)
    for action, label in zip(action_names, cluster_labels):
        clusters[label].append(action)

    results = {
        'labels': cluster_labels,
        'clusters': dict(clusters),
        'silhouette_score': silhouette,
        'centroids': kmeans.cluster_centers_
    }

    return cluster_labels, results


def reduce_dimensionality(feature_matrix: np.ndarray,
                          method: str = 'tsne') -> np.ndarray:
    """
    Reduce dimensionality for visualization
    """
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(feature_matrix)
    return reduced


def visualize_action_clusters(reduced_coords: np.ndarray,
                              action_names: List[str],
                              cluster_labels: np.ndarray,
                              action_stats: Dict) -> go.Figure:
    """
    Create interactive scatter plot of action clusters
    """
    # Create hover text
    hover_texts = []
    for action in action_names:
        stats = action_stats[action]
        hover_text = f"<b>{action}</b><br>"
        hover_text += f"Mean sentiment: {stats['mean_sentiment']:.2f}<br>"
        hover_text += f"Mean layers: {stats['mean_layers']:.2f}<br>"
        hover_text += f"Occurrences: {stats['total_occurrences']}"
        hover_texts.append(hover_text)

    # Create figure
    fig = go.Figure()

    # Add scatter points colored by cluster
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        fig.add_trace(go.Scatter(
            x=reduced_coords[mask, 0],
            y=reduced_coords[mask, 1],
            mode='markers+text',
            name=f'Cluster {cluster_id}',
            text=[action_names[i] for i in range(len(action_names)) if mask[i]],
            textposition='top center',
            textfont=dict(size=8),
            hovertext=[hover_texts[i] for i in range(len(hover_texts)) if mask[i]],
            hoverinfo='text',
            marker=dict(
                size=10,
                line=dict(width=1, color='white')
            )
        ))

    fig.update_layout(
        title='Cognitive Action Clusters (t-SNE Projection)',
        title_x=0.5,
        xaxis=dict(title='Component 1', showgrid=False),
        yaxis=dict(title='Component 2', showgrid=False),
        width=1400,
        height=900,
        hovermode='closest',
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def analyze_cluster_characteristics(clusters: Dict,
                                    action_stats: Dict) -> pd.DataFrame:
    """
    Analyze characteristics of each cluster
    """
    results = []

    for cluster_id, actions in clusters.items():
        # Aggregate statistics
        sentiments = [action_stats[a]['mean_sentiment'] for a in actions]
        layers = [action_stats[a]['mean_layers'] for a in actions]
        pos_ratios = [action_stats[a]['pos_ratio'] for a in actions]
        neg_ratios = [action_stats[a]['neg_ratio'] for a in actions]
        trans_ratios = [action_stats[a]['trans_ratio'] for a in actions]

        results.append({
            'cluster_id': cluster_id,
            'num_actions': len(actions),
            'mean_sentiment': np.mean(sentiments),
            'std_sentiment': np.std(sentiments),
            'mean_layers': np.mean(layers),
            'mean_pos_ratio': np.mean(pos_ratios),
            'mean_neg_ratio': np.mean(neg_ratios),
            'mean_trans_ratio': np.mean(trans_ratios),
            'example_actions': ', '.join(actions[:5])
        })

    df = pd.DataFrame(results).sort_values('cluster_id')
    return df


def compute_action_statistics(analyses: List[dict]) -> Dict[str, Dict]:
    """
    Compute summary statistics for each action
    """
    action_data = defaultdict(lambda: {
        'sentiments': [],
        'layers': [],
        'pattern_types': Counter(),
        'total': 0
    })

    for analysis in analyses:
        for action, layer_count, confidence in analysis['top_actions']:
            action_data[action]['sentiments'].append(analysis['sentiment_avg'])
            action_data[action]['layers'].append(layer_count)
            action_data[action]['pattern_types'][analysis['pattern_type']] += 1
            action_data[action]['total'] += 1

    # Convert to summary stats
    action_stats = {}
    for action, data in action_data.items():
        total = data['total']
        action_stats[action] = {
            'mean_sentiment': np.mean(data['sentiments']),
            'mean_layers': np.mean(data['layers']),
            'total_occurrences': total,
            'pos_ratio': data['pattern_types']['positive'] / total,
            'neg_ratio': data['pattern_types']['negative'] / total,
            'trans_ratio': data['pattern_types']['transformation'] / total
        }

    return action_stats


def create_cluster_summary_visualization(cluster_chars: pd.DataFrame) -> go.Figure:
    """
    Create visualization of cluster characteristics
    """
    fig = go.Figure()

    # Sentiment by cluster
    fig.add_trace(go.Bar(
        name='Mean Sentiment',
        x=cluster_chars['cluster_id'].astype(str),
        y=cluster_chars['mean_sentiment'],
        marker=dict(color='rgba(102, 126, 234, 0.7)')
    ))

    fig.update_layout(
        title='Cluster Characteristics: Mean Sentiment',
        title_x=0.5,
        xaxis_title='Cluster ID',
        yaxis_title='Mean Sentiment',
        width=1000,
        height=500
    )

    return fig


def create_cluster_radar_chart(cluster_chars: pd.DataFrame) -> go.Figure:
    """
    Create radar chart showing cluster profiles
    """
    fig = go.Figure()

    categories = ['Mean Sentiment', 'Mean Layers', 'Positive Ratio', 'Negative Ratio', 'Trans Ratio']

    for _, row in cluster_chars.iterrows():
        # Normalize values to 0-1 range for visualization
        values = [
            (row['mean_sentiment'] + 5) / 10,  # Assuming sentiment range -5 to 5
            row['mean_layers'] / 15,  # Assuming max 15 layers
            row['mean_pos_ratio'],
            row['mean_neg_ratio'],
            row['mean_trans_ratio']
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f"Cluster {row['cluster_id']}"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Cluster Profiles (Normalized)',
        title_x=0.5,
        width=800,
        height=800
    )

    return fig


def main():
    batches_dir = Path("results/positive_patterns_analysis/batches")
    output_dir = Path("results/positive_patterns_analysis/clustering_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COGNITIVE ACTION CLUSTERING ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading analyses...")
    analyses = load_analyses(batches_dir)
    print(f"   Loaded {len(analyses)} analyses")

    # Compute action statistics
    print("\n2. Computing action statistics...")
    action_stats = compute_action_statistics(analyses)
    print(f"   Found {len(action_stats)} unique actions")

    # Create feature vectors
    print("\n3. Creating action feature vectors...")
    feature_vectors, action_names = create_action_context_vectors(analyses)
    feature_matrix = np.array([feature_vectors[action] for action in action_names])
    print(f"   Feature matrix shape: {feature_matrix.shape}")

    # Perform clustering
    print("\n4. Performing K-means clustering...")
    n_clusters = 6
    cluster_labels, cluster_results = perform_clustering(feature_matrix, action_names, n_clusters)
    print(f"   Silhouette score: {cluster_results['silhouette_score']:.3f}")

    # Save clusters (convert numpy int keys to regular int)
    clusters_dict = {int(k): v for k, v in cluster_results['clusters'].items()}
    clusters_data = {
        'clusters': clusters_dict,
        'silhouette_score': float(cluster_results['silhouette_score'])
    }
    with open(output_dir / 'clusters.json', 'w') as f:
        json.dump(clusters_data, f, indent=2)

    print("\n   Clusters:")
    for cluster_id, actions in cluster_results['clusters'].items():
        print(f"     Cluster {cluster_id}: {len(actions)} actions")
        print(f"        {', '.join(actions[:5])}...")

    # Analyze cluster characteristics
    print("\n5. Analyzing cluster characteristics...")
    cluster_chars = analyze_cluster_characteristics(cluster_results['clusters'], action_stats)
    cluster_chars.to_csv(output_dir / 'cluster_characteristics.csv', index=False)

    print("\n   Cluster Characteristics:")
    print(cluster_chars.to_string(index=False))

    # Dimensionality reduction
    print("\n6. Reducing dimensionality (t-SNE)...")
    reduced_coords = reduce_dimensionality(feature_matrix, method='tsne')

    # Create visualizations
    print("\n7. Creating visualizations...")

    print("   - Cluster scatter plot...")
    fig_scatter = visualize_action_clusters(reduced_coords, action_names, cluster_labels, action_stats)
    fig_scatter.write_html(output_dir / 'cluster_scatter.html')

    print("   - Cluster summary...")
    fig_summary = create_cluster_summary_visualization(cluster_chars)
    fig_summary.write_html(output_dir / 'cluster_summary.html')

    print("   - Cluster radar chart...")
    fig_radar = create_cluster_radar_chart(cluster_chars)
    fig_radar.write_html(output_dir / 'cluster_radar.html')

    # Save action statistics
    action_stats_df = pd.DataFrame.from_dict(action_stats, orient='index')
    action_stats_df.index.name = 'action'
    action_stats_df = action_stats_df.reset_index()
    action_stats_df.to_csv(output_dir / 'action_statistics.csv', index=False)

    print("\n" + "="*80)
    print("✅ CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - clusters.json")
    print("  - cluster_characteristics.csv")
    print("  - cluster_scatter.html")
    print("  - cluster_summary.html")
    print("  - cluster_radar.html")
    print("  - action_statistics.csv")


if __name__ == "__main__":
    main()