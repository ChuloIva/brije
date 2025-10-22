"""
Master script to run all comprehensive analyses
"""

import subprocess
import sys


def run_script(script_path: str, description: str):
    """Run a Python script and report status"""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)

    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def main():
    print("="*80)
    print("RUNNING ALL COMPREHENSIVE ANALYSES")
    print("="*80)

    scripts = [
        ("src/probes/comprehensive_analysis.py", "Comprehensive Statistical Analysis"),
        ("src/probes/action_clustering.py", "Action Clustering & Dimensionality Reduction"),
        ("src/probes/visualize_networks.py", "Network Visualizations & Dashboard"),
    ]

    results = []
    for script_path, description in scripts:
        success = run_script(script_path, description)
        results.append((description, success))

    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)

    for description, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")

    # Output locations
    print("\n" + "="*80)
    print("OUTPUT LOCATIONS")
    print("="*80)
    print("\n📁 results/positive_patterns_analysis/")
    print("   ├── comprehensive_analysis/")
    print("   │   ├── comprehensive_report.json")
    print("   │   ├── cooccurrence_*.csv")
    print("   │   ├── action_sentiment_correlation.csv")
    print("   │   ├── transformation_effectiveness.csv")
    print("   │   ├── layer_patterns.csv")
    print("   │   ├── diversity_*.csv")
    print("   │   └── confidence_statistics.csv")
    print("   │")
    print("   ├── clustering_analysis/")
    print("   │   ├── clusters.json")
    print("   │   ├── cluster_characteristics.csv")
    print("   │   ├── cluster_scatter.html 🌐")
    print("   │   ├── cluster_summary.html 🌐")
    print("   │   ├── cluster_radar.html 🌐")
    print("   │   └── action_statistics.csv")
    print("   │")
    print("   └── network_dashboard.html 🌐 (Main Dashboard)")

    print("\n🌐 = Interactive HTML visualization - open in browser")

    if all(success for _, success in results):
        print("\n✅ All analyses completed successfully!")
    else:
        print("\n⚠️  Some analyses failed. Check logs above.")


if __name__ == "__main__":
    main()