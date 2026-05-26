import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.visualizer import MetricsVisualizer


def main():
    print("PROJECT VIMAAN - METRICS VISUALIZATION")
    print("=" * 70)
    
    visualizer = MetricsVisualizer()
    
    version = None
    if len(sys.argv) > 1:
        try:
            version = int(sys.argv[1])
            print(f"\nGenerating detailed graphs for v{version}")
        except ValueError:
            print(f"\n  Invalid version: {sys.argv[1]}")
            print(f"Using latest version instead")
    
    visualizer.generate_all_visualizations(latest_version=version)
    
    print(f"\n Done! Check ML/evaluation/results/visualizations/")


if __name__ == "__main__":
    main()
