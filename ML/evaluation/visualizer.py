import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class MetricsVisualizer:
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = os.path.join(os.path.dirname(__file__), "results")
        
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def load_all_metrics(self):
        metrics_files = glob.glob(os.path.join(self.results_dir, "metrics_v*.json"))
        
        all_metrics = {}
        for filepath in metrics_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                version = data['metadata']['model_version']
                all_metrics[version] = data
        
        print(f"Loaded metrics for {len(all_metrics)} versions")
        return dict(sorted(all_metrics.items()))
    
    def load_comparison_json(self):
        comparison_files = glob.glob(os.path.join(self.results_dir, "comparison_all_versions_*.json"))
        
        if comparison_files:
            latest = max(comparison_files, key=os.path.getctime)
            with open(latest, 'r') as f:
                return json.load(f)
        return None
    
    def plot_progress_metrics(self, all_metrics):
        versions = sorted(all_metrics.keys())
        accuracies = [all_metrics[v]['overall_accuracy'] * 100 for v in versions]
        weighted_f1s = [all_metrics[v]['weighted_f1'] for v in versions]
        macro_f1s = [all_metrics[v]['macro_f1'] for v in versions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        #Accuracy
        ax1.plot(versions, accuracies, marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', label='Accuracy')
        ax1.fill_between(versions, accuracies, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Model Version', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([min(accuracies) - 5, 100])
        
        for v, acc in zip(versions, accuracies):
            ax1.text(v, acc + 1, f'{acc:.1f}%', ha='center', fontsize=9)
        
        #F1 Scores
        ax2.plot(versions, weighted_f1s, marker='s', linewidth=2, markersize=8,
                color='#A23B72', label='Weighted F1')
        ax2.plot(versions, macro_f1s, marker='^', linewidth=2, markersize=8,
                color='#F18F01', label='Macro F1')
        ax2.set_xlabel('Model Version', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('F1 Score Progression', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([min(min(weighted_f1s), min(macro_f1s)) - 0.05, 1.0])
        
        for v, wf1, mf1 in zip(versions, weighted_f1s, macro_f1s):
            ax2.text(v, wf1 + 0.01, f'{wf1:.3f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "01_progress_metrics.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_detailed_comparison(self, all_metrics):
        versions = sorted(all_metrics.keys())
        
        metrics_data = {
            'Accuracy (%)': [all_metrics[v]['overall_accuracy'] * 100 for v in versions],
            'Weighted F1': [all_metrics[v]['weighted_f1'] * 100 for v in versions],
            'Confidence': [all_metrics[v]['confidence']['mean'] * 100 for v in versions],
            'Cohen\'s Kappa': [all_metrics[v]['cohens_kappa'] * 100 for v in versions]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(versions, values, color=colors[idx], alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Model Version', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name} by Version', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "02_detailed_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_confusion_matrix(self, version, metrics):
        cm_data = metrics['confusion_matrix']
        labels = cm_data['labels']
        matrix = np.array(cm_data['matrix'])
        
        matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        matrix_normalized = np.nan_to_num(matrix_normalized)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        #heatmap
        sns.heatmap(matrix_normalized, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Normalized Frequency'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'Confusion Matrix - Model v{version}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Intent', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Intent', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"03_confusion_matrix_v{version}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_inference_time(self, all_metrics):
        versions = sorted(all_metrics.keys())
        
        means = [all_metrics[v]['inference_time_ms']['mean'] for v in versions]
        mins = [all_metrics[v]['inference_time_ms']['min'] for v in versions]
        maxs = [all_metrics[v]['inference_time_ms']['max'] for v in versions]
        p95s = [all_metrics[v]['inference_time_ms']['p95'] for v in versions]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(versions))
        width = 0.2
        
        ax.bar(x - width*1.5, mins, width, label='Min', color='#06D6A0', alpha=0.8)
        ax.bar(x - width/2, means, width, label='Mean', color='#2E86AB', alpha=0.8)
        ax.bar(x + width/2, p95s, width, label='95th Percentile', color='#F18F01', alpha=0.8)
        ax.bar(x + width*1.5, maxs, width, label='Max', color='#C73E1D', alpha=0.8)
        
        ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'v{v}' for v in versions])
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "04_inference_time.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_per_intent_performance(self, version, metrics, top_n=10):
        per_intent = metrics['per_intent']
        
        #Sort by F1 score
        sorted_intents = sorted(per_intent.items(), 
                               key=lambda x: x[1]['f1'], 
                               reverse=True)[:top_n]
        
        intents = [item[0] for item in sorted_intents]
        precisions = [item[1]['precision'] * 100 for item in sorted_intents]
        recalls = [item[1]['recall'] * 100 for item in sorted_intents]
        f1s = [item[1]['f1'] * 100 for item in sorted_intents]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(intents))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', color='#2E86AB', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', color='#A23B72', alpha=0.8)
        ax.bar(x + width, f1s, width, label='F1 Score', color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Intent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Intent Performance - Model v{version}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(intents, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"05_per_intent_v{version}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self, latest_version=None):
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*70}\n")

        all_metrics = self.load_all_metrics()
        
        if not all_metrics:
            print("No metrics files found!")
            return
        
        print("\nGenerating Graph 1: Progress Metrics...")
        self.plot_progress_metrics(all_metrics)
        
        print("\nGenerating Graph 2: Detailed Comparison...")
        self.plot_detailed_comparison(all_metrics)
        
        print("\nGenerating Graph 4: Inference Time...")
        self.plot_inference_time(all_metrics)
        
        if latest_version is None:
            latest_version = max(all_metrics.keys())
        
        print(f"\nGenerating Graph 3: Confusion Matrix (v{latest_version})...")
        self.plot_confusion_matrix(latest_version, all_metrics[latest_version])
        
        print(f"\nGenerating Graph 5: Per-Intent Performance (v{latest_version})...")
        self.plot_per_intent_performance(latest_version, all_metrics[latest_version])
        
        print(f"\n\n{'='*70}")
        print(f"ALL VISUALIZATIONS COMPLETE!")
        print(f"{'='*70}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"\nGenerated files:")
        for i in range(1, 6):
            files = glob.glob(os.path.join(self.output_dir, f"0{i}_*.png"))
            for f in files:
                print(f"  - {os.path.basename(f)}")


def main():
    visualizer = MetricsVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
