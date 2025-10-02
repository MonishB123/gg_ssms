"""
Distance Metric Benchmarking for GraphSSM

This script tests different distance metrics (cosine, euclidean, gaussian, manhattan, norm2)
to measure their impact on GraphSSM inference speed and creates visualization plots.

REQUIRES: Modified main.py with distance_metric parameter support

Usage:
    python benchmark_distances.py --seq_len 96 --pred_len 24 --batch_size 32
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd

# Add paths (same as eval_forecasting.py)
gg_ssms_path = os.path.expanduser("/workspace")
mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")
sys.path.append(mamba_ts_path)
sys.path.append(os.path.join(gg_ssms_path, "core", "graph_ssm"))

from main import GraphSSM


def benchmark_distance_metric(distance_metric, args, num_iterations=50, warmup=5):
    """
    Benchmark a specific distance metric
    
    Args:
        distance_metric: Name of distance metric to test
        args: Argparse arguments
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations
    
    Returns:
        dict: Results containing timing statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Create model with specific distance metric
    model = GraphSSM(
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        distance_metric=distance_metric  # THIS NOW WORKS!
    ).to(device)
    
    model.eval()
    
    # Create dummy input
    batch_size = args.batch_size
    seq_len = args.seq_len
    x = torch.randn(batch_size, seq_len, args.d_model).to(device)
    context_len = min(seq_len, 4)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, context_len)
    
    # Benchmark
    times = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            output = model(x, context_len)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'metric': distance_metric,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'all_times': times
    }


def create_visualization(results, args):
    """
    Create matplotlib visualizations of benchmark results
    
    Args:
        results: List of result dictionaries
        args: Argparse arguments
    """
    metrics = [r['metric'] for r in results]
    mean_times = [r['mean_ms'] for r in results]
    std_times = [r['std_ms'] for r in results]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f'GraphSSM Distance Metric Performance Comparison\n'
                 f'seq_len={args.seq_len}, batch_size={args.batch_size}, d_model={args.d_model}',
                 fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(metrics)))
    
    # 1. Bar chart with error bars
    ax1 = axes[0, 0]
    bars = ax1.bar(metrics, mean_times, yerr=std_times, capsize=5, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance Metric', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Inference Time (with std dev)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, mean_times, std_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.2f}±{std_val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Box plot showing distribution
    ax2 = axes[0, 1]
    all_times_data = [r['all_times'] for r in results]
    bp = ax2.boxplot(all_times_data, labels=metrics, patch_artist=True,
                     showmeans=True, meanline=True,
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5),
                     medianprops=dict(linewidth=2, color='red'),
                     meanprops=dict(linewidth=2, color='blue', linestyle='--'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance Metric', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Inference Times', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.legend([bp['medians'][0], bp['means'][0]], 
               ['Median', 'Mean'], loc='upper right')
    
    # 3. Relative speedup chart
    ax3 = axes[1, 0]
    baseline_time = mean_times[0]  # Use first metric as baseline
    speedups = [baseline_time / t for t in mean_times]
    bars3 = ax3.bar(metrics, speedups, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                label=f'Baseline ({metrics[0]})', zorder=5)
    ax3.set_ylabel('Relative Speed', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Distance Metric', fontsize=12, fontweight='bold')
    ax3.set_title(f'Relative Speedup (vs {metrics[0]})', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    
    # Add percentage labels
    for bar, speedup, mean_t in zip(bars3, speedups, mean_times):
        height = bar.get_height()
        percentage = (speedup - 1) * 100
        color = 'green' if percentage > 0 else 'red'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.3f}x\n({percentage:+.1f}%)\n{mean_t:.2f}ms',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=color)
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            r['metric'],
            f"{r['mean_ms']:.3f}",
            f"{r['std_ms']:.3f}",
            f"{r['min_ms']:.3f}",
            f"{r['max_ms']:.3f}",
            f"{r['median_ms']:.3f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.14, 0.14, 0.13, 0.13, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternate row colors and highlight fastest
    fastest_idx = mean_times.index(min(mean_times))
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i - 1 == fastest_idx:
                table[(i, j)].set_facecolor('#90EE90')  # Light green for fastest
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f5f5f5')
    
    ax4.set_title('Summary Statistics (all times in ms)', 
                  fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(args.output_dir, 'distance_metric_benchmark.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = os.path.join(args.output_dir, 'distance_metric_benchmark.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"📊 PDF version saved to: {pdf_path}")
    
    if not args.no_display:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Benchmark GraphSSM distance metrics')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=16, help='State dimension')
    parser.add_argument('--d_conv', type=int, default=4, help='Conv kernel size')
    parser.add_argument('--expand', type=int, default=2, help='Expansion ratio')
    
    # Benchmark parameters
    parser.add_argument('--seq_len', type=int, default=96, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=50, 
                       help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=5, 
                       help='Number of warmup iterations')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./benchmark_results', 
                       help='Directory to save results')
    parser.add_argument('--metrics', type=str, nargs='+', 
                       default=['cosine', 'euclidean', 'gaussian', 'manhattan', 'norm2'],
                       help='Distance metrics to benchmark')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plot (only save to file)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 70)
    print("GraphSSM Distance Metric Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model dimension: {args.d_model}")
    print(f"Iterations: {args.num_iterations} (+ {args.warmup} warmup)")
    print(f"Metrics to test: {', '.join(args.metrics)}")
    print("=" * 70)
    
    # Run benchmarks
    results = []
    for i, metric in enumerate(args.metrics, 1):
        print(f"\n[{i}/{len(args.metrics)}] Benchmarking: {metric.upper()}")
        print("-" * 70)
        
        try:
            result = benchmark_distance_metric(metric, args, args.num_iterations, args.warmup)
            results.append(result)
            
            print(f"  Mean time: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
            print(f"  Min time:  {result['min_ms']:.3f} ms")
            print(f"  Max time:  {result['max_ms']:.3f} ms")
            print(f"  Median:    {result['median_ms']:.3f} ms")
        except Exception as e:
            print(f"  ERROR: Failed to benchmark {metric}: {e}")
            continue
    
    if len(results) == 0:
        print("\nNo successful benchmarks. Exiting.")
        return
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['mean_ms'])
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    
    print(f"FASTEST: {fastest['metric']} ({fastest['mean_ms']:.3f} ms)")
    print(f"SLOWEST: {slowest['metric']} ({slowest['mean_ms']:.3f} ms)")
    print(f"Speedup: {slowest['mean_ms'] / fastest['mean_ms']:.3f}x faster")
    
    print("\nRankings (fastest to slowest):")
    for i, r in enumerate(sorted_results, 1):
        speedup = fastest['mean_ms'] / r['mean_ms']
        print(f"  {i}. {r['metric']:12s} - {r['mean_ms']:7.3f} ms ({speedup:.3f}x)")
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'benchmark_results.csv')
    df = pd.DataFrame([{
        'metric': r['metric'],
        'mean_ms': r['mean_ms'],
        'std_ms': r['std_ms'],
        'min_ms': r['min_ms'],
        'max_ms': r['max_ms'],
        'median_ms': r['median_ms']
    } for r in results])
    df = df.sort_values('mean_ms')  # Sort by speed
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    create_visualization(results, args)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()