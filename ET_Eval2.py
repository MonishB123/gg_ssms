"""
GraphSSM Distance Metrics Performance Testing Script

This script tests the performance and speed of different distance formulas
used in the GraphSSM model. It generates synthetic data to benchmark:
- norm2_distance
- cosine_distance  
- gaussian_distance
- euclidean_distance
- manhattan_distance

Results are visualized in matplotlib charts showing computation time and
accuracy metrics for each distance function.
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
GraphSSM Distance Metrics Performance Testing Script

This script tests the performance and speed of different distance formulas
used in the GraphSSM model. It generates synthetic data to benchmark:
- norm2_distance
- cosine_distance  
- gaussian_distance
- euclidean_distance
- manhattan_distance

Results are visualized in matplotlib charts showing computation time and
accuracy metrics for each distance function.
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add the GraphSSM path
gg_ssms_path = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(gg_ssms_path, "core", "graph_ssm")
if os.path.exists(core_path):
    sys.path.append(core_path)
    from main import GraphSSM, norm2_distance, cosine_distance, gaussian_distance, euclidean_distance, manhattan_distance
else:
    print(f"Warning: Could not find core/graph_ssm at {core_path}")
    print("Importing from current directory...")
    # Try to import from current directory structure
    try:
        from core.graph_ssm.main import GraphSSM, norm2_distance, cosine_distance, gaussian_distance, euclidean_distance, manhattan_distance
    except ImportError:
        print("Could not import GraphSSM. Please ensure the core/graph_ssm/main.py file exists.")
        sys.exit(1)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(batch_size: int, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Generate synthetic data for testing distance functions"""
    # Create data with some structure to make distance calculations meaningful
    data = torch.randn(batch_size, d_model, seq_len, device=device)
    
    # Add some temporal structure
    for i in range(1, seq_len):
        # Each timestep is somewhat similar to the previous one
        data[:, :, i] = 0.7 * data[:, :, i-1] + 0.3 * data[:, :, i]
    
    return data


def time_distance_function(distance_fn, data1: torch.Tensor, data2: torch.Tensor, num_runs: int = 100) -> Tuple[float, torch.Tensor]:
    """Time a distance function and return average execution time and result"""
    # Warm up
    for _ in range(10):
        _ = distance_fn(data1, data2)
    
    # Synchronize GPU if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time the function
    start_time = time.time()
    
    for _ in range(num_runs):
        result = distance_fn(data1, data2)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time, result


def test_distance_accuracy(distance_functions: Dict, test_data: torch.Tensor) -> Dict:
    """Test accuracy/behavior of distance functions on synthetic data"""
    results = {}
    
    # Create test pairs with known relationships
    seq_len = test_data.shape[-1]
    
    # Test 1: Identical vectors should have minimum distance (or maximum similarity)
    identical_data = test_data[:, :, :2]  # Take first two timesteps
    
    # Test 2: Opposite vectors should have maximum distance
    opposite_data = torch.stack([test_data[:, :, 0], -test_data[:, :, 0]], dim=-1)
    
    # Test 3: Gradually changing vectors
    gradual_data = torch.stack([
        test_data[:, :, 0], 
        test_data[:, :, 0] * 0.9,
        test_data[:, :, 0] * 0.5,
        test_data[:, :, 0] * 0.1
    ], dim=-1)
    
    for name, dist_fn in distance_functions.items():
        results[name] = {}
        
        # Test identical vectors
        identical_dist = dist_fn(identical_data[:, :, :1], identical_data[:, :, 1:2])
        results[name]['identical'] = identical_dist.mean().item()
        
        # Test opposite vectors  
        opposite_dist = dist_fn(opposite_data[:, :, :1], opposite_data[:, :, 1:2])
        results[name]['opposite'] = opposite_dist.mean().item()
        
        # Test gradual change
        gradual_dists = []
        for i in range(3):
            dist = dist_fn(gradual_data[:, :, :1], gradual_data[:, :, i+1:i+2])
            gradual_dists.append(dist.mean().item())
        results[name]['gradual'] = gradual_dists
        
        # Compute variance (measure of discrimination ability)
        all_dists = [identical_dist.mean().item(), opposite_dist.mean().item()] + gradual_dists
        results[name]['variance'] = np.var(all_dists)
    
    return results


def benchmark_distance_functions(batch_sizes: List[int], seq_lens: List[int], d_model: int = 64, device: torch.device = torch.device('cpu')) -> Dict:
    """Benchmark all distance functions across different input sizes"""
    
    distance_functions = {
        'norm2': norm2_distance,
        'cosine': cosine_distance,
        'gaussian': gaussian_distance,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance
    }
    
    results = {
        'batch_sizes': batch_sizes,
        'seq_lens': seq_lens,
        'times': {name: [] for name in distance_functions.keys()},
        'accuracy': {},
        'memory_usage': {name: [] for name in distance_functions.keys()}
    }
    
    print(f"Benchmarking distance functions on {device}")
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Testing sequence lengths: {seq_lens}")
    print("-" * 50)
    
    # Test different configurations
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
            
            # Generate test data
            test_data = generate_synthetic_data(batch_size, seq_len, d_model, device)
            
            # Create pairs for distance computation
            data1 = test_data[:, :, :-1]  # All but last timestep
            data2 = test_data[:, :, 1:]   # All but first timestep
            
            batch_times = {name: [] for name in distance_functions.keys()}
            
            for name, dist_fn in distance_functions.items():
                try:
                    # Measure memory before
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        mem_before = torch.cuda.memory_allocated()
                    
                    # Time the function
                    avg_time, _ = time_distance_function(dist_fn, data1, data2, num_runs=50)
                    batch_times[name].append(avg_time)
                    
                    # Measure memory after
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated()
                        mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
                        results['memory_usage'][name].append(mem_used)
                    else:
                        results['memory_usage'][name].append(0)
                    
                    print(f"  {name}: {avg_time*1000:.3f}ms")
                    
                except Exception as e:
                    print(f"  {name}: ERROR - {e}")
                    batch_times[name].append(float('inf'))
                    results['memory_usage'][name].append(0)
            
            # Store results
            for name in distance_functions.keys():
                results['times'][name].extend(batch_times[name])
            
            print()
    
    # Test accuracy on a standard configuration
    print("Testing accuracy/behavior...")
    test_data = generate_synthetic_data(2, 10, d_model, device)
    accuracy_results = test_distance_accuracy(distance_functions, test_data)
    results['accuracy'] = accuracy_results
    
    return results


def plot_performance_results(results: Dict, save_path: str = None):
    """Create comprehensive plots of the performance results"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GraphSSM Distance Functions Performance Analysis', fontsize=16, fontweight='bold')
    
    # Colors for each distance function
    colors = {
        'norm2': '#1f77b4',
        'cosine': '#ff7f0e', 
        'gaussian': '#2ca02c',
        'euclidean': '#d62728',
        'manhattan': '#9467bd'
    }
    
    # 1. Execution Time Comparison (Bar Chart)
    ax1 = axes[0, 0]
    dist_names = list(results['times'].keys())
    avg_times = [np.mean(results['times'][name]) * 1000 for name in dist_names]  # Convert to ms
    bars = ax1.bar(dist_names, avg_times, color=[colors[name] for name in dist_names])
    ax1.set_title('Average Execution Time', fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xlabel('Distance Function')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    # 2. Execution Time Distribution (Box Plot)
    ax2 = axes[0, 1]
    time_data = [np.array(results['times'][name]) * 1000 for name in dist_names]
    bp = ax2.boxplot(time_data, labels=dist_names, patch_artist=True)
    for patch, name in zip(bp['boxes'], dist_names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    ax2.set_title('Execution Time Distribution', fontweight='bold')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xlabel('Distance Function')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Memory Usage Comparison
    ax3 = axes[0, 2]
    if any(sum(results['memory_usage'][name]) > 0 for name in dist_names):
        avg_memory = [np.mean(results['memory_usage'][name]) for name in dist_names]
        bars = ax3.bar(dist_names, avg_memory, color=[colors[name] for name in dist_names])
        ax3.set_title('Average Memory Usage', fontweight='bold')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xlabel('Distance Function')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mem_val in zip(bars, avg_memory):
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{mem_val:.1f}MB', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Memory usage data\nnot available\n(CPU mode)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Memory Usage (N/A)', fontweight='bold')
    
    # 4. Accuracy Test: Distance for Identical vs Opposite Vectors
    ax4 = axes[1, 0]
    identical_scores = [results['accuracy'][name]['identical'] for name in dist_names]
    opposite_scores = [results['accuracy'][name]['opposite'] for name in dist_names]
    
    x = np.arange(len(dist_names))
    width = 0.35
    bars1 = ax4.bar(x - width/2, identical_scores, width, label='Identical Vectors', alpha=0.8)
    bars2 = ax4.bar(x + width/2, opposite_scores, width, label='Opposite Vectors', alpha=0.8)
    
    ax4.set_title('Distance Values: Identical vs Opposite Vectors', fontweight='bold')
    ax4.set_ylabel('Distance Value')
    ax4.set_xlabel('Distance Function')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dist_names, rotation=45)
    ax4.legend()
    
    # 5. Discrimination Ability (Variance)
    ax5 = axes[1, 1]
    variances = [results['accuracy'][name]['variance'] for name in dist_names]
    bars = ax5.bar(dist_names, variances, color=[colors[name] for name in dist_names])
    ax5.set_title('Discrimination Ability (Variance)', fontweight='bold')
    ax5.set_ylabel('Variance in Distance Values')
    ax5.set_xlabel('Distance Function')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, var_val in zip(bars, variances):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{var_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Gradual Change Response
    ax6 = axes[1, 2]
    for name in dist_names:
        gradual_values = results['accuracy'][name]['gradual']
        similarity_levels = ['90%', '50%', '10%']
        ax6.plot(similarity_levels, gradual_values, marker='o', linewidth=2, 
                label=name, color=colors[name])
    
    ax6.set_title('Response to Gradual Changes', fontweight='bold')
    ax6.set_ylabel('Distance Value')
    ax6.set_xlabel('Vector Similarity Level')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {save_path}")
    
    plt.show()


def print_summary_report(results: Dict):
    """Print a detailed summary report of the benchmark results"""
    print("\n" + "="*70)
    print("GRAPHSSM DISTANCE FUNCTIONS PERFORMANCE REPORT")
    print("="*70)
    
    dist_names = list(results['times'].keys())
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 30)
    
    # Find fastest and slowest
    avg_times = {name: np.mean(results['times'][name]) * 1000 for name in dist_names}
    fastest = min(avg_times.items(), key=lambda x: x[1])
    slowest = max(avg_times.items(), key=lambda x: x[1])
    
    print(f"🏆 Fastest: {fastest[0]} ({fastest[1]:.3f}ms)")
    print(f"🐌 Slowest: {slowest[0]} ({slowest[1]:.3f}ms)")
    print(f"📈 Speed Ratio: {slowest[1]/fastest[1]:.1f}x")
    
    print(f"\n⏱️  Detailed Execution Times:")
    for name in dist_names:
        avg_time = avg_times[name]
        std_time = np.std(results['times'][name]) * 1000
        print(f"   {name:12}: {avg_time:7.3f}ms ± {std_time:5.3f}ms")
    
    # Accuracy Analysis
    print(f"\n🎯 ACCURACY & BEHAVIOR ANALYSIS")
    print("-" * 35)
    
    print("Distance Values for Test Cases:")
    print(f"{'Function':<12} {'Identical':<10} {'Opposite':<10} {'Variance':<10}")
    print("-" * 45)
    
    for name in dist_names:
        identical = results['accuracy'][name]['identical']
        opposite = results['accuracy'][name]['opposite']
        variance = results['accuracy'][name]['variance']
        print(f"{name:<12} {identical:8.4f}   {opposite:8.4f}   {variance:8.4f}")
    
    # Best discrimination
    variances = {name: results['accuracy'][name]['variance'] for name in dist_names}
    best_discrimination = max(variances.items(), key=lambda x: x[1])
    
    print(f"\n🎯 Best Discrimination: {best_discrimination[0]} (variance: {best_discrimination[1]:.4f})")
    
    # Memory Usage
    if any(sum(results['memory_usage'][name]) > 0 for name in dist_names):
        print(f"\n💾 MEMORY USAGE")
        print("-" * 15)
        for name in dist_names:
            avg_mem = np.mean(results['memory_usage'][name])
            if avg_mem > 0:
                print(f"   {name:12}: {avg_mem:6.2f}MB")
            else:
                print(f"   {name:12}: N/A")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 18)
    print(f"• For speed: Use '{fastest[0]}' (fastest execution)")
    print(f"• For discrimination: Use '{best_discrimination[0]}' (highest variance)")
    
    # Determine best overall
    # Score based on speed (inverse) and discrimination ability
    scores = {}
    for name in dist_names:
        speed_score = fastest[1] / avg_times[name]  # Higher is better
        discrimination_score = variances[name] / best_discrimination[1]  # Higher is better
        overall_score = (speed_score + discrimination_score) / 2
        scores[name] = overall_score
    
    best_overall = max(scores.items(), key=lambda x: x[1])
    print(f"• Overall best: '{best_overall[0]}' (balanced speed & accuracy)")
    
    print("\n" + "="*70)


def main():
    """Main function to run the distance function performance tests"""
    parser = argparse.ArgumentParser(description="Benchmark GraphSSM distance functions")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4, 8, 16], 
                       help="Batch sizes to test")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[8, 16, 32, 64], 
                       help="Sequence lengths to test")
    parser.add_argument("--d_model", type=int, default=64, 
                       help="Model dimension")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument("--save_plot", type=str, default="distance_performance.png", 
                       help="Path to save the performance plot")
    parser.add_argument("--no_plot", action="store_true", 
                       help="Skip plotting")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Run benchmarks
    print("Starting distance function benchmarks...")
    results = benchmark_distance_functions(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens, 
        d_model=args.d_model,
        device=device
    )
    
    # Print summary report
    print_summary_report(results)
    
    # Create plots
    if not args.no_plot:
        plot_performance_results(results, args.save_plot)
    
    print("\nBenchmarking completed!")


if __name__ == "__main__":
    main()