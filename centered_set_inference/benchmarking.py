"""
Benchmarking module for measuring efficiency of Centered Set Inference.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import psutil
import os
from sentence_transformers import SentenceTransformer
from .simplified_testing import SimplifiedValueCenter

def measure_performance(func: Callable, *args, **kwargs) -> Dict:
    """
    Measure performance metrics of a function.
    
    Args:
        func: Function to measure
        args, kwargs: Arguments to pass to the function
        
    Returns:
        Dictionary with performance metrics
    """
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Get final memory usage
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = end_memory - start_memory
    
    return {
        "execution_time": execution_time,
        "memory_used": memory_used,
        "result": result
    }

def benchmark_value_center_creation(value_counts: List[int], 
                                   description_lengths: List[int],
                                   embedding_model: str = "all-MiniLM-L6-v2") -> Dict:
    """
    Benchmark the creation of value centers with different numbers of values.
    
    Args:
        value_counts: List of different numbers of values to test
        description_lengths: List of different description lengths to test
        embedding_model: Name of the embedding model to use
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "value_counts": value_counts,
        "description_lengths": description_lengths,
        "creation_times": np.zeros((len(value_counts), len(description_lengths))),
        "memory_usage": np.zeros((len(value_counts), len(description_lengths)))
    }
    
    # Generate placeholder descriptions of different lengths
    descriptions = {}
    for length in description_lengths:
        descriptions[length] = "x" * length
    
    for i, count in enumerate(value_counts):
        for j, length in enumerate(description_lengths):
            # Create values dictionary
            values = {f"value_{k}": descriptions[length] for k in range(count)}
            
            # Measure performance
            perf = measure_performance(SimplifiedValueCenter, values, embedding_model)
            
            # Store results
            results["creation_times"][i, j] = perf["execution_time"]
            results["memory_usage"][i, j] = perf["memory_used"]
    
    return results

def benchmark_alignment_measurement(center: SimplifiedValueCenter, 
                                   text_lengths: List[int],
                                   num_iterations: int = 10) -> Dict:
    """
    Benchmark the measurement of alignment for texts of different lengths.
    
    Args:
        center: The value center to use
        text_lengths: List of different text lengths to test
        num_iterations: Number of iterations for each measurement
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "text_lengths": text_lengths,
        "measurement_times": [],
        "memory_usage": []
    }
    
    # Generate texts of different lengths
    texts = {length: "x" * length for length in text_lengths}
    
    for length in text_lengths:
        text = texts[length]
        
        # Measure average performance over multiple iterations
        total_time = 0
        total_memory = 0
        
        for _ in range(num_iterations):
            perf = measure_performance(center.measure_alignment, text)
            total_time += perf["execution_time"]
            total_memory += perf["memory_used"]
        
        # Store average results
        results["measurement_times"].append(total_time / num_iterations)
        results["memory_usage"].append(total_memory / num_iterations)
    
    return results

def compare_bounded_vs_centered(num_samples: int = 100, 
                               text_length: int = 200) -> Dict:
    """
    Compare efficiency of bounded set vs. centered set approaches.
    
    Args:
        num_samples: Number of text samples to process
        text_length: Length of each text sample
        
    Returns:
        Dictionary with comparison results
    """
    # Generate random text samples
    texts = [f"Sample text {i} " + "x" * text_length for i in range(num_samples)]
    
    # Define a simple value center
    center = SimplifiedValueCenter({
        "value1": "Description of value 1",
        "value2": "Description of value 2",
        "value3": "Description of value 3"
    })
    
    # Simulate bounded set approach (binary classification)
    def bounded_set_classify(texts):
        results = []
        for text in texts:
            # Simple threshold-based classification
            scores = center.measure_alignment(text)
            classification = "acceptable" if scores["overall"] > 0.5 else "unacceptable"
            results.append(classification)
        return results
    
    # Simulate centered set approach (directional alignment)
    def centered_set_measure(texts):
        results = []
        for text in texts:
            # Full alignment measurement
            scores = center.measure_alignment(text)
            results.append(scores)
        return results
    
    # Measure performance
    bounded_perf = measure_performance(bounded_set_classify, texts)
    centered_perf = measure_performance(centered_set_measure, texts)
    
    return {
        "bounded_set": {
            "execution_time": bounded_perf["execution_time"],
            "memory_used": bounded_perf["memory_used"],
            "time_per_sample": bounded_perf["execution_time"] / num_samples
        },
        "centered_set": {
            "execution_time": centered_perf["execution_time"],
            "memory_used": centered_perf["memory_used"],
            "time_per_sample": centered_perf["execution_time"] / num_samples
        },
        "ratio": {
            "execution_time": centered_perf["execution_time"] / bounded_perf["execution_time"],
            "memory_used": centered_perf["memory_used"] / bounded_perf["memory_used"]
        }
    }

def plot_benchmark_results(results: Dict, title: str, xlabel: str, ylabel: str):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    
    if "value_counts" in results and "description_lengths" in results:
        # Plot heatmap for 2D results
        plt.imshow(results["creation_times"], cmap="viridis")
        plt.colorbar(label="Execution Time (s)")
        plt.xlabel(xlabel)
        plt.ylabel("Number of Values")
        plt.title(title)
        plt.xticks(range(len(results["description_lengths"])), results["description_lengths"])
        plt.yticks(range(len(results["value_counts"])), results["value_counts"])
    else:
        # Plot line chart for 1D results
        plt.plot(results["text_lengths"], results["measurement_times"], marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comparison_results(results: Dict):
    """Plot comparison results between bounded and centered approaches."""
    # Bar chart for execution time
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(["Bounded Set", "Centered Set"], 
           [results["bounded_set"]["execution_time"], results["centered_set"]["execution_time"]])
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Comparison")
    
    # Bar chart for memory usage
    plt.subplot(1, 2, 2)
    plt.bar(["Bounded Set", "Centered Set"], 
           [results["bounded_set"]["memory_used"], results["centered_set"]["memory_used"]])
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed Comparison Results:")
    print(f"Bounded Set Approach:")
    print(f"  Execution Time: {results['bounded_set']['execution_time']:.4f} seconds")
    print(f"  Memory Used: {max(0, results['bounded_set']['memory_used']):.2f} MB")  # Avoid negative values
    print(f"  Time per Sample: {results['bounded_set']['time_per_sample']*1000:.2f} ms")
    
    print(f"\nCentered Set Approach:")
    print(f"  Execution Time: {results['centered_set']['execution_time']:.4f} seconds")
    print(f"  Memory Used: {max(0, results['centered_set']['memory_used']):.2f} MB")  # Avoid negative values
    print(f"  Time per Sample: {results['centered_set']['time_per_sample']*1000:.2f} ms")
    
    print(f"\nCentered/Bounded Ratio:")
    print(f"  Execution Time: {results['ratio']['execution_time']:.2f}x")
    # Avoid division by zero or negative values
    if results['bounded_set']['memory_used'] > 0:
        print(f"  Memory Used: {results['ratio']['memory_used']:.2f}x")
    else:
        print("  Memory Used: N/A (measurement issue)")

def estimate_realistic_production_costs(
    requests_per_day: int,
    avg_text_length: int = 200,
    llm_cost_per_1k_tokens: float = 0.002,  # Cost for a model like GPT-3.5-Turbo
    embedding_cost_per_1k_tokens: float = 0.0001,  # Cost for embedding models
    avg_tokens_per_request: int = 500,  # Average tokens per request (prompt + response)
    bounded_retry_rate: float = 0.15,  # Percentage of bounded approach responses that need retry
    centered_improvement_rate: float = 0.50,  # Percentage of centered approach responses that need improvement
    avg_improvement_attempts: float = 1.5  # Average number of improvement attempts when needed
) -> Dict:
    """
    Estimate realistic production costs including LLM generation costs.
    
    Args:
        requests_per_day: Number of requests per day
        avg_text_length: Average text length in characters
        llm_cost_per_1k_tokens: Cost per 1000 tokens for LLM generation
        embedding_cost_per_1k_tokens: Cost per 1000 tokens for embeddings
        avg_tokens_per_request: Average tokens per request
        bounded_retry_rate: Percentage of bounded approach responses that need retry
        centered_improvement_rate: Percentage of centered approach responses that need improvement
        avg_improvement_attempts: Average number of improvement attempts when needed
        
    Returns:
        Dictionary with cost estimates
    """
    # Calculate tokens per day
    tokens_per_day = requests_per_day * avg_tokens_per_request / 1000  # in thousands
    
    # Bounded Set Approach
    bounded_initial_generations = requests_per_day
    bounded_retries = requests_per_day * bounded_retry_rate
    bounded_total_generations = bounded_initial_generations + bounded_retries
    bounded_total_tokens = bounded_total_generations * avg_tokens_per_request / 1000
    
    bounded_generation_cost = bounded_total_tokens * llm_cost_per_1k_tokens
    bounded_embedding_cost = tokens_per_day * embedding_cost_per_1k_tokens
    bounded_total_cost = bounded_generation_cost + bounded_embedding_cost
    
    # Centered Set Approach
    centered_initial_generations = requests_per_day
    centered_improvements_needed = requests_per_day * centered_improvement_rate
    centered_additional_generations = centered_improvements_needed * avg_improvement_attempts
    centered_total_generations = centered_initial_generations + centered_additional_generations
    centered_total_tokens = centered_total_generations * avg_tokens_per_request / 1000
    
    centered_generation_cost = centered_total_tokens * llm_cost_per_1k_tokens
    centered_embedding_cost = tokens_per_day * embedding_cost_per_1k_tokens * 1.2  # 20% more embedding evaluations
    centered_total_cost = centered_generation_cost + centered_embedding_cost
    
    # Calculate monthly costs (30 days)
    bounded_monthly_cost = bounded_total_cost * 30
    centered_monthly_cost = centered_total_cost * 30
    
    return {
        "requests_per_day": requests_per_day,
        "avg_text_length": avg_text_length,
        "bounded_set": {
            "initial_generations": bounded_initial_generations,
            "retries": bounded_retries,
            "total_generations": bounded_total_generations,
            "generation_cost": bounded_generation_cost,
            "embedding_cost": bounded_embedding_cost,
            "total_daily_cost": bounded_total_cost,
            "monthly_cost": bounded_monthly_cost
        },
        "centered_set": {
            "initial_generations": centered_initial_generations,
            "improvements_needed": centered_improvements_needed,
            "additional_generations": centered_additional_generations,
            "total_generations": centered_total_generations,
            "generation_cost": centered_generation_cost,
            "embedding_cost": centered_embedding_cost,
            "total_daily_cost": centered_total_cost,
            "monthly_cost": centered_monthly_cost
        },
        "cost_difference": {
            "daily": centered_total_cost - bounded_total_cost,
            "monthly": centered_monthly_cost - bounded_monthly_cost,
            "ratio": centered_total_cost / bounded_total_cost
        }
    }

def print_realistic_cost_estimates(cost_estimates: Dict):
    """Print realistic cost estimates in a readable format."""
    print("\n=== Realistic Production Cost Estimates ===")
    print(f"Based on {cost_estimates['requests_per_day']:,} requests per day")
    
    print("\nBounded Set Approach:")
    print(f"  Initial Generations: {cost_estimates['bounded_set']['initial_generations']:,}")
    print(f"  Retries Due to Rejection: {cost_estimates['bounded_set']['retries']:,}")
    print(f"  Total Generations: {cost_estimates['bounded_set']['total_generations']:,}")
    print(f"  Generation Cost: ${cost_estimates['bounded_set']['generation_cost']:.2f}")
    print(f"  Embedding/Evaluation Cost: ${cost_estimates['bounded_set']['embedding_cost']:.2f}")
    print(f"  Total Daily Cost: ${cost_estimates['bounded_set']['total_daily_cost']:.2f}")
    print(f"  Monthly Cost: ${cost_estimates['bounded_set']['monthly_cost']:.2f}")
    
    print("\nCentered Set Approach:")
    print(f"  Initial Generations: {cost_estimates['centered_set']['initial_generations']:,}")
    print(f"  Responses Needing Improvement: {cost_estimates['centered_set']['improvements_needed']:,}")
    print(f"  Additional Generations for Improvement: {cost_estimates['centered_set']['additional_generations']:,}")
    print(f"  Total Generations: {cost_estimates['centered_set']['total_generations']:,}")
    print(f"  Generation Cost: ${cost_estimates['centered_set']['generation_cost']:.2f}")
    print(f"  Embedding/Evaluation Cost: ${cost_estimates['centered_set']['embedding_cost']:.2f}")
    print(f"  Total Daily Cost: ${cost_estimates['centered_set']['total_daily_cost']:.2f}")
    print(f"  Monthly Cost: ${cost_estimates['centered_set']['monthly_cost']:.2f}")
    
    print("\nCost Difference:")
    diff = cost_estimates['cost_difference']['daily']
    if diff > 0:
        print(f"  The centered set approach costs ${diff:.2f} more per day")
    else:
        print(f"  The centered set approach saves ${-diff:.2f} per day")
    
    print(f"  Cost Ratio: {cost_estimates['cost_difference']['ratio']:.2f}x")
    
    # Add recommendation
    if cost_estimates['cost_difference']['ratio'] < 1.2:
        print("\nRecommendation: The additional cost of the centered set approach is minimal")
        print("compared to the benefits of more nuanced alignment. Consider using the centered set approach.")
    elif cost_estimates['cost_difference']['ratio'] < 1.5:
        print("\nRecommendation: The centered set approach is moderately more expensive.")
        print("The improved alignment quality likely justifies this additional cost.")
    elif cost_estimates['cost_difference']['ratio'] < 2.0:
        print("\nRecommendation: The centered set approach is significantly more expensive.")
        print("Consider using it for high-value interactions where alignment quality is critical.")
    else:
        print("\nRecommendation: The centered set approach is substantially more expensive.")
        print("Consider a hybrid approach or optimizing the implementation to reduce costs.")

def plot_cost_comparison(cost_estimates: Dict):
    """Plot cost comparison between bounded and centered approaches."""
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot generation counts
    plt.subplot(2, 2, 1)
    labels = ['Initial', 'Retries/Improvements', 'Total']
    bounded_values = [
        cost_estimates['bounded_set']['initial_generations'],
        cost_estimates['bounded_set']['retries'],
        cost_estimates['bounded_set']['total_generations']
    ]
    centered_values = [
        cost_estimates['centered_set']['initial_generations'],
        cost_estimates['centered_set']['additional_generations'],
        cost_estimates['centered_set']['total_generations']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, bounded_values, width, label='Bounded Set')
    plt.bar(x + width/2, centered_values, width, label='Centered Set')
    
    plt.ylabel('Number of Generations')
    plt.title('Generation Counts')
    plt.xticks(x, labels)
    plt.legend()
    
    # Plot daily costs breakdown
    plt.subplot(2, 2, 2)
    labels = ['Generation', 'Embedding', 'Total']
    bounded_values = [
        cost_estimates['bounded_set']['generation_cost'],
        cost_estimates['bounded_set']['embedding_cost'],
        cost_estimates['bounded_set']['total_daily_cost']
    ]
    centered_values = [
        cost_estimates['centered_set']['generation_cost'],
        cost_estimates['centered_set']['embedding_cost'],
        cost_estimates['centered_set']['total_daily_cost']
    ]
    
    x = np.arange(len(labels))
    
    plt.bar(x - width/2, bounded_values, width, label='Bounded Set')
    plt.bar(x + width/2, centered_values, width, label='Centered Set')
    
    plt.ylabel('Daily Cost ($)')
    plt.title('Daily Cost Breakdown')
    plt.xticks(x, labels)
    plt.legend()
    
    # Plot monthly costs
    plt.subplot(2, 2, 3)
    labels = ['Bounded Set', 'Centered Set']
    values = [
        cost_estimates['bounded_set']['monthly_cost'],
        cost_estimates['centered_set']['monthly_cost']
    ]
    
    plt.bar(labels, values)
    plt.ylabel('Monthly Cost ($)')
    plt.title('Monthly Cost Comparison')
    
    # Plot cost ratio at different request volumes
    plt.subplot(2, 2, 4)
    request_volumes = [1000, 10000, 100000, 1000000]
    ratios = []
    
    for volume in request_volumes:
        est = estimate_realistic_production_costs(requests_per_day=volume)
        ratios.append(est['cost_difference']['ratio'])
    
    plt.plot(request_volumes, ratios, marker='o')
    plt.xscale('log')
    plt.xlabel('Requests per Day')
    plt.ylabel('Cost Ratio (Centered/Bounded)')
    plt.title('Cost Ratio vs. Request Volume')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run benchmarks and display results."""
    print("=== Centered Set Inference Benchmarking ===")
    
    # Benchmark value center creation
    print("\nBenchmarking value center creation...")
    creation_results = benchmark_value_center_creation(
        value_counts=[3, 5, 10],
        description_lengths=[50, 100, 200]
    )
    plot_benchmark_results(creation_results, "Value Center Creation Performance", 
                          "Description Length", "Execution Time (s)")
    
    # Create a value center for further benchmarks
    print("\nCreating value center for alignment benchmarks...")
    center = SimplifiedValueCenter({
        "value1": "Description of value 1",
        "value2": "Description of value 2",
        "value3": "Description of value 3"
    })
    
    # Benchmark alignment measurement
    print("\nBenchmarking alignment measurement...")
    alignment_results = benchmark_alignment_measurement(
        center=center,
        text_lengths=[100, 200, 500, 1000]
    )
    plot_benchmark_results(alignment_results, "Alignment Measurement Performance", 
                          "Text Length (chars)", "Execution Time (s)")
    
    # Compare bounded vs. centered approaches (embedding only)
    print("\nComparing bounded set vs. centered set approaches (embedding only)...")
    comparison_results = compare_bounded_vs_centered(num_samples=50)
    plot_comparison_results(comparison_results)
    
    # Estimate realistic production costs including LLM generation
    print("\nEstimating realistic production costs including LLM generation...")
    
    # Different request volumes
    for requests in [1000, 10000, 100000]:
        # Default parameters
        cost_estimates = estimate_realistic_production_costs(
            requests_per_day=requests,
            avg_text_length=200,
            llm_cost_per_1k_tokens=0.002,  # $0.002 per 1K tokens (GPT-3.5-Turbo)
            embedding_cost_per_1k_tokens=0.0001,  # $0.0001 per 1K tokens
            avg_tokens_per_request=500,  # 500 tokens per request
            bounded_retry_rate=0.15,  # 15% of bounded responses need retry
            centered_improvement_rate=0.50,  # 50% of centered responses need improvement
            avg_improvement_attempts=1.5  # 1.5 improvement attempts on average
        )
        print_realistic_cost_estimates(cost_estimates)
    
    # Plot detailed cost comparison for 10,000 requests/day
    detailed_cost = estimate_realistic_production_costs(requests_per_day=10000)
    plot_cost_comparison(detailed_cost)
    
    # Sensitivity analysis
    print("\n=== Sensitivity Analysis ===")
    print("How does the cost ratio change with different improvement rates?")
    
    improvement_rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ratios = []
    
    for rate in improvement_rates:
        est = estimate_realistic_production_costs(
            requests_per_day=10000,
            centered_improvement_rate=rate
        )
        ratios.append(est['cost_difference']['ratio'])
        print(f"Improvement rate {rate*100:.0f}%: Cost ratio {est['cost_difference']['ratio']:.2f}x")
    
    plt.figure(figsize=(8, 5))
    plt.plot(improvement_rates, ratios, marker='o')
    plt.xlabel('Centered Set Improvement Rate')
    plt.ylabel('Cost Ratio (Centered/Bounded)')
    plt.title('Cost Sensitivity to Improvement Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 