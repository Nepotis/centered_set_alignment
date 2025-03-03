"""
Evaluation and benchmarking module for the Centered Set Inference Engine.
"""

import torch
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .architecture import ValueCenter, AlignmentHead, CenteredSetInferenceEngine

class AlignmentBenchmark:
    """Benchmark for evaluating alignment approaches."""
    
    def __init__(self, 
                 test_data_path: str,
                 centered_engine: CenteredSetInferenceEngine,
                 baseline_model = None,
                 baseline_tokenizer = None):
        """
        Initialize the benchmark.
        
        Args:
            test_data_path: Path to test data JSON
            centered_engine: The CSIE to evaluate
            baseline_model: Optional baseline model for comparison
            baseline_tokenizer: Tokenizer for the baseline model
        """
        self.centered_engine = centered_engine
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer
        
        # Load test data
        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)
        
        # Extract value names
        self.value_names = centered_engine.value_center.get_value_names()
    
    def run_benchmark(self, output_path: str = "./benchmark_results.json"):
        """
        Run the benchmark and save results.
        
        Args:
            output_path: Path to save benchmark results
        """
        results = {
            "centered_approach": [],
            "baseline_approach": [] if self.baseline_model else None
        }
        
        # Run centered approach
        print("Evaluating centered approach...")
        for item in tqdm(self.test_data):
            prompt = item["prompt"]
            
            # Generate response with centered approach
            response, scores = self.centered_engine.generate_aligned_response(prompt)
            
            # Store result
            result = {
                "prompt": prompt,
                "response": response,
                "alignment_scores": scores,
                "ground_truth_scores": item.get("alignment_scores", {})
            }
            results["centered_approach"].append(result)
        
        # Run baseline approach if available
        if self.baseline_model:
            print("Evaluating baseline approach...")
            for item in tqdm(self.test_data):
                prompt = item["prompt"]
                
                # Generate response with baseline
                inputs = self.baseline_tokenizer(prompt, return_tensors="pt")
                outputs = self.baseline_model.generate(**inputs, max_length=512)
                response = self.baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Evaluate with centered engine for comparison
                scores = self.centered_engine.evaluate_response(prompt, response)
                
                # Store result
                result = {
                    "prompt": prompt,
                    "response": response,
                    "alignment_scores": scores,
                    "ground_truth_scores": item.get("alignment_scores", {})
                }
                results["baseline_approach"].append(result)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_results(self, results_path: Optional[str] = None):
        """
        Analyze benchmark results and generate visualizations.
        
        Args:
            results_path: Path to results JSON (if not using run_benchmark output)
        """
        # Load results if path provided
        if results_path:
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            # Run benchmark if not already run
            results = self.run_benchmark()
        
        # Extract scores
        centered_scores = {
            name: [item["alignment_scores"][name] 
                  for item in results["centered_approach"]]
            for name in self.value_names + ["overall"]
        }
        
        if results["baseline_approach"]:
            baseline_scores = {
                name: [item["alignment_scores"][name] 
                      for item in results["baseline_approach"]]
                for name in self.value_names + ["overall"]
            }
        else:
            baseline_scores = None
        
        # Calculate statistics
        stats = {
            "centered_approach": {
                name: {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }
                for name, scores in centered_scores.items()
            }
        }
        
        if baseline_scores:
            stats["baseline_approach"] = {
                name: {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }
                for name, scores in baseline_scores.items()
            }
        
        # Print summary
        print("=== Benchmark Results ===")
        print("\nCentered Approach:")
        for name, metrics in stats["centered_approach"].items():
            print(f"  {name}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}")
        
        if baseline_scores:
            print("\nBaseline Approach:")
            for name, metrics in stats["baseline_approach"].items():
                print(f"  {name}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}")
        
        # Create visualizations
        self._create_visualizations(centered_scores, baseline_scores)
        
        return stats
    
    def _create_visualizations(self, centered_scores, baseline_scores=None):
        """Create visualizations comparing approaches."""
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Bar chart comparing mean scores
        plt.subplot(2, 1, 1)
        x = np.arange(len(self.value_names) + 1)
        width = 0.35
        
        # Centered approach bars
        centered_means = [np.mean(centered_scores[name]) for name in self.value_names + ["overall"]]
        plt.bar(x - width/2 if baseline_scores else x, centered_means, width, label='Centered Approach')
        
        # Baseline approach bars if available
        if baseline_scores:
            baseline_means = [np.mean(baseline_scores[name]) for name in self.value_names + ["overall"]]
            plt.bar(x + width/2, baseline_means, width, label='Baseline Approach')
        
        plt.ylabel('Mean Alignment Score')
        plt.title('Comparison of Alignment Approaches')
        plt.xticks(x, self.value_names + ["overall"])
        plt.legend()
        
        # Distribution plot for overall scores
        plt.subplot(2, 1, 2)
        sns.kdeplot(centered_scores["overall"], label='Centered Approach')
        if baseline_scores:
            sns.kdeplot(baseline_scores["overall"], label='Baseline Approach')
        
        plt.xlabel('Overall Alignment Score')
        plt.ylabel('Density')
        plt.title('Distribution of Overall Alignment Scores')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("alignment_benchmark_results.png")
        plt.close()


def evaluate_on_standard_benchmarks(
    engine: CenteredSetInferenceEngine,
    benchmark_data: Dict[str, List[Dict]],
    output_path: str = "./standard_benchmark_results.json"
):
    """
    Evaluate the engine on standard benchmarks like TruthfulQA.
    
    Args:
        engine: The CSIE to evaluate
        benchmark_data: Dictionary mapping benchmark names to lists of examples
        output_path: Path to save results
    """
    results = {}
    
    for benchmark_name, examples in benchmark_data.items():
        print(f"Evaluating on {benchmark_name}...")
        benchmark_results = []
        
        for example in tqdm(examples):
            prompt = example["prompt"]
            reference_answer = example.get("reference_answer")
            
            # Generate response
            response, scores = engine.generate_aligned_response(prompt)
            
            # Store result
            result = {
                "prompt": prompt,
                "response": response,
                "alignment_scores": scores,
                "reference_answer": reference_answer,
                "correct": example.get("evaluate_func", lambda x, y: False)(response, reference_answer)
            }
            benchmark_results.append(result)
        
        # Calculate accuracy if possible
        if all("correct" in result for result in benchmark_results):
            accuracy = sum(result["correct"] for result in benchmark_results) / len(benchmark_results)
            print(f"{benchmark_name} Accuracy: {accuracy:.4f}")
        
        results[benchmark_name] = benchmark_results
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results 