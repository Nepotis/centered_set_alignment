"""
Main script to run the Centered Set Inference POC.
"""

import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .architecture import ValueCenter, AlignmentHead, CenteredSetInferenceEngine
from .training import train_alignment_head
from .evaluation import AlignmentBenchmark, evaluate_on_standard_benchmarks
from .use_cases.therapeutic_chatbot import create_chatbot_interface
from .utils.data_generator import generate_synthetic_data, generate_test_prompts

def setup_directories():
    """Create necessary directories for the project."""
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

def generate_training_data():
    """Generate synthetic training data if it doesn't exist."""
    if not os.path.exists("./data/synthetic_training_data.json"):
        # Create base examples
        base_examples = [
            {
                "prompt": "I've been feeling really down lately and I don't know why.",
                "response": "I'm sorry to hear you've been feeling down. It's common to experience periods of low mood, sometimes without an obvious cause. This could be related to stress, sleep patterns, or even biological factors. Would you like to talk more about what's been going on in your life recently? Sometimes discussing it can help identify patterns or triggers you might not have noticed."
            },
            {
                "prompt": "I had a panic attack yesterday and I'm scared it will happen again.",
                "response": "That sounds really frightening. Panic attacks can be incredibly overwhelming, and the fear of having another one is very common. Remember that panic attacks, while extremely uncomfortable, aren't dangerous. Some people find it helpful to practice deep breathing exercises or grounding techniques when they feel anxiety building. Have you spoken with a healthcare provider about this? They can offer specific strategies and support for managing panic attacks."
            },
            {
                "prompt": "My partner and I keep having the same argument over and over.",
                "response": "Recurring arguments can be really frustrating. It often happens when there's an underlying issue that isn't being fully addressed. Sometimes it helps to try discussing the topic at a neutral time, not in the heat of an argument. Using 'I' statements to express how you feel rather than 'you' statements that can sound accusatory might also help. Would you feel comfortable sharing what the argument is usually about? That might help us explore some different approaches."
            },
            {
                "prompt": "I can't stop thinking about a mistake I made at work.",
                "response": "It's really common to ruminate on mistakes, especially work-related ones. Our minds often fixate on negative experiences as a way to process them or try to prevent similar situations in the future. Try to remember that everyone makes mistakes - they're how we learn and grow. Would it help to talk about what happened? Sometimes articulating the situation can help put it in perspective."
            },
            {
                "prompt": "I feel like no one understands what I'm going through.",
                "response": "Feeling misunderstood can be incredibly isolating. It takes courage to share your experiences, and it can be disappointing when others don't seem to get it. Your feelings and experiences are valid, even if others haven't gone through the same things. Would you like to share more about what you're experiencing? I'm here to listen without judgment."
            }
        ]
        
        # Generate variations
        generate_synthetic_data(
            base_examples=base_examples,
            num_variations=10,
            output_path="./data/synthetic_training_data.json"
        )
    
    # Generate test prompts if they don't exist
    if not os.path.exists("./data/test_prompts.json"):
        generate_test_prompts(
            num_prompts=30,
            output_path="./data/test_prompts.json"
        )

def load_or_train_model(args):
    """
    Load or train the model based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (engine, tokenizer)
    """
    # Load base model
    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Define value center
    value_center = ValueCenter({
        "empathy": 1.0,
        "supportiveness": 1.0,
        "safety": 1.5,
        "helpfulness": 1.0,
        "truthfulness": 1.0
    })
    
    # Check if alignment head exists
    alignment_head_path = os.path.join("./models", "alignment_head.pt")
    
    if os.path.exists(alignment_head_path) and not args.retrain:
        print("Loading existing alignment head")
        input_dim = model.config.hidden_size
        alignment_head = AlignmentHead(input_dim, value_center)
        alignment_head.load_state_dict(torch.load(alignment_head_path))
    else:
        print("Training alignment head")
        # Train alignment head
        alignment_head = train_alignment_head(
            language_model=model,
            tokenizer=tokenizer,
            value_center=value_center,
            train_data_path="./data/synthetic_training_data.json",
            output_dir="./models",
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs
        )
    
    # Create inference engine
    engine = CenteredSetInferenceEngine(
        language_model=model,
        alignment_head=alignment_head,
        tokenizer=tokenizer,
        min_acceptable_score=args.min_score
    )
    
    return engine, tokenizer

def run_benchmark(engine, tokenizer, args):
    """
    Run benchmarks on the model.
    
    Args:
        engine: The CSIE engine
        tokenizer: Tokenizer for the model
        args: Command line arguments
    """
    if not args.benchmark:
        return
    
    print("Running benchmark...")
    
    # Load test prompts
    with open("./data/test_prompts.json", 'r') as f:
        test_data = json.load(f)
    
    # Create benchmark
    benchmark = AlignmentBenchmark(
        test_data_path="./data/test_prompts.json",
        centered_engine=engine
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        output_path="./results/benchmark_results.json"
    )
    
    # Analyze results
    stats = benchmark.analyze_results()
    
    # Save stats
    with open("./results/benchmark_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

def run_chatbot(engine, tokenizer, args):
    """
    Run the chatbot interface.
    
    Args:
        engine: The CSIE engine
        tokenizer: Tokenizer for the model
        args: Command line arguments
    """
    if not args.chatbot:
        return
    
    print("Starting chatbot interface...")
    demo = create_chatbot_interface(engine, tokenizer)
    demo.launch(share=args.share)

def main():
    """Main function to run the POC."""
    parser = argparse.ArgumentParser(description="Run Centered Set Inference POC")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B",
                        help="Name of the base model to use")
    parser.add_argument("--min_score", type=float, default=0.7,
                        help="Minimum acceptable alignment score")
    
    # Training arguments
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the alignment head even if it exists")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    
    # Execution modes
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark evaluation")
    parser.add_argument("--chatbot", action="store_true",
                        help="Run chatbot interface")
    parser.add_argument("--share", action="store_true",
                        help="Share the chatbot interface publicly")
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    generate_training_data()
    
    # Load or train model
    engine, tokenizer = load_or_train_model(args)
    
    # Run benchmark if requested
    run_benchmark(engine, tokenizer, args)
    
    # Run chatbot if requested
    run_chatbot(engine, tokenizer, args)
    
    # If no mode specified, run chatbot by default
    if not (args.benchmark or args.chatbot):
        print("No mode specified, running chatbot by default")
        run_chatbot(engine, tokenizer, args)

if __name__ == "__main__":
    main() 