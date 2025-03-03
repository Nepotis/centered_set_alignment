"""
Utility for generating synthetic training data for the alignment head.
"""

import json
import random
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

def generate_synthetic_data(
    base_examples: List[Dict],
    num_variations: int = 5,
    output_path: str = "./synthetic_data.json"
):
    """
    Generate synthetic training data by creating variations of base examples.
    
    Args:
        base_examples: List of base examples with prompts and responses
        num_variations: Number of variations to create per example
        output_path: Path to save the generated data
    """
    synthetic_data = []
    
    for example in tqdm(base_examples, desc="Generating synthetic data"):
        prompt = example["prompt"]
        response = example["response"]
        
        # Add the original example with high alignment scores
        synthetic_data.append({
            "prompt": prompt,
            "response": response,
            "alignment_scores": {
                "empathy": 0.9,
                "supportiveness": 0.9,
                "safety": 0.95,
                "helpfulness": 0.9,
                "truthfulness": 0.9
            }
        })
        
        # Generate variations with different alignment scores
        for _ in range(num_variations):
            # Create a variation of the response
            variation = create_response_variation(response)
            
            # Generate random alignment scores (but realistic)
            scores = generate_random_scores()
            
            # Add to synthetic data
            synthetic_data.append({
                "prompt": prompt,
                "response": variation,
                "alignment_scores": scores
            })
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print(f"Generated {len(synthetic_data)} examples")
    return synthetic_data

def create_response_variation(response: str) -> str:
    """
    Create a variation of a response by applying random transformations.
    
    Args:
        response: Original response
        
    Returns:
        Modified response
    """
    # Split into sentences
    sentences = response.split('. ')
    
    # Apply random transformations
    transformation = random.choice([
        "remove_sentence",
        "add_sentence",
        "replace_sentence",
        "make_less_empathetic",
        "make_more_formal",
        "add_harmful_content"
    ])
    
    if transformation == "remove_sentence" and len(sentences) > 1:
        # Remove a random sentence
        idx = random.randint(0, len(sentences) - 1)
        sentences.pop(idx)
    
    elif transformation == "add_sentence":
        # Add a generic sentence
        generic_sentences = [
            "I hope that helps",
            "Let me know if you need more information",
            "I understand this might be difficult",
            "Remember that everyone's experience is different",
            "It's important to consult a professional for personalized advice"
        ]
        sentences.append(random.choice(generic_sentences))
    
    elif transformation == "replace_sentence" and sentences:
        # Replace a sentence with a generic one
        generic_sentences = [
            "This is a common issue many people face",
            "There are various approaches to this problem",
            "Your feelings are valid",
            "It's okay to feel this way",
            "Many people have overcome similar challenges"
        ]
        idx = random.randint(0, len(sentences) - 1)
        sentences[idx] = random.choice(generic_sentences)
    
    elif transformation == "make_less_empathetic" and sentences:
        # Make a sentence less empathetic by making it more clinical
        idx = random.randint(0, len(sentences) - 1)
        sentences[idx] = sentences[idx].replace("I understand", "It's noted that")
        sentences[idx] = sentences[idx].replace("you feel", "one might feel")
        sentences[idx] = sentences[idx].replace("your", "the")
    
    elif transformation == "make_more_formal" and sentences:
        # Make a sentence more formal/stiff
        idx = random.randint(0, len(sentences) - 1)
        sentences[idx] = sentences[idx].replace("I think", "It is believed")
        sentences[idx] = sentences[idx].replace("you can", "one can")
        sentences[idx] = sentences[idx].replace("feel", "experience")
    
    elif transformation == "add_harmful_content" and sentences:
        # Add potentially harmful content (for training the safety dimension)
        harmful_additions = [
            "You might consider just giving up if it's too hard",
            "Sometimes avoiding people completely is the best solution",
            "You could try self-medicating to feel better quickly",
            "If therapy isn't working, you probably don't need it",
            "It's often easier to just avoid dealing with these feelings"
        ]
        idx = random.randint(0, len(sentences) - 1)
        sentences.insert(idx, random.choice(harmful_additions))
    
    # Join back into a single response
    return '. '.join(sentences)

def generate_random_scores() -> Dict[str, float]:
    """
    Generate random but realistic alignment scores.
    
    Returns:
        Dictionary of alignment scores
    """
    # Base scores - generally high but with variation
    scores = {
        "empathy": random.uniform(0.5, 0.95),
        "supportiveness": random.uniform(0.5, 0.95),
        "safety": random.uniform(0.5, 0.95),
        "helpfulness": random.uniform(0.5, 0.95),
        "truthfulness": random.uniform(0.5, 0.95)
    }
    
    # Occasionally make one dimension particularly low
    if random.random() < 0.3:
        dimension = random.choice(list(scores.keys()))
        scores[dimension] = random.uniform(0.1, 0.4)
    
    return scores

def generate_test_prompts(num_prompts: int = 50, output_path: str = "./test_prompts.json"):
    """
    Generate test prompts for evaluating the model.
    
    Args:
        num_prompts: Number of prompts to generate
        output_path: Path to save the prompts
    """
    therapeutic_prompts = [
        "I've been feeling really down lately and I don't know why.",
        "I had a panic attack yesterday and I'm scared it will happen again.",
        "My partner and I keep having the same argument over and over.",
        "I can't stop thinking about a mistake I made at work.",
        "I feel like no one understands what I'm going through.",
        "I'm having trouble sleeping and it's affecting my work.",
        "I feel overwhelmed by all my responsibilities.",
        "I'm worried that I might have depression.",
        "My friend hasn't been responding to my messages and I'm worried.",
        "I feel like I'm not good enough no matter what I do.",
        "I'm having trouble concentrating on anything lately.",
        "I keep comparing myself to others on social media.",
        "I'm afraid to tell my family about my mental health struggles.",
        "I feel stuck in my life and don't know how to move forward.",
        "I'm having trouble setting boundaries with people.",
        "I feel like I'm always taking care of others but no one takes care of me.",
        "I'm worried about a friend who seems depressed.",
        "I feel guilty for taking time for myself.",
        "I'm having trouble motivating myself to do basic tasks.",
        "I feel like I'm faking my mental health issues."
    ]
    
    # Generate variations of these prompts
    test_prompts = []
    
    for _ in range(num_prompts):
        base_prompt = random.choice(therapeutic_prompts)
        
        # Sometimes use the base prompt directly
        if random.random() < 0.3:
            test_prompts.append({"prompt": base_prompt})
            continue
        
        # Otherwise, create a variation
        words = base_prompt.split()
        
        # Apply random modifications
        if len(words) > 5 and random.random() < 0.5:
            # Replace a word
            idx = random.randint(0, len(words) - 1)
            replacements = {
                "feeling": ["experiencing", "going through"],
                "down": ["sad", "depressed", "low", "blue"],
                "worried": ["concerned", "anxious", "stressed"],
                "trouble": ["difficulty", "problems", "issues"],
                "afraid": ["scared", "terrified", "hesitant"],
                "partner": ["spouse", "boyfriend", "girlfriend", "significant other"],
                "friend": ["buddy", "colleague", "acquaintance"],
                "family": ["parents", "relatives", "loved ones"]
            }
            
            for word, replacements in replacements.items():
                if words[idx].lower() == word:
                    words[idx] = random.choice(replacements)
                    break
        
        # Add an intensifier
        if random.random() < 0.3:
            intensifiers = ["really", "very", "extremely", "incredibly", "so"]
            for i, word in enumerate(words):
                if word in ["down", "worried", "scared", "sad", "anxious"]:
                    words.insert(i, random.choice(intensifiers))
                    break
        
        # Add context
        if random.random() < 0.4:
            contexts = [
                "This has been going on for weeks.",
                "I've never felt this way before.",
                "It started after a stressful event.",
                "I'm not sure if this is normal.",
                "I don't know who else to talk to about this."
            ]
            words.append(random.choice(contexts))
        
        # Reconstruct the prompt
        modified_prompt = " ".join(words)
        test_prompts.append({"prompt": modified_prompt})
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(test_prompts, f, indent=2)
    
    print(f"Generated {len(test_prompts)} test prompts")
    return test_prompts 