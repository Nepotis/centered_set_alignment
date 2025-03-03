"""
Architecture for Centered Set Inference POC.
This module defines the core components of the CSIE framework.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

class ValueCenter:
    """Represents the 'center' of values that responses should align with."""
    
    def __init__(self, values: Dict[str, float]):
        """
        Initialize a value center with named dimensions and their weights.
        
        Args:
            values: Dictionary mapping value names (e.g., "helpfulness") to their weights
        """
        self.values = values
        self.value_names = list(values.keys())
        self.weights = np.array([values[name] for name in self.value_names])
        # Normalize weights to sum to 1
        self.weights = self.weights / np.sum(self.weights)
        
    def get_value_dimensions(self) -> int:
        """Return the number of value dimensions."""
        return len(self.values)
    
    def get_value_names(self) -> List[str]:
        """Return the names of the values."""
        return self.value_names
    
    def get_weights(self) -> np.ndarray:
        """Return the weights of each value dimension."""
        return self.weights
    
    def adjust_weight(self, value_name: str, new_weight: float):
        """
        Adjust the weight of a specific value dimension.
        
        Args:
            value_name: The name of the value to adjust
            new_weight: The new weight for this value
        """
        if value_name not in self.values:
            raise ValueError(f"Value '{value_name}' not found in center")
        
        self.values[value_name] = new_weight
        # Recalculate normalized weights
        self.weights = np.array([self.values[name] for name in self.value_names])
        self.weights = self.weights / np.sum(self.weights)


class AlignmentHead(nn.Module):
    """
    Neural network head that predicts alignment scores for each value dimension.
    This is attached to a language model to evaluate outputs.
    """
    
    def __init__(self, input_dim: int, value_center: ValueCenter):
        """
        Initialize the alignment head.
        
        Args:
            input_dim: Dimension of the input embeddings from the language model
            value_center: The ValueCenter object defining the values to align with
        """
        super().__init__()
        self.value_center = value_center
        self.value_dim = value_center.get_value_dimensions()
        
        # Simple MLP for alignment prediction
        self.alignment_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.value_dim),
            nn.Sigmoid()  # Outputs between 0 and 1 for each value dimension
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict alignment scores for each value dimension.
        
        Args:
            embeddings: Hidden state embeddings from the language model
            
        Returns:
            Tensor of alignment scores, one per value dimension
        """
        return self.alignment_net(embeddings)
    
    def get_overall_alignment(self, value_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate overall alignment score using weighted average.
        
        Args:
            value_scores: Tensor of individual value dimension scores
            
        Returns:
            Tensor containing the overall alignment score
        """
        weights = torch.tensor(self.value_center.get_weights(), 
                              device=value_scores.device)
        return torch.sum(value_scores * weights, dim=-1)


class CenteredSetInferenceEngine:
    """
    Main engine that evaluates responses and guides generation toward the center.
    """
    
    def __init__(self, 
                 language_model,
                 alignment_head: AlignmentHead,
                 tokenizer,
                 min_acceptable_score: float = 0.7):
        """
        Initialize the inference engine.
        
        Args:
            language_model: The base language model
            alignment_head: The alignment head for evaluating responses
            tokenizer: Tokenizer for the language model
            min_acceptable_score: Minimum acceptable overall alignment score
        """
        self.language_model = language_model
        self.alignment_head = alignment_head
        self.tokenizer = tokenizer
        self.min_acceptable_score = min_acceptable_score
        self.value_center = alignment_head.value_center
    
    def evaluate_response(self, 
                         prompt: str, 
                         response: str) -> Dict[str, float]:
        """
        Evaluate a response for alignment with the center values.
        
        Args:
            prompt: The user prompt
            response: The model's response to evaluate
            
        Returns:
            Dictionary with alignment scores for each value and overall
        """
        # Tokenize the input
        inputs = self.tokenizer(prompt + response, return_tensors="pt")
        
        # Get the model's hidden states
        with torch.no_grad():
            outputs = self.language_model(**inputs, output_hidden_states=True)
            # Use the last hidden state of the last token as representation
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            
            # Get alignment scores
            value_scores = self.alignment_head(last_hidden_state).squeeze()
            overall_score = self.alignment_head.get_overall_alignment(value_scores)
        
        # Convert to dictionary
        scores = {name: score.item() for name, score in 
                 zip(self.value_center.get_value_names(), value_scores)}
        scores["overall"] = overall_score.item()
        
        return scores
    
    def generate_aligned_response(self, 
                                 prompt: str, 
                                 max_attempts: int = 2) -> Tuple[str, Dict[str, float]]:
        """
        Generate a response aligned with the center values.
        
        Args:
            prompt: The user prompt
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Tuple of (final_response, alignment_scores)
        """
        # First attempt
        response = self._generate_response(prompt)
        scores = self.evaluate_response(prompt, response)
        
        # If alignment is sufficient, return
        if scores["overall"] >= self.min_acceptable_score:
            return response, scores
        
        # Otherwise, try to improve (up to max_attempts)
        for attempt in range(max_attempts - 1):
            # Create a guidance prompt based on low-scoring values
            low_values = [name for name, score in scores.items() 
                         if name != "overall" and score < self.min_acceptable_score]
            
            if not low_values:
                # If no specific low values but overall is low, try general improvement
                guidance = "Please provide a more helpful and appropriate response."
            else:
                guidance = f"Please revise your response to be more {', '.join(low_values)}."
            
            # Generate improved response
            improved_prompt = f"{prompt}\n\nYour previous response: {response}\n\n{guidance}"
            improved_response = self._generate_response(improved_prompt)
            
            # Evaluate the improved response
            improved_scores = self.evaluate_response(prompt, improved_response)
            
            # If improved, update response and scores
            if improved_scores["overall"] > scores["overall"]:
                response = improved_response
                scores = improved_scores
                
                # If good enough, return
                if scores["overall"] >= self.min_acceptable_score:
                    break
        
        return response, scores
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.language_model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 