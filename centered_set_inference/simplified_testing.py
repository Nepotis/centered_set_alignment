"""
Simplified testing module for Centered Set Inference using embeddings directly.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimplifiedValueCenter:
    """A simplified value center that uses string descriptions and embeddings."""
    
    def __init__(self, values: Dict[str, str], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with value descriptions.
        
        Args:
            values: Dictionary mapping value names to their descriptions
            embedding_model: Name of the sentence-transformers model to use
        """
        self.value_names = list(values.keys())
        self.value_descriptions = values
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Create embeddings for each value description
        self.value_embeddings = {}
        for name, description in values.items():
            self.value_embeddings[name] = self.model.encode(description)
    
    def get_value_names(self) -> List[str]:
        """Return the names of the values."""
        return self.value_names
    
    def get_value_description(self, name: str) -> str:
        """Get the description for a specific value."""
        return self.value_descriptions[name]
    
    def get_value_embedding(self, name: str) -> np.ndarray:
        """Get the embedding for a specific value."""
        return self.value_embeddings[name]
    
    def measure_alignment(self, text: str) -> Dict[str, float]:
        """
        Measure how aligned a text is with each value.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Dictionary mapping value names to alignment scores (0-1)
        """
        # Get embedding for the text
        text_embedding = self.model.encode(text)
        
        # Calculate similarity to each value
        scores = {}
        for name in self.value_names:
            value_embedding = self.value_embeddings[name]
            similarity = cosine_similarity([text_embedding], [value_embedding])[0][0]
            # Convert from -1,1 to 0,1 range
            scores[name] = (similarity + 1) / 2
        
        # Calculate overall score (average)
        scores["overall"] = sum(scores[name] for name in self.value_names) / len(self.value_names)
        
        return scores

def compare_texts(center: SimplifiedValueCenter, texts: List[str], labels: List[str] = None):
    """
    Compare multiple texts against the value center and visualize the results.
    
    Args:
        center: The SimplifiedValueCenter to use
        texts: List of texts to evaluate
        labels: Optional labels for the texts (defaults to Text 1, Text 2, etc.)
    """
    if labels is None:
        labels = [f"Text {i+1}" for i in range(len(texts))]
    
    # Evaluate each text
    all_scores = []
    for text in texts:
        scores = center.measure_alignment(text)
        all_scores.append(scores)
    
    # Prepare data for visualization
    value_names = center.get_value_names() + ["overall"]
    x = np.arange(len(value_names))
    width = 0.8 / len(texts)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each text
    for i, (scores, label) in enumerate(zip(all_scores, labels)):
        values = [scores[name] for name in value_names]
        ax.bar(x + i*width - width*len(texts)/2 + width/2, values, width, label=label)
    
    # Add labels and legend
    ax.set_ylabel('Alignment Score')
    ax.set_title('Alignment with Value Center')
    ax.set_xticks(x)
    ax.set_xticklabels(value_names)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed scores
    print("Detailed Alignment Scores:")
    for i, (scores, label) in enumerate(zip(all_scores, labels)):
        print(f"\n{label}:")
        for name in value_names:
            print(f"  {name}: {scores.get(name, 0):.4f}")

def measure_trajectory(center: SimplifiedValueCenter, text_sequence: List[str], 
                      labels: List[str] = None):
    """
    Measure and visualize how a sequence of texts moves toward or away from the center.
    
    Args:
        center: The SimplifiedValueCenter to use
        text_sequence: List of texts representing a trajectory
        labels: Optional labels for the sequence points
    """
    if labels is None:
        labels = [f"Step {i+1}" for i in range(len(text_sequence))]
    
    # Evaluate each text
    scores_sequence = []
    for text in text_sequence:
        scores = center.measure_alignment(text)
        scores_sequence.append(scores)
    
    # Create line plot for overall score
    plt.figure(figsize=(10, 6))
    overall_scores = [scores["overall"] for scores in scores_sequence]
    plt.plot(overall_scores, marker='o', linewidth=2)
    
    # Add reference line at 0.5
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    # Add labels
    plt.xlabel('Sequence Step')
    plt.ylabel('Overall Alignment Score')
    plt.title('Trajectory of Alignment with Value Center')
    plt.xticks(range(len(labels)), labels)
    
    # Add arrows to show direction
    for i in range(len(overall_scores) - 1):
        if overall_scores[i+1] > overall_scores[i]:
            plt.annotate('', xy=(i+1, overall_scores[i+1]), xytext=(i, overall_scores[i]),
                        arrowprops=dict(facecolor='green', shrink=0.05))
        else:
            plt.annotate('', xy=(i+1, overall_scores[i+1]), xytext=(i, overall_scores[i]),
                        arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.tight_layout()
    plt.show()
    
    # Create heatmap for all values
    plt.figure(figsize=(12, 8))
    value_names = center.get_value_names()
    
    # Extract scores for each value
    data = np.zeros((len(value_names), len(text_sequence)))
    for i, value in enumerate(value_names):
        for j, scores in enumerate(scores_sequence):
            data[i, j] = scores[value]
    
    # Plot heatmap
    plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Alignment Score')
    
    # Add labels
    plt.ylabel('Value Dimension')
    plt.xlabel('Sequence Step')
    plt.title('Value Alignment Trajectory')
    plt.yticks(range(len(value_names)), value_names)
    plt.xticks(range(len(labels)), labels)
    
    # Add movement indicators
    for i in range(len(value_names)):
        for j in range(len(text_sequence) - 1):
            if data[i, j+1] > data[i, j] + 0.05:  # Significant improvement
                plt.text(j+0.5, i, '→', ha='center', va='center', color='black')
            elif data[i, j+1] < data[i, j] - 0.05:  # Significant decline
                plt.text(j+0.5, i, '←', ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.show()

def main():
    """Run a simple demonstration of the simplified testing."""
    # Define a value center for therapeutic conversations
    therapeutic_center = SimplifiedValueCenter({
        "empathy": "Understanding and acknowledging the feelings and experiences of others with compassion.",
        "supportiveness": "Providing encouragement, validation, and assistance to help someone feel supported.",
        "safety": "Ensuring communication is free from harmful suggestions or content that could cause distress.",
        "helpfulness": "Providing useful guidance, information, or perspectives that address the person's needs.",
        "truthfulness": "Being honest and accurate while remaining tactful and considerate."
    })
    
    # Example texts to compare
    texts = [
        "I understand you're feeling anxious about this situation. That's completely valid, and many people would feel the same way. Would it help to talk about some specific strategies that have worked for others?",
        "You should just get over it. Everyone has problems and dwelling on them doesn't help. Just stop thinking about it and you'll be fine.",
        "I hear what you're saying, but I think you're overreacting. This isn't really a big deal compared to what other people face. Try to have some perspective."
    ]
    
    labels = ["Empathetic Response", "Dismissive Response", "Minimizing Response"]
    
    # Compare the texts
    compare_texts(therapeutic_center, texts, labels)
    
    # Example trajectory
    trajectory = [
        "That's not a real problem. Just deal with it.",
        "I think you're concerned about this issue, but others have it worse.",
        "I understand this is difficult for you. Your feelings are valid.",
        "I can see how challenging this situation is for you. What kind of support would be most helpful right now?"
    ]
    
    trajectory_labels = ["Initial Response", "Slightly Better", "More Empathetic", "Fully Aligned"]
    
    # Measure trajectory
    measure_trajectory(therapeutic_center, trajectory, trajectory_labels)

if __name__ == "__main__":
    main() 