"""
Training module for the Centered Set Inference Engine.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os
from tqdm import tqdm

from .architecture import ValueCenter, AlignmentHead, CenteredSetInferenceEngine

class AlignmentDataset(Dataset):
    """Dataset for training the alignment head."""
    
    def __init__(self, data_path: str, tokenizer, model, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file
            tokenizer: Tokenizer for the language model
            model: Language model to extract embeddings
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Extract value names from the first item
        self.value_names = list(self.data[0]["alignment_scores"].keys())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]
        
        # Tokenize
        inputs = self.tokenizer(
            prompt + response, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use the last hidden state of the last token
            embedding = outputs.hidden_states[-1][0, -1, :]
        
        # Get alignment scores
        alignment_scores = torch.tensor([
            item["alignment_scores"][name] for name in self.value_names
        ])
        
        return {
            "embedding": embedding,
            "alignment_scores": alignment_scores
        }


def train_alignment_head(
    language_model,
    tokenizer,
    value_center: ValueCenter,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "./models",
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 5
):
    """
    Train the alignment head.
    
    Args:
        language_model: Base language model
        tokenizer: Tokenizer for the language model
        value_center: ValueCenter object defining the values
        train_data_path: Path to training data
        val_data_path: Path to validation data (optional)
        output_dir: Directory to save the trained model
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
    """
    # Create datasets
    train_dataset = AlignmentDataset(train_data_path, tokenizer, language_model)
    
    if val_data_path:
        val_dataset = AlignmentDataset(val_data_path, tokenizer, language_model)
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize alignment head
    input_dim = train_dataset[0]["embedding"].shape[0]
    alignment_head = AlignmentHead(input_dim, value_center)
    
    # Set up optimizer
    optimizer = optim.Adam(alignment_head.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        alignment_head.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = batch["embedding"]
            target_scores = batch["alignment_scores"]
            
            # Forward pass
            pred_scores = alignment_head(embeddings)
            loss = criterion(pred_scores, target_scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        
        # Validation
        if val_dataset:
            alignment_head.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch["embedding"]
                    target_scores = batch["alignment_scores"]
                    
                    pred_scores = alignment_head(embeddings)
                    loss = criterion(pred_scores, target_scores)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Save model
                torch.save(alignment_head.state_dict(), 
                          os.path.join(output_dir, "alignment_head.pt"))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # If no validation set, save the final model
    if not val_dataset:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(alignment_head.state_dict(), 
                  os.path.join(output_dir, "alignment_head.pt"))
    
    return alignment_head 