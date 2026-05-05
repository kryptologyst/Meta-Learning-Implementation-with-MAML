#!/usr/bin/env python3
"""
Modernized Meta-learning Implementation with MAML

This is a clean, reproducible implementation of Model-Agnostic Meta-Learning (MAML)
for few-shot learning, featuring proper error handling, type hints, and device support.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import learn2learn as l2l
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from utils import set_seed, get_device, setup_logging
from models import SimpleCNN
from data import FewShotDataset, create_few_shot_loaders
from metrics import FewShotEvaluator


def main():
    """Main function demonstrating MAML meta-learning."""
    
    # Setup logging
    logger = setup_logging("./logs")
    logger.info("Starting MAML meta-learning demonstration")
    
    # Set random seed for reproducibility
    set_seed(42)
    logger.info("Random seed set to 42")
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load MNIST dataset
    logger.info("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    mnist_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Split dataset into training and validation sets
    train_dataset, val_dataset = random_split(mnist_dataset, [55000, 5000])
    
    # Create few-shot loaders
    logger.info("Creating few-shot data loaders...")
    train_loader, val_loader, test_loader = create_few_shot_loaders(
        train_dataset,
        val_dataset,
        mnist_dataset,  # Use full dataset as test
        n_way=5,
        k_shot=1,
        query_shots=15,
        batch_size=32,
        num_tasks_per_epoch=100
    )
    
    # Create model
    logger.info("Creating SimpleCNN model...")
    model = SimpleCNN(input_channels=1, num_classes=10, hidden_dim=128)
    model = model.to(device)
    
    # Create MAML wrapper
    logger.info("Creating MAML wrapper...")
    maml = l2l.algorithms.MAML(model, lr=0.01)
    
    # Setup optimizer
    optimizer = optim.Adam(maml.parameters(), lr=0.001)
    
    # Training loop
    logger.info("Starting meta-training...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            support_data = batch['support_data'].to(device)
            support_labels = batch['support_labels'].to(device)
            query_data = batch['query_data'].to(device)
            query_labels = batch['query_labels'].to(device)
            
            # Meta-learning step
            learner = maml.clone()
            
            # Inner loop: adapt to support set
            for step in range(5):  # 5 inner steps
                support_logits = learner(support_data)
                support_loss = nn.functional.cross_entropy(support_logits, support_labels)
                learner.adapt(support_loss)
            
            # Outer loop: evaluate on query set
            query_logits = learner(query_data)
            query_loss = nn.functional.cross_entropy(query_logits, query_labels)
            
            # Backward pass
            optimizer.zero_grad()
            query_loss.backward()
            optimizer.step()
            
            total_loss += query_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {query_loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # Test the meta-learned model
    logger.info("Evaluating meta-learned model...")
    model.eval()
    
    # Create evaluator
    evaluator = FewShotEvaluator(model, device, n_way=5, k_shot=1)
    
    # Evaluate on validation set
    val_metrics = evaluator.evaluate_dataset(val_loader, num_tasks=50)
    
    logger.info("Validation Results:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Test on a few tasks
    logger.info("Testing on individual tasks...")
    test_tasks = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if test_tasks >= 10:  # Test on 10 tasks
                break
            
            # Process each task in the batch
            batch_size = batch['support_data'].size(0)
            for i in range(batch_size):
                # Adapt model to support set
                adapted_model = evaluator._adapt_to_support(
                    batch['support_data'][i],
                    batch['support_labels'][i],
                    adaptation_steps=5
                )
                
                # Evaluate on query set
                query_logits = adapted_model(batch['query_data'][i])
                _, predicted = torch.max(query_logits, 1)
                
                correct = (predicted == batch['query_labels'][i]).sum().item()
                total = batch['query_labels'][i].size(0)
                
                correct_predictions += correct
                total_predictions += total
                test_tasks += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    logger.info(f'Accuracy on test tasks: {100 * accuracy:.2f}%')
    
    logger.info("MAML demonstration completed successfully!")


if __name__ == "__main__":
    main()
