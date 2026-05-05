"""Data loading and preprocessing utilities for meta-learning."""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
import learn2learn as l2l
from pathlib import Path


class FewShotDataset(Dataset):
    """Dataset wrapper for few-shot learning tasks."""
    
    def __init__(
        self, 
        dataset: Dataset, 
        n_way: int = 5, 
        k_shot: int = 1, 
        query_shots: int = 15,
        num_tasks: int = 100
    ):
        """Initialize few-shot dataset.
        
        Args:
            dataset: Base dataset (e.g., MNIST).
            n_way: Number of classes per task.
            k_shot: Number of support examples per class.
            query_shots: Number of query examples per class.
            num_tasks: Number of tasks to generate.
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_shots = query_shots
        self.num_tasks = num_tasks
        
        # Group data by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        
    def __len__(self) -> int:
        """Return number of tasks."""
        return self.num_tasks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a few-shot task.
        
        Args:
            idx: Task index.
            
        Returns:
            Dictionary containing support and query sets.
        """
        # Sample n_way classes
        selected_classes = np.random.choice(
            self.classes, size=self.n_way, replace=False
        )
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            # Get indices for this class
            class_indices = self.class_to_indices[class_label]
            
            # Sample support and query examples
            sampled_indices = np.random.choice(
                class_indices, 
                size=self.k_shot + self.query_shots, 
                replace=False
            )
            
            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:]
            
            # Add support examples
            for support_idx in support_indices:
                data, _ = self.dataset[support_idx]
                support_data.append(data)
                support_labels.append(class_idx)
            
            # Add query examples
            for query_idx in query_indices:
                data, _ = self.dataset[query_idx]
                query_data.append(data)
                query_labels.append(class_idx)
        
        return {
            'support_data': torch.stack(support_data),
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_data': torch.stack(query_data),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'n_way': self.n_way,
            'k_shot': self.k_shot
        }


def get_mnist_transforms(normalize: bool = True) -> transforms.Compose:
    """Get MNIST data transforms.
    
    Args:
        normalize: Whether to normalize data.
        
    Returns:
        Composed transforms.
    """
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    return transforms.Compose(transform_list)


def load_mnist_data(
    data_dir: str = "./data",
    download: bool = True,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    normalize: bool = True
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load MNIST dataset with train/val/test splits.
    
    Args:
        data_dir: Directory to store data.
        download: Whether to download data.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        normalize: Whether to normalize data.
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    transform = get_mnist_transforms(normalize)
    
    # Load full dataset
    full_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=download, 
        transform=transform
    )
    
    # Split into train/val
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Load test dataset
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=download, 
        transform=transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_few_shot_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    n_way: int = 5,
    k_shot: int = 1,
    query_shots: int = 15,
    batch_size: int = 32,
    num_tasks_per_epoch: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create few-shot data loaders.
    
    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        n_way: Number of classes per task.
        k_shot: Number of support examples per class.
        query_shots: Number of query examples per class.
        batch_size: Batch size for training.
        num_tasks_per_epoch: Number of tasks per epoch.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create few-shot datasets
    train_few_shot = FewShotDataset(
        train_dataset, n_way, k_shot, query_shots, num_tasks_per_epoch
    )
    val_few_shot = FewShotDataset(
        val_dataset, n_way, k_shot, query_shots, num_tasks_per_epoch // 4
    )
    test_few_shot = FewShotDataset(
        test_dataset, n_way, k_shot, query_shots, num_tasks_per_epoch // 4
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_few_shot, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_few_shot, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_few_shot, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for few-shot tasks.
    
    Args:
        batch: List of task dictionaries.
        
    Returns:
        Batched task dictionary.
    """
    # Stack all support and query data
    support_data = torch.cat([task['support_data'] for task in batch], dim=0)
    support_labels = torch.cat([task['support_labels'] for task in batch], dim=0)
    query_data = torch.cat([task['query_data'] for task in batch], dim=0)
    query_labels = torch.cat([task['query_labels'] for task in batch], dim=0)
    
    return {
        'support_data': support_data,
        'support_labels': support_labels,
        'query_data': query_data,
        'query_labels': query_labels,
        'n_way': batch[0]['n_way'],
        'k_shot': batch[0]['k_shot']
    }
