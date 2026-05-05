"""Visualization utilities for meta-learning."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """Plot training history.
    
    Args:
        history: Training history dictionary.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_few_shot_comparison(results: Dict[str, Dict], save_path: Optional[Path] = None):
    """Plot few-shot learning comparison.
    
    Args:
        results: Results dictionary with method comparisons.
        save_path: Path to save the plot.
    """
    methods = list(results.keys())
    k_shots = list(results[methods[0]].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    for method in methods:
        accuracies = [results[method][k]['accuracy'] for k in k_shots]
        axes[0].plot(k_shots, accuracies, 'o-', label=method, linewidth=2, markersize=6)
    
    axes[0].set_xlabel('K-shot')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Few-shot Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 score comparison
    for method in methods:
        f1_scores = [results[method][k]['f1_macro'] for k in k_shots]
        axes[1].plot(k_shots, f1_scores, 'o-', label=method, linewidth=2, markersize=6)
    
    axes[1].set_xlabel('K-shot')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Few-shot F1 Score Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Class names for labels.
        save_path: Path to save the plot.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_task_visualization(
    support_data: torch.Tensor,
    support_labels: torch.Tensor,
    query_data: torch.Tensor,
    query_labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    n_way: int = 5,
    k_shot: int = 1,
    save_path: Optional[Path] = None
):
    """Visualize a few-shot task.
    
    Args:
        support_data: Support set data.
        support_labels: Support set labels.
        query_data: Query set data.
        query_labels: Query set labels.
        predictions: Model predictions (optional).
        n_way: Number of classes.
        k_shot: Number of support examples per class.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, n_way, figsize=(15, 6))
    
    # Support set
    for i in range(n_way):
        class_indices = torch.where(support_labels == i)[0]
        if len(class_indices) > 0:
            img = support_data[class_indices[0]].squeeze().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Support Class {i}')
            axes[0, i].axis('off')
    
    # Query set
    for i in range(n_way):
        class_indices = torch.where(query_labels == i)[0]
        if len(class_indices) > 0:
            img = query_data[class_indices[0]].squeeze().numpy()
            axes[1, i].imshow(img, cmap='gray')
            
            # Add prediction if available
            if predictions is not None:
                pred_class = predictions[class_indices[0]].item()
                correct = pred_class == i
                color = 'green' if correct else 'red'
                axes[1, i].set_title(f'Query Class {i}\nPred: {pred_class}', color=color)
            else:
                axes[1, i].set_title(f'Query Class {i}')
            
            axes[1, i].axis('off')
    
    plt.suptitle(f'{n_way}-way {k_shot}-shot Task')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_adaptation_curves(
    adaptation_results: Dict[int, Dict],
    save_path: Optional[Path] = None
):
    """Plot adaptation curves for different numbers of steps.
    
    Args:
        adaptation_results: Results from adaptation speed evaluation.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = sorted(adaptation_results.keys())
    means = [adaptation_results[step]['mean'] for step in steps]
    lowers = [adaptation_results[step]['lower'] for step in steps]
    uppers = [adaptation_results[step]['upper'] for step in steps]
    
    ax.plot(steps, means, 'b-o', label='Mean Accuracy', linewidth=2, markersize=8)
    ax.fill_between(steps, lowers, uppers, alpha=0.3, label='95% Confidence Interval')
    
    ax.set_xlabel('Adaptation Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Adaptation Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_results_table(results: Dict[str, float], save_path: Optional[Path] = None):
    """Create a results table.
    
    Args:
        results: Results dictionary.
        save_path: Path to save the table.
    """
    import pandas as pd
    
    df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    df['Value'] = df['Value'].round(4)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
