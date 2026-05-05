#!/usr/bin/env python3
"""Evaluation script for meta-learning implementation."""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, load_config, setup_logging
from data import load_mnist_data, create_few_shot_loaders
from models import create_model
from metrics import FewShotEvaluator, compute_confidence_intervals


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate meta-learning model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/maml_mnist.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, mps, cpu, auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="Number of test tasks to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./assets",
        help="Directory for output files"
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        device: Device to load checkpoint on.
        
    Returns:
        Checkpoint data.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def evaluate_adaptation_speed(
    evaluator: FewShotEvaluator,
    test_loader: torch.utils.data.DataLoader,
    adaptation_steps: list,
    num_tasks: int = 50
) -> dict:
    """Evaluate model adaptation speed.
    
    Args:
        evaluator: Model evaluator.
        test_loader: Test data loader.
        adaptation_steps: List of adaptation steps to test.
        num_tasks: Number of tasks to evaluate.
        
    Returns:
        Dictionary of adaptation speed results.
    """
    results = {}
    
    for steps in adaptation_steps:
        print(f"Evaluating with {steps} adaptation steps...")
        
        task_scores = []
        for i, batch in enumerate(test_loader):
            if i >= num_tasks:
                break
                
            # Process each task in the batch
            batch_size = batch['support_data'].size(0)
            for j in range(batch_size):
                task_metrics = evaluator.evaluate_task(
                    batch['support_data'][j],
                    batch['support_labels'][j],
                    batch['query_data'][j],
                    batch['query_labels'][j],
                    adaptation_steps=steps
                )
                task_scores.append(task_metrics['accuracy'])
        
        mean_score, lower_bound, upper_bound = compute_confidence_intervals(task_scores)
        results[steps] = {
            'mean': mean_score,
            'lower': lower_bound,
            'upper': upper_bound,
            'scores': task_scores
        }
    
    return results


def plot_adaptation_speed(results: dict, output_dir: Path):
    """Plot adaptation speed results.
    
    Args:
        results: Adaptation speed results.
        output_dir: Output directory for plots.
    """
    steps = list(results.keys())
    means = [results[step]['mean'] for step in steps]
    lowers = [results[step]['lower'] for step in steps]
    uppers = [results[step]['upper'] for step in steps]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, 'b-o', label='Mean Accuracy')
    plt.fill_between(steps, lowers, uppers, alpha=0.3, label='95% Confidence Interval')
    
    plt.xlabel('Adaptation Steps')
    plt.ylabel('Accuracy')
    plt.title('Model Adaptation Speed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'adaptation_speed.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_few_shot_performance(results: dict, output_dir: Path):
    """Plot few-shot performance comparison.
    
    Args:
        results: Performance results.
        output_dir: Output directory for plots.
    """
    k_shots = list(results.keys())
    accuracies = [results[k]['accuracy'] for k in k_shots]
    f1_scores = [results[k]['f1_macro'] for k in k_shots]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy plot
    ax1.plot(k_shots, accuracies, 'b-o', label='Accuracy')
    ax1.set_xlabel('K-shot')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Few-shot Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 score plot
    ax2.plot(k_shots, f1_scores, 'r-o', label='F1 Score')
    ax2.set_xlabel('K-shot')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Few-shot F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'few_shot_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging("./logs")
    logger.info(f"Starting evaluation with config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Setup device
    if args.device == "auto":
        device = get_device(
            use_cuda=config['device']['cuda'],
            use_mps=config['device']['mps']
        )
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_dataset, val_dataset, test_dataset = load_mnist_data(
        data_dir=config['data']['data_dir'],
        download=config['data']['download'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        normalize=config['data']['normalize']
    )
    
    # Create test loader
    _, _, test_loader = create_few_shot_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        n_way=config['task']['n_way'],
        k_shot=config['task']['k_shot'],
        query_shots=config['task']['query_shots'],
        batch_size=1,  # Evaluate one task at a time
        num_tasks_per_epoch=args.num_tasks
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        architecture=config['model']['architecture'],
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim']
    )
    
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Best validation score: {checkpoint['best_val_score']:.4f}")
    
    # Create evaluator
    evaluator = FewShotEvaluator(
        model=model,
        device=device,
        n_way=config['task']['n_way'],
        k_shot=config['task']['k_shot']
    )
    
    # Basic evaluation
    logger.info("Running basic evaluation...")
    test_metrics = evaluator.evaluate_dataset(test_loader, num_tasks=args.num_tasks)
    
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Adaptation speed evaluation
    logger.info("Evaluating adaptation speed...")
    adaptation_steps = [1, 3, 5, 10, 15]
    adaptation_results = evaluate_adaptation_speed(
        evaluator, test_loader, adaptation_steps, num_tasks=50
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    logger.info("Creating plots...")
    plot_adaptation_speed(adaptation_results, output_dir)
    
    # Save detailed results
    results = {
        'config': config,
        'checkpoint_info': {
            'epoch': checkpoint['epoch'],
            'best_val_score': checkpoint['best_val_score']
        },
        'test_metrics': test_metrics,
        'adaptation_speed': adaptation_results
    }
    
    results_path = output_dir / 'evaluation_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Plots saved to {output_dir}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
