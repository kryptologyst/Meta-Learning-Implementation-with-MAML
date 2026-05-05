#!/usr/bin/env python3
"""Training script for meta-learning implementation."""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, load_config, setup_logging
from data import load_mnist_data, create_few_shot_loaders
from models import create_model
from train import MAMLTrainer, BaselineTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train meta-learning model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/maml_mnist.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["maml", "baseline"],
        default="maml",
        help="Type of model to train"
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
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logging"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed != 42:
        config['seed'] = args.seed
    if args.log_dir != "./logs":
        config['logging']['log_dir'] = args.log_dir
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Random seed: {config['seed']}")
    
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
    
    # Create few-shot loaders
    logger.info("Creating few-shot data loaders...")
    train_loader, val_loader, test_loader = create_few_shot_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        n_way=config['task']['n_way'],
        k_shot=config['task']['k_shot'],
        query_shots=config['task']['query_shots'],
        batch_size=config['training']['batch_size'],
        num_tasks_per_epoch=config['task']['num_tasks_per_epoch']
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
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    if args.model_type == "maml":
        trainer = MAMLTrainer(
            model=model,
            device=device,
            config=config,
            log_dir=config['logging']['log_dir']
        )
    else:
        trainer = BaselineTrainer(
            model=model,
            device=device,
            config=config,
            log_dir=config['logging']['log_dir']
        )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {trainer.best_val_score:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_metrics['f1_macro']:.4f}")
    
    # Save final results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'best_val_score': trainer.best_val_score
    }
    
    results_path = Path(config['logging']['log_dir']) / 'training_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
