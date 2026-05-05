#!/usr/bin/env python3
"""
Complete Meta-Learning Workflow Demonstration

This script demonstrates the complete workflow from training to evaluation
of the MAML meta-learning implementation.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import set_seed, get_device, load_config, setup_logging
from data import load_mnist_data, create_few_shot_loaders
from models import create_model
from train import MAMLTrainer, BaselineTrainer
from metrics import FewShotEvaluator
from viz import plot_training_history, plot_adaptation_curves


def run_complete_workflow():
    """Run the complete meta-learning workflow."""
    
    print("🚀 Starting Complete Meta-Learning Workflow")
    print("=" * 50)
    
    # Setup
    set_seed(42)
    device = get_device()
    logger = setup_logging("./logs")
    
    print(f"✅ Setup complete - Using device: {device}")
    
    # Load data
    print("\n📊 Loading MNIST dataset...")
    train_dataset, val_dataset, test_dataset = load_mnist_data()
    
    # Create few-shot loaders
    train_loader, val_loader, test_loader = create_few_shot_loaders(
        train_dataset, val_dataset, test_dataset,
        n_way=5, k_shot=1, query_shots=15,
        batch_size=32, num_tasks_per_epoch=100
    )
    
    print("✅ Data loading complete")
    
    # Train MAML model
    print("\n🧠 Training MAML model...")
    
    # Load MAML config
    config = load_config("configs/maml_mnist.yaml")
    
    # Create model
    model = create_model(
        architecture=config['model']['architecture'],
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim']
    )
    model = model.to(device)
    
    # Create trainer
    trainer = MAMLTrainer(model, device, config)
    
    # Train (reduced epochs for demo)
    config['training']['epochs'] = 10  # Reduced for demo
    history = trainer.train(train_loader, val_loader)
    
    print("✅ MAML training complete")
    
    # Train baseline for comparison
    print("\n📈 Training baseline model...")
    
    baseline_config = load_config("configs/baseline_mnist.yaml")
    baseline_model = create_model(
        architecture=baseline_config['model']['architecture'],
        input_channels=baseline_config['model']['input_channels'],
        num_classes=baseline_config['model']['num_classes'],
        hidden_dim=baseline_config['model']['hidden_dim']
    )
    baseline_model = baseline_model.to(device)
    
    baseline_trainer = BaselineTrainer(baseline_model, device, baseline_config)
    baseline_config['training']['epochs'] = 10  # Reduced for demo
    baseline_history = baseline_trainer.train(train_loader, val_loader)
    
    print("✅ Baseline training complete")
    
    # Evaluate both models
    print("\n🔬 Evaluating models...")
    
    # MAML evaluation
    maml_evaluator = FewShotEvaluator(model, device, n_way=5, k_shot=1)
    maml_metrics = maml_evaluator.evaluate_dataset(test_loader, num_tasks=50)
    
    # Baseline evaluation
    baseline_evaluator = FewShotEvaluator(baseline_model, device, n_way=5, k_shot=1)
    baseline_metrics = baseline_evaluator.evaluate_dataset(test_loader, num_tasks=50)
    
    print("✅ Evaluation complete")
    
    # Display results
    print("\n📊 Results Summary")
    print("=" * 30)
    print(f"MAML Accuracy:     {maml_metrics['accuracy']:.4f}")
    print(f"MAML F1 Score:     {maml_metrics['f1_macro']:.4f}")
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline F1 Score: {baseline_metrics['f1_macro']:.4f}")
    
    improvement = maml_metrics['accuracy'] - baseline_metrics['accuracy']
    print(f"\nMAML Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Create assets directory
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Plot training history
    plot_training_history(history, assets_dir / "maml_training_history.png")
    plot_training_history(baseline_history, assets_dir / "baseline_training_history.png")
    
    print("✅ Visualizations saved to assets/")
    
    # Save results
    results = {
        'maml_metrics': maml_metrics,
        'baseline_metrics': baseline_metrics,
        'improvement': improvement,
        'config': config
    }
    
    with open(assets_dir / "results.yaml", 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)
    
    print("✅ Results saved to assets/results.yaml")
    
    print("\n🎉 Complete workflow finished successfully!")
    print("\nNext steps:")
    print("1. Run 'streamlit run demo/app.py' for interactive demo")
    print("2. Check assets/ for visualizations and results")
    print("3. Run 'python scripts/evaluate.py --checkpoint checkpoints/best_model.pth' for detailed evaluation")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run complete meta-learning workflow")
    parser.add_argument("--quick", action="store_true", help="Run quick demo with reduced epochs")
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Running quick demo...")
    
    run_complete_workflow()


if __name__ == "__main__":
    main()
