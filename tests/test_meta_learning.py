"""Test suite for meta-learning implementation."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, get_device, count_parameters
from models import SimpleCNN, ResNet18, LogisticRegression, create_model
from data import FewShotDataset, get_mnist_transforms
from metrics import MetaLearningMetrics, FewShotEvaluator


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seeds are actually set
        assert torch.initial_seed() is not None
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleCNN()
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


class TestModels:
    """Test model implementations."""
    
    def test_simple_cnn(self):
        """Test SimpleCNN model."""
        model = SimpleCNN(input_channels=1, num_classes=10, hidden_dim=128)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_resnet18(self):
        """Test ResNet18 model."""
        model = ResNet18(input_channels=1, num_classes=10)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_logistic_regression(self):
        """Test LogisticRegression model."""
        model = LogisticRegression(input_dim=784, num_classes=10)
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
    
    def test_create_model(self):
        """Test model creation function."""
        # Test SimpleCNN creation
        model = create_model("simple_cnn", input_channels=1, num_classes=10)
        assert isinstance(model, SimpleCNN)
        
        # Test ResNet18 creation
        model = create_model("resnet18", input_channels=1, num_classes=10)
        assert isinstance(model, ResNet18)
        
        # Test LogisticRegression creation
        model = create_model("logistic_regression", input_channels=1, num_classes=10)
        assert isinstance(model, LogisticRegression)
        
        # Test unknown architecture
        with pytest.raises(ValueError):
            create_model("unknown_architecture")


class TestData:
    """Test data loading and processing."""
    
    def test_mnist_transforms(self):
        """Test MNIST transforms."""
        transform = get_mnist_transforms(normalize=True)
        
        # Create dummy image
        img = torch.randn(28, 28)
        
        # Apply transforms
        transformed = transform(img.numpy())
        
        assert transformed.shape == (1, 28, 28)
        assert transformed.dtype == torch.float32
    
    def test_few_shot_dataset(self):
        """Test FewShotDataset."""
        # Create dummy dataset
        class DummyDataset:
            def __init__(self):
                self.data = []
                for i in range(100):
                    self.data.append((torch.randn(1, 28, 28), i % 10))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = DummyDataset()
        few_shot_dataset = FewShotDataset(dataset, n_way=5, k_shot=1, query_shots=3)
        
        # Test dataset length
        assert len(few_shot_dataset) == 100
        
        # Test getting a task
        task = few_shot_dataset[0]
        
        assert 'support_data' in task
        assert 'support_labels' in task
        assert 'query_data' in task
        assert 'query_labels' in task
        
        assert task['support_data'].shape[0] == 5  # n_way * k_shot
        assert task['query_data'].shape[0] == 15   # n_way * query_shots


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_meta_learning_metrics(self):
        """Test MetaLearningMetrics."""
        metrics = MetaLearningMetrics(num_classes=5)
        
        # Test reset
        metrics.reset()
        assert len(metrics.predictions) == 0
        
        # Test update
        predictions = torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.1, 0.9, 0.0]])
        targets = torch.tensor([1, 3])
        
        metrics.update(predictions, targets, 0.5)
        
        assert len(metrics.predictions) == 2
        assert len(metrics.targets) == 2
        assert len(metrics.losses) == 1
        
        # Test compute
        computed_metrics = metrics.compute()
        
        assert 'accuracy' in computed_metrics
        assert 'f1_macro' in computed_metrics
        assert 'loss' in computed_metrics
        assert 0 <= computed_metrics['accuracy'] <= 1
    
    def test_few_shot_evaluator(self):
        """Test FewShotEvaluator."""
        model = SimpleCNN(input_channels=1, num_classes=5, hidden_dim=64)
        device = torch.device('cpu')
        evaluator = FewShotEvaluator(model, device, n_way=5, k_shot=1)
        
        # Test evaluation on dummy task
        support_data = torch.randn(5, 1, 28, 28)
        support_labels = torch.tensor([0, 1, 2, 3, 4])
        query_data = torch.randn(15, 1, 28, 28)
        query_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        
        metrics = evaluator.evaluate_task(
            support_data, support_labels, query_data, query_labels
        )
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
