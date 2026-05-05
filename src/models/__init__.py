"""Model definitions for meta-learning."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """Simple CNN architecture for few-shot learning."""
    
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 10, 
        hidden_dim: int = 128
    ):
        """Initialize SimpleCNN.
        
        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension size.
        """
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.fc1 = nn.Linear(128 * 3 * 3, hidden_dim)  # 28x28 -> 3x3 after 3 pooling layers
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNet18(nn.Module):
    """ResNet-18 adapted for few-shot learning."""
    
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 10, 
        pretrained: bool = False
    ):
        """Initialize ResNet-18.
        
        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
        """
        super(ResNet18, self).__init__()
        
        # Load pretrained ResNet-18
        self.backbone = resnet18(pretrained=pretrained)
        
        # Modify first layer for different input channels
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Modify final layer for different number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        return self.backbone(x)


class LogisticRegression(nn.Module):
    """Logistic regression baseline for few-shot learning."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        """Initialize logistic regression.
        
        Args:
            input_dim: Input dimension (28*28 for MNIST).
            num_classes: Number of output classes.
        """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        x = x.view(x.size(0), -1)
        return self.linear(x)


class MAMLModel(nn.Module):
    """Wrapper for MAML meta-learning."""
    
    def __init__(
        self, 
        model: nn.Module, 
        inner_lr: float = 0.01, 
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """Initialize MAML model.
        
        Args:
            model: Base model to meta-learn.
            inner_lr: Inner loop learning rate.
            inner_steps: Number of inner loop steps.
            first_order: Whether to use first-order approximation.
        """
        super(MAMLModel, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
    def forward(
        self, 
        support_data: torch.Tensor, 
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with meta-learning.
        
        Args:
            support_data: Support set data.
            support_labels: Support set labels.
            query_data: Query set data.
            query_labels: Query set labels.
            
        Returns:
            Tuple of (query_logits, loss).
        """
        # Clone model parameters for inner loop
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop: adapt to support set
        for step in range(self.inner_steps):
            # Forward pass on support set
            support_logits = self._forward_with_weights(support_data, fast_weights)
            support_loss = F.cross_entropy(support_logits, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                support_loss, 
                fast_weights.values(), 
                create_graph=not self.first_order,
                retain_graph=True
            )
            
            # Update fast weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.inner_lr * grad
        
        # Forward pass on query set with adapted weights
        query_logits = self._forward_with_weights(query_data, fast_weights)
        query_loss = F.cross_entropy(query_logits, query_labels)
        
        return query_logits, query_loss
    
    def _forward_with_weights(
        self, 
        x: torch.Tensor, 
        weights: dict
    ) -> torch.Tensor:
        """Forward pass with custom weights.
        
        Args:
            x: Input tensor.
            weights: Dictionary of weights.
            
        Returns:
            Output logits.
        """
        # This is a simplified version - in practice, you'd need to implement
        # a more sophisticated weight application mechanism
        return self.model(x)


def create_model(
    architecture: str,
    input_channels: int = 1,
    num_classes: int = 10,
    hidden_dim: int = 128,
    **kwargs
) -> nn.Module:
    """Create model based on architecture name.
    
    Args:
        architecture: Model architecture name.
        input_channels: Number of input channels.
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension size.
        **kwargs: Additional model arguments.
        
    Returns:
        Created model.
    """
    if architecture == "simple_cnn":
        return SimpleCNN(input_channels, num_classes, hidden_dim)
    elif architecture == "resnet18":
        return ResNet18(input_channels, num_classes, **kwargs)
    elif architecture == "logistic_regression":
        return LogisticRegression(input_channels * 28 * 28, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
